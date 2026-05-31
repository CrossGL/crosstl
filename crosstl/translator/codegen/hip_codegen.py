"""
CrossGL to HIP Code Generator

This module provides code generation functionality to convert CrossGL AST to HIP source code.
HIP (Heterogeneous-Compute Interface for Portability) is AMD's CUDA-compatible runtime API
for GPU programming.
"""

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ASTNode,
    BreakNode,
    CbufferNode,
    ContinueNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    LiteralNode,
    MatchNode,
    MemberAccessNode,
    MeshOpNode,
    PrimitiveType,
    RangeNode,
    RayQueryOpNode,
    RayTracingOpNode,
    ReturnNode,
    ShaderNode,
    StructNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
)
from .array_utils import parse_array_type
from .match_utils import (
    generate_match_expression_assignment,
    generate_ordered_conditional_match,
    generate_switch_match,
    infer_match_expression_result_type,
    is_switch_lowerable_match,
)
from .resource_arrays import format_array_declarator
from .resource_diagnostics import ResourceDiagnosticMixin
from .resource_query import ResourceQueryMixin
from .stage_utils import normalize_stage_name, stage_matches
from .vector_arithmetic import VectorArithmeticMixin

HIP_WAVE_OP_ARITIES = {
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
    "WaveActiveBallot": 1,
    "WaveActiveCountBits": 1,
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

HIP_WAVE_PREDICATE_ARGUMENT_OPS = {
    "WaveActiveAllTrue",
    "WaveActiveAnyTrue",
    "WaveActiveBallot",
    "WaveActiveCountBits",
    "WavePrefixCountBits",
    "WaveMultiPrefixCountBits",
}

HIP_WAVE_UINT_RESULT_OPS = {
    "WaveGetLaneCount",
    "WaveGetLaneIndex",
    "WaveActiveCountBits",
    "WavePrefixCountBits",
    "WaveMultiPrefixCountBits",
}

HIP_WAVE_BOOL_RESULT_OPS = {
    "WaveIsFirstLane",
    "WaveActiveAllTrue",
    "WaveActiveAnyTrue",
    "WaveActiveAllEqual",
}

HIP_WAVE_UVEC4_RESULT_OPS = {
    "WaveActiveBallot",
    "WaveMatch",
}


class HipCodeGen(VectorArithmeticMixin, ResourceQueryMixin, ResourceDiagnosticMixin):
    """Emit HIP source from the shared CrossGL translator AST."""

    resource_diagnostic_backend = "HIP"
    synchronization_builtins = {
        "barrier",
        "groupMemoryBarrier",
        "memoryBarrier",
        "memoryBarrierShared",
        "memoryBarrierBuffer",
        "memoryBarrierImage",
        "allMemoryBarrier",
        "deviceMemoryBarrier",
        "workgroupBarrier",
    }
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
        "SamplerComparisonState": "sampler",
    }

    def __init__(self):
        """Initialize HIP type maps and per-generation visitor state."""
        self.indent_level = 0
        self.code_lines = []
        self.current_function = None
        self.current_function_return_type = None
        self.variable_counter = 0
        self.match_temp_variable_index = 0
        self.variable_types = {}
        self.image_resource_accesses = {}
        self.buffer_resource_accesses = {}
        self.glsl_buffer_block_accesses = {}
        self.glsl_buffer_block_layouts = {}
        self.hip_resource_binding_cursors = {}
        self.hip_used_resource_bindings = {}
        self.struct_member_types = {}
        self.struct_member_semantics = {}
        self.function_return_types = {}
        self.helper_functions = {}
        self.query_resource_names = set()
        self.query_metadata_function_params = {}
        self.query_functions_by_name = {}
        self.structured_buffer_length_names = set()
        self.structured_buffer_length_function_params = {}
        self.current_structured_buffer_length_parameters = {}
        self.current_function_name = None
        self.current_stage_name = None
        self.resource_query_info_required = False
        self.assignment_lhs_depth = 0

        # CrossGL to HIP type mapping
        self.type_map = {
            # Basic types
            "int": "int",
            "float": "float",
            "double": "double",
            "bool": "bool",
            "void": "void",
            "uint": "unsigned int",
            # Vector types
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "vec2<f32>": "float2",
            "vec3<f32>": "float3",
            "vec4<f32>": "float4",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "vec2<i32>": "int2",
            "vec3<i32>": "int3",
            "vec4<i32>": "int4",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            "vec2<u32>": "uint2",
            "vec3<u32>": "uint3",
            "vec4<u32>": "uint4",
            "dvec2": "double2",
            "dvec3": "double3",
            "dvec4": "double4",
            "vec2<f64>": "double2",
            "vec3<f64>": "double3",
            "vec4<f64>": "double4",
            "bvec2": "uchar2",
            "bvec3": "uchar3",
            "bvec4": "uchar4",
            "vec2<bool>": "uchar2",
            "vec3<bool>": "uchar3",
            "vec4<bool>": "uchar4",
            "bool2": "uchar2",
            "bool3": "uchar3",
            "bool4": "uchar4",
            # Matrix types
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "mat2x2": "float2x2",
            "mat2x3": "float2x3",
            "mat2x4": "float2x4",
            "mat3x2": "float3x2",
            "mat3x3": "float3x3",
            "mat3x4": "float3x4",
            "mat4x2": "float4x2",
            "mat4x3": "float4x3",
            "mat4x4": "float4x4",
            "dmat2": "double2x2",
            "dmat3": "double3x3",
            "dmat4": "double4x4",
            "dmat2x2": "double2x2",
            "dmat2x3": "double2x3",
            "dmat2x4": "double2x4",
            "dmat3x2": "double3x2",
            "dmat3x3": "double3x3",
            "dmat3x4": "double3x4",
            "dmat4x2": "double4x2",
            "dmat4x3": "double4x3",
            "dmat4x4": "double4x4",
            # Texture/resource types
            "sampler": "hipTextureObject_t",
            "sampler1D": "hipTextureObject_t",
            "sampler1DArray": "hipTextureObject_t",
            "sampler2D": "hipTextureObject_t",
            "sampler3D": "hipTextureObject_t",
            "samplerCube": "hipTextureObject_t",
            "sampler2DArray": "hipTextureObject_t",
            "sampler2DShadow": "hipTextureObject_t",
            "sampler2DArrayShadow": "hipTextureObject_t",
            "samplerCubeShadow": "hipTextureObject_t",
            "samplerCubeArray": "hipTextureObject_t",
            "samplerCubeArrayShadow": "hipTextureObject_t",
            "sampler2DMS": "hipTextureObject_t",
            "sampler2DMSArray": "hipTextureObject_t",
            "image1D": "hipSurfaceObject_t",
            "image1DArray": "hipSurfaceObject_t",
            "image2D": "hipSurfaceObject_t",
            "image3D": "hipSurfaceObject_t",
            "imageCube": "hipSurfaceObject_t",
            "imageCubeArray": "hipSurfaceObject_t",
            "image2DArray": "hipSurfaceObject_t",
            "image2DMS": "hipSurfaceObject_t",
            "image2DMSArray": "hipSurfaceObject_t",
            "iimage1D": "hipSurfaceObject_t",
            "iimage1DArray": "hipSurfaceObject_t",
            "iimage2D": "hipSurfaceObject_t",
            "iimage3D": "hipSurfaceObject_t",
            "iimageCube": "hipSurfaceObject_t",
            "iimageCubeArray": "hipSurfaceObject_t",
            "iimage2DArray": "hipSurfaceObject_t",
            "iimage2DMS": "hipSurfaceObject_t",
            "iimage2DMSArray": "hipSurfaceObject_t",
            "uimage1D": "hipSurfaceObject_t",
            "uimage1DArray": "hipSurfaceObject_t",
            "uimage2D": "hipSurfaceObject_t",
            "uimage3D": "hipSurfaceObject_t",
            "uimageCube": "hipSurfaceObject_t",
            "uimageCubeArray": "hipSurfaceObject_t",
            "uimage2DArray": "hipSurfaceObject_t",
            "uimage2DMS": "hipSurfaceObject_t",
            "uimage2DMSArray": "hipSurfaceObject_t",
            "buffer": "hipDeviceptr_t",
            "accelerationStructureEXT": "CglRayTracingAccelerationStructure",
            "AccelerationStructure": "CglRayTracingAccelerationStructure",
            "acceleration_structure": "CglRayTracingAccelerationStructure",
            "RaytracingAccelerationStructure": "CglRayTracingAccelerationStructure",
            "RayTracingAccelerationStructure": "CglRayTracingAccelerationStructure",
            "RayDesc": "CglRayDesc",
            "RayQuery": "CglRayQuery",
            "BuiltInTriangleIntersectionAttributes": (
                "CglBuiltInTriangleIntersectionAttributes"
            ),
        }

        # CrossGL to HIP function mapping
        self.function_map = {
            # Math functions
            "sin": "sinf",
            "cos": "cosf",
            "tan": "tanf",
            "asin": "asinf",
            "acos": "acosf",
            "atan": "atanf",
            "atan2": "atan2f",
            "sinh": "sinhf",
            "cosh": "coshf",
            "tanh": "tanhf",
            "exp": "expf",
            "exp2": "exp2f",
            "log": "logf",
            "log2": "log2f",
            "sqrt": "sqrtf",
            "inversesqrt": "rsqrtf",
            "pow": "powf",
            "abs": "fabsf",
            "floor": "floorf",
            "ceil": "ceilf",
            "round": "roundf",
            "trunc": "truncf",
            "mod": "fmodf",
            "min": "fminf",
            "max": "fmaxf",
            "clamp": "fmaxf(fminf",  # Special handling needed
            "step": "step",
            "smoothstep": "smoothstep",
            # Vector functions
            "length": "length",
            "distance": "distance",
            "dot": "dot",
            "cross": "cross",
            "normalize": "normalize",
            "reflect": "reflect",
            "refract": "refract",
            # Geometric functions
            "faceforward": "faceforward",
            # Vector constructors
            "vec2": "make_float2",
            "vec3": "make_float3",
            "vec4": "make_float4",
            "float2": "make_float2",
            "float3": "make_float3",
            "float4": "make_float4",
            "vec2<f32>": "make_float2",
            "vec3<f32>": "make_float3",
            "vec4<f32>": "make_float4",
            "ivec2": "make_int2",
            "ivec3": "make_int3",
            "ivec4": "make_int4",
            "int2": "make_int2",
            "int3": "make_int3",
            "int4": "make_int4",
            "vec2<i32>": "make_int2",
            "vec3<i32>": "make_int3",
            "vec4<i32>": "make_int4",
            "uvec2": "make_uint2",
            "uvec3": "make_uint3",
            "uvec4": "make_uint4",
            "uint2": "make_uint2",
            "uint3": "make_uint3",
            "uint4": "make_uint4",
            "vec2<u32>": "make_uint2",
            "vec3<u32>": "make_uint3",
            "vec4<u32>": "make_uint4",
            "dvec2": "make_double2",
            "dvec3": "make_double3",
            "dvec4": "make_double4",
            "double2": "make_double2",
            "double3": "make_double3",
            "double4": "make_double4",
            "vec2<f64>": "make_double2",
            "vec3<f64>": "make_double3",
            "vec4<f64>": "make_double4",
            "bvec2": "make_uchar2",
            "bvec3": "make_uchar3",
            "bvec4": "make_uchar4",
            "uchar2": "make_uchar2",
            "uchar3": "make_uchar3",
            "uchar4": "make_uchar4",
            "vec2<bool>": "make_uchar2",
            "vec3<bool>": "make_uchar3",
            "vec4<bool>": "make_uchar4",
            "bool2": "make_uchar2",
            "bool3": "make_uchar3",
            "bool4": "make_uchar4",
            # Matrix constructors
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "mat2x2": "float2x2",
            "mat2x3": "float2x3",
            "mat2x4": "float2x4",
            "mat3x2": "float3x2",
            "mat3x3": "float3x3",
            "mat3x4": "float3x4",
            "mat4x2": "float4x2",
            "mat4x3": "float4x3",
            "mat4x4": "float4x4",
            "dmat2": "double2x2",
            "dmat3": "double3x3",
            "dmat4": "double4x4",
            "dmat2x2": "double2x2",
            "dmat2x3": "double2x3",
            "dmat2x4": "double2x4",
            "dmat3x2": "double3x2",
            "dmat3x3": "double3x3",
            "dmat3x4": "double3x4",
            "dmat4x2": "double4x2",
            "dmat4x3": "double4x3",
            "dmat4x4": "double4x4",
            # Atomic operations
            "atomicAdd": "atomicAdd",
            "atomicSub": "atomicSub",
            "atomicMin": "atomicMin",
            "atomicMax": "atomicMax",
            "atomicAnd": "atomicAnd",
            "atomicOr": "atomicOr",
            "atomicXor": "atomicXor",
            "atomicExchange": "atomicExch",
            "atomicCompareExchange": "atomicCAS",
            "atomicCompSwap": "atomicCAS",
            # Synchronization
            "barrier": "__syncthreads",
            "groupMemoryBarrier": "__threadfence_block",
            "memoryBarrier": "__threadfence",
            "memoryBarrierShared": "__threadfence_block",
            "memoryBarrierBuffer": "__threadfence",
            "memoryBarrierImage": "__threadfence",
            "allMemoryBarrier": "__threadfence",
            "deviceMemoryBarrier": "__threadfence",
            "workgroupBarrier": "__syncthreads",
            # Ray tracing placeholders
            "RayDesc": "CglRayDesc",
            "RayQuery": "CglRayQuery",
            "BuiltInTriangleIntersectionAttributes": (
                "CglBuiltInTriangleIntersectionAttributes"
            ),
            "TraceRay": "cgl_trace_ray",
            "CallShader": "cgl_call_shader",
            "ReportHit": "cgl_report_hit",
            "IgnoreHit": "cgl_ignore_hit",
            "AcceptHitAndEndSearch": "cgl_accept_hit_and_end_search",
            # Texture functions
            "texture": "tex2D",
            "textureLod": "tex2DLod",
            "textureGrad": "tex2DGrad",
        }

        # Built-in variable mappings
        self.builtin_map = {
            "gl_LocalInvocationID.x": "threadIdx.x",
            "gl_LocalInvocationID.y": "threadIdx.y",
            "gl_LocalInvocationID.z": "threadIdx.z",
            "gl_WorkGroupID.x": "blockIdx.x",
            "gl_WorkGroupID.y": "blockIdx.y",
            "gl_WorkGroupID.z": "blockIdx.z",
            "gl_WorkGroupSize.x": "blockDim.x",
            "gl_WorkGroupSize.y": "blockDim.y",
            "gl_WorkGroupSize.z": "blockDim.z",
            "gl_NumWorkGroups.x": "gridDim.x",
            "gl_NumWorkGroups.y": "gridDim.y",
            "gl_NumWorkGroups.z": "gridDim.z",
            "gl_GlobalInvocationID.x": "(blockIdx.x * blockDim.x + threadIdx.x)",
            "gl_GlobalInvocationID.y": "(blockIdx.y * blockDim.y + threadIdx.y)",
            "gl_GlobalInvocationID.z": "(blockIdx.z * blockDim.z + threadIdx.z)",
            "gl_LocalInvocationIndex": (
                "(threadIdx.z * blockDim.y * blockDim.x + "
                "threadIdx.y * blockDim.x + threadIdx.x)"
            ),
            "SV_GroupThreadID": "make_uint3(threadIdx.x, threadIdx.y, threadIdx.z)",
            "SV_GroupThreadID.x": "threadIdx.x",
            "SV_GroupThreadID.y": "threadIdx.y",
            "SV_GroupThreadID.z": "threadIdx.z",
            "SV_GroupID": "make_uint3(blockIdx.x, blockIdx.y, blockIdx.z)",
            "SV_GroupID.x": "blockIdx.x",
            "SV_GroupID.y": "blockIdx.y",
            "SV_GroupID.z": "blockIdx.z",
            "SV_DispatchThreadID": (
                "make_uint3((blockIdx.x * blockDim.x + threadIdx.x), "
                "(blockIdx.y * blockDim.y + threadIdx.y), "
                "(blockIdx.z * blockDim.z + threadIdx.z))"
            ),
            "SV_DispatchThreadID.x": "(blockIdx.x * blockDim.x + threadIdx.x)",
            "SV_DispatchThreadID.y": "(blockIdx.y * blockDim.y + threadIdx.y)",
            "SV_DispatchThreadID.z": "(blockIdx.z * blockDim.z + threadIdx.z)",
            "SV_GroupIndex": (
                "(threadIdx.z * blockDim.y * blockDim.x + "
                "threadIdx.y * blockDim.x + threadIdx.x)"
            ),
        }

    def generate(self, node: ASTNode) -> str:
        """Generate complete HIP source for a CrossGL AST."""
        self.code_lines = []
        self.indent_level = 0
        self.validate_supported_stage_types(node)
        self.variable_types = {}
        self.image_resource_accesses = {}
        self.buffer_resource_accesses = {}
        self.glsl_buffer_block_accesses = {}
        self.glsl_buffer_block_layouts = {}
        self.hip_resource_binding_cursors = {}
        self.hip_used_resource_bindings = {}
        self.current_function_return_type = None
        self.match_temp_variable_index = 0
        self.struct_member_types = {}
        self.struct_member_semantics = {}
        self.function_return_types = self.collect_function_return_types(node)
        self.helper_functions = {}
        self.resource_query_info_required = False
        self.assignment_lhs_depth = 0
        self.current_stage_name = None
        (
            self.query_resource_names,
            self.query_metadata_function_params,
        ) = self.collect_resource_query_requirements(node)
        (
            self.structured_buffer_length_names,
            self.structured_buffer_length_function_params,
        ) = self.collect_structured_buffer_length_requirements(node)
        self.query_functions_by_name = {
            getattr(func, "name", None): func
            for func in self.query_collect_functions(node)
        }
        self.query_functions_by_name = {
            name: func for name, func in self.query_functions_by_name.items() if name
        }
        self.reserve_explicit_hip_resource_bindings(node)

        self.add_includes()
        if self.has_geometry_stage(node):
            self.add_generated_code(self.generate_hip_geometry_stream_helpers())
            self.add_line()
        if self.has_tessellation_stage(node):
            self.add_generated_code(self.generate_hip_tessellation_patch_helpers())
            self.add_line()
        if self.has_mesh_task_stage(node):
            self.add_generated_code(self.generate_hip_mesh_task_helpers())
            self.add_line()
        self.visit(node)
        self.insert_helper_functions()

        return "\n".join(self.code_lines)

    def unsupported_stage_types(self):
        return set()

    def hip_ray_stage_names(self):
        return {
            "ray_any_hit",
            "ray_callable",
            "ray_closest_hit",
            "ray_generation",
            "ray_intersection",
            "ray_miss",
        }

    def hip_ray_stage_metadata_name(self, stage_name):
        return {
            "ray_any_hit": "any_hit",
            "ray_callable": "callable",
            "ray_closest_hit": "closest_hit",
            "ray_generation": "ray_generation",
            "ray_intersection": "intersection",
            "ray_miss": "miss",
        }.get(stage_name, stage_name)

    def has_geometry_stage(self, ast_node, target_stage=None):
        if not stage_matches(target_stage, "geometry"):
            return False

        return self.has_stage(ast_node, "geometry")

    def has_tessellation_stage(self, ast_node, target_stage=None):
        if not (
            stage_matches(target_stage, "tessellation_control")
            or stage_matches(target_stage, "tessellation_evaluation")
        ):
            return False

        return self.has_stage(ast_node, "tessellation_control") or self.has_stage(
            ast_node, "tessellation_evaluation"
        )

    def hip_mesh_task_stage_names(self):
        return {"mesh", "task", "object", "amplification"}

    def has_mesh_task_stage(self, ast_node, target_stage=None):
        stage_names = self.hip_mesh_task_stage_names()
        if target_stage is not None and not any(
            stage_matches(target_stage, stage_name) for stage_name in stage_names
        ):
            return False

        return any(self.has_stage(ast_node, stage_name) for stage_name in stage_names)

    def has_stage(self, ast_node, stage_name):
        for stage_type in getattr(ast_node, "stages", {}) or {}:
            if normalize_stage_name(stage_type) == stage_name:
                return True

        for func in getattr(ast_node, "functions", []) or []:
            qualifiers = list(getattr(func, "qualifiers", []) or [])
            qualifier = getattr(func, "qualifier", None)
            if qualifier:
                qualifiers.append(qualifier)
            if any(
                normalize_stage_name(qualifier) == stage_name
                for qualifier in qualifiers
            ):
                return True

        return False

    def generate_hip_geometry_stream_helpers(self):
        return (
            "template <typename T>\n"
            "struct CglHipPointStream {\n"
            "    __device__ void Append(const T& value) { (void)value; }\n"
            "    __device__ void RestartStrip() { }\n"
            "};\n\n"
            "template <typename T>\n"
            "struct CglHipLineStream {\n"
            "    __device__ void Append(const T& value) { (void)value; }\n"
            "    __device__ void RestartStrip() { }\n"
            "};\n\n"
            "template <typename T>\n"
            "struct CglHipTriangleStream {\n"
            "    __device__ void Append(const T& value) { (void)value; }\n"
            "    __device__ void RestartStrip() { }\n"
            "};\n"
        )

    def generate_hip_tessellation_patch_helpers(self):
        return (
            "template <typename T, int N>\n"
            "struct CglHipInputPatch {\n"
            "    T data[N];\n"
            "    __device__ const T& operator[](int index) const { return data[index]; }\n"
            "    __device__ T& operator[](int index) { return data[index]; }\n"
            "};\n\n"
            "template <typename T, int N>\n"
            "struct CglHipOutputPatch {\n"
            "    T data[N];\n"
            "    __device__ const T& operator[](int index) const { return data[index]; }\n"
            "    __device__ T& operator[](int index) { return data[index]; }\n"
            "};\n"
        )

    def generate_hip_mesh_task_helpers(self):
        return (
            "__device__ inline void cgl_hip_set_mesh_output_counts(\n"
            "    unsigned int vertex_count, unsigned int primitive_count)\n"
            "{\n"
            "    (void)vertex_count;\n"
            "    (void)primitive_count;\n"
            "}\n\n"
            "__device__ inline void cgl_hip_dispatch_mesh(\n"
            "    unsigned int group_count_x,\n"
            "    unsigned int group_count_y,\n"
            "    unsigned int group_count_z)\n"
            "{\n"
            "    (void)group_count_x;\n"
            "    (void)group_count_y;\n"
            "    (void)group_count_z;\n"
            "}\n\n"
            "template <typename Payload>\n"
            "__device__ inline void cgl_hip_dispatch_mesh(\n"
            "    unsigned int group_count_x,\n"
            "    unsigned int group_count_y,\n"
            "    unsigned int group_count_z,\n"
            "    const Payload& payload)\n"
            "{\n"
            "    (void)group_count_x;\n"
            "    (void)group_count_y;\n"
            "    (void)group_count_z;\n"
            "    (void)payload;\n"
            "}\n"
        )

    def validate_supported_stage_types(self, ast_node, target_stage=None):
        unsupported_stages = set()
        for stage_type in getattr(ast_node, "stages", {}) or {}:
            stage_name = normalize_stage_name(stage_type)
            if stage_name in self.unsupported_stage_types() and stage_matches(
                target_stage, stage_name
            ):
                unsupported_stages.add(stage_name)

        for func in getattr(ast_node, "functions", []) or []:
            for qualifier in getattr(func, "qualifiers", []) or []:
                stage_name = normalize_stage_name(qualifier)
                if stage_name in self.unsupported_stage_types() and stage_matches(
                    target_stage, stage_name
                ):
                    unsupported_stages.add(stage_name)

        if unsupported_stages:
            stage_list = ", ".join(sorted(unsupported_stages))
            raise ValueError(f"HIP output does not support stage type(s): {stage_list}")

    def add_includes(self):
        """Emit the standard HIP runtime include block."""
        self.code_lines.extend(
            [
                "#include <hip/hip_runtime.h>",
                "#include <hip/hip_runtime_api.h>",
                "#include <hip/math_functions.h>",
                "#include <hip/device_functions.h>",
                "",
            ]
        )

    def indent(self) -> str:
        """Return whitespace for the current indentation level."""
        return "    " * self.indent_level

    def add_line(self, line: str = ""):
        """Append one HIP output line using the current indentation level."""
        if line:
            self.code_lines.append(self.indent() + line)
        else:
            self.code_lines.append("")

    def add_generated_code(self, code: str):
        """Append pre-indented generated code to the output stream."""
        for line in code.rstrip("\n").splitlines():
            self.code_lines.append(line.rstrip())

    def visit(self, node: ASTNode) -> str:
        """Dispatch an AST node to its HIP visitor method."""
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ASTNode) -> str:
        """Raise a clear error for unsupported AST nodes."""
        raise NotImplementedError(
            f"Code generation not implemented for {type(node).__name__}"
        )

    def visit_ShaderNode(self, node: ShaderNode) -> str:
        """Render a full shader/program AST as a HIP translation unit."""
        structs = getattr(node, "structs", [])
        for struct in structs:
            self.visit(struct)

        global_vars = getattr(node, "global_variables", [])
        for var in global_vars:
            self.visit(var)

        cbuffers = getattr(node, "cbuffers", [])
        for cbuffer in cbuffers:
            self.visit(cbuffer)

        functions = getattr(node, "functions", [])
        for func in functions:
            self.visit(func)

        # Handle shader stages (new AST structure)
        if hasattr(node, "stages") and node.stages:
            emitted_local_functions = set()
            stage_entry_name_counts = self.stage_entry_name_counts(node.stages)
            for stage_type, stage in node.stages.items():
                if hasattr(stage, "entry_point"):
                    # Set the stage type context for proper qualifier handling
                    stage_name = (
                        str(stage_type).split(".")[-1].lower()
                        if hasattr(stage_type, "name")
                        else str(stage_type).lower()
                    )

                    # Temporarily set qualifier for compute stages
                    if stage_name == "compute" or "compute" in stage_name:
                        # Set the function qualifier to compute for proper __global__ generation
                        if hasattr(stage.entry_point, "qualifiers"):
                            if "compute" not in stage.entry_point.qualifiers:
                                stage.entry_point.qualifiers.append("compute")
                        else:
                            stage.entry_point.qualifiers = ["compute"]

                    for func in getattr(stage, "local_functions", []):
                        if id(func) in emitted_local_functions:
                            continue
                        self.visit(func)
                        emitted_local_functions.add(id(func))

                    saved_stage_name = self.current_stage_name
                    self.current_stage_name = stage_name
                    saved_override = getattr(
                        self, "current_stage_entry_function_name", None
                    )
                    self.current_stage_entry_function_name = (
                        self.stage_entry_function_name(
                            stage_name, stage.entry_point, stage_entry_name_counts
                        )
                    )
                    self.visit(stage.entry_point)
                    self.current_stage_entry_function_name = saved_override
                    self.current_stage_name = saved_stage_name

        return ""

    def stage_entry_name_counts(self, stages):
        counts = {}
        for stage in stages.values():
            entry_point = getattr(stage, "entry_point", None)
            name = getattr(entry_point, "name", None)
            if name:
                counts[name] = counts.get(name, 0) + 1
        return counts

    def stage_entry_function_name(self, stage_name, entry_point, name_counts):
        name = getattr(entry_point, "name", None)
        if not name:
            return name
        if stage_name in self.hip_ray_stage_names() and name == "main":
            return f"{stage_name}_{name}"
        if (
            stage_name
            in {"geometry", "tessellation_control", "tessellation_evaluation"}
            | self.hip_mesh_task_stage_names()
            and name == "main"
        ):
            return f"{stage_name}_{name}"
        if name_counts.get(name, 0) > 1:
            return f"{stage_name}_{name}"
        return name

    def visit_FunctionNode(self, node: FunctionNode) -> str:
        """Render a CrossGL function or compute entry point as HIP code."""
        saved_variable_types = self.variable_types.copy()
        saved_image_resource_accesses = self.image_resource_accesses.copy()
        saved_buffer_resource_accesses = self.buffer_resource_accesses.copy()
        saved_glsl_buffer_block_accesses = self.glsl_buffer_block_accesses.copy()
        saved_glsl_buffer_block_layouts = self.glsl_buffer_block_layouts.copy()
        self.current_function = node.name
        saved_current_function_return_type = self.current_function_return_type
        saved_current_function_name = self.current_function_name
        saved_structured_buffer_length_parameters = (
            self.current_structured_buffer_length_parameters
        )
        self.current_function_name = node.name
        self.current_structured_buffer_length_parameters = {}

        qualifiers = []
        if hasattr(node, "qualifiers") and node.qualifiers:
            for qualifier in node.qualifiers:
                qualifier_name = normalize_stage_name(qualifier)
                if qualifier_name in {"kernel", "compute"}:
                    qualifiers.append("__global__")
                elif qualifier_name == "device":
                    qualifiers.append("__device__")
                else:
                    qualifiers.append("__device__")
        elif hasattr(node, "qualifier") and node.qualifier:
            qualifier_name = normalize_stage_name(node.qualifier)
            if qualifier_name in {"kernel", "compute"}:
                qualifiers.append("__global__")
            elif qualifier_name == "device":
                qualifiers.append("__device__")
            else:
                qualifiers.append("__device__")
        else:
            qualifiers.append("__device__")

        if hasattr(node, "return_type"):
            self.current_function_return_type = node.return_type
            return_type = self.map_type(node.return_type)
        else:
            self.current_function_return_type = "void"
            return_type = "void"

        stage_name = self.function_stage_name(node)
        return_semantic = self.semantic_from_node(node)
        self.validate_hip_return_semantic(
            stage_name, self.current_function_return_type, return_semantic
        )
        self.validate_hip_struct_return_semantics(
            stage_name, self.current_function_return_type
        )

        param_list = getattr(node, "parameters", getattr(node, "params", []))
        self.validate_hip_stage_parameter_semantics(stage_name, param_list)
        if stage_name == "geometry":
            self.validate_hip_geometry_stage(node, param_list)
        if stage_name in {"tessellation_control", "tessellation_evaluation"}:
            self.validate_hip_tessellation_stage(node, stage_name)
        if stage_name in self.hip_mesh_task_stage_names():
            self.validate_hip_mesh_task_stage(node, stage_name)
        param_declarations = []
        for param in param_list:
            param_declarations.append(self.visit_parameter(param))
            param_type = self.get_parameter_type(param)
            param_name = getattr(param, "name", getattr(param, "param_name", None))
            metadata_param = self.query_metadata_parameter(param_name, param_type)
            if metadata_param:
                param_declarations.append(metadata_param)
            length_param = self.structured_buffer_length_parameter(
                node.name, param_name, param_type
            )
            if length_param:
                self.current_structured_buffer_length_parameters[param_name] = (
                    self.structured_buffer_length_name(param_name)
                )
                param_declarations.append(length_param)
            counter_param = self.structured_buffer_counter_parameter(
                param_name, param_type
            )
            if counter_param:
                param_declarations.append(counter_param)
        params = ", ".join(param_declarations)

        qualifier_str = " ".join(qualifiers)
        function_name = getattr(self, "current_stage_entry_function_name", None)
        if not function_name:
            function_name = node.name
        signature = f"{qualifier_str} {return_type} {function_name}({params})"

        if stage_name in self.hip_ray_stage_names():
            self.add_line(
                "// CrossGL ray stage: "
                f"{self.hip_ray_stage_metadata_name(stage_name)}"
            )
        if stage_name == "geometry":
            self.add_generated_code(
                self.generate_hip_geometry_stage_comments(node, param_list)
            )
        if stage_name in {"tessellation_control", "tessellation_evaluation"}:
            self.add_generated_code(
                self.generate_hip_tessellation_stage_comments(node, stage_name)
            )
        if stage_name in self.hip_mesh_task_stage_names():
            self.add_generated_code(
                self.generate_hip_mesh_task_stage_comments(node, stage_name)
            )
        if return_semantic:
            self.add_line(
                f"// CrossGL return semantic: {self.map_semantic(return_semantic)}"
            )
        if stage_name:
            for param in param_list:
                param_semantic = self.semantic_from_node(param)
                if param_semantic:
                    param_name = getattr(
                        param, "name", getattr(param, "param_name", "param")
                    )
                    self.add_line(
                        f"// CrossGL parameter semantic: {param_name}: "
                        f"{self.map_semantic(param_semantic)}"
                    )
        self.add_line(signature)

        body = getattr(node, "body", [])
        if body:
            self.add_line("{")
            self.indent_level += 1
            self.emit_body(body)

            self.indent_level -= 1
            self.add_line("}")
        else:
            self.add_line(";")

        self.add_line()
        self.current_function = None
        self.current_function_return_type = saved_current_function_return_type
        self.variable_types = saved_variable_types
        self.image_resource_accesses = saved_image_resource_accesses
        self.buffer_resource_accesses = saved_buffer_resource_accesses
        self.glsl_buffer_block_accesses = saved_glsl_buffer_block_accesses
        self.glsl_buffer_block_layouts = saved_glsl_buffer_block_layouts
        self.current_function_name = saved_current_function_name
        self.current_structured_buffer_length_parameters = (
            saved_structured_buffer_length_parameters
        )
        return ""

    def visit_parameter(self, param) -> str:
        if isinstance(param, dict):
            param_type = param.get("type", "int")
            param_name = param.get("name", "param")
        else:
            if hasattr(param, "param_type"):
                param_type = param.param_type
            elif hasattr(param, "vtype"):
                param_type = param.vtype
            else:
                param_type = "int"

            param_name = getattr(param, "name", "param")

        param_type = self.resource_type_with_access(param_type, param)
        geometry_stream_declaration = self.format_hip_geometry_stream_parameter(
            param_type,
            param_name,
        )
        if geometry_stream_declaration is not None:
            self.register_variable_type(
                param_name,
                self.hip_geometry_stream_mapped_type(param_type),
                param,
            )
            return geometry_stream_declaration

        self.register_variable_type(param_name, param_type, param)
        declaration = self.format_typed_declarator(param_type, param_name)
        return self.apply_resource_memory_qualifiers_to_declaration(
            declaration, param_name, param
        )

    def semantic_from_node(self, node):
        semantic = getattr(node, "semantic", None)
        if semantic:
            return semantic
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if self.is_semantic_name(attr_name):
                return attr_name
        return None

    def is_semantic_name(self, name):
        if name is None:
            return False
        name = str(name)
        upper_name = name.upper()
        if name.startswith("gl_"):
            return True
        if upper_name in {
            "BINORMAL",
            "CALLABLE_DATA",
            "CALLABLEDATAEXT",
            "CALLABLEDATAINEXT",
            "COLOR",
            "HIT_ATTRIBUTE",
            "HITATTRIBUTEEXT",
            "NORMAL",
            "PAYLOAD",
            "POSITION",
            "RAYPAYLOADEXT",
            "RAYPAYLOADINEXT",
            "SV_DEPTH",
            "SV_DISPATCHRAYSDIMENSIONS",
            "SV_DISPATCHRAYSINDEX",
            "SV_DISPATCHTHREADID",
            "SV_DRAWID",
            "SV_GROUPID",
            "SV_GROUPINDEX",
            "SV_GROUPTHREADID",
            "SV_INSTANCEID",
            "SV_ISFRONTFACE",
            "SV_POSITION",
            "SV_PRIMITIVEID",
            "SV_SAMPLEINDEX",
            "SV_STARTINSTANCELOCATION",
            "SV_STARTVERTEXLOCATION",
            "SV_TARGET",
            "SV_VERTEXID",
            "TANGENT",
            "TEXCOORD",
        }:
            return True
        for prefix in ("COLOR", "SV_TARGET", "TEXCOORD"):
            if upper_name.startswith(prefix) and upper_name[len(prefix) :].isdigit():
                return True
        return False

    def map_semantic(self, semantic):
        if semantic is None:
            return ""
        semantic_name = str(semantic)
        lower_name = semantic_name.lower()
        upper_name = semantic_name.upper()
        if lower_name == "gl_position" or upper_name == "SV_POSITION":
            return "position"
        if lower_name == "gl_fragdepth" or upper_name == "SV_DEPTH":
            return "depth(any)"
        if lower_name == "gl_fragcolor":
            return "target(0)"
        if lower_name.startswith("gl_fragcolor"):
            suffix = lower_name[len("gl_fragcolor") :]
            if suffix.isdigit():
                return f"target({suffix})"
        if upper_name == "SV_TARGET":
            return "target(0)"
        if upper_name.startswith("SV_TARGET"):
            suffix = upper_name[len("SV_TARGET") :]
            if suffix.isdigit():
                return f"target({suffix})"
        if upper_name == "TEXCOORD":
            return "texcoord"
        if (
            upper_name.startswith("TEXCOORD")
            and upper_name[len("TEXCOORD") :].isdigit()
        ):
            return f"texcoord({upper_name[len('TEXCOORD'):]})"
        if upper_name == "COLOR":
            return "color"
        if upper_name.startswith("COLOR") and upper_name[len("COLOR") :].isdigit():
            return f"color({upper_name[len('COLOR'):]})"
        generic_semantics = {
            "BINORMAL": "binormal",
            "NORMAL": "normal",
            "POSITION": "position",
            "TANGENT": "tangent",
            "gl_FragCoord": "position",
            "gl_FrontFacing": "front_facing",
            "gl_PointCoord": "point_coord",
            "gl_VertexID": "vertex_id",
            "gl_InstanceID": "instance_id",
            "gl_BaseVertex": "start_vertex_location",
            "gl_BaseInstance": "start_instance_location",
            "gl_DrawID": "draw_id",
            "gl_WorkGroupID": "workgroup_id",
            "gl_LocalInvocationID": "local_invocation_id",
            "gl_GlobalInvocationID": "global_invocation_id",
            "gl_LocalInvocationIndex": "local_invocation_index",
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
            "SV_DispatchThreadID": "global_invocation_id",
            "SV_DrawID": "draw_id",
            "SV_GroupID": "workgroup_id",
            "SV_GroupIndex": "local_invocation_index",
            "SV_GroupThreadID": "local_invocation_id",
            "SV_InstanceID": "instance_id",
            "SV_IsFrontFace": "front_facing",
            "SV_PrimitiveID": "primitive_id",
            "SV_SampleIndex": "sample_index",
            "SV_StartInstanceLocation": "start_instance_location",
            "SV_StartVertexLocation": "start_vertex_location",
            "SV_VertexID": "vertex_id",
        }
        return generic_semantics.get(semantic_name, semantic_name)

    def hip_semantic_output_kind(self, semantic):
        if semantic is None:
            return None
        semantic_name = str(semantic)
        lower_name = semantic_name.lower()
        upper_name = semantic_name.upper()
        if lower_name in {
            "gl_fragcoord",
            "gl_frontfacing",
            "gl_globalinvocationid",
            "gl_instanceid",
            "gl_localinvocationid",
            "gl_localinvocationindex",
            "gl_pointcoord",
            "gl_vertexid",
            "gl_workgroupid",
        } or upper_name in {
            "SV_DISPATCHTHREADID",
            "SV_GROUPID",
            "SV_GROUPINDEX",
            "SV_GROUPTHREADID",
            "SV_INSTANCEID",
            "SV_ISFRONTFACE",
            "SV_VERTEXID",
        }:
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

    def is_hip_float4_type(self, type_name):
        return self.map_type(type_name) == "float4"

    def is_hip_float_scalar_type(self, type_name):
        return self.map_type(type_name) == "float"

    def validate_hip_builtin_semantic_type(self, semantic, type_name, context):
        kind = self.hip_semantic_output_kind(semantic)
        if kind is None or kind == "input_only":
            return
        if kind in {"position", "color"}:
            if self.is_hip_float4_type(type_name):
                return
            raise ValueError(
                f"Unsupported {semantic} {context} for HIP codegen; "
                "expected vec4-compatible type"
            )
        if kind == "depth" and not self.is_hip_float_scalar_type(type_name):
            raise ValueError(
                f"Unsupported {semantic} {context} for HIP codegen; "
                "expected float type"
            )

    def validate_hip_output_semantic_stage(self, stage_name, semantic, context):
        kind = self.hip_semantic_output_kind(semantic)
        if kind is None:
            return
        if kind == "input_only":
            raise ValueError(
                f"Unsupported {semantic} {context} for HIP codegen; "
                "input-only builtin semantics cannot be used as outputs"
            )
        if stage_name is None:
            return
        allowed_stages = {
            "position": {"vertex"},
            "color": {"fragment"},
            "depth": {"fragment"},
        }[kind]
        if stage_name not in allowed_stages:
            allowed = ", ".join(sorted(allowed_stages))
            raise ValueError(
                f"Unsupported {semantic} {context} for HIP {stage_name} stage; "
                f"valid stage is {allowed}"
            )

    def function_stage_name(self, node):
        if self.current_stage_name:
            return self.current_stage_name
        qualifiers = list(getattr(node, "qualifiers", []) or [])
        qualifier = getattr(node, "qualifier", None)
        if qualifier:
            qualifiers.append(qualifier)
        supported_stage_names = (
            {
                "compute",
                "fragment",
                "geometry",
                "tessellation_control",
                "tessellation_evaluation",
                "vertex",
            }
            | self.hip_ray_stage_names()
            | self.hip_mesh_task_stage_names()
        )
        for qualifier in qualifiers:
            qualifier_name = normalize_stage_name(qualifier)
            if qualifier_name in supported_stage_names:
                return qualifier_name
        return None

    def hip_stage_attribute_arguments(self, func, attribute_name):
        requested = str(attribute_name).strip().lower().replace("-", "_")
        for attr in getattr(func, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name is None:
                continue
            normalized = str(attr_name).strip().lower().replace("-", "_")
            if normalized == requested:
                return self.attribute_arguments(attr)
        return []

    def literal_int_value(self, node):
        if isinstance(node, LiteralNode):
            value = node.value
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                literal_type = getattr(
                    getattr(node, "literal_type", None), "name", None
                )
                if literal_type in {"char", "string"}:
                    return None
                try:
                    return int(value, 0)
                except ValueError:
                    return None
            return None

        if isinstance(node, UnaryOpNode) and not getattr(node, "is_postfix", False):
            operator = getattr(node, "operator", getattr(node, "op", None))
            if operator not in {"+", "-"}:
                return None
            value = self.literal_int_value(node.operand)
            if value is None:
                return None
            if operator == "-":
                return -value
            return value

        return None

    def hip_geometry_maxvertexcount(self, func):
        arguments = self.hip_stage_attribute_arguments(func, "maxvertexcount")
        if not arguments:
            raise ValueError("HIP geometry stage requires maxvertexcount attribute")
        if len(arguments) != 1:
            raise ValueError(
                "HIP geometry stage maxvertexcount requires exactly one argument"
            )
        value_text = self.attribute_value_to_string(arguments[0])
        value = self.literal_int_value(arguments[0])
        if value is not None and value <= 0:
            raise ValueError(
                f"HIP geometry stage maxvertexcount ({value}) must be positive"
            )
        return value_text

    def hip_stage_attribute_value(self, func, attribute_name, stage_name):
        arguments = self.hip_stage_attribute_arguments(func, attribute_name)
        if not arguments:
            return None
        if len(arguments) != 1:
            raise ValueError(
                f"HIP {stage_name} stage {attribute_name} requires exactly one "
                "argument"
            )
        return self.attribute_value_to_string(arguments[0])

    def hip_tessellation_output_control_points(self, func):
        arguments = self.hip_stage_attribute_arguments(func, "outputcontrolpoints")
        if not arguments:
            raise ValueError(
                "HIP tessellation_control stage requires outputcontrolpoints "
                "attribute"
            )
        if len(arguments) != 1:
            raise ValueError(
                "HIP tessellation_control stage outputcontrolpoints requires "
                "exactly one argument"
            )
        value_text = self.attribute_value_to_string(arguments[0])
        value = self.literal_int_value(arguments[0])
        if value is not None and value <= 0:
            raise ValueError(
                "HIP tessellation_control stage outputcontrolpoints "
                f"({value}) must be positive"
            )
        return value_text

    def canonical_hip_tessellation_domain(self, value):
        if value is None:
            return None
        normalized = str(value).strip().lower()
        return {
            "tri": "tri",
            "triangle": "tri",
            "triangles": "tri",
            "quad": "quad",
            "quads": "quad",
            "isoline": "isoline",
            "isolines": "isoline",
        }.get(normalized)

    def hip_tessellation_domain(self, func, stage_name, required=False):
        value = self.hip_stage_attribute_value(func, "domain", stage_name)
        if value is None:
            if required:
                raise ValueError(f"HIP {stage_name} stage requires domain attribute")
            return None
        canonical = self.canonical_hip_tessellation_domain(value)
        if canonical is None:
            raise ValueError(
                f"HIP {stage_name} stage domain '{value}' must be tri, quad, "
                "or isoline"
            )
        return canonical

    def hip_tessellation_partitioning(self, func, stage_name):
        value = self.hip_stage_attribute_value(func, "partitioning", stage_name)
        if value is None:
            return None
        normalized = str(value).strip().lower()
        valid_partitioning = {
            "fractional_even",
            "fractional_odd",
            "integer",
            "pow2",
        }
        if normalized not in valid_partitioning:
            valid = ", ".join(sorted(valid_partitioning))
            raise ValueError(
                f"HIP {stage_name} stage partitioning '{value}' must be one of: "
                f"{valid}"
            )
        return normalized

    def validate_hip_tessellation_stage(self, func, stage_name):
        if stage_name == "tessellation_control":
            self.hip_tessellation_output_control_points(func)
            self.hip_tessellation_domain(func, stage_name)
        elif stage_name == "tessellation_evaluation":
            self.hip_tessellation_domain(func, stage_name, required=True)
        self.hip_tessellation_partitioning(func, stage_name)

    def generate_hip_tessellation_stage_comments(self, func, stage_name):
        lines = []
        if stage_name == "tessellation_control":
            output_points = self.hip_tessellation_output_control_points(func)
            lines.append(
                "// CrossGL tessellation control stage: "
                f"outputcontrolpoints={output_points}\n"
            )
        else:
            domain = self.hip_tessellation_domain(func, stage_name, required=True)
            lines.append(
                "// CrossGL tessellation evaluation stage: " f"domain={domain}\n"
            )

        domain = self.hip_tessellation_domain(
            func, stage_name, required=stage_name == "tessellation_evaluation"
        )
        if stage_name == "tessellation_control" and domain is not None:
            lines.append(f"// CrossGL tessellation domain: {domain}\n")

        partitioning = self.hip_tessellation_partitioning(func, stage_name)
        if partitioning is not None:
            lines.append(f"// CrossGL tessellation partitioning: {partitioning}\n")

        output_topology = self.hip_stage_attribute_value(
            func, "outputtopology", stage_name
        )
        if output_topology is not None:
            lines.append(
                "// CrossGL tessellation output topology: " f"{output_topology}\n"
            )

        patch_constant_function = self.hip_stage_attribute_value(
            func, "patchconstantfunc", stage_name
        )
        if patch_constant_function is not None:
            lines.append(
                "// CrossGL tessellation patch constant function: "
                f"{patch_constant_function}\n"
            )

        return "".join(lines)

    def hip_mesh_task_stage_label(self, stage_name):
        return {
            "amplification": "amplification/task",
            "object": "object/task",
        }.get(stage_name, stage_name)

    def hip_mesh_task_positive_attribute(self, func, attribute_name, stage_name):
        value = self.hip_stage_attribute_value(func, attribute_name, stage_name)
        if value is None:
            return None
        value_int = self.literal_int_value(
            self.hip_stage_attribute_arguments(func, attribute_name)[0]
        )
        if value_int is not None and value_int <= 0:
            raise ValueError(
                f"HIP {stage_name} stage {attribute_name} ({value_int}) "
                "must be positive"
            )
        return value

    def hip_mesh_task_numthreads(self, func, stage_name):
        for attribute_name in ("numthreads", "local_size", "workgroup_size"):
            arguments = self.hip_stage_attribute_arguments(func, attribute_name)
            if not arguments:
                continue
            if len(arguments) != 3:
                raise ValueError(
                    f"HIP {stage_name} stage {attribute_name} requires exactly "
                    "three arguments"
                )
            values = []
            for axis, argument in zip("xyz", arguments):
                value = self.literal_int_value(argument)
                if value is not None and value <= 0:
                    raise ValueError(
                        f"HIP {stage_name} stage {attribute_name} {axis} "
                        f"dimension ({value}) must be positive"
                    )
                values.append(self.attribute_value_to_string(argument))
            return ", ".join(values)
        return None

    def hip_mesh_task_output_topology(self, func, stage_name):
        value = self.hip_stage_attribute_value(func, "outputtopology", stage_name)
        if value is None:
            return None
        normalized = str(value).strip().lower()
        topology_aliases = {
            "point": "point",
            "points": "point",
            "line": "line",
            "lines": "line",
            "triangle": "triangle",
            "triangles": "triangle",
        }
        topology = topology_aliases.get(normalized)
        if topology is None:
            raise ValueError(
                f"HIP {stage_name} stage outputtopology '{value}' must be "
                "point, line, or triangle"
            )
        return topology

    def validate_hip_mesh_task_stage(self, func, stage_name):
        self.hip_mesh_task_numthreads(func, stage_name)
        if stage_name == "mesh":
            self.hip_mesh_task_output_topology(func, stage_name)
            self.hip_mesh_task_positive_attribute(func, "max_vertices", stage_name)
            self.hip_mesh_task_positive_attribute(func, "max_primitives", stage_name)

    def generate_hip_mesh_task_stage_comments(self, func, stage_name):
        label = self.hip_mesh_task_stage_label(stage_name)
        lines = [f"// CrossGL {label} stage\n"]

        numthreads = self.hip_mesh_task_numthreads(func, stage_name)
        if numthreads is not None:
            lines.append(f"// CrossGL mesh/task numthreads: {numthreads}\n")

        if stage_name == "mesh":
            topology = self.hip_mesh_task_output_topology(func, stage_name)
            if topology is not None:
                lines.append(f"// CrossGL mesh output topology: {topology}\n")
            for attribute_name in ("max_vertices", "max_primitives"):
                value = self.hip_mesh_task_positive_attribute(
                    func, attribute_name, stage_name
                )
                if value is not None:
                    lines.append(f"// CrossGL mesh {attribute_name}: {value}\n")

        return "".join(lines)

    def parameter_raw_type(self, parameter):
        if hasattr(parameter, "param_type"):
            return parameter.param_type
        if hasattr(parameter, "vtype"):
            return parameter.vtype
        return "void"

    def hip_parameter_qualifiers(self, parameter):
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
            primitive = self.attribute_value_to_string(arguments[0])
            normalized = str(primitive).lower()
            if normalized in allowed_qualifiers:
                qualifiers.append(normalized)
        return qualifiers

    def hip_geometry_input_primitive_qualifier(self, parameter):
        primitive_qualifiers = {
            "point",
            "line",
            "triangle",
            "lineadj",
            "triangleadj",
        }
        for qualifier in self.hip_parameter_qualifiers(parameter):
            if qualifier in primitive_qualifiers:
                return qualifier
        return None

    def hip_parameter_is_array(self, parameter):
        raw_type = self.parameter_raw_type(parameter)
        if (
            hasattr(raw_type, "element_type")
            and str(type(raw_type)).find("ArrayType") != -1
        ):
            return True
        type_name = self.type_name_string(raw_type)
        return bool(type_name and "[" in type_name and "]" in type_name)

    def hip_parameter_array_count(self, parameter):
        raw_type = self.parameter_raw_type(parameter)
        if (
            hasattr(raw_type, "element_type")
            and str(type(raw_type)).find("ArrayType") != -1
        ):
            size = getattr(raw_type, "size", None)
            if size is None:
                return None
            if isinstance(size, int):
                return size
            return self.literal_int_value(size)

        type_name = self.type_name_string(raw_type)
        if not type_name or "[" not in type_name or "]" not in type_name:
            return None
        _base_type, array_size = parse_array_type(type_name)
        return array_size

    def validate_hip_geometry_input_primitive_arity(self, parameters):
        expected_counts = {
            "point": 1,
            "line": 2,
            "triangle": 3,
            "lineadj": 4,
            "triangleadj": 6,
        }

        for parameter in parameters:
            primitive = self.hip_geometry_input_primitive_qualifier(parameter)
            if primitive is None:
                continue

            expected_count = expected_counts[primitive]
            if not self.hip_parameter_is_array(parameter):
                raise ValueError(
                    "HIP geometry stage "
                    f"{primitive} input primitive parameter '{parameter.name}' "
                    f"must be an array with {expected_count} element(s)"
                )

            array_count = self.hip_parameter_array_count(parameter)
            if array_count is None:
                continue
            if array_count != expected_count:
                raise ValueError(
                    "HIP geometry stage "
                    f"{primitive} input primitive parameter '{parameter.name}' "
                    f"must have {expected_count} element(s), got {array_count}"
                )

    def validate_hip_geometry_stage(self, func, parameters):
        self.hip_geometry_maxvertexcount(func)

        if not any(
            self.hip_geometry_stream_info(self.parameter_raw_type(param))
            for param in parameters
        ):
            raise ValueError(
                "HIP geometry stage parameters must include a PointStream, "
                "LineStream, or TriangleStream output parameter"
            )

        if not any(
            self.hip_geometry_input_primitive_qualifier(param) for param in parameters
        ):
            raise ValueError(
                "HIP geometry stage parameters must include an input primitive "
                "parameter qualified as point, line, triangle, lineadj, or triangleadj"
            )

        self.validate_hip_geometry_input_primitive_arity(parameters)

    def generate_hip_geometry_stage_comments(self, func, parameters):
        maxvertexcount = self.hip_geometry_maxvertexcount(func)
        input_descriptions = []
        stream_descriptions = []

        for parameter in parameters:
            primitive = self.hip_geometry_input_primitive_qualifier(parameter)
            if primitive is not None:
                input_descriptions.append(f"{primitive}:{parameter.name}")

            stream_info = self.hip_geometry_stream_info(
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
                "// CrossGL geometry output stream: "
                f"{', '.join(stream_descriptions)}\n"
            )
        return "".join(lines)

    def validate_hip_return_semantic(self, stage_name, return_type, semantic):
        if semantic is None:
            return
        if self.map_type(return_type) == "void":
            raise ValueError(
                f"Unsupported {semantic} return semantic for HIP codegen; "
                "void return type"
            )
        self.validate_hip_output_semantic_stage(stage_name, semantic, "return semantic")
        self.validate_hip_builtin_semantic_type(
            semantic, return_type, "return semantic"
        )

    def validate_hip_struct_return_semantics(self, stage_name, return_type):
        if stage_name is None:
            return
        base_type = self.type_name_string(return_type)
        if not base_type:
            return
        base_type = base_type.split("<", 1)[0].split("[", 1)[0].strip()
        member_semantics = self.struct_member_semantics.get(base_type, {})
        member_types = self.struct_member_types.get(base_type, {})
        for member_name, semantic in member_semantics.items():
            context = f"struct return semantic '{base_type}.{member_name}'"
            self.validate_hip_output_semantic_stage(stage_name, semantic, context)
            self.validate_hip_builtin_semantic_type(
                semantic, member_types.get(member_name, "float"), context
            )

    def hip_stage_parameter_semantic_key(self, semantic):
        semantic_name = str(semantic)
        lower_name = semantic_name.lower()
        if lower_name.startswith("gl_"):
            return lower_name
        return semantic_name.upper()

    def hip_stage_parameter_semantic_rules(self):
        ray_stages = self.hip_ray_stage_names()
        ray_hit_stages = {"ray_any_hit", "ray_closest_hit", "ray_intersection"}
        return {
            "gl_vertexid": ("vertex_id", "unsigned int", {"vertex"}),
            "gl_instanceid": ("instance_id", "unsigned int", {"vertex"}),
            "gl_basevertex": ("start_vertex_location", "int", {"vertex"}),
            "gl_baseinstance": ("start_instance_location", "unsigned int", {"vertex"}),
            "gl_drawid": ("draw_id", "unsigned int", {"vertex"}),
            "SV_VERTEXID": ("vertex_id", "unsigned int", {"vertex"}),
            "SV_INSTANCEID": ("instance_id", "unsigned int", {"vertex"}),
            "SV_STARTVERTEXLOCATION": ("start_vertex_location", "int", {"vertex"}),
            "SV_STARTINSTANCELOCATION": (
                "start_instance_location",
                "unsigned int",
                {"vertex"},
            ),
            "SV_DRAWID": ("draw_id", "unsigned int", {"vertex"}),
            "gl_position": ("position", "float4", {"fragment"}),
            "gl_fragcoord": ("position", "float4", {"fragment"}),
            "gl_frontfacing": ("front_facing", "bool", {"fragment"}),
            "gl_pointcoord": ("point_coord", "float2", {"fragment"}),
            "gl_primitiveid": (
                "primitive_id",
                "unsigned int",
                {"fragment"} | ray_hit_stages,
            ),
            "gl_sampleid": ("sample_index", "unsigned int", {"fragment"}),
            "SV_POSITION": ("position", "float4", {"fragment"}),
            "SV_ISFRONTFACE": ("front_facing", "bool", {"fragment"}),
            "SV_PRIMITIVEID": (
                "primitive_id",
                "unsigned int",
                {"fragment"} | ray_hit_stages,
            ),
            "SV_SAMPLEINDEX": ("sample_index", "unsigned int", {"fragment"}),
            "gl_workgroupid": ("workgroup_id", "uint3", {"compute"}),
            "gl_localinvocationid": ("local_invocation_id", "uint3", {"compute"}),
            "gl_globalinvocationid": ("global_invocation_id", "uint3", {"compute"}),
            "gl_localinvocationindex": (
                "local_invocation_index",
                "unsigned int",
                {"compute"},
            ),
            "SV_GROUPID": ("workgroup_id", "uint3", {"compute"}),
            "SV_GROUPTHREADID": ("local_invocation_id", "uint3", {"compute"}),
            "SV_DISPATCHTHREADID": ("global_invocation_id", "uint3", {"compute"}),
            "SV_GROUPINDEX": ("local_invocation_index", "unsigned int", {"compute"}),
            "gl_launchidext": ("launch_id", "uint3", ray_stages),
            "gl_launchsizeext": ("launch_size", "uint3", ray_stages),
            "gl_hittext": ("hit_t", "float", ray_hit_stages),
            "gl_hitkindext": ("hit_kind", "unsigned int", {"ray_any_hit"}),
            "SV_DISPATCHRAYSINDEX": ("launch_id", "uint3", ray_stages),
            "SV_DISPATCHRAYSDIMENSIONS": ("launch_size", "uint3", ray_stages),
        }

    def validate_hip_stage_parameter_semantic_type(
        self, param, semantic, expected_type
    ):
        actual_type = self.map_type(self.get_parameter_type(param))
        if actual_type == expected_type:
            return
        raise ValueError(
            f"Unsupported {semantic} stage parameter semantic for HIP codegen; "
            f"expected {expected_type} type"
        )

    def validate_hip_stage_parameter_semantics(self, stage_name, parameters):
        if stage_name is None:
            return

        rules = self.hip_stage_parameter_semantic_rules()
        seen_system_semantics = {}
        for param in parameters or []:
            semantic = self.semantic_from_node(param)
            if semantic is None:
                continue

            semantic_key = self.hip_stage_parameter_semantic_key(semantic)
            rule = rules.get(semantic_key)
            if rule is not None:
                mapped_semantic, expected_type, allowed_stages = rule
                if stage_name not in allowed_stages:
                    allowed = ", ".join(sorted(allowed_stages))
                    raise ValueError(
                        f"Unsupported {semantic} stage parameter semantic for HIP "
                        f"{stage_name} stage; valid stage is {allowed}"
                    )
                param_name = getattr(param, "name", getattr(param, "param_name", None))
                previous_name = seen_system_semantics.get(mapped_semantic)
                if previous_name is not None:
                    raise ValueError(
                        f"Duplicate HIP stage parameter semantic {mapped_semantic} "
                        f"on '{previous_name}' and '{param_name}'"
                    )
                seen_system_semantics[mapped_semantic] = param_name
                self.validate_hip_stage_parameter_semantic_type(
                    param, semantic, expected_type
                )
                continue

            kind = self.hip_semantic_output_kind(semantic)
            if kind in {"color", "depth"}:
                raise ValueError(
                    f"Unsupported {semantic} stage parameter semantic for HIP "
                    f"{stage_name} stage; output-only builtin semantics cannot "
                    "be used as inputs"
                )

    def visit_StructNode(self, node: StructNode) -> str:
        if getattr(node, "is_cbuffer", False):
            return self.emit_cbuffer_members_as_constant_memory(node)
        self.add_line(f"struct {node.name}")
        self.add_line("{")
        self.indent_level += 1

        members = getattr(node, "members", [])
        member_types = {}
        member_semantics = {}
        for member in members:
            if hasattr(member, "member_type"):
                member_type = member.member_type
            elif hasattr(member, "vtype"):
                member_type = member.vtype
            elif hasattr(member, "var_type"):
                member_type = member.var_type
            else:
                member_type = "float"

            member_type = self.resource_type_with_access(member_type, member)
            member_types[member.name] = member_type
            semantic = self.semantic_from_node(member)
            if semantic:
                member_semantics[member.name] = semantic
                self.validate_hip_builtin_semantic_type(
                    semantic, member_type, "struct member semantic"
                )
            semantic_comment = (
                f" // CrossGL semantic: {self.map_semantic(semantic)}"
                if semantic
                else ""
            )
            declaration = self.format_typed_declarator(member_type, member.name)
            declaration = self.apply_resource_memory_qualifiers_to_declaration(
                declaration, member.name, member
            )
            self.add_line(f"{declaration};{semantic_comment}")

        self.struct_member_types[node.name] = member_types
        self.struct_member_semantics[node.name] = member_semantics
        self.indent_level -= 1
        self.add_line("};")
        self.add_line()
        return ""

    def cbuffer_member_type(self, member):
        if hasattr(member, "member_type"):
            return member.member_type
        if hasattr(member, "vtype"):
            return member.vtype
        if hasattr(member, "var_type"):
            return member.var_type
        return "int"

    def emit_cbuffer_members_as_constant_memory(self, node) -> str:
        metadata_comment = self.hip_resource_metadata_comment(
            node, getattr(node, "name", None), kind="cbuffer"
        )
        if metadata_comment:
            self.add_line(metadata_comment)
        self.add_line(f"// Constant buffer: {node.name}")
        for member in getattr(node, "members", []) or []:
            member_type = self.cbuffer_member_type(member)
            declaration = self.format_typed_declarator(member_type, member.name)
            self.add_line(f"__constant__ {declaration};")
        self.add_line()
        return ""

    def visit_VariableNode(self, node: VariableNode) -> str:
        var_type = self.get_variable_node_type(node)
        initial_value = getattr(node, "initial_value", getattr(node, "value", None))
        if isinstance(initial_value, MatchNode):
            self.emit_match_expression_variable(node, initial_value, var_type)
            return ""

        metadata_comment = self.hip_resource_metadata_comment(node, var_type)
        if metadata_comment:
            self.add_line(metadata_comment)
        self.add_line(f"{self.format_variable_declaration(node)};")
        metadata_declaration = self.query_metadata_declaration(node.name, var_type)
        if metadata_declaration:
            self.add_line(f"{metadata_declaration};")
        length_declaration = self.structured_buffer_length_declaration(
            node.name, var_type
        )
        if length_declaration:
            self.add_line(f"{length_declaration};")
        counter_declaration = self.structured_buffer_counter_declaration(
            node.name, var_type
        )
        if counter_declaration:
            self.add_line(f"{counter_declaration};")
        return ""

    def format_variable_declaration(self, node: VariableNode) -> str:
        initial_value = getattr(node, "initial_value", getattr(node, "value", None))
        var_type = self.get_variable_node_type(node)

        if var_type is None and initial_value is not None:
            inferred_type = self.expression_result_type(initial_value)
            self.register_variable_type(node.name, inferred_type, node, initial_value)
            declaration = self.format_typed_declarator(
                inferred_type or "auto", node.name
            )
        else:
            var_type = var_type or "int"
            self.register_variable_type(node.name, var_type, node, initial_value)
            var_type = self.glsl_buffer_block_declaration_type(var_type, node)
            declaration = self.format_typed_declarator(var_type, node.name)

        declaration = self.apply_resource_memory_qualifiers_to_declaration(
            declaration, node.name, node
        )

        qualifiers = self.variable_memory_qualifiers(node)
        if qualifiers:
            declaration = f"{' '.join(qualifiers)} {declaration}"

        if initial_value is not None:
            declaration += f" = {self.visit(initial_value)}"

        return declaration

    def variable_memory_qualifiers(self, node: VariableNode):
        qualifiers = []
        for qualifier in getattr(node, "qualifiers", []) or []:
            qualifier_name = str(qualifier).lower()
            if "workgroup" in qualifier_name or "shared" in qualifier_name:
                qualifiers.append("__shared__")
            elif "uniform" in qualifier_name:
                qualifiers.append("__constant__")
        return qualifiers

    def declaration_qualifier_names(self, node):
        """Return normalized declaration qualifier and attribute names."""
        names = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name:
                names.add(attr_name)
        return names

    def apply_resource_memory_qualifiers_to_declaration(self, declaration, name, node):
        """Apply HIP-native pointer qualifiers for resource-like declarations."""
        if "*" not in declaration or not name:
            return declaration

        names = self.declaration_qualifier_names(node)
        if not names:
            return declaration

        if (
            names & {"const", "constant", "readonly"}
            and "writeonly" not in names
            and not declaration.startswith("const ")
        ):
            declaration = f"const {declaration}"

        if "volatile" in names:
            if declaration.startswith("const "):
                declaration = f"const volatile {declaration[len('const '):]}"
            elif not declaration.startswith("volatile "):
                declaration = f"volatile {declaration}"

        if "restrict" in names and "__restrict__" not in declaration:
            name_token = f" {name}"
            if name_token in declaration:
                declaration = declaration.replace(
                    name_token, f" __restrict__{name_token}", 1
                )

        return declaration

    def emit_match_expression_variable(self, node, match_node, var_type):
        if var_type is None:
            var_type = infer_match_expression_result_type(self, match_node)
        var_type = var_type or "auto"
        self.register_variable_type(node.name, var_type, node)

        declaration = self.format_typed_declarator(var_type, node.name)
        declaration = self.apply_resource_memory_qualifiers_to_declaration(
            declaration, node.name, node
        )
        qualifiers = self.variable_memory_qualifiers(node)
        if qualifiers:
            declaration = f"{' '.join(qualifiers)} {declaration}"

        self.add_line(f"{declaration};")
        self.add_generated_code(
            generate_match_expression_assignment(
                self,
                match_node,
                node.name,
                var_type,
                self.indent_level,
                "HIP",
            )
        )

    def visit_CbufferNode(self, node: CbufferNode) -> str:
        return self.emit_cbuffer_members_as_constant_memory(node)

    def visit_list(self, node_list) -> str:
        for node in node_list:
            self.emit_statement(node)
        return ""

    def emit_statement(self, node):
        """Render and append one statement node when it produces code."""
        if node is None:
            return

        result = self.visit(node)
        if isinstance(result, str) and result.strip():
            self.add_line(f"{result};")

    def emit_body(self, body):
        """Render a list-like or block-like function body."""
        if isinstance(body, list):
            for stmt in body:
                self.emit_statement(stmt)
        elif hasattr(body, "statements"):
            for stmt in body.statements:
                self.emit_statement(stmt)
        else:
            self.emit_statement(body)

    def statement_list(self, body):
        if body is None:
            return []
        if isinstance(body, list):
            return body
        if hasattr(body, "statements"):
            return body.statements
        return [body]

    def statement_body_terminates(self, body):
        statements = self.statement_list(body)
        if not statements:
            return False
        return isinstance(statements[-1], (BreakNode, ContinueNode, ReturnNode))

    def visit_IfNode(self, node) -> str:
        condition = self.visit(node.if_condition)
        self.add_line(f"if ({condition})")
        self.add_line("{")
        self.indent_level += 1
        self.emit_body(node.if_body)
        self.indent_level -= 1
        self.add_line("}")

        if hasattr(node, "else_body") and node.else_body:
            self.add_line("else")
            self.add_line("{")
            self.indent_level += 1
            self.emit_body(node.else_body)
            self.indent_level -= 1
            self.add_line("}")

        return ""

    def visit_ForNode(self, node) -> str:
        if isinstance(node.init, VariableNode):
            init = self.format_variable_declaration(node.init)
        elif hasattr(node.init, "expression"):
            init = self.visit(node.init.expression)
        else:
            init = self.visit(node.init) if node.init else ""
        condition = self.visit(node.condition) if node.condition else ""
        update = self.visit(node.update) if node.update else ""

        self.add_line(f"for ({init}; {condition}; {update})")
        self.add_line("{")
        self.indent_level += 1
        self.emit_body(node.body)
        self.indent_level -= 1
        self.add_line("}")

        return ""

    def visit_ForInNode(self, node) -> str:
        pattern = getattr(node, "pattern", "item")
        iterable = getattr(node, "iterable", None)

        if isinstance(iterable, RangeNode):
            start = self.visit(iterable.start)
            end = self.visit(iterable.end)
            comparator = "<=" if iterable.inclusive else "<"
        else:
            start = "0"
            end = self.visit(iterable)
            comparator = "<"

        self.add_line(
            f"for (int {pattern} = {start}; {pattern} {comparator} {end}; ++{pattern})"
        )
        self.add_line("{")
        self.indent_level += 1
        self.emit_body(getattr(node, "body", []))
        self.indent_level -= 1
        self.add_line("}")

        return ""

    def visit_WhileNode(self, node) -> str:
        condition = self.visit(node.condition) if node.condition else ""

        self.add_line(f"while ({condition})")
        self.add_line("{")
        self.indent_level += 1
        self.emit_body(node.body)
        self.indent_level -= 1
        self.add_line("}")

        return ""

    def visit_DoWhileNode(self, node) -> str:
        condition = self.visit(node.condition) if node.condition else ""

        self.add_line("do")
        self.add_line("{")
        self.indent_level += 1
        self.emit_body(node.body)
        self.indent_level -= 1
        self.add_line(f"}} while ({condition});")

        return ""

    def visit_SwitchNode(self, node) -> str:
        expression = self.visit(node.expression)

        self.add_line(f"switch ({expression})")
        self.add_line("{")
        self.indent_level += 1
        for case in getattr(node, "cases", []):
            self.visit(case)
        self.indent_level -= 1
        self.add_line("}")

        return ""

    def visit_MatchNode(self, node) -> str:
        if is_switch_lowerable_match(node):
            code = generate_switch_match(self, node, self.indent_level)
        else:
            code = generate_ordered_conditional_match(
                self, node, self.indent_level, "HIP"
            )
        self.add_generated_code(code)
        return ""

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

    def statement_body_has_statements(self, body):
        return bool(self.statement_list(body))

    def generate_scoped_statement_body(self, body, indent):
        saved_lines = self.code_lines
        saved_indent = self.indent_level
        self.code_lines = []
        self.indent_level = indent
        try:
            self.emit_body(body)
            if not self.code_lines:
                return ""
            return "\n".join(self.code_lines) + "\n"
        finally:
            self.code_lines = saved_lines
            self.indent_level = saved_indent

    def generate_expression(self, node):
        if node is None:
            return ""
        return self.visit(node)

    def generate_expression_with_expected(self, node, _expected_type):
        return self.generate_expression(node)

    def visit_CaseNode(self, node) -> str:
        if getattr(node, "value", None) is None:
            self.add_line("default:")
        else:
            value = self.visit(node.value)
            self.add_line(f"case {value}:")

        self.indent_level += 1
        self.emit_body(getattr(node, "statements", []))
        self.indent_level -= 1

        return ""

    def visit_ReturnNode(self, node) -> str:
        if node.value:
            if isinstance(node.value, MatchNode):
                return self.emit_match_expression_return(node.value)
            value = self.visit(node.value)
            self.add_line(f"return {value};")
        else:
            self.add_line("return;")
        return ""

    def emit_match_expression_return(self, match_node) -> str:
        return_type = self.type_name_string(self.current_function_return_type)
        if not return_type or return_type == "void":
            raise ValueError(
                "Unsupported match expression for HIP codegen; return context "
                "requires a concrete non-void result type"
            )

        result_name = self.next_hip_temp_variable("match_value")
        self.add_line(f"{self.format_typed_declarator(return_type, result_name)};")
        self.add_generated_code(
            generate_match_expression_assignment(
                self,
                match_node,
                result_name,
                return_type,
                self.indent_level,
                "HIP",
            )
        )
        self.add_line(f"return {result_name};")
        return ""

    def visit_AssignmentNode(self, node) -> str:
        operator = getattr(node, "operator", getattr(node, "op", "="))
        diagnostic = self.glsl_buffer_block_write_diagnostic(node.left, "assignment")
        if diagnostic is not None:
            return diagnostic
        if operator == "=":
            diagnostic = self.structured_buffer_element_write_diagnostic(
                node.left, "assignment"
            )
        else:
            diagnostic = self.structured_buffer_element_read_write_diagnostic(
                node.left, "assignment"
            )
        if diagnostic is not None:
            return diagnostic
        self.assignment_lhs_depth += 1
        try:
            left = self.visit(node.left)
        finally:
            self.assignment_lhs_depth -= 1
        right = self.visit(node.right)
        compound_binary_ops = {
            "+=": "+",
            "-=": "-",
            "*=": "*",
            "/=": "/",
            "%=": "%",
        }
        if operator in compound_binary_ops:
            lowered_right = self.lower_vector_binary_operation(
                node.left,
                left,
                node.right,
                right,
                compound_binary_ops[operator],
            )
            if lowered_right is not None:
                return f"{left} = {lowered_right}"
            if operator == "%=":
                modulo = self.lower_scalar_modulo_operation(
                    node.left,
                    left,
                    node.right,
                    right,
                )
                if modulo is not None:
                    return f"{left} = {modulo}"
        compound_bitwise_ops = {
            "&=": "&",
            "|=": "|",
            "^=": "^",
            "<<=": "<<",
            ">>=": ">>",
        }
        if operator in compound_bitwise_ops:
            lowered_right = self.lower_vector_bitwise_operation(
                node.left,
                left,
                node.right,
                right,
                compound_bitwise_ops[operator],
            )
            if lowered_right is not None:
                return f"{left} = {lowered_right}"
        return f"{left} {operator} {right}"

    def visit_BinaryOpNode(self, node) -> str:
        left = self.visit(node.left)
        right = self.visit(node.right)
        operator = getattr(node, "operator", getattr(node, "op", "+"))

        logical = self.lower_vector_logical_operation(
            node.left,
            left,
            node.right,
            right,
            operator,
        )
        if logical is not None:
            return logical

        bitwise = self.lower_vector_bitwise_operation(
            node.left,
            left,
            node.right,
            right,
            operator,
        )
        if bitwise is not None:
            return bitwise

        # Handle special operators
        if operator == "and":
            return f"({left} && {right})"
        elif operator == "or":
            return f"({left} || {right})"
        comparison = self.lower_vector_comparison_operation(
            node.left,
            left,
            node.right,
            right,
            operator,
        )
        if comparison is not None:
            return comparison
        lowered = self.lower_vector_binary_operation(
            node.left,
            left,
            node.right,
            right,
            operator,
        )
        if lowered is not None:
            return lowered
        if operator == "%":
            modulo = self.lower_scalar_modulo_operation(
                node.left,
                left,
                node.right,
                right,
            )
            if modulo is not None:
                return modulo
        return f"({left} {operator} {right})"

    def visit_UnaryOpNode(self, node) -> str:
        operand = self.visit(node.operand)

        if node.op in {"!", "not"}:
            lowered = self.lower_vector_unary_operation(node.operand, operand, node.op)
            if lowered is not None:
                return lowered
            return f"!{operand}"
        elif node.op in ["++", "--"]:
            if getattr(node, "is_postfix", getattr(node, "postfix", False)):
                return f"{operand}{node.op}"
            else:
                return f"{node.op}{operand}"
        lowered = self.lower_vector_unary_operation(node.operand, operand, node.op)
        if lowered is not None:
            return lowered
        return f"{node.op}{operand}"

    def visit_FunctionCallNode(self, node) -> str:
        func_expr = (
            node.function if hasattr(node, "function") else getattr(node, "name", None)
        )
        func_name = None
        if isinstance(func_expr, MemberAccessNode):
            callee = self.visit(func_expr)
            func_name = callee
        elif hasattr(func_expr, "name") and getattr(func_expr, "name", None):
            func_name = func_expr.name
            callee = func_name
        elif isinstance(func_expr, str):
            func_name = func_expr
            callee = func_expr
        else:
            callee = self.visit(func_expr)
        raw_args = getattr(node, "args", getattr(node, "arguments", []))
        args = [self.visit(arg) for arg in raw_args]

        if func_name == "lambda":
            return self.generate_lambda_expression(raw_args)
        if func_name in HIP_WAVE_OP_ARITIES:
            return self.generate_wave_operation(func_name, raw_args, args)
        if func_name in {"SetMeshOutputCounts", "DispatchMesh"}:
            return self.generate_mesh_task_call_expression(func_name, raw_args, args)

        is_user_function = self.is_user_defined_function(func_name)
        if not is_user_function:
            ray_tracing_call = self.generate_ray_tracing_call_expression(
                func_name, raw_args
            )
            if ray_tracing_call is not None:
                return ray_tracing_call
        if not is_user_function:
            buffer_call = self.generate_buffer_call(
                func_expr, func_name, raw_args, args
            )
            if buffer_call is not None:
                return buffer_call

            structured_atomic_call = self.generate_structured_buffer_atomic_call(
                func_name, raw_args, args
            )
            if structured_atomic_call is not None:
                return structured_atomic_call

            plain_atomic_call = self.generate_plain_atomic_call(
                func_name, raw_args, args
            )
            if plain_atomic_call is not None:
                return plain_atomic_call

            resource_call = self.generate_resource_call(func_name, raw_args, args)
            if resource_call is not None:
                return resource_call

        if is_user_function:
            args = self.hip_user_function_call_arguments(func_name, raw_args, args)
            return f"{callee}({', '.join(args)})"

        if func_name in self.synchronization_builtins and raw_args:
            raise ValueError(
                f"HIP synchronization builtin '{func_name}' requires 0 "
                f"arguments; got {len(raw_args)}"
            )

        args = self.query_metadata_call_arguments(func_name, raw_args, args)

        # Map function name
        mapped_name = self.function_map.get(func_name, func_name)

        # Handle special functions
        if func_name == "abs":
            if len(args) == 1:
                abs_call = self.generate_abs_call(raw_args, args)
                if abs_call is not None:
                    return abs_call
        elif func_name == "sign":
            if len(args) == 1:
                sign_call = self.generate_sign_call(raw_args, args)
                if sign_call is not None:
                    return sign_call
        elif func_name == "mod":
            if len(args) == 2:
                mod_call = self.generate_mod_call(raw_args, args)
                if mod_call is not None:
                    return mod_call
        elif func_name in {"fract", "frac"}:
            if len(args) == 1:
                return self.generate_fract_call(raw_args, args)
        elif func_name in {"min", "max"}:
            if len(args) == 2:
                min_max_call = self.generate_min_max_call(func_name, raw_args, args)
                if min_max_call is not None:
                    return min_max_call
        elif func_name == "atan2":
            if len(args) == 2:
                atan2_call = self.generate_atan2_call(raw_args, args)
                if atan2_call is not None:
                    return atan2_call
        elif func_name in {"dot", "cross", "length", "normalize"}:
            geometric_call = self.generate_vector_geometric_call(
                func_name,
                raw_args,
                args,
            )
            if geometric_call is not None:
                return geometric_call
        elif func_name == "clamp":
            if len(args) == 3:
                return self.generate_clamp_call(raw_args, args)
        elif func_name == "mix":
            if len(args) == 3:
                mix_call = self.generate_mix_call(raw_args, args)
                if mix_call is not None:
                    return mix_call
        elif func_name == "saturate":
            if len(args) == 1:
                saturate_call = self.generate_saturate_call(raw_args, args)
                if saturate_call is not None:
                    return saturate_call
        elif func_name in ["texture", "tex2D"]:
            # Handle texture sampling
            if len(args) >= 2:
                return (
                    f"tex2D<float4>({args[0]}, "
                    f"{self.coord_component(args[1], 'x')}, "
                    f"{self.coord_component(args[1], 'y')})"
                )

        vector_info = self.vector_type_info(func_name)
        if vector_info:
            splat_call = self.generate_vector_scalar_splat_call(
                vector_info, raw_args, args
            )
            if splat_call is not None:
                return splat_call
            constructor_call = self.generate_vector_constructor_single_eval_call(
                vector_info, raw_args, args
            )
            if constructor_call is not None:
                return constructor_call
            args = self.generate_vector_constructor_args(vector_info, raw_args, args)

        scalar_math_call = self.generate_scalar_math_call(func_name, raw_args, args)
        if scalar_math_call is not None:
            return scalar_math_call

        args_str = ", ".join(args)
        target = mapped_name if mapped_name is not None else callee
        return f"{target}({args_str})"

    def visit_MeshOpNode(self, node: MeshOpNode) -> str:
        operation = getattr(node, "operation", "")
        raw_args = getattr(node, "arguments", []) or []
        args = [self.visit(arg) for arg in raw_args]
        return self.generate_mesh_task_call_expression(operation, raw_args, args)

    def generate_mesh_task_call_expression(self, operation, raw_args, args):
        stage_name = getattr(self, "current_stage_name", None)
        if operation == "SetMeshOutputCounts":
            if stage_name != "mesh":
                raise ValueError("HIP SetMeshOutputCounts is only valid in mesh stages")
            if len(raw_args) != 2:
                raise ValueError(
                    "HIP SetMeshOutputCounts requires exactly two arguments"
                )
            return f"cgl_hip_set_mesh_output_counts({', '.join(args)})"

        if operation == "DispatchMesh":
            if stage_name not in {"task", "object", "amplification"}:
                raise ValueError(
                    "HIP DispatchMesh is only valid in task, object, or "
                    "amplification stages"
                )
            if len(raw_args) not in {3, 4}:
                raise ValueError(
                    "HIP DispatchMesh requires exactly three arguments, or four "
                    "with a task payload"
                )
            return f"cgl_hip_dispatch_mesh({', '.join(args)})"

        return None

    def visit_RayTracingOpNode(self, node: RayTracingOpNode) -> str:
        operation = getattr(node, "operation", "")
        if operation in self.function_return_types:
            args = ", ".join(
                self.visit(arg) for arg in getattr(node, "arguments", []) or []
            )
            return f"{operation}({args})"
        return self.generate_ray_tracing_call_expression(
            operation,
            getattr(node, "arguments", []),
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

        self.require_hip_ray_runtime_helpers()
        generated_args = [self.visit(arg) for arg in arguments or []]
        if operation == "TraceRay" and len(generated_args) == 11:
            ray_desc = (
                "CglRayDesc("
                f"{generated_args[6]}, {generated_args[7]}, "
                f"{generated_args[8]}, {generated_args[9]}"
                ")"
            )
            generated_args = generated_args[:6] + [ray_desc, generated_args[10]]
        elif operation == "ReportHit" and len(generated_args) == 2:
            generated_args.append("CglBuiltInTriangleIntersectionAttributes{}")

        helper_name = self.function_map.get(operation, operation)
        return f"{helper_name}({', '.join(generated_args)})"

    def visit_RayQueryOpNode(self, node: RayQueryOpNode) -> str:
        operation = getattr(node, "operation", "")
        helper_name = self.require_hip_ray_query_helper(operation)
        args = [self.visit(getattr(node, "query_expr", None))]
        args.extend(self.visit(arg) for arg in getattr(node, "arguments", []) or [])
        return f"{helper_name}({', '.join(args)})"

    def visit_WaveOpNode(self, node: WaveOpNode) -> str:
        raw_args = list(getattr(node, "arguments", []) or [])
        args = [self.visit(arg) for arg in raw_args]
        return self.generate_wave_operation(
            getattr(node, "operation", "WaveOp"), raw_args, args
        )

    def generate_wave_operation(self, operation, raw_args, args):
        expected_count = HIP_WAVE_OP_ARITIES.get(operation)
        if expected_count is None:
            raise ValueError(f"Unsupported HIP wave intrinsic {operation}")
        if len(raw_args) != expected_count:
            raise ValueError(
                f"HIP wave intrinsic {operation} requires {expected_count} "
                f"argument{'s' if expected_count != 1 else ''}, got {len(raw_args)}"
            )

        if operation == "WaveGetLaneCount":
            return "warpSize"
        if operation == "WaveGetLaneIndex":
            return "(threadIdx.x & (warpSize - 1))"
        if operation == "WaveIsFirstLane":
            return "((threadIdx.x & (warpSize - 1)) == 0)"

        diagnostic = self.wave_operation_diagnostic(operation, raw_args)
        if diagnostic is not None:
            return diagnostic

        helper_name = self.require_wave_helper(operation, raw_args)
        return f"{helper_name}({', '.join(args)})"

    def wave_operation_diagnostic(self, operation, raw_args):
        reason = self.unsupported_wave_operation_reason(operation, raw_args)
        if reason is None:
            return None
        fallback = self.diagnostic_zero_value_for_type(
            self.wave_result_type(operation, raw_args)
        )
        return (
            f"/* unsupported {self.resource_backend_name()} wave intrinsic: "
            f"{operation} {reason} */ {fallback}"
        )

    def unsupported_wave_operation_reason(self, operation, raw_args):
        if not raw_args:
            return None

        if operation in HIP_WAVE_PREDICATE_ARGUMENT_OPS:
            predicate_type = self.wave_diagnostic_type_name(raw_args[0])
            if not self.wave_is_scalar_component_type(predicate_type, {"bool"}):
                return f"predicate must be bool, got {predicate_type}"

        if operation in {"WaveReadLaneAt", "QuadReadLaneAt"} and len(raw_args) > 1:
            lane_type = self.wave_diagnostic_type_name(raw_args[1])
            if not self.wave_is_scalar_component_type(lane_type, {"int", "uint"}):
                return f"lane index must be scalar int or uint, got {lane_type}"

        if operation.startswith("WaveMultiPrefix") and len(raw_args) > 1:
            mask_type = self.wave_diagnostic_type_name(raw_args[1])
            if self.map_type(mask_type) != "uint4":
                return f"partition mask must be uvec4, got {mask_type}"

        if operation in {
            "WaveActiveSum",
            "WaveActiveProduct",
            "WaveActiveMin",
            "WaveActiveMax",
            "WavePrefixSum",
            "WavePrefixProduct",
            "WaveMultiPrefixSum",
            "WaveMultiPrefixProduct",
        }:
            value_type = self.wave_diagnostic_type_name(raw_args[0])
            if not self.wave_has_scalar_or_vector_component(
                value_type, {"float", "double", "int", "uint"}
            ):
                return "value must be numeric scalar or vector, " f"got {value_type}"

        if operation in {
            "WaveActiveBitAnd",
            "WaveActiveBitOr",
            "WaveActiveBitXor",
            "WaveMultiPrefixBitAnd",
            "WaveMultiPrefixBitOr",
            "WaveMultiPrefixBitXor",
        }:
            value_type = self.wave_diagnostic_type_name(raw_args[0])
            if not self.wave_has_scalar_or_vector_component(
                value_type, {"int", "uint"}
            ):
                return "value must be integer scalar or vector, " f"got {value_type}"

        if operation in {
            "WaveActiveAllEqual",
            "WaveMatch",
            "WaveReadLaneAt",
            "WaveReadLaneFirst",
            "QuadReadLaneAt",
            "QuadReadAcrossX",
            "QuadReadAcrossY",
            "QuadReadAcrossDiagonal",
        }:
            value_type = self.wave_diagnostic_type_name(raw_args[0])
            if not self.wave_has_scalar_or_vector_component(
                value_type, {"bool", "float", "double", "int", "uint"}
            ):
                return "value must be scalar or vector, " f"got {value_type}"

        return None

    def wave_diagnostic_type_name(self, arg):
        type_name = self.expression_result_type(arg)
        if type_name is None:
            return "unknown"
        return self.map_type(type_name)

    def wave_is_scalar_component_type(self, type_name, components):
        if self.vector_type_info(type_name) is not None:
            return False
        return self.scalar_component_type(type_name) in components

    def wave_has_scalar_or_vector_component(self, type_name, components):
        vector_info = self.vector_type_info(type_name)
        if vector_info is not None:
            return vector_info["component_type"] in components
        return self.scalar_component_type(type_name) in components

    def require_wave_helper(self, operation, raw_args):
        result_type = self.map_type(self.wave_result_type(operation, raw_args))
        arg_types = [
            self.wave_argument_type(operation, index, arg)
            for index, arg in enumerate(raw_args)
        ]
        helper_name = self.wave_helper_name(operation, result_type, arg_types)
        if helper_name in self.helper_functions:
            return helper_name

        parameter_names = self.wave_helper_parameter_names(operation, len(arg_types))
        params = [
            f"{arg_type} {parameter_name}"
            for parameter_name, arg_type in zip(parameter_names, arg_types)
        ]
        helper = (
            f"__device__ inline {result_type} {helper_name}({', '.join(params)})\n"
            "{\n"
            f"    return {self.wave_helper_return_expression(operation, result_type)};\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def wave_result_type(self, operation, raw_args):
        if operation in HIP_WAVE_UINT_RESULT_OPS:
            return "uint"
        if operation in HIP_WAVE_BOOL_RESULT_OPS:
            return "bool"
        if operation in HIP_WAVE_UVEC4_RESULT_OPS:
            return "uvec4"
        if raw_args:
            return self.expression_result_type(raw_args[0]) or "uint"
        return "uint"

    def wave_argument_type(self, operation, index, arg):
        if index == 0 and operation in HIP_WAVE_PREDICATE_ARGUMENT_OPS:
            return self.map_type("bool")
        if index == 1 and operation.startswith("WaveMultiPrefix"):
            return self.map_type("uvec4")
        if index == 1 and operation in {"WaveReadLaneAt", "QuadReadLaneAt"}:
            return self.map_type("uint")
        return self.map_type(self.expression_result_type(arg) or "uint")

    def wave_helper_name(self, operation, result_type, arg_types):
        suffix_parts = [self.wave_type_suffix(result_type)]
        suffix_parts.extend(self.wave_type_suffix(arg_type) for arg_type in arg_types)
        return (
            f"cgl_hip_{self.wave_operation_suffix(operation)}_{'_'.join(suffix_parts)}"
        )

    def wave_operation_suffix(self, operation):
        name = operation[0].lower() + operation[1:]
        suffix = []
        for char in name:
            if char.isupper():
                suffix.append("_")
                suffix.append(char.lower())
            else:
                suffix.append(char)
        return "".join(suffix)

    def wave_type_suffix(self, type_name):
        suffix = type_name.replace("unsigned int", "uint")
        suffix = suffix.replace(" ", "_").replace("*", "_ptr")
        suffix = suffix.replace("&", "_ref").replace("::", "_")
        return "".join(char if char.isalnum() else "_" for char in suffix).strip("_")

    def wave_helper_parameter_names(self, operation, arg_count):
        if arg_count == 1:
            if operation in HIP_WAVE_PREDICATE_ARGUMENT_OPS:
                return ["predicate"]
            return ["value"]
        if operation in {"WaveReadLaneAt", "QuadReadLaneAt"}:
            return ["value", "lane"]
        if operation == "WaveMultiPrefixCountBits":
            return ["predicate", "mask"]
        return ["value", "mask"]

    def wave_helper_return_expression(self, operation, result_type):
        if operation in {"WaveActiveAllTrue", "WaveActiveAnyTrue"}:
            return "predicate"
        if operation == "WaveActiveAllEqual":
            return "true"
        if operation in {
            "WaveActiveCountBits",
            "WavePrefixCountBits",
            "WaveMultiPrefixCountBits",
        }:
            return "(predicate ? 1u : 0u)"
        if operation == "WaveActiveBallot":
            return "make_uint4((predicate ? 1u : 0u), 0u, 0u, 0u)"
        if operation == "WaveMatch":
            return "make_uint4(0u, 0u, 0u, 0u)"
        if result_type == "bool":
            return "false"
        return "value"

    def generate_lambda_expression(self, args):
        """Render CrossGL's pseudo-lambda as a HIP device lambda."""
        if not args:
            return "[&] __device__ () {}"

        params = ", ".join(self.generate_lambda_parameter(arg) for arg in args[:-1])
        body = self.generate_lambda_body(args[-1])
        return f"[&] __device__ ({params}) {body}"

    def generate_lambda_parameter(self, arg):
        raw = self.lambda_raw_argument_text(arg).strip()
        typed_param = self.split_lambda_typed_parameter(raw)
        if typed_param is None:
            param_name = self.lambda_fallback_parameter_name(raw)
            return f"auto {param_name}" if param_name else "auto"

        type_name, param_name = typed_param
        mapped_type = self.lambda_parameter_type(type_name)
        return f"{mapped_type} {param_name}"

    def generate_lambda_body(self, arg):
        raw = self.lambda_raw_argument_text(arg).strip()
        if raw.startswith("{") and raw.endswith("}"):
            return raw
        if raw:
            return f"{{ return {raw}; }}"
        return "{}"

    def lambda_raw_argument_text(self, arg):
        if isinstance(arg, IdentifierNode):
            return arg.name
        if isinstance(arg, str):
            return arg
        return self.visit(arg)

    def split_lambda_typed_parameter(self, raw):
        if not raw:
            return None
        if any(char in raw for char in "{}()"):
            return None
        parts = raw.rsplit(None, 1)
        if len(parts) != 2:
            return None
        type_name, param_name = parts
        if not param_name.isidentifier():
            return None
        return type_name, param_name

    def lambda_parameter_type(self, type_name):
        if "<" in type_name or ">" in type_name:
            return "auto"
        return self.map_type(type_name)

    def lambda_fallback_parameter_name(self, raw):
        if not raw:
            return ""
        candidate = raw.rsplit(None, 1)[-1]
        if candidate.isidentifier():
            return candidate
        return raw

    def generate_vector_constructor_args(self, vector_info, raw_args, args):
        """Flatten vector arguments passed to HIP make_* constructors."""
        if len(args) == 1:
            arg_type = self.expression_result_type(raw_args[0])
            if arg_type is not None and not self.vector_type_info(arg_type):
                return args * len(vector_info["components"])

        generated_args = []
        for raw_arg, arg_expr in zip(raw_args, args):
            arg_info = self.vector_type_info(self.expression_result_type(raw_arg))
            if arg_info is None:
                generated_args.append(arg_expr)
            else:
                generated_args.extend(
                    self.vector_argument_lane_expressions(
                        raw_arg,
                        arg_expr,
                        arg_info,
                    )
                )

            if len(generated_args) >= len(vector_info["components"]):
                return generated_args[: len(vector_info["components"])]

        return generated_args

    def vector_argument_lane_expressions(self, raw_arg, arg_expr, arg_info):
        swizzle_components = self.member_swizzle_components(raw_arg)
        if swizzle_components is not None:
            object_node = getattr(
                raw_arg,
                "object_expr",
                getattr(raw_arg, "object", None),
            )
            object_expr = self.visit(object_node)
            return [f"{object_expr}.{component}" for component in swizzle_components]

        return [f"{arg_expr}.{component}" for component in arg_info["components"]]

    def member_swizzle_components(self, node):
        if not isinstance(node, MemberAccessNode):
            return None

        object_node = getattr(node, "object_expr", getattr(node, "object", None))
        vector_info = self.vector_type_info(self.expression_result_type(object_node))
        if vector_info is None:
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
        member = getattr(node, "member", "")
        components = [component_aliases.get(component) for component in member]
        if not components or any(component is None for component in components):
            return None

        available_components = vector_info["components"]
        if any(component not in available_components for component in components):
            return None

        return components

    def generate_fract_call(self, raw_args, args):
        arg_type = self.expression_result_type(raw_args[0])
        vector_info = self.vector_type_info(arg_type)
        if vector_info is not None:
            helper_name = self.require_vector_fract_helper(vector_info)
            if helper_name is not None:
                return f"{helper_name}({args[0]})"

        scalar_type = self.fract_scalar_type(arg_type)
        helper_name = self.require_scalar_fract_helper(scalar_type)
        return f"{helper_name}({args[0]})"

    def fract_scalar_type(self, type_name):
        if type_name is not None and not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        mapped_type = self.map_type(type_name) if type_name is not None else None
        if mapped_type == "double" or type_name in {"double", "f64"}:
            return "double"
        return "float"

    def require_scalar_fract_helper(self, scalar_type):
        helper_name = f"cgl_fract_{scalar_type}"
        if helper_name in self.helper_functions:
            return helper_name

        floor_name = "floor" if scalar_type == "double" else "floorf"
        helper = (
            f"__device__ inline {scalar_type} {helper_name}({scalar_type} value)\n"
            "{\n"
            f"    return value - {floor_name}(value);\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def require_vector_fract_helper(self, vector_info):
        component_type = vector_info["component_type"]
        if component_type not in {"float", "double"}:
            return None

        vector_type = vector_info["type"]
        helper_name = f"cgl_{vector_type}_fract"
        if helper_name in self.helper_functions:
            return helper_name

        scalar_helper_name = self.require_scalar_fract_helper(component_type)
        components = [
            f"{scalar_helper_name}(value.{component})"
            for component in vector_info["components"]
        ]
        helper = (
            f"__device__ inline {vector_type} {helper_name}({vector_type} value)\n"
            "{\n"
            f"    return {vector_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def generate_clamp_call(self, raw_args, args):
        value_type = self.expression_result_type(raw_args[0])
        value_info = self.vector_type_info(value_type)
        if not value_info:
            scalar_type = self.clamp_scalar_type(value_type)
            scalar_call = self.generate_scalar_clamp_single_eval_call(
                scalar_type,
                raw_args,
                args,
            )
            if scalar_call is not None:
                return scalar_call
            return self.format_clamp_component(
                scalar_type,
                args[0],
                args[1],
                args[2],
            )

        min_info = self.vector_type_info(self.expression_result_type(raw_args[1]))
        max_info = self.vector_type_info(self.expression_result_type(raw_args[2]))
        if min_info and len(min_info["components"]) != len(value_info["components"]):
            return f"fmaxf({args[1]}, fminf({args[2]}, {args[0]}))"
        if max_info and len(max_info["components"]) != len(value_info["components"]):
            return f"fmaxf({args[1]}, fminf({args[2]}, {args[0]}))"

        helper_name = self.require_vector_clamp_helper(
            value_info,
            min_is_vector=min_info is not None,
            max_is_vector=max_info is not None,
        )
        if helper_name is None:
            return f"fmaxf({args[1]}, fminf({args[2]}, {args[0]}))"
        return f"{helper_name}({args[0]}, {args[1]}, {args[2]})"

    def generate_saturate_call(self, raw_args, args):
        value_type = self.expression_result_type(raw_args[0])
        value_info = self.vector_type_info(value_type)
        component_type = (
            value_info["component_type"]
            if value_info is not None
            else self.scalar_component_type(value_type)
        )
        if component_type not in {"float", "double"}:
            return None

        float_type = PrimitiveType("float")
        return self.generate_clamp_call(
            [
                raw_args[0],
                LiteralNode(0.0, float_type),
                LiteralNode(1.0, float_type),
            ],
            [args[0], "0.0", "1.0"],
        )

    def generate_mix_call(self, raw_args, args):
        left_type = self.expression_result_type(raw_args[0])
        right_type = self.expression_result_type(raw_args[1])
        factor_type = self.expression_result_type(raw_args[2])
        left_info = self.vector_type_info(left_type)
        right_info = self.vector_type_info(right_type)
        factor_info = self.vector_type_info(factor_type)

        if not left_info and not right_info and not factor_info:
            scalar_bool_mix = self.lower_bool_scalar_mix_operation(
                raw_args[0],
                args[0],
                raw_args[1],
                args[1],
                raw_args[2],
                args[2],
            )
            if scalar_bool_mix is not None:
                return scalar_bool_mix
            scalar_mix_call = self.generate_scalar_mix_single_eval_call(
                raw_args,
                args,
            )
            if scalar_mix_call is not None:
                return scalar_mix_call
            return self.format_mix_component(args[0], args[1], args[2])

        if left_info is None or right_info is None:
            return None
        if (
            len(left_info["components"]) != len(right_info["components"])
            or left_info["component_type"] != right_info["component_type"]
        ):
            return None

        bool_mix = self.lower_bool_vector_mix_operation(
            raw_args[0],
            args[0],
            raw_args[1],
            args[1],
            raw_args[2],
            args[2],
            left_info,
            right_info,
            factor_info,
        )
        if bool_mix is not None:
            return bool_mix

        if factor_info is not None and (
            len(factor_info["components"]) != len(left_info["components"])
            or factor_info["component_type"] != left_info["component_type"]
        ):
            return None

        helper_name = self.require_vector_mix_helper(
            left_info,
            factor_is_vector=factor_info is not None,
        )
        if helper_name is None:
            return None
        return f"{helper_name}({args[0]}, {args[1]}, {args[2]})"

    def require_vector_mix_helper(self, vector_info, factor_is_vector):
        component_type = vector_info["component_type"]
        if component_type not in {"float", "double"}:
            return None

        vector_type = vector_info["type"]
        factor_shape = "vector" if factor_is_vector else "scalar"
        helper_name = f"cgl_{vector_type}_mix_{factor_shape}"
        if helper_name in self.helper_functions:
            return helper_name

        factor_type = (
            vector_type
            if factor_is_vector
            else self.vector_scalar_parameter_type(vector_info)
        )
        components = []
        for component in vector_info["components"]:
            factor_component = f"a.{component}" if factor_is_vector else "a"
            components.append(
                self.format_mix_component(
                    f"x.{component}",
                    f"y.{component}",
                    factor_component,
                )
            )

        helper = (
            f"__device__ inline {vector_type} {helper_name}"
            f"({vector_type} x, {vector_type} y, {factor_type} a)\n"
            "{\n"
            f"    return {vector_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def format_mix_component(self, left, right, factor):
        return f"({left} + (({right} - {left}) * {factor}))"

    def generate_atan2_call(self, raw_args, args):
        y_type = self.expression_result_type(raw_args[0])
        x_type = self.expression_result_type(raw_args[1])
        y_info = self.vector_type_info(y_type)
        x_info = self.vector_type_info(x_type)

        if y_info is None and x_info is None:
            return None
        if y_info is None or x_info is None:
            return None
        if (
            len(y_info["components"]) != len(x_info["components"])
            or y_info["component_type"] != x_info["component_type"]
        ):
            return None

        helper_name = self.require_vector_atan2_helper(y_info)
        if helper_name is None:
            return None
        return f"{helper_name}({args[0]}, {args[1]})"

    def require_vector_atan2_helper(self, vector_info):
        component_type = vector_info["component_type"]
        if component_type not in {"float", "double"}:
            return None

        vector_type = vector_info["type"]
        helper_name = f"cgl_{vector_type}_atan2"
        if helper_name in self.helper_functions:
            return helper_name

        scalar_func = "atan2" if component_type == "double" else "atan2f"
        components = [
            f"{scalar_func}(y.{component}, x.{component})"
            for component in vector_info["components"]
        ]
        helper = (
            f"__device__ inline {vector_type} {helper_name}"
            f"({vector_type} y, {vector_type} x)\n"
            "{\n"
            f"    return {vector_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def clamp_scalar_type(self, type_name):
        if type_name is None:
            return "float"
        if not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        mapped_type = self.map_type(type_name)
        if mapped_type in {"float", "double"}:
            return mapped_type
        if mapped_type in {
            "bool",
            "char",
            "unsigned char",
            "short",
            "unsigned short",
            "int",
            "unsigned int",
            "long long",
            "unsigned long long",
        }:
            return mapped_type
        return "float"

    def require_vector_clamp_helper(self, vector_info, min_is_vector, max_is_vector):
        if vector_info["component_type"] == "bool":
            return None

        vector_type = vector_info["type"]
        scalar_type = self.vector_scalar_parameter_type(vector_info)
        min_shape = "vector" if min_is_vector else "scalar"
        max_shape = "vector" if max_is_vector else "scalar"
        helper_name = f"cgl_{vector_type}_clamp"
        if min_shape != "vector" or max_shape != "vector":
            helper_name += f"_{min_shape}_min_{max_shape}_max"
        if helper_name in self.helper_functions:
            return helper_name

        min_type = vector_type if min_is_vector else scalar_type
        max_type = vector_type if max_is_vector else scalar_type
        components = vector_info["components"]
        constructor = vector_info["constructor"]
        args = []
        for component in components:
            value_component = f"value.{component}"
            min_component = f"min_value.{component}" if min_is_vector else "min_value"
            max_component = f"max_value.{component}" if max_is_vector else "max_value"
            args.append(
                self.format_clamp_component(
                    vector_info["component_type"],
                    value_component,
                    min_component,
                    max_component,
                )
            )

        helper = (
            f"__device__ inline {vector_type} {helper_name}"
            f"({vector_type} value, {min_type} min_value, {max_type} max_value)\n"
            "{\n"
            f"    return {constructor}({', '.join(args)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def format_clamp_component(self, component_type, value, min_value, max_value):
        if component_type == "float":
            return f"fmaxf({min_value}, fminf({max_value}, {value}))"
        if component_type == "double":
            return f"fmax({min_value}, fmin({max_value}, {value}))"
        return (
            f"(({value}) < ({min_value}) ? ({min_value}) : "
            f"(({value}) > ({max_value}) ? ({max_value}) : ({value})))"
        )

    def insert_helper_functions(self):
        if not self.helper_functions:
            return
        helpers = []
        if self.resource_query_info_required:
            helpers.extend(
                [
                    "struct CglResourceQueryInfo {",
                    "    int width;",
                    "    int height;",
                    "    int depth;",
                    "    int elements;",
                    "    int levels;",
                    "    int samples;",
                    "};",
                    "",
                ]
            )
        for helper in self.helper_functions.values():
            helpers.extend(helper.splitlines())
            helpers.append("")
        self.code_lines[5:5] = helpers

    def require_hip_ray_runtime_helpers(self):
        helper_name = "cgl_ray_runtime_helpers"
        if helper_name in self.helper_functions:
            return

        self.helper_functions[helper_name] = (
            "struct CglRayTracingAccelerationStructure\n"
            "{\n"
            "    unsigned long long handle;\n"
            "};\n"
            "\n"
            "struct CglRayDesc\n"
            "{\n"
            "    float3 origin;\n"
            "    float t_min;\n"
            "    float3 direction;\n"
            "    float t_max;\n"
            "\n"
            "    __host__ __device__ CglRayDesc()\n"
            "        : origin(make_float3(0.0f, 0.0f, 0.0f)),\n"
            "          t_min(0.0f),\n"
            "          direction(make_float3(0.0f, 0.0f, 1.0f)),\n"
            "          t_max(0.0f)\n"
            "    {\n"
            "    }\n"
            "\n"
            "    __host__ __device__ CglRayDesc(\n"
            "        float3 origin_value,\n"
            "        float t_min_value,\n"
            "        float3 direction_value,\n"
            "        float t_max_value)\n"
            "        : origin(origin_value),\n"
            "          t_min(t_min_value),\n"
            "          direction(direction_value),\n"
            "          t_max(t_max_value)\n"
            "    {\n"
            "    }\n"
            "};\n"
            "\n"
            "struct CglRayQuery\n"
            "{\n"
            "    unsigned int state;\n"
            "};\n"
            "\n"
            "struct CglBuiltInTriangleIntersectionAttributes\n"
            "{\n"
            "    float2 barycentrics;\n"
            "};\n"
            "\n"
            "__device__ inline uint3 cgl_ray_launch_id()\n"
            "{\n"
            "    return make_uint3(0u, 0u, 0u);\n"
            "}\n"
            "\n"
            "__device__ inline uint3 cgl_ray_launch_size()\n"
            "{\n"
            "    return make_uint3(0u, 0u, 0u);\n"
            "}\n"
            "\n"
            "__device__ inline float cgl_ray_hit_t()\n"
            "{\n"
            "    return 0.0f;\n"
            "}\n"
            "\n"
            "__device__ inline unsigned int cgl_ray_hit_kind()\n"
            "{\n"
            "    return 0u;\n"
            "}\n"
            "\n"
            "__device__ inline float3 cgl_ray_world_origin()\n"
            "{\n"
            "    return make_float3(0.0f, 0.0f, 0.0f);\n"
            "}\n"
            "\n"
            "__device__ inline float3 cgl_ray_world_direction()\n"
            "{\n"
            "    return make_float3(0.0f, 0.0f, 0.0f);\n"
            "}\n"
            "\n"
            "__device__ inline float3 cgl_ray_object_origin()\n"
            "{\n"
            "    return make_float3(0.0f, 0.0f, 0.0f);\n"
            "}\n"
            "\n"
            "__device__ inline float3 cgl_ray_object_direction()\n"
            "{\n"
            "    return make_float3(0.0f, 0.0f, 0.0f);\n"
            "}\n"
            "\n"
            "__device__ inline float cgl_ray_t_min()\n"
            "{\n"
            "    return 0.0f;\n"
            "}\n"
            "\n"
            "__device__ inline unsigned int cgl_ray_incoming_flags()\n"
            "{\n"
            "    return 0u;\n"
            "}\n"
            "\n"
            "__device__ inline unsigned int cgl_ray_instance_custom_index()\n"
            "{\n"
            "    return 0u;\n"
            "}\n"
            "\n"
            "__device__ inline unsigned int cgl_ray_geometry_index()\n"
            "{\n"
            "    return 0u;\n"
            "}\n"
            "\n"
            "template <typename AS, typename Flags, typename Mask, "
            "typename HitGroup, typename Multiplier, typename Miss, "
            "typename Ray, typename Payload>\n"
            "__device__ inline void cgl_trace_ray(\n"
            "    const AS&,\n"
            "    const Flags&,\n"
            "    const Mask&,\n"
            "    const HitGroup&,\n"
            "    const Multiplier&,\n"
            "    const Miss&,\n"
            "    const Ray&,\n"
            "    const Payload&)\n"
            "{\n"
            "}\n"
            "\n"
            "template <typename Index, typename Data>\n"
            "__device__ inline void cgl_call_shader(const Index&, const Data&)\n"
            "{\n"
            "}\n"
            "\n"
            "template <typename Distance, typename Kind, typename Attributes>\n"
            "__device__ inline bool cgl_report_hit(\n"
            "    const Distance&,\n"
            "    const Kind&,\n"
            "    const Attributes&)\n"
            "{\n"
            "    return false;\n"
            "}\n"
            "\n"
            "__device__ inline void cgl_ignore_hit()\n"
            "{\n"
            "}\n"
            "\n"
            "__device__ inline void cgl_accept_hit_and_end_search()\n"
            "{\n"
            "}\n"
        )

    def require_hip_ray_query_helper(self, operation):
        self.require_hip_ray_runtime_helpers()
        helper_name = f"cgl_ray_query_{self.hip_snake_case_name(operation)}"
        if helper_name in self.helper_functions:
            return helper_name

        return_type = "bool" if operation == "Proceed" else "unsigned int"
        return_value = "false" if return_type == "bool" else "0u"
        self.helper_functions[helper_name] = (
            "template <typename Query, typename... Args>\n"
            f"__device__ inline {return_type} {helper_name}(Query&, Args&&...)\n"
            "{\n"
            f"    return {return_value};\n"
            "}"
        )
        return helper_name

    def hip_snake_case_name(self, name):
        text = str(name)
        result = []
        for index, char in enumerate(text):
            if char.isupper() and index > 0:
                previous = text[index - 1]
                next_char = text[index + 1] if index + 1 < len(text) else ""
                if previous.islower() or previous.isdigit() or next_char.islower():
                    result.append("_")
            result.append(char.lower())
        return "".join(result).replace("__", "_").strip("_")

    def register_variable_type(self, name, type_name, node=None, source_node=None):
        if not name or type_name is None:
            return
        if not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        self.variable_types[name] = type_name
        if self.is_glsl_buffer_block_node(node):
            access = self.explicit_resource_access(node) or "readwrite"
            self.glsl_buffer_block_accesses[name] = access
            self.glsl_buffer_block_layouts[name] = self.glsl_buffer_block_layout(node)
        else:
            self.glsl_buffer_block_accesses.pop(name, None)
            self.glsl_buffer_block_layouts.pop(name, None)
        if self.is_buffer_resource_type(type_name):
            access = self.explicit_resource_access(node)
            if access is None and source_node is not None:
                access = self.buffer_resource_access(source_node)
            if access is None:
                self.buffer_resource_accesses.pop(name, None)
            else:
                self.buffer_resource_accesses[name] = access
        else:
            self.buffer_resource_accesses.pop(name, None)
        if not self.is_storage_image_type(type_name):
            self.image_resource_accesses.pop(name, None)
            return

        access = self.explicit_resource_access(node)
        if access is None and source_node is not None:
            access = self.image_resource_access(source_node)
        if access is None:
            self.image_resource_accesses.pop(name, None)
        else:
            self.image_resource_accesses[name] = access

    @property
    def local_variable_types(self):
        return self.variable_types

    @local_variable_types.setter
    def local_variable_types(self, value):
        self.variable_types = value

    def next_hip_temp_variable(self, prefix):
        index = self.match_temp_variable_index
        self.match_temp_variable_index += 1
        return f"__crossgl_{prefix}_{index}"

    def get_expression_name(self, node):
        if isinstance(node, IdentifierNode):
            return node.name
        if isinstance(node, VariableNode):
            return node.name
        if isinstance(node, str):
            return node
        if isinstance(node, ArrayAccessNode):
            array_node = getattr(node, "array", getattr(node, "array_expr", None))
            return self.get_expression_name(array_node)
        return None

    def get_expression_type(self, node):
        name = self.get_expression_name(node)
        if name is None:
            return None
        return self.variable_types.get(name)

    def resource_expression_type(self, node):
        type_name = self.get_expression_type(node)
        if type_name is None:
            type_name = self.expression_result_type(node)
        if type_name is not None and not isinstance(type_name, str):
            return self.convert_type_node_to_string(type_name)
        return type_name

    def is_buffer_resource_type(self, type_name):
        """Return whether a type is a structured or byte-address buffer resource."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return False
        return (
            self.structured_buffer_type_parts(type_name) is not None
            or self.byte_address_buffer_base_type(type_name) is not None
        )

    def buffer_resource_access(self, buffer_arg):
        """Return explicit access metadata for a HIP buffer resource expression."""
        if isinstance(buffer_arg, ArrayAccessNode):
            array_node = getattr(
                buffer_arg, "array", getattr(buffer_arg, "array_expr", None)
            )
            return self.buffer_resource_access(array_node)

        buffer_name = self.get_expression_name(buffer_arg)
        if not buffer_name:
            return None
        return self.buffer_resource_accesses.get(buffer_name)

    def map_vector_arithmetic_type(self, type_name):
        return self.map_type(type_name)

    def require_surface_read_helper(self, helper_name):
        helpers = {
            "cgl_surf1Dread": (
                "template <typename T>\n"
                "__device__ T cgl_surf1Dread(hipSurfaceObject_t surfObj, int x)\n"
                "{\n"
                "    T value;\n"
                "    surf1Dread(&value, surfObj, x);\n"
                "    return value;\n"
                "}"
            ),
            "cgl_surf1DLayeredread": (
                "template <typename T>\n"
                "__device__ T cgl_surf1DLayeredread(hipSurfaceObject_t surfObj, int x, int layer)\n"
                "{\n"
                "    T value;\n"
                "    surf1DLayeredread(&value, surfObj, x, layer);\n"
                "    return value;\n"
                "}"
            ),
            "cgl_surf2Dread": (
                "template <typename T>\n"
                "__device__ T cgl_surf2Dread(hipSurfaceObject_t surfObj, int x, int y)\n"
                "{\n"
                "    T value;\n"
                "    surf2Dread(&value, surfObj, x, y);\n"
                "    return value;\n"
                "}"
            ),
            "cgl_surf3Dread": (
                "template <typename T>\n"
                "__device__ T cgl_surf3Dread(hipSurfaceObject_t surfObj, int x, int y, int z)\n"
                "{\n"
                "    T value;\n"
                "    surf3Dread(&value, surfObj, x, y, z);\n"
                "    return value;\n"
                "}"
            ),
            "cgl_surf2DLayeredread": (
                "template <typename T>\n"
                "__device__ T cgl_surf2DLayeredread(hipSurfaceObject_t surfObj, int x, int y, int layer)\n"
                "{\n"
                "    T value;\n"
                "    surf2DLayeredread(&value, surfObj, x, y, layer);\n"
                "    return value;\n"
                "}"
            ),
            "cgl_surfCubemapread": (
                "template <typename T>\n"
                "__device__ T cgl_surfCubemapread(hipSurfaceObject_t surfObj, int x, int y, int face)\n"
                "{\n"
                "    T value;\n"
                "    surfCubemapread(&value, surfObj, x, y, face);\n"
                "    return value;\n"
                "}"
            ),
            "cgl_surfCubemapLayeredread": (
                "template <typename T>\n"
                "__device__ T cgl_surfCubemapLayeredread(hipSurfaceObject_t surfObj, int x, int y, int face, int layer)\n"
                "{\n"
                "    T value;\n"
                "    surfCubemapLayeredread(&value, surfObj, x, y, face, layer);\n"
                "    return value;\n"
                "}"
            ),
        }
        self.require_helper_function(helper_name, helpers[helper_name])

    def vector_component_count_for_type(self, type_name):
        if not isinstance(type_name, str):
            return None

        base_type = type_name.split("[", 1)[0].strip()
        scalar_types = {
            "bool",
            "double",
            "f16",
            "f32",
            "f64",
            "float",
            "half",
            "i32",
            "int",
            "u32",
            "uint",
        }
        if base_type in scalar_types:
            return 1

        vector_prefixes = (
            "bvec",
            "dvec",
            "ivec",
            "uvec",
            "vec",
            "bool",
            "double",
            "float",
            "half",
            "int",
            "uint",
        )
        for prefix in vector_prefixes:
            if not base_type.startswith(prefix):
                continue
            suffix = base_type[len(prefix) :]
            if suffix and suffix[0] in {"2", "3", "4"}:
                return int(suffix[0])
        return None

    def texel_fetch_coordinate_count(self, raw_coord):
        coord_type = self.get_expression_type(raw_coord)
        if coord_type is None:
            coord_type = self.expression_result_type(raw_coord)
        return self.vector_component_count_for_type(coord_type)

    def expected_texel_fetch_coordinate_count(self, texture_type):
        return {
            "sampler1D": 1,
            "sampler1DArray": 2,
            "sampler2D": 2,
            "sampler2DArray": 3,
            "sampler3D": 3,
        }.get(texture_type)

    def expected_texture_coordinate_count(self, texture_type):
        return {
            "sampler1D": 1,
            "sampler1DArray": 2,
            "sampler2D": 2,
            "sampler2DArray": 3,
            "sampler3D": 3,
            "samplerCube": 3,
            "samplerCubeArray": 4,
        }.get(texture_type)

    def texture_coordinate_rank_diagnostic(self, func_name, texture_type, raw_coord):
        expected_coord_count = self.expected_texture_coordinate_count(texture_type)
        actual_coord_count = self.texel_fetch_coordinate_count(raw_coord)
        if (
            expected_coord_count is not None
            and actual_coord_count is not None
            and actual_coord_count != expected_coord_count
        ):
            return self.unsupported_sampled_resource_call(
                f"{func_name} coordinate rank",
                texture_type,
                [],
            )
        return None

    def expected_texture_gradient_coordinate_counts(self, texture_type):
        return {
            "sampler1D": {1},
            "sampler1DArray": {1},
            "sampler2D": {2},
            "sampler2DArray": {2},
            "sampler3D": {3, 4},
            "samplerCube": {3, 4},
            "samplerCubeArray": {3, 4},
        }.get(texture_type)

    def texture_gradient_rank_diagnostic(self, texture_type, *raw_gradients):
        expected_gradient_counts = self.expected_texture_gradient_coordinate_counts(
            texture_type
        )
        if expected_gradient_counts is None:
            return None

        for raw_gradient in raw_gradients:
            actual_gradient_count = self.texel_fetch_coordinate_count(raw_gradient)
            if (
                actual_gradient_count is not None
                and actual_gradient_count not in expected_gradient_counts
            ):
                return self.unsupported_sampled_resource_call(
                    "textureGrad derivative rank",
                    texture_type,
                    [],
                )
        return None

    def expected_projected_texture_coordinate_count(self, texture_type):
        return {
            "sampler1D": 2,
            "sampler2D": 3,
            "sampler3D": 4,
            "samplerCube": 4,
        }.get(texture_type)

    def projected_texture_coordinate_rank_diagnostic(
        self, func_name, texture_type, raw_coord
    ):
        expected_coord_count = self.expected_projected_texture_coordinate_count(
            texture_type
        )
        actual_coord_count = self.texel_fetch_coordinate_count(raw_coord)
        if (
            expected_coord_count is not None
            and actual_coord_count is not None
            and actual_coord_count != expected_coord_count
        ):
            return self.unsupported_sampled_resource_call(
                f"{func_name} coordinate rank",
                texture_type,
                [],
            )
        return None

    def projected_coord_args(self, texture_type, texture_name, coord):
        projection_component = {
            "sampler1D": "y",
            "sampler2D": "z",
            "sampler3D": "w",
            "samplerCube": "w",
        }.get(texture_type)
        if projection_component is None:
            return None

        projection = self.coord_component(coord, projection_component)
        components = ["x"]
        if texture_type in {"sampler2D", "sampler3D", "samplerCube"}:
            components.append("y")
        if texture_type in {"sampler3D", "samplerCube"}:
            components.append("z")
        projected_components = [
            f"({self.coord_component(coord, component)} / {projection})"
            for component in components
        ]
        return ", ".join([texture_name, *projected_components])

    def expected_texel_fetch_offset_count(self, texture_type):
        return {
            "sampler1D": 1,
            "sampler1DArray": 1,
            "sampler2D": 2,
            "sampler2DArray": 2,
            "sampler3D": 3,
        }.get(texture_type)

    def texel_fetch_offset_rank_diagnostic(self, texture_type, raw_offset):
        expected_offset_count = self.expected_texel_fetch_offset_count(texture_type)
        actual_offset_count = self.texel_fetch_coordinate_count(raw_offset)
        if (
            expected_offset_count is not None
            and actual_offset_count is not None
            and actual_offset_count != expected_offset_count
        ):
            return self.unsupported_sampled_resource_call(
                "texelFetchOffset offset rank",
                texture_type,
                [],
            )
        return None

    def offset_coord_component(self, coord, offset, component):
        return (
            f"({self.coord_component(coord, component)} + "
            f"{self.coord_component(offset, component)})"
        )

    def hip_texture_gradient_argument(self, texture_type, raw_gradient, gradient):
        if texture_type not in {"sampler3D", "samplerCube", "samplerCubeArray"}:
            return gradient

        gradient_count = self.texel_fetch_coordinate_count(raw_gradient)
        if gradient_count == 3:
            return (
                f"make_float4({self.coord_component(gradient, 'x')}, "
                f"{self.coord_component(gradient, 'y')}, "
                f"{self.coord_component(gradient, 'z')}, 0.0f)"
            )
        return gradient

    def expected_image_coordinate_count(self, image_type):
        image_type = self.resource_base_type(image_type)
        if not isinstance(image_type, str):
            return None
        if "1DArray" in image_type:
            return 2
        if "1D" in image_type:
            return 1
        if "CubeArray" in image_type:
            return 4
        if "3D" in image_type:
            return 3
        if "Cube" in image_type:
            return 3
        if "Array" in image_type:
            return 3
        if "2D" in image_type:
            return 2
        return None

    def unsupported_image_coordinate_rank_call(self, func_name, image_type):
        fallback = "((void)0)"
        if func_name == "imageLoad":
            fallback = self.zero_value_for_type(self.image_value_type(image_type))
        return (
            f"/* unsupported {self.resource_backend_name()} image resource call: "
            f"{func_name} coordinate rank on {image_type} */ {fallback}"
        )

    def image_coordinate_rank_diagnostic(self, func_name, image_type, raw_coord):
        expected_coord_count = self.expected_image_coordinate_count(image_type)
        actual_coord_count = self.texel_fetch_coordinate_count(raw_coord)
        if (
            expected_coord_count is not None
            and actual_coord_count is not None
            and actual_coord_count != expected_coord_count
        ):
            return self.unsupported_image_coordinate_rank_call(func_name, image_type)
        return None

    def image_resource_access(self, image_arg):
        if isinstance(image_arg, ArrayAccessNode):
            array_node = getattr(
                image_arg, "array", getattr(image_arg, "array_expr", None)
            )
            return self.image_resource_access(array_node)

        image_name = self.get_expression_name(image_arg)
        if not image_name:
            return None
        return self.image_resource_accesses.get(image_name)

    def unsupported_image_access_call(self, func_name, image_type, reason):
        image_type = image_type or "unknown resource"
        fallback = "((void)0)"
        if func_name == "imageLoad":
            fallback = self.zero_value_for_type(self.image_value_type(image_type))
        return (
            f"/* unsupported {self.resource_backend_name()} image access: "
            f"{func_name} {reason} on {image_type} */ {fallback}"
        )

    def image_access_diagnostic(self, func_name, image_type, raw_image):
        access = self.image_resource_access(raw_image)
        if func_name == "imageLoad" and access == "writeonly":
            return self.unsupported_image_access_call(
                func_name,
                image_type,
                "requires readable image resource",
            )
        if func_name == "imageStore" and access == "readonly":
            return self.unsupported_image_access_call(
                func_name,
                image_type,
                "requires writable image resource",
            )
        return None

    def unsupported_image_atomic_coordinate_rank_call(self, func_name, image_type):
        image_type = image_type or "unknown resource"
        return (
            f"/* unsupported {self.resource_backend_name()} image atomic resource call: "
            f"{func_name} coordinate rank on {image_type} */ "
            f"{self.image_atomic_zero_value(image_type)}"
        )

    def image_atomic_coordinate_rank_diagnostic(self, func_name, image_type, raw_coord):
        expected_coord_count = self.expected_image_coordinate_count(image_type)
        actual_coord_count = self.texel_fetch_coordinate_count(raw_coord)
        if (
            expected_coord_count is not None
            and actual_coord_count is not None
            and actual_coord_count != expected_coord_count
        ):
            return self.unsupported_image_atomic_coordinate_rank_call(
                func_name, image_type
            )
        return None

    def unsupported_image_atomic_access_call(self, func_name, image_type):
        image_type = image_type or "unknown resource"
        return (
            f"/* unsupported {self.resource_backend_name()} image atomic resource call: "
            f"{func_name} requires readwrite image resource on {image_type} */ "
            f"{self.image_atomic_zero_value(image_type)}"
        )

    def image_atomic_access_diagnostic(self, func_name, image_type, raw_image):
        if self.image_resource_access(raw_image) in {"readonly", "writeonly"}:
            return self.unsupported_image_atomic_access_call(func_name, image_type)
        return None

    def image_surface_x_offset(self, coord, value_type, image_type):
        if "1D" in image_type and "Array" not in image_type:
            return f"{coord} * sizeof({value_type})"
        return self.surface_x_offset(coord, value_type)

    def is_storage_image_type(self, type_name):
        if not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        base_type = self.resource_base_type(type_name)
        if not isinstance(base_type, str):
            return False
        image_shapes = {
            "image1D",
            "image1DArray",
            "image2D",
            "image2DArray",
            "image3D",
            "imageCube",
            "imageCubeArray",
            "image2DMS",
            "image2DMSArray",
        }
        return base_type in image_shapes or any(
            base_type == f"{prefix}{shape}"
            for prefix in ("i", "u")
            for shape in image_shapes
        )

    def unsupported_scalar_resource_query_call(self, func_name, resource_type):
        resource_type = resource_type or "unknown resource"
        return (
            f"/* unsupported {self.resource_backend_name()} resource query: "
            f"{func_name} on {resource_type} */ 0"
        )

    def unsupported_dimension_resource_query_call(self, func_name, resource_type):
        spec = self.dimension_query_spec(resource_type)
        if spec is None:
            return None
        return_type = self.query_return_type(spec["dimensions"])
        fallback = self.query_constructor(
            return_type,
            ["0"] * len(spec["dimensions"]),
        )
        return (
            f"/* unsupported {self.resource_backend_name()} resource query: "
            f"{func_name} on {resource_type} */ {fallback}"
        )

    def generate_dimension_query(self, func_name, raw_args, args):
        if not raw_args:
            return None

        resource_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
        if resource_type is None:
            return None
        if func_name == "textureSize" and not self.is_sampled_resource_type(
            resource_type
        ):
            return self.unsupported_dimension_resource_query_call(
                func_name, resource_type
            )
        if func_name == "imageSize" and not self.is_storage_image_type(resource_type):
            return self.unsupported_dimension_resource_query_call(
                func_name, resource_type
            )
        return ResourceQueryMixin.generate_dimension_query(
            self, func_name, raw_args, args
        )

    def generate_sample_count_query(self, func_name, raw_args, args):
        if not raw_args:
            return None

        resource_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
        if resource_type is None:
            return None

        expected_resource = (
            self.is_sampled_resource_type(resource_type)
            if func_name == "textureSamples"
            else self.is_storage_image_type(resource_type)
        )
        sample_count_query = None
        if expected_resource:
            sample_count_query = ResourceQueryMixin.generate_sample_count_query(
                self, func_name, raw_args, args
            )
        if sample_count_query is not None:
            return sample_count_query
        return self.unsupported_scalar_resource_query_call(func_name, resource_type)

    def generate_texture_query_levels(self, raw_args):
        if not raw_args:
            return None

        resource_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
        if resource_type is None:
            return None
        spec = self.dimension_query_spec(resource_type)
        if (
            not self.is_sampled_resource_type(resource_type)
            or spec is None
            or not spec["mip"]
        ):
            return self.unsupported_scalar_resource_query_call(
                "textureQueryLevels", resource_type
            )
        return ResourceQueryMixin.generate_texture_query_levels(self, raw_args)

    def is_sampler_state_resource_type(self, type_name):
        return self.resource_base_type(type_name) == "sampler"

    def has_explicit_sampler_argument(self, raw_args):
        if len(raw_args) < 3:
            return False
        return self.is_sampler_state_resource_type(
            self.resource_expression_type(raw_args[1])
        )

    def normalize_split_sampler_texture_args(self, func_name, raw_args, args):
        split_sampler_functions = {
            "texture",
            "textureLod",
            "textureGrad",
            "textureGather",
            "textureGatherOffset",
            "textureGatherOffsets",
            "textureCompare",
            "textureCompareLod",
            "textureCompareGrad",
            "textureCompareOffset",
            "textureCompareLodOffset",
            "textureCompareGradOffset",
            "textureCompareProj",
            "textureCompareProjOffset",
            "textureCompareProjLod",
            "textureCompareProjLodOffset",
            "textureCompareProjGrad",
            "textureCompareProjGradOffset",
            "textureGatherCompare",
            "textureGatherCompareOffset",
            "textureOffset",
            "textureLodOffset",
            "textureGradOffset",
            "textureProj",
            "textureProjOffset",
            "textureProjLod",
            "textureProjLodOffset",
            "textureProjGrad",
            "textureProjGradOffset",
            "textureQueryLod",
        }
        if (
            func_name not in split_sampler_functions
            or not self.has_explicit_sampler_argument(raw_args)
        ):
            return raw_args, args
        return [raw_args[0], *raw_args[2:]], [args[0], *args[2:]]

    def generate_resource_call(self, func_name, raw_args, args):
        if func_name in {"textureSize", "imageSize"}:
            return self.generate_dimension_query(func_name, raw_args, args)

        if func_name in {"textureSamples", "imageSamples"}:
            return self.generate_sample_count_query(func_name, raw_args, args)

        if func_name == "textureQueryLevels":
            return self.generate_texture_query_levels(raw_args)

        raw_args, args = self.normalize_split_sampler_texture_args(
            func_name, raw_args, args
        )

        if func_name == "textureQueryLod" and len(args) >= 2:
            texture_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
            )
            if texture_type is not None:
                return self.unsupported_resource_query_call(
                    func_name, texture_type, args
                )

        if (
            func_name
            in {
                "texture",
                "textureLod",
                "textureGrad",
                "textureGather",
                "textureCompare",
                "textureCompareLod",
                "textureCompareGrad",
                "textureCompareOffset",
                "textureCompareLodOffset",
                "textureCompareGradOffset",
                "textureCompareProj",
                "textureCompareProjOffset",
                "textureCompareProjLod",
                "textureCompareProjLodOffset",
                "textureCompareProjGrad",
                "textureCompareProjGradOffset",
                "textureGatherCompare",
                "textureGatherCompareOffset",
            }
            and len(args) >= 2
        ):
            texture_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
            )
            if self.is_shadow_resource_type(texture_type):
                return self.unsupported_shadow_resource_call(
                    func_name, texture_type, args
                )

        if func_name == "textureGather" and len(args) >= 2:
            texture_gather = self.generate_texture_gather_call(raw_args, args)
            if texture_gather is not None:
                return texture_gather

        if (
            func_name
            in {
                "textureGather",
                "textureGatherOffset",
                "textureGatherOffsets",
            }
            and len(args) >= 2
        ):
            texture_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
            )
            if texture_type is not None:
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )

        if (
            func_name
            in {
                "textureProj",
                "textureProjLod",
                "textureProjGrad",
            }
            and len(args) >= 2
        ):
            projected_texture = self.generate_projected_texture_call(
                func_name, raw_args, args
            )
            if projected_texture is not None:
                return projected_texture

        if func_name == "texelFetchOffset" and len(args) >= 4:
            texel_fetch_offset = self.generate_texel_fetch_offset_call(raw_args, args)
            if texel_fetch_offset is not None:
                return texel_fetch_offset

        if (
            func_name
            in {
                "textureOffset",
                "textureLodOffset",
                "textureGradOffset",
                "textureProj",
                "textureProjOffset",
                "textureProjLod",
                "textureProjLodOffset",
                "textureProjGrad",
                "textureProjGradOffset",
                "texelFetchOffset",
            }
            and raw_args
        ):
            texture_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
            )
            if texture_type is not None:
                if self.is_shadow_resource_type(texture_type):
                    return self.unsupported_shadow_resource_call(
                        func_name, texture_type, args
                    )
                if self.is_multisample_resource_type(texture_type):
                    return self.unsupported_multisample_resource_call(
                        func_name, texture_type, args
                    )
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )

        if func_name in {
            "imageAtomicAdd",
            "imageAtomicMin",
            "imageAtomicMax",
            "imageAtomicAnd",
            "imageAtomicOr",
            "imageAtomicXor",
            "imageAtomicExchange",
            "imageAtomicCompSwap",
        }:
            image_type = None
            if raw_args:
                image_type = self.resource_base_type(
                    self.get_expression_type(raw_args[0])
                )
                access_diagnostic = self.image_atomic_access_diagnostic(
                    func_name, image_type, raw_args[0]
                )
                if access_diagnostic is not None:
                    return access_diagnostic
            if len(raw_args) >= 2:
                coordinate_diagnostic = self.image_atomic_coordinate_rank_diagnostic(
                    func_name, image_type, raw_args[1]
                )
                if coordinate_diagnostic is not None:
                    return coordinate_diagnostic
            return self.unsupported_image_atomic_resource_call(
                func_name, image_type, args
            )

        if func_name in {"texture", "textureLod", "textureGrad"} and len(args) >= 2:
            texture_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
            )
            if self.is_multisample_resource_type(texture_type):
                return self.unsupported_multisample_resource_call(
                    func_name, texture_type, args
                )
            coordinate_diagnostic = self.texture_coordinate_rank_diagnostic(
                func_name, texture_type, raw_args[1]
            )
            if coordinate_diagnostic is not None:
                return coordinate_diagnostic
            grad_x = None
            grad_y = None
            if func_name == "textureGrad" and len(args) >= 4:
                gradient_diagnostic = self.texture_gradient_rank_diagnostic(
                    texture_type,
                    raw_args[2],
                    raw_args[3],
                )
                if gradient_diagnostic is not None:
                    return gradient_diagnostic
                grad_x = self.hip_texture_gradient_argument(
                    texture_type,
                    raw_args[2],
                    args[2],
                )
                grad_y = self.hip_texture_gradient_argument(
                    texture_type,
                    raw_args[3],
                    args[3],
                )

            texture_name = args[0]
            coord = args[1]
            if texture_type == "sampler1D":
                if func_name == "texture":
                    return f"tex1D<float4>({texture_name}, {coord})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex1DLod<float4>({texture_name}, {coord}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return (
                        f"tex1DGrad<float4>"
                        f"({texture_name}, {coord}, {grad_x}, {grad_y})"
                    )

            if texture_type == "sampler2D":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}"
                )
                if func_name == "texture":
                    return f"tex2D<float4>({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex2DLod<float4>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return f"tex2DGrad<float4>" f"({coord_args}, {grad_x}, {grad_y})"

            if texture_type == "sampler1DArray":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}"
                )
                if func_name == "texture":
                    return f"tex1DLayered<float4>({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex1DLayeredLod<float4>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return (
                        f"tex1DLayeredGrad<float4>"
                        f"({coord_args}, {grad_x}, {grad_y})"
                    )

            if texture_type == "sampler2DArray":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}"
                )
                if func_name == "texture":
                    return f"tex2DLayered<float4>({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex2DLayeredLod<float4>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return (
                        f"tex2DLayeredGrad<float4>"
                        f"({coord_args}, {grad_x}, {grad_y})"
                    )

            if texture_type == "sampler3D":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}"
                )
                if func_name == "texture":
                    return f"tex3D<float4>({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex3DLod<float4>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return f"tex3DGrad<float4>({coord_args}, {grad_x}, {grad_y})"

            if texture_type == "samplerCube":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}"
                )
                if func_name == "texture":
                    return f"texCubemap<float4>({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"texCubemapLod<float4>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return (
                        f"texCubemapGrad<float4>" f"({coord_args}, {grad_x}, {grad_y})"
                    )

            if texture_type == "samplerCubeArray":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}, "
                    f"{self.coord_component(coord, 'w')}"
                )
                if func_name == "texture":
                    return f"texCubemapLayered<float4>({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"texCubemapLayeredLod<float4>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return (
                        f"texCubemapLayeredGrad<float4>"
                        f"({coord_args}, {grad_x}, {grad_y})"
                    )

        if func_name == "texelFetch" and len(args) >= 3:
            texture_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
            )
            if texture_type is not None and not self.is_sampled_resource_type(
                texture_type
            ):
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )
            if self.is_shadow_resource_type(texture_type):
                return self.unsupported_shadow_resource_call(
                    func_name, texture_type, args
                )
            if self.is_multisample_resource_type(texture_type):
                return self.unsupported_multisample_resource_call(
                    func_name, texture_type, args
                )
            if texture_type in {"samplerCube", "samplerCubeArray"}:
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )

            texture_name = args[0]
            coord = args[1]
            expected_coord_count = self.expected_texel_fetch_coordinate_count(
                texture_type
            )
            actual_coord_count = self.texel_fetch_coordinate_count(raw_args[1])
            if (
                expected_coord_count is not None
                and actual_coord_count is not None
                and actual_coord_count != expected_coord_count
            ):
                return self.unsupported_sampled_resource_call(
                    "texelFetch coordinate rank",
                    texture_type,
                    args,
                )
            if texture_type == "sampler1D":
                return f"tex1Dfetch<float4>({texture_name}, {coord})"
            if texture_type == "sampler1DArray":
                return (
                    f"tex1DLayered<float4>({texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')})"
                )
            if texture_type == "sampler2D":
                return (
                    f"tex2D<float4>({texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')})"
                )
            if texture_type == "sampler2DArray":
                return (
                    f"tex2DLayered<float4>({texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')})"
                )
            if texture_type == "sampler3D":
                return (
                    f"tex3D<float4>({texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')})"
                )

        if func_name == "imageLoad" and len(args) >= 2:
            image_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
            if image_type is None:
                return None
            if self.is_multisample_resource_type(image_type):
                return self.unsupported_multisample_resource_call(
                    func_name, image_type, args
                )
            access_diagnostic = self.image_access_diagnostic(
                func_name, image_type, raw_args[0]
            )
            if access_diagnostic is not None:
                return access_diagnostic

            image_name = args[0]
            coord = args[1]
            coordinate_diagnostic = self.image_coordinate_rank_diagnostic(
                func_name, image_type, raw_args[1]
            )
            if coordinate_diagnostic is not None:
                return coordinate_diagnostic
            value_type = self.image_value_type(image_type)
            x = self.image_surface_x_offset(coord, value_type, image_type)
            y = self.coord_component(coord, "y")

            if "1DArray" in image_type:
                self.require_surface_read_helper("cgl_surf1DLayeredread")
                layer = self.coord_component(coord, "y")
                return (
                    f"cgl_surf1DLayeredread<{value_type}>"
                    f"({image_name}, {x}, {layer})"
                )
            if "1D" in image_type:
                self.require_surface_read_helper("cgl_surf1Dread")
                return f"cgl_surf1Dread<{value_type}>({image_name}, {x})"
            if "CubeArray" in image_type:
                self.require_surface_read_helper("cgl_surfCubemapLayeredread")
                face = self.coord_component(coord, "z")
                layer = self.coord_component(coord, "w")
                return (
                    f"cgl_surfCubemapLayeredread<{value_type}>"
                    f"({image_name}, {x}, {y}, {face}, {layer})"
                )
            if "Cube" in image_type:
                self.require_surface_read_helper("cgl_surfCubemapread")
                face = self.coord_component(coord, "z")
                return (
                    f"cgl_surfCubemapread<{value_type}>"
                    f"({image_name}, {x}, {y}, {face})"
                )
            if "3D" in image_type:
                self.require_surface_read_helper("cgl_surf3Dread")
                z = self.coord_component(coord, "z")
                return f"cgl_surf3Dread<{value_type}>({image_name}, {x}, {y}, {z})"
            if "Array" in image_type:
                self.require_surface_read_helper("cgl_surf2DLayeredread")
                layer = self.coord_component(coord, "z")
                return (
                    f"cgl_surf2DLayeredread<{value_type}>"
                    f"({image_name}, {x}, {y}, {layer})"
                )
            if "2D" in image_type:
                self.require_surface_read_helper("cgl_surf2Dread")
                return f"cgl_surf2Dread<{value_type}>({image_name}, {x}, {y})"

        if func_name == "imageStore" and len(args) >= 3:
            image_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
            if image_type is None:
                return None
            if self.is_multisample_resource_type(image_type):
                return self.unsupported_multisample_resource_call(
                    func_name, image_type, args
                )
            access_diagnostic = self.image_access_diagnostic(
                func_name, image_type, raw_args[0]
            )
            if access_diagnostic is not None:
                return access_diagnostic

            image_name = args[0]
            coord = args[1]
            value = args[2]
            coordinate_diagnostic = self.image_coordinate_rank_diagnostic(
                func_name, image_type, raw_args[1]
            )
            if coordinate_diagnostic is not None:
                return coordinate_diagnostic
            value_type = self.image_value_type(image_type)
            x = self.image_surface_x_offset(coord, value_type, image_type)
            y = self.coord_component(coord, "y")

            if "1DArray" in image_type:
                layer = self.coord_component(coord, "y")
                return f"surf1DLayeredwrite({value}, {image_name}, {x}, {layer})"
            if "1D" in image_type:
                return f"surf1Dwrite({value}, {image_name}, {x})"
            if "CubeArray" in image_type:
                face = self.coord_component(coord, "z")
                layer = self.coord_component(coord, "w")
                return (
                    f"surfCubemapLayeredwrite"
                    f"({value}, {image_name}, {x}, {y}, {face}, {layer})"
                )
            if "Cube" in image_type:
                face = self.coord_component(coord, "z")
                return f"surfCubemapwrite({value}, {image_name}, {x}, {y}, {face})"
            if "3D" in image_type:
                z = self.coord_component(coord, "z")
                return f"surf3Dwrite({value}, {image_name}, {x}, {y}, {z})"
            if "Array" in image_type:
                layer = self.coord_component(coord, "z")
                return f"surf2DLayeredwrite({value}, {image_name}, {x}, {y}, {layer})"
            if "2D" in image_type:
                return f"surf2Dwrite({value}, {image_name}, {x}, {y})"

        return None

    def generate_texel_fetch_offset_call(self, raw_args, args):
        texture_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
        if texture_type is not None and not self.is_sampled_resource_type(texture_type):
            return self.unsupported_sampled_resource_call(
                "texelFetchOffset", texture_type, args
            )
        if self.is_shadow_resource_type(texture_type):
            return self.unsupported_shadow_resource_call(
                "texelFetchOffset", texture_type, args
            )
        if self.is_multisample_resource_type(texture_type):
            return self.unsupported_multisample_resource_call(
                "texelFetchOffset", texture_type, args
            )
        if texture_type in {"samplerCube", "samplerCubeArray"}:
            return self.unsupported_sampled_resource_call(
                "texelFetchOffset", texture_type, args
            )

        coordinate_diagnostic = self.texture_coordinate_rank_diagnostic(
            "texelFetchOffset",
            texture_type,
            raw_args[1],
        )
        if coordinate_diagnostic is not None:
            return coordinate_diagnostic

        offset_diagnostic = self.texel_fetch_offset_rank_diagnostic(
            texture_type,
            raw_args[3],
        )
        if offset_diagnostic is not None:
            return offset_diagnostic

        texture_name = args[0]
        coord = args[1]
        offset = args[3]
        if texture_type == "sampler1D":
            return f"tex1Dfetch<float4>({texture_name}, ({coord} + {offset}))"
        if texture_type == "sampler1DArray":
            return (
                f"tex1DLayered<float4>({texture_name}, "
                f"({self.coord_component(coord, 'x')} + {offset}), "
                f"{self.coord_component(coord, 'y')})"
            )
        if texture_type == "sampler2D":
            return (
                f"tex2D<float4>({texture_name}, "
                f"{self.offset_coord_component(coord, offset, 'x')}, "
                f"{self.offset_coord_component(coord, offset, 'y')})"
            )
        if texture_type == "sampler2DArray":
            return (
                f"tex2DLayered<float4>({texture_name}, "
                f"{self.offset_coord_component(coord, offset, 'x')}, "
                f"{self.offset_coord_component(coord, offset, 'y')}, "
                f"{self.coord_component(coord, 'z')})"
            )
        if texture_type == "sampler3D":
            return (
                f"tex3D<float4>({texture_name}, "
                f"{self.offset_coord_component(coord, offset, 'x')}, "
                f"{self.offset_coord_component(coord, offset, 'y')}, "
                f"{self.offset_coord_component(coord, offset, 'z')})"
            )
        return None

    def generate_projected_texture_call(self, func_name, raw_args, args):
        texture_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
        if texture_type is None:
            return None
        if self.is_shadow_resource_type(texture_type):
            return self.unsupported_shadow_resource_call(func_name, texture_type, args)
        if self.is_multisample_resource_type(texture_type):
            return self.unsupported_multisample_resource_call(
                func_name, texture_type, args
            )

        coord = args[1]
        coordinate_diagnostic = self.projected_texture_coordinate_rank_diagnostic(
            func_name, texture_type, raw_args[1]
        )
        if coordinate_diagnostic is not None:
            return coordinate_diagnostic

        grad_x = None
        grad_y = None
        if func_name == "textureProjGrad" and len(args) >= 4:
            gradient_diagnostic = self.texture_gradient_rank_diagnostic(
                texture_type,
                raw_args[2],
                raw_args[3],
            )
            if gradient_diagnostic is not None:
                return gradient_diagnostic
            grad_x = self.hip_texture_gradient_argument(
                texture_type,
                raw_args[2],
                args[2],
            )
            grad_y = self.hip_texture_gradient_argument(
                texture_type,
                raw_args[3],
                args[3],
            )

        texture_name = args[0]
        coord_args = self.projected_coord_args(texture_type, texture_name, coord)
        if coord_args is None:
            return self.unsupported_sampled_resource_call(func_name, texture_type, args)

        if texture_type == "sampler1D":
            if func_name == "textureProj":
                return f"tex1D<float4>({coord_args})"
            if func_name == "textureProjLod" and len(args) >= 3:
                return f"tex1DLod<float4>({coord_args}, {args[2]})"
            if func_name == "textureProjGrad" and grad_x is not None:
                return f"tex1DGrad<float4>({coord_args}, {grad_x}, {grad_y})"

        if texture_type == "sampler2D":
            if func_name == "textureProj":
                return f"tex2D<float4>({coord_args})"
            if func_name == "textureProjLod" and len(args) >= 3:
                return f"tex2DLod<float4>({coord_args}, {args[2]})"
            if func_name == "textureProjGrad" and grad_x is not None:
                return f"tex2DGrad<float4>({coord_args}, {grad_x}, {grad_y})"

        if texture_type == "sampler3D":
            if func_name == "textureProj":
                return f"tex3D<float4>({coord_args})"
            if func_name == "textureProjLod" and len(args) >= 3:
                return f"tex3DLod<float4>({coord_args}, {args[2]})"
            if func_name == "textureProjGrad" and grad_x is not None:
                return f"tex3DGrad<float4>({coord_args}, {grad_x}, {grad_y})"

        if texture_type == "samplerCube":
            if func_name == "textureProj":
                return f"texCubemap<float4>({coord_args})"
            if func_name == "textureProjLod" and len(args) >= 3:
                return f"texCubemapLod<float4>({coord_args}, {args[2]})"
            if func_name == "textureProjGrad" and grad_x is not None:
                return f"texCubemapGrad<float4>({coord_args}, {grad_x}, {grad_y})"

        return self.unsupported_sampled_resource_call(func_name, texture_type, args)

    def generate_texture_gather_call(self, raw_args, args):
        texture_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
        if texture_type != "sampler2D":
            return None

        coordinate_index = 1
        component = None
        component_arg = None
        if len(args) == 2:
            pass
        elif len(args) == 3:
            component = args[2]
            component_arg = raw_args[2]
        else:
            return None

        coordinate_diagnostic = self.texture_coordinate_rank_diagnostic(
            "textureGather",
            texture_type,
            raw_args[coordinate_index],
        )
        if coordinate_diagnostic is not None:
            return coordinate_diagnostic

        component_diagnostic = self.texture_gather_component_diagnostic(
            texture_type, component_arg
        )
        if component_diagnostic is not None:
            return component_diagnostic

        texture_name = args[0]
        coord = args[coordinate_index]
        gather_args = (
            f"{texture_name}, "
            f"{self.coord_component(coord, 'x')}, "
            f"{self.coord_component(coord, 'y')}"
        )
        if component is not None:
            return f"tex2Dgather<float4>({gather_args}, {component})"
        return f"tex2Dgather<float4>({gather_args})"

    def texture_gather_component_diagnostic(self, texture_type, component_arg):
        if not self.is_texture_gather_literal_component(component_arg):
            return None
        component_value = self.literal_int_value(component_arg)
        if component_value in {0, 1, 2, 3}:
            return None
        return self.unsupported_sampled_resource_call(
            "textureGather component literal must be 0, 1, 2, or 3",
            texture_type,
            [],
        )

    def is_texture_gather_literal_component(self, node):
        if isinstance(node, LiteralNode):
            return True
        if isinstance(node, UnaryOpNode) and not getattr(node, "is_postfix", False):
            operator = getattr(node, "operator", getattr(node, "op", None))
            if operator in {"+", "-"}:
                return self.is_texture_gather_literal_component(node.operand)
        return False

    def generate_buffer_call(self, function_expr, func_name, raw_args, args):
        """Lower structured and byte-address buffer calls to HIP pointer forms."""
        byte_address_call = self.generate_byte_address_buffer_call(
            function_expr, func_name, raw_args, args
        )
        if byte_address_call is not None:
            return byte_address_call

        member_call = self.structured_buffer_member_call(function_expr)
        if member_call is not None:
            buffer_expr, operation, buffer_type = member_call
            if operation == "Append" and args:
                return self.generate_structured_buffer_append(
                    buffer_expr, buffer_type, args[0], operation
                )
            if operation == "Consume":
                return self.generate_structured_buffer_consume(
                    buffer_expr, buffer_type, operation
                )
            if operation == "GetDimensions":
                return self.generate_structured_buffer_dimensions(
                    buffer_expr, buffer_type, raw_args, args, operation
                )
            access = self.format_structured_buffer_access(buffer_expr, raw_args, args)
            if access is None:
                return None
            if operation == "Load":
                parts = self.structured_buffer_type_parts(buffer_type)
                fallback = self.diagnostic_zero_value_for_type(
                    parts[1] if parts is not None else None
                )
                diagnostic = self.structured_buffer_read_diagnostic(
                    buffer_expr, buffer_type, operation, fallback
                )
                if diagnostic is not None:
                    return diagnostic
                return access
            if operation == "Store":
                diagnostic = self.structured_buffer_write_diagnostic(
                    buffer_expr, buffer_type, operation
                )
                if diagnostic is not None:
                    return diagnostic
                if self.structured_buffer_is_writable(buffer_type) and len(args) >= 2:
                    return f"{access} = {args[1]}"
                return self.structured_buffer_diagnostic_call(
                    "Store", buffer_type, "((void)0)"
                )
            return None

        if func_name == "buffer_load" and len(args) >= 2:
            buffer_type = self.expression_result_type(raw_args[0])
            parts = self.structured_buffer_type_parts(buffer_type)
            if parts is None:
                return None
            fallback = self.diagnostic_zero_value_for_type(parts[1])
            diagnostic = self.structured_buffer_read_diagnostic(
                raw_args[0], buffer_type, func_name, fallback
            )
            if diagnostic is not None:
                return diagnostic
            return f"{args[0]}[{args[1]}]"

        if func_name == "buffer_append" and len(args) >= 2:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.structured_buffer_type_parts(buffer_type) is None:
                return None
            return self.generate_structured_buffer_append(
                raw_args[0], buffer_type, args[1], func_name
            )

        if func_name == "buffer_consume" and args:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.structured_buffer_type_parts(buffer_type) is None:
                return None
            return self.generate_structured_buffer_consume(
                raw_args[0], buffer_type, func_name
            )

        if func_name == "buffer_dimensions" and args:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.structured_buffer_type_parts(buffer_type) is None:
                return None
            return self.generate_structured_buffer_dimensions(
                raw_args[0], buffer_type, raw_args[1:], args[1:], func_name
            )

        if func_name == "buffer_store" and len(args) >= 3:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.structured_buffer_type_parts(buffer_type) is None:
                return None
            diagnostic = self.structured_buffer_write_diagnostic(
                raw_args[0], buffer_type, func_name
            )
            if diagnostic is not None:
                return diagnostic
            if self.structured_buffer_is_writable(buffer_type):
                return f"{args[0]}[{args[1]}] = {args[2]}"
            return self.structured_buffer_diagnostic_call(
                "buffer_store", buffer_type, "((void)0)"
            )

        return None

    def generate_byte_address_buffer_call(
        self, function_expr, func_name, raw_args, args
    ):
        """Lower byte-address buffer loads and stores to typed HIP helpers."""
        member_call = self.byte_address_buffer_member_call(function_expr)
        if member_call is not None:
            buffer_expr, operation, buffer_type = member_call
            if operation == "GetDimensions":
                return self.generate_byte_address_buffer_dimensions(
                    buffer_expr, buffer_type, raw_args, args, operation
                )

            if operation in self.byte_address_buffer_atomic_operations():
                return self.generate_byte_address_buffer_atomic(
                    buffer_expr, buffer_type, operation, args
                )

            component_count = self.byte_address_buffer_component_count(operation)
            if component_count is None:
                return None

            buffer_name = self.visit(buffer_expr)
            if operation.startswith("Load") and len(args) >= 1:
                fallback = self.diagnostic_zero_value_for_type(
                    self.byte_address_buffer_value_type(component_count)
                )
                diagnostic = self.byte_address_buffer_read_diagnostic(
                    buffer_expr, buffer_type, operation, fallback
                )
                if diagnostic is not None:
                    return diagnostic
                helper_name = self.require_byte_address_load_helper(component_count)
                return f"{helper_name}({buffer_name}, {args[0]})"

            if operation.startswith("Store") and len(args) >= 2:
                diagnostic = self.byte_address_buffer_write_diagnostic(
                    buffer_expr, buffer_type, operation
                )
                if diagnostic is not None:
                    return diagnostic
                if self.byte_address_buffer_is_writable(buffer_type):
                    helper_name = self.require_byte_address_store_helper(
                        component_count
                    )
                    return f"{helper_name}({buffer_name}, {args[0]}, {args[1]})"
                return self.byte_address_buffer_diagnostic_call(
                    operation, buffer_type, "((void)0)"
                )
            return None

        if func_name == "buffer_load" and len(args) >= 2:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.byte_address_buffer_base_type(buffer_type) is None:
                return None
            diagnostic = self.byte_address_buffer_read_diagnostic(
                raw_args[0], buffer_type, func_name, "0u"
            )
            if diagnostic is not None:
                return diagnostic
            helper_name = self.require_byte_address_load_helper(1)
            return f"{helper_name}({args[0]}, {args[1]})"

        if func_name == "buffer_dimensions" and args:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.byte_address_buffer_base_type(buffer_type) is None:
                return None
            return self.generate_byte_address_buffer_dimensions(
                raw_args[0], buffer_type, raw_args[1:], args[1:], func_name
            )

        if func_name == "buffer_store" and len(args) >= 3:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.byte_address_buffer_base_type(buffer_type) is None:
                return None
            diagnostic = self.byte_address_buffer_write_diagnostic(
                raw_args[0], buffer_type, func_name
            )
            if diagnostic is not None:
                return diagnostic
            if self.byte_address_buffer_is_writable(buffer_type):
                helper_name = self.require_byte_address_store_helper(1)
                return f"{helper_name}({args[0]}, {args[1]}, {args[2]})"
            return self.byte_address_buffer_diagnostic_call(
                "buffer_store", buffer_type, "((void)0)"
            )

        return None

    def generate_structured_buffer_dimensions(
        self, buffer_expr, buffer_type, raw_dimension_args, dimension_args, operation
    ):
        """Lower structured-buffer dimensions through an explicit length sidecar."""
        if self.structured_buffer_type_parts(buffer_type) is None:
            return None

        length_expr = self.structured_buffer_length_expression(buffer_expr)
        if length_expr is None:
            length_expr = (
                "0 /* HIP structured buffer dimensions "
                "requires explicit length sidecar */"
            )

        if dimension_args:
            return f"{dimension_args[0]} = {length_expr}"
        return length_expr

    def generate_structured_buffer_append(
        self, buffer_expr, buffer_type, value, operation
    ):
        """Lower AppendStructuredBuffer writes through an explicit counter."""
        parts = self.structured_buffer_type_parts(buffer_type)
        if parts is None or parts[0] != "AppendStructuredBuffer":
            return self.structured_buffer_diagnostic_call(
                operation, buffer_type, "((void)0)"
            )

        counter = self.structured_buffer_counter_expression(buffer_expr)
        if counter is None:
            return self.structured_buffer_diagnostic_call(
                operation, buffer_type, "((void)0)"
            )

        helper_name = self.require_structured_buffer_append_helper()
        buffer_name = self.visit(buffer_expr)
        return f"{helper_name}({buffer_name}, {counter}, {value})"

    def generate_structured_buffer_consume(self, buffer_expr, buffer_type, operation):
        """Lower ConsumeStructuredBuffer reads through an explicit counter."""
        parts = self.structured_buffer_type_parts(buffer_type)
        if parts is None or parts[0] != "ConsumeStructuredBuffer":
            fallback = self.diagnostic_zero_value_for_type(
                parts[1] if parts is not None else None
            )
            return self.structured_buffer_diagnostic_call(
                operation, buffer_type, fallback
            )

        counter = self.structured_buffer_counter_expression(buffer_expr)
        if counter is None:
            return self.structured_buffer_diagnostic_call(
                operation,
                buffer_type,
                self.diagnostic_zero_value_for_type(parts[1]),
            )

        helper_name = self.require_structured_buffer_consume_helper()
        buffer_name = self.visit(buffer_expr)
        return f"{helper_name}({buffer_name}, {counter})"

    def require_structured_buffer_append_helper(self):
        """Register the HIP helper for AppendStructuredBuffer operations."""
        helper_name = "cgl_append_structured_buffer"
        if helper_name in self.helper_functions:
            return helper_name

        helper = (
            "template <typename T>\n"
            "__device__ inline void "
            f"{helper_name}(T* buffer, uint* counter, const T& value)\n"
            "{\n"
            "    uint index = atomicAdd(counter, 1u);\n"
            "    buffer[index] = value;\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def require_structured_buffer_consume_helper(self):
        """Register the HIP helper for ConsumeStructuredBuffer operations."""
        helper_name = "cgl_consume_structured_buffer"
        if helper_name in self.helper_functions:
            return helper_name

        helper = (
            "template <typename T>\n"
            "__device__ inline T "
            f"{helper_name}(const T* buffer, uint* counter)\n"
            "{\n"
            "    uint index = atomicSub(counter, 1u) - 1u;\n"
            "    return buffer[index];\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def structured_buffer_atomic_operations(self):
        """Return generic atomic mappings for RWStructuredBuffer element targets."""
        integer_kinds = {"int", "uint"}
        add_exchange_kinds = {*integer_kinds, "float"}
        return {
            "atomicAdd": ("atomicAdd", 2, add_exchange_kinds, "int/uint/float"),
            "atomicSub": ("atomicSub", 2, integer_kinds, "int/uint"),
            "atomicMin": ("atomicMin", 2, integer_kinds, "int/uint"),
            "atomicMax": ("atomicMax", 2, integer_kinds, "int/uint"),
            "atomicAnd": ("atomicAnd", 2, integer_kinds, "int/uint"),
            "atomicOr": ("atomicOr", 2, integer_kinds, "int/uint"),
            "atomicXor": ("atomicXor", 2, integer_kinds, "int/uint"),
            "atomicInc": ("atomicInc", 2, {"uint"}, "uint"),
            "atomicDec": ("atomicDec", 2, {"uint"}, "uint"),
            "atomicExchange": ("atomicExch", 2, add_exchange_kinds, "int/uint/float"),
            "atomicCompareExchange": ("atomicCAS", 3, integer_kinds, "int/uint"),
            "atomicCompSwap": ("atomicCAS", 3, integer_kinds, "int/uint"),
        }

    def generate_structured_buffer_atomic_call(self, func_name, raw_args, args):
        """Lower atomics on RWStructuredBuffer element lvalues to HIP atomics."""
        operation = self.structured_buffer_atomic_operations().get(func_name)
        if operation is None or not raw_args:
            return None

        target = self.structured_buffer_atomic_target(raw_args[0])
        if target is None:
            return None

        intrinsic, required_arg_count, supported_kinds, supported_kinds_label = (
            operation
        )
        target_type = target["target_type"]
        fallback = self.diagnostic_zero_value_for_type(target_type)

        if len(args) != required_arg_count:
            return self.unsupported_structured_buffer_atomic_call(
                func_name,
                target["buffer_type"],
                f"requires {required_arg_count} argument(s)",
                fallback,
            )

        buffer_base_type, _ = self.structured_buffer_type_parts(target["buffer_type"])
        access_diagnostic = self.structured_buffer_read_write_diagnostic(
            target["buffer_expr"],
            target["buffer_type"],
            func_name,
            fallback,
        )
        if access_diagnostic is not None:
            return access_diagnostic
        if buffer_base_type != "RWStructuredBuffer":
            return self.unsupported_structured_buffer_atomic_call(
                func_name,
                target["buffer_type"],
                "requires RWStructuredBuffer target",
                fallback,
            )

        scalar_kind = self.hip_atomic_scalar_kind(target_type)
        if scalar_kind not in supported_kinds:
            return self.unsupported_structured_buffer_atomic_call(
                func_name,
                target["buffer_type"],
                f"requires supported scalar {supported_kinds_label} target",
                fallback,
            )

        target_expr = args[0]
        value_args = ", ".join(args[1:])
        return f"{intrinsic}(&{target_expr}, {value_args})"

    def generate_plain_atomic_call(self, func_name, raw_args, args):
        """Lower HIP atomics on ordinary scalar lvalues to pointer operands."""
        operation = self.structured_buffer_atomic_operations().get(func_name)
        if operation is None:
            return None

        intrinsic, required_arg_count, supported_kinds, supported_kinds_label = (
            operation
        )
        target_expr = raw_args[0] if raw_args else None
        block_access, _ = self.glsl_buffer_block_access(target_expr)
        if block_access is None:
            return None
        target_type = self.expression_result_type(target_expr) if target_expr else None
        fallback = self.diagnostic_zero_value_for_type(target_type)

        if len(args) != required_arg_count:
            return self.unsupported_plain_atomic_call(
                func_name,
                f"requires {required_arg_count} argument(s)",
                fallback,
            )
        if target_expr is None or not self.is_plain_atomic_lvalue(target_expr):
            return self.unsupported_plain_atomic_call(
                func_name,
                "requires assignable scalar target",
                fallback,
            )

        access_diagnostic = self.glsl_buffer_block_read_write_diagnostic(
            target_expr, func_name, fallback
        )
        if access_diagnostic is not None:
            return access_diagnostic

        scalar_kind = self.hip_atomic_scalar_kind(target_type)
        if scalar_kind not in supported_kinds:
            type_label = self.type_name_string(target_type) or "unknown target"
            return self.unsupported_plain_atomic_call(
                func_name,
                f"on {type_label} requires supported scalar "
                f"{supported_kinds_label} target",
                fallback,
            )

        target_code = args[0]
        value_args = ", ".join(args[1:])
        return f"{intrinsic}(&{target_code}, {value_args})"

    def is_plain_atomic_lvalue(self, expr):
        """Return whether an expression can be addressed for a HIP atomic."""
        return isinstance(expr, (IdentifierNode, ArrayAccessNode, MemberAccessNode))

    def unsupported_plain_atomic_call(self, operation, reason, fallback):
        """Return diagnostic code for unsupported ordinary HIP atomics."""
        return (
            f"/* unsupported {self.resource_backend_name()} atomic: "
            f"{operation} {reason} */ {fallback}"
        )

    def structured_buffer_atomic_target(self, target_expr):
        """Return RWStructuredBuffer target metadata for an atomic lvalue."""
        element_access = self.structured_buffer_element_access(target_expr)
        if element_access is None:
            return None

        array_expr = getattr(
            element_access, "array_expr", getattr(element_access, "array", None)
        )
        buffer_type = self.expression_result_type(array_expr)
        parts = self.structured_buffer_type_parts(buffer_type)
        if parts is None:
            return None

        target_type = self.expression_result_type(target_expr) or parts[1]
        return {
            "buffer_expr": array_expr,
            "buffer_type": buffer_type,
            "target_type": target_type,
        }

    def structured_buffer_element_access(self, target_expr):
        """Return the structured-buffer element access inside an atomic lvalue."""
        if isinstance(target_expr, ArrayAccessNode):
            array_expr = getattr(
                target_expr, "array_expr", getattr(target_expr, "array", None)
            )
            buffer_type = self.expression_result_type(array_expr)
            if self.structured_buffer_type_parts(buffer_type) is not None:
                return target_expr

        if isinstance(target_expr, MemberAccessNode):
            object_expr = getattr(
                target_expr, "object_expr", getattr(target_expr, "object", None)
            )
            return self.structured_buffer_element_access(object_expr)

        return None

    def structured_buffer_data_access_info(self, expr):
        """Return direct structured-buffer element access metadata, if any."""
        if not isinstance(expr, ArrayAccessNode):
            return None

        array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
        buffer_type = self.expression_result_type(array_expr)
        parts = self.structured_buffer_type_parts(buffer_type)
        if parts is None:
            return None

        result_type = self.expression_result_type(expr)
        if self.structured_buffer_type_parts(result_type) is not None:
            return None

        return {
            "buffer_expr": array_expr,
            "buffer_type": buffer_type,
            "element_type": result_type or parts[1],
        }

    def structured_buffer_element_read_diagnostic(self, expr, operation):
        info = self.structured_buffer_data_access_info(expr)
        if info is None:
            return None
        fallback = self.diagnostic_zero_value_for_type(info["element_type"])
        return self.structured_buffer_read_diagnostic(
            info["buffer_expr"], info["buffer_type"], operation, fallback
        )

    def structured_buffer_element_write_diagnostic(self, expr, operation):
        info = self.structured_buffer_data_access_info(expr)
        if info is None:
            return None
        return self.structured_buffer_write_diagnostic(
            info["buffer_expr"], info["buffer_type"], operation
        )

    def structured_buffer_element_read_write_diagnostic(self, expr, operation):
        info = self.structured_buffer_data_access_info(expr)
        if info is None:
            return None
        fallback = self.diagnostic_zero_value_for_type(info["element_type"])
        return self.structured_buffer_read_write_diagnostic(
            info["buffer_expr"], info["buffer_type"], operation, fallback
        )

    def hip_atomic_scalar_kind(self, type_name):
        """Return the HIP atomic scalar kind supported for structured buffers."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None

        mapped_type = self.map_type(type_name)
        if type_name in {"uint", "u32"} or mapped_type in {"uint", "unsigned int"}:
            return "uint"
        if type_name in {"int", "i32"} or mapped_type == "int":
            return "int"
        if type_name in {"float", "f32"} or mapped_type == "float":
            return "float"
        return None

    def unsupported_structured_buffer_atomic_call(
        self, operation, buffer_type, reason, fallback
    ):
        """Return diagnostic code for unsupported structured-buffer atomics."""
        buffer_type = self.type_name_string(buffer_type) or "unknown buffer"
        return (
            f"/* unsupported {self.resource_backend_name()} structured buffer atomic: "
            f"{operation} on {buffer_type} {reason} */ {fallback}"
        )

    def generate_byte_address_buffer_dimensions(
        self, buffer_expr, buffer_type, raw_dimension_args, dimension_args, operation
    ):
        """Lower byte-address buffer dimensions through a byte-length sidecar."""
        if self.byte_address_buffer_base_type(buffer_type) is None:
            return None

        length_expr = self.structured_buffer_length_expression(buffer_expr)
        if length_expr is None:
            length_expr = (
                "0 /* HIP byte-address buffer dimensions "
                "requires explicit byte-length sidecar */"
            )

        if dimension_args:
            return f"{dimension_args[0]} = {length_expr}"
        return length_expr

    def structured_buffer_member_call(self, function_expr):
        """Return structured-buffer member-call pieces, if applicable."""
        if not isinstance(function_expr, MemberAccessNode):
            return None

        buffer_expr = getattr(
            function_expr, "object_expr", getattr(function_expr, "object", None)
        )
        buffer_type = self.expression_result_type(buffer_expr)
        if self.structured_buffer_type_parts(buffer_type) is None:
            return None

        return buffer_expr, getattr(function_expr, "member", ""), buffer_type

    def format_structured_buffer_access(self, buffer_expr, raw_args, args):
        """Format one HIP pointer access for a structured-buffer operation."""
        if not raw_args or not args:
            return None
        buffer_name = self.visit(buffer_expr)
        return f"{buffer_name}[{args[0]}]"

    def structured_buffer_diagnostic_call(self, operation, buffer_type, fallback):
        """Return diagnostic code for rejected structured-buffer operations."""
        buffer_type = self.type_name_string(buffer_type) or "unknown buffer"
        return (
            f"/* unsupported {self.resource_backend_name()} structured buffer call: "
            f"{operation} on {buffer_type} */ {fallback}"
        )

    def structured_buffer_access_diagnostic_call(
        self, operation, buffer_type, reason, fallback
    ):
        """Return diagnostic code for rejected structured-buffer access metadata."""
        buffer_type = self.type_name_string(buffer_type) or "unknown buffer"
        return (
            f"/* unsupported {self.resource_backend_name()} structured buffer "
            f"access: {operation} {reason} on {buffer_type} */ {fallback}"
        )

    def structured_buffer_read_diagnostic(
        self, buffer_expr, buffer_type, operation, fallback
    ):
        access = self.buffer_resource_access(buffer_expr)
        if access == "writeonly":
            return self.structured_buffer_access_diagnostic_call(
                operation,
                buffer_type,
                "requires readable buffer access",
                fallback,
            )
        return None

    def structured_buffer_write_diagnostic(self, buffer_expr, buffer_type, operation):
        access = self.buffer_resource_access(buffer_expr)
        if access == "readonly":
            return self.structured_buffer_access_diagnostic_call(
                operation,
                buffer_type,
                "requires writable buffer access",
                "((void)0)",
            )
        return None

    def structured_buffer_read_write_diagnostic(
        self, buffer_expr, buffer_type, operation, fallback
    ):
        access = self.buffer_resource_access(buffer_expr)
        if access in {"readonly", "writeonly"}:
            return self.structured_buffer_access_diagnostic_call(
                operation,
                buffer_type,
                "requires readwrite buffer access",
                fallback,
            )
        return None

    def diagnostic_zero_value_for_type(self, type_name):
        """Return a HIP fallback expression for rejected value-producing calls."""
        if type_name is None:
            return "0"

        mapped_type = self.map_type(type_name)
        vector_info = self.vector_type_info(mapped_type) or self.vector_type_info(
            self.type_name_string(type_name)
        )
        if vector_info is not None:
            component_type = vector_info["component_type"]
            if component_type == "uint":
                zero = "0u"
            elif component_type == "double":
                zero = "0.0"
            elif component_type == "float":
                zero = "0.0f"
            elif component_type == "bool":
                zero = "false"
            else:
                zero = "0"
            values = ", ".join([zero] * len(vector_info["components"]))
            return f"{vector_info['constructor']}({values})"

        if mapped_type == "bool":
            return "false"
        if mapped_type in {"float", "half"}:
            return "0.0f"
        if mapped_type == "double":
            return "0.0"
        if mapped_type in {"uint", "unsigned int", "u32"}:
            return "0u"
        if mapped_type in {
            "int",
            "short",
            "char",
            "long long",
            "unsigned char",
            "unsigned short",
            "unsigned long long",
        }:
            return "0"
        return f"{mapped_type}{{}}"

    def collect_structured_buffer_length_requirements(self, root):
        """Collect buffers that need explicit length sidecar parameters."""
        functions = self.query_collect_functions(root)
        functions_by_name = {getattr(func, "name", None): func for func in functions}
        functions_by_name = {
            name: func for name, func in functions_by_name.items() if name
        }
        param_names = {
            func_name: {
                getattr(param, "name", None)
                for param in getattr(func, "parameters", getattr(func, "params", []))
            }
            for func_name, func in functions_by_name.items()
        }
        param_names = {
            func_name: {name for name in names if name}
            for func_name, names in param_names.items()
        }

        global_length_names = set()
        function_param_length_names = {
            func_name: set() for func_name in functions_by_name
        }

        def mark_resource_name(func_name, resource_name):
            if not resource_name:
                return False
            if resource_name in param_names.get(func_name, set()):
                before = len(function_param_length_names[func_name])
                function_param_length_names[func_name].add(resource_name)
                return len(function_param_length_names[func_name]) != before
            before = len(global_length_names)
            global_length_names.add(resource_name)
            return len(global_length_names) != before

        for func_name, func in functions_by_name.items():
            for call in self.query_walk_nodes(getattr(func, "body", [])):
                if not isinstance(call, FunctionCallNode):
                    continue
                buffer_expr = self.structured_buffer_dimensions_target(call)
                if buffer_expr is None:
                    continue
                mark_resource_name(func_name, self.get_expression_name(buffer_expr))

        changed = True
        while changed:
            changed = False
            for caller_name, caller in functions_by_name.items():
                caller_params = param_names.get(caller_name, set())
                for call in self.query_walk_nodes(getattr(caller, "body", [])):
                    if not isinstance(call, FunctionCallNode):
                        continue
                    callee_name = self.raw_function_call_name(call)
                    callee = functions_by_name.get(callee_name)
                    if callee is None:
                        continue

                    callee_required = function_param_length_names.get(
                        callee_name, set()
                    )
                    if not callee_required:
                        continue

                    callee_params = getattr(
                        callee, "parameters", getattr(callee, "params", [])
                    )
                    raw_args = getattr(call, "arguments", getattr(call, "args", []))
                    for index, param in enumerate(callee_params):
                        if index >= len(raw_args):
                            continue
                        param_name = getattr(param, "name", None)
                        if param_name not in callee_required:
                            continue

                        arg_name = self.get_expression_name(raw_args[index])
                        if not arg_name:
                            continue
                        if arg_name in caller_params:
                            before = len(function_param_length_names[caller_name])
                            function_param_length_names[caller_name].add(arg_name)
                            changed = (
                                changed
                                or len(function_param_length_names[caller_name])
                                != before
                            )
                        else:
                            before = len(global_length_names)
                            global_length_names.add(arg_name)
                            changed = changed or len(global_length_names) != before

        return (
            global_length_names,
            {
                func_name: names
                for func_name, names in function_param_length_names.items()
                if names
            },
        )

    def structured_buffer_dimensions_target(self, call):
        """Return the buffer expression queried by a dimensions call."""
        func_name = self.raw_function_call_name(call)
        raw_args = getattr(call, "arguments", getattr(call, "args", []))
        if func_name == "buffer_dimensions" and raw_args:
            return raw_args[0]

        function_expr = getattr(call, "function", getattr(call, "name", None))
        if isinstance(function_expr, MemberAccessNode):
            member = getattr(function_expr, "member", None)
            if member == "GetDimensions":
                return getattr(
                    function_expr,
                    "object_expr",
                    getattr(function_expr, "object", None),
                )
        return None

    def structured_buffer_length_name(self, name):
        """Return the sidecar length parameter/declaration name for a buffer."""
        return f"{name}_length"

    def structured_buffer_requires_length(self, name):
        """Return whether a global buffer needs a length sidecar."""
        return bool(name and name in self.structured_buffer_length_names)

    def structured_buffer_parameter_requires_length(
        self, func_name, name, type_name=None
    ):
        """Return whether a function parameter needs a length sidecar."""
        if not name:
            return False
        if name not in self.structured_buffer_length_function_params.get(
            func_name, set()
        ):
            return False
        return type_name is None or self.buffer_type_supports_length(type_name)

    def buffer_type_supports_length(self, type_name):
        """Return whether a resource type can use a HIP length sidecar."""
        return (
            self.structured_buffer_type_parts(type_name) is not None
            or self.byte_address_buffer_base_type(type_name) is not None
        )

    def structured_buffer_length_declaration(self, name, type_name):
        """Format a global/local sidecar length declaration when required."""
        if not self.structured_buffer_requires_length(name):
            return None
        if not self.buffer_type_supports_length(type_name):
            return None
        return self.format_structured_buffer_length_declarator(type_name, name)

    def structured_buffer_length_parameter(self, func_name, name, type_name):
        """Format a sidecar length parameter when required."""
        if not self.structured_buffer_parameter_requires_length(
            func_name, name, type_name
        ):
            return None
        return self.format_structured_buffer_length_declarator(type_name, name)

    def format_structured_buffer_length_declarator(self, type_name, name):
        """Format the HIP uint pointer sidecar matching a buffer declarator."""
        type_name = self.type_name_string(type_name)
        length_name = self.structured_buffer_length_name(name)
        if "[" not in type_name or "]" not in type_name:
            return f"const uint* {length_name}"

        array_suffix = type_name[type_name.find("[") :]
        return format_array_declarator("const uint*", length_name, array_suffix)

    def structured_buffer_length_data_expression(self, buffer_expr):
        """Return the sidecar length pointer paired with a buffer expression."""
        if isinstance(buffer_expr, ArrayAccessNode):
            array_expr = getattr(
                buffer_expr, "array_expr", getattr(buffer_expr, "array", None)
            )
            index_expr = getattr(
                buffer_expr, "index_expr", getattr(buffer_expr, "index", None)
            )
            base_length = self.structured_buffer_length_data_expression(array_expr)
            if base_length is None:
                return None
            return f"{base_length}[{self.visit(index_expr)}]"

        name = self.get_expression_name(buffer_expr)
        if not name:
            return None

        length_parameter = self.current_structured_buffer_length_parameters.get(name)
        if length_parameter is not None:
            return length_parameter

        if self.structured_buffer_requires_length(name):
            return self.structured_buffer_length_name(name)
        return None

    def structured_buffer_length_expression(self, buffer_expr):
        """Return the scalar length expression for a buffer resource."""
        length_data = self.structured_buffer_length_data_expression(buffer_expr)
        if length_data is None:
            return None
        return f"{length_data}[0]"

    def structured_buffer_counter_name(self, name):
        """Return the sidecar counter parameter/declaration name for a buffer."""
        return f"{name}_counter"

    def structured_buffer_counter_declaration(self, name, type_name):
        """Format a global/local sidecar counter declaration when required."""
        if not self.structured_buffer_requires_counter(type_name):
            return None
        return self.format_structured_buffer_counter_declarator(type_name, name)

    def structured_buffer_counter_parameter(self, name, type_name):
        """Format a sidecar counter parameter when required."""
        if not self.structured_buffer_requires_counter(type_name):
            return None
        return self.format_structured_buffer_counter_declarator(type_name, name)

    def format_structured_buffer_counter_declarator(self, type_name, name):
        """Format the HIP uint pointer sidecar matching a buffer declarator."""
        type_name = self.type_name_string(type_name)
        counter_name = self.structured_buffer_counter_name(name)
        if "[" not in type_name or "]" not in type_name:
            return f"uint* {counter_name}"

        array_suffix = type_name[type_name.find("[") :]
        return format_array_declarator("uint*", counter_name, array_suffix)

    def structured_buffer_counter_expression(self, buffer_expr):
        """Return the sidecar counter expression paired with a buffer expression."""
        if isinstance(buffer_expr, ArrayAccessNode):
            array_expr = getattr(
                buffer_expr, "array_expr", getattr(buffer_expr, "array", None)
            )
            index_expr = getattr(
                buffer_expr, "index_expr", getattr(buffer_expr, "index", None)
            )
            base_counter = self.structured_buffer_counter_expression(array_expr)
            if base_counter is None:
                return None
            return f"{base_counter}[{self.visit(index_expr)}]"

        name = self.get_expression_name(buffer_expr)
        if not name:
            return None
        return self.structured_buffer_counter_name(name)

    def hip_user_function_call_arguments(self, func_name, raw_args, args):
        """Expand user calls with HIP sidecar resource arguments."""
        callee = self.query_functions_by_name.get(func_name)
        if callee is None:
            return args

        params = getattr(callee, "parameters", getattr(callee, "params", []))
        query_params = self.query_metadata_function_params.get(func_name, set())
        expanded_args = []
        for index, arg in enumerate(args):
            expanded_args.append(arg)
            if index >= len(params) or index >= len(raw_args):
                continue

            param = params[index]
            param_name = getattr(param, "name", None)
            if param_name in query_params:
                metadata_arg = self.query_metadata_expression(raw_args[index])
                if metadata_arg:
                    expanded_args.append(metadata_arg)

            param_type = self.get_parameter_type(param)
            if self.structured_buffer_parameter_requires_length(
                func_name, param_name, param_type
            ):
                length_arg = self.structured_buffer_length_data_expression(
                    raw_args[index]
                )
                if length_arg:
                    expanded_args.append(length_arg)

            if self.structured_buffer_requires_counter(param_type):
                counter_arg = self.structured_buffer_counter_expression(raw_args[index])
                if counter_arg:
                    expanded_args.append(counter_arg)

        return expanded_args

    def byte_address_buffer_member_call(self, function_expr):
        """Return byte-address buffer member-call pieces, if applicable."""
        if not isinstance(function_expr, MemberAccessNode):
            return None

        buffer_expr = getattr(
            function_expr, "object_expr", getattr(function_expr, "object", None)
        )
        buffer_type = self.expression_result_type(buffer_expr)
        if self.byte_address_buffer_base_type(buffer_type) is None:
            return None

        return buffer_expr, getattr(function_expr, "member", ""), buffer_type

    def byte_address_buffer_component_count(self, operation):
        """Return the uint lane count for byte-address load/store operations."""
        operation_counts = {
            "Load": 1,
            "Load2": 2,
            "Load3": 3,
            "Load4": 4,
            "Store": 1,
            "Store2": 2,
            "Store3": 3,
            "Store4": 4,
        }
        return operation_counts.get(operation)

    def byte_address_buffer_atomic_operations(self):
        """Return supported RWByteAddressBuffer atomic method mappings."""
        return {
            "InterlockedAdd": ("add", "atomicAdd", 2),
            "InterlockedMin": ("min", "atomicMin", 2),
            "InterlockedMax": ("max", "atomicMax", 2),
            "InterlockedAnd": ("and", "atomicAnd", 2),
            "InterlockedOr": ("or", "atomicOr", 2),
            "InterlockedXor": ("xor", "atomicXor", 2),
            "InterlockedExchange": ("exchange", "atomicExch", 2),
            "InterlockedCompareExchange": ("compare_exchange", "atomicCAS", 3),
        }

    def generate_byte_address_buffer_atomic(
        self, buffer_expr, buffer_type, operation, args
    ):
        """Lower RWByteAddressBuffer Interlocked* methods to HIP atomics."""
        operation_info = self.byte_address_buffer_atomic_operations().get(operation)
        if operation_info is None:
            return None

        operation_name, intrinsic, required_args = operation_info
        has_out_arg = len(args) == required_args + 1
        fallback = "((void)0)" if has_out_arg else "0u"
        if len(args) < required_args or len(args) > required_args + 1:
            return self.byte_address_buffer_diagnostic_call(
                operation, buffer_type, fallback
            )
        diagnostic = self.byte_address_buffer_read_write_diagnostic(
            buffer_expr, buffer_type, operation, fallback
        )
        if diagnostic is not None:
            return diagnostic
        if not self.byte_address_buffer_is_writable(buffer_type):
            return self.byte_address_buffer_diagnostic_call(
                operation, buffer_type, fallback
            )

        helper_name = self.require_byte_address_atomic_helper(operation_name, intrinsic)
        buffer_name = self.visit(buffer_expr)
        helper_args = [buffer_name, *args[:required_args]]
        call = f"{helper_name}({', '.join(helper_args)})"
        if has_out_arg:
            return f"{args[required_args]} = {call}"
        return call

    def byte_address_atomic_helper_name(self, operation):
        """Return a stable HIP helper name for a byte-address atomic operation."""
        return f"cgl_byte_address_atomic_{operation}_uint"

    def require_byte_address_atomic_helper(self, operation, intrinsic):
        """Register a HIP byte-address atomic helper and return its name."""
        helper_name = self.byte_address_atomic_helper_name(operation)
        if helper_name in self.helper_functions:
            return helper_name

        pointer_expr = "reinterpret_cast<unsigned int*>(buffer + offset)"
        if operation == "compare_exchange":
            helper = (
                "__device__ inline uint "
                f"{helper_name}(unsigned char* buffer, uint offset, "
                "uint compare_value, uint value)\n"
                "{\n"
                f"    return {intrinsic}({pointer_expr}, compare_value, value);\n"
                "}"
            )
            self.helper_functions[helper_name] = helper
            return helper_name

        helper = (
            "__device__ inline uint "
            f"{helper_name}(unsigned char* buffer, uint offset, uint value)\n"
            "{\n"
            f"    return {intrinsic}({pointer_expr}, value);\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def byte_address_buffer_value_type(self, component_count):
        """Return the HIP uint vector type for a byte-address operation."""
        if component_count == 1:
            return "uint"
        return f"uint{component_count}"

    def byte_address_helper_suffix(self, component_count):
        """Return a stable helper-name suffix for byte-address operations."""
        if component_count == 1:
            return "uint"
        return f"uint{component_count}"

    def require_byte_address_load_helper(self, component_count):
        """Register a HIP byte-address load helper and return its name."""
        helper_name = (
            f"cgl_byte_address_load_{self.byte_address_helper_suffix(component_count)}"
        )
        if helper_name in self.helper_functions:
            return helper_name

        if component_count == 1:
            helper = (
                "__device__ inline uint "
                f"{helper_name}(const unsigned char* buffer, uint offset)\n"
                "{\n"
                "    return *reinterpret_cast<const uint*>(buffer + offset);\n"
                "}"
            )
            self.helper_functions[helper_name] = helper
            return helper_name

        scalar_helper = self.require_byte_address_load_helper(1)
        components = ("x", "y", "z", "w")[:component_count]
        args = [
            f"{scalar_helper}(buffer, offset + {index * 4}u)"
            for index, _ in enumerate(components)
        ]
        value_type = self.byte_address_buffer_value_type(component_count)
        constructor = self.function_map.get(value_type, value_type)
        helper = (
            f"__device__ inline {value_type} "
            f"{helper_name}(const unsigned char* buffer, uint offset)\n"
            "{\n"
            f"    return {constructor}({', '.join(args)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def require_byte_address_store_helper(self, component_count):
        """Register a HIP byte-address store helper and return its name."""
        helper_name = (
            f"cgl_byte_address_store_{self.byte_address_helper_suffix(component_count)}"
        )
        if helper_name in self.helper_functions:
            return helper_name

        value_type = self.byte_address_buffer_value_type(component_count)
        if component_count == 1:
            helper = (
                "__device__ inline void "
                f"{helper_name}(unsigned char* buffer, uint offset, uint value)\n"
                "{\n"
                "    *reinterpret_cast<uint*>(buffer + offset) = value;\n"
                "}"
            )
            self.helper_functions[helper_name] = helper
            return helper_name

        scalar_helper = self.require_byte_address_store_helper(1)
        components = ("x", "y", "z", "w")[:component_count]
        lines = [
            f"    {scalar_helper}(buffer, offset + {index * 4}u, value.{component});"
            for index, component in enumerate(components)
        ]
        helper = (
            f"__device__ inline void {helper_name}"
            f"(unsigned char* buffer, uint offset, {value_type} value)\n"
            "{\n" + "\n".join(lines) + "\n}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def byte_address_buffer_diagnostic_call(self, operation, buffer_type, fallback):
        """Return diagnostic code for rejected byte-address buffer operations."""
        buffer_type = self.type_name_string(buffer_type) or "unknown buffer"
        return (
            f"/* unsupported {self.resource_backend_name()} byte-address buffer call: "
            f"{operation} on {buffer_type} */ {fallback}"
        )

    def byte_address_buffer_access_diagnostic_call(
        self, operation, buffer_type, reason, fallback
    ):
        """Return diagnostic code for rejected byte-address access metadata."""
        buffer_type = self.type_name_string(buffer_type) or "unknown buffer"
        return (
            f"/* unsupported {self.resource_backend_name()} byte-address buffer "
            f"access: {operation} {reason} on {buffer_type} */ {fallback}"
        )

    def byte_address_buffer_read_diagnostic(
        self, buffer_expr, buffer_type, operation, fallback
    ):
        access = self.buffer_resource_access(buffer_expr)
        if access == "writeonly":
            return self.byte_address_buffer_access_diagnostic_call(
                operation,
                buffer_type,
                "requires readable buffer access",
                fallback,
            )
        return None

    def byte_address_buffer_write_diagnostic(self, buffer_expr, buffer_type, operation):
        access = self.buffer_resource_access(buffer_expr)
        if access == "readonly":
            return self.byte_address_buffer_access_diagnostic_call(
                operation,
                buffer_type,
                "requires writable buffer access",
                "((void)0)",
            )
        return None

    def byte_address_buffer_read_write_diagnostic(
        self, buffer_expr, buffer_type, operation, fallback
    ):
        access = self.buffer_resource_access(buffer_expr)
        if access in {"readonly", "writeonly"}:
            return self.byte_address_buffer_access_diagnostic_call(
                operation,
                buffer_type,
                "requires readwrite buffer access",
                fallback,
            )
        return None

    def visit_str(self, node) -> str:
        return str(node)

    def visit_int(self, node) -> str:
        return str(node)

    def visit_float(self, node) -> str:
        return str(node)

    def visit_ArrayAccessNode(self, node) -> str:
        if self.assignment_lhs_depth == 0:
            diagnostic = self.glsl_buffer_block_read_diagnostic(node, "load")
            if diagnostic is not None:
                return diagnostic
            diagnostic = self.structured_buffer_element_read_diagnostic(node, "load")
            if diagnostic is not None:
                return diagnostic
        array = self.visit(node.array)
        index = self.visit(node.index)
        return f"{array}[{index}]"

    def visit_ArrayLiteralNode(self, node: ArrayLiteralNode) -> str:
        elements = ", ".join(self.visit(element) for element in node.elements)
        return f"{{{elements}}}"

    def visit_MemberAccessNode(self, node) -> str:
        if self.assignment_lhs_depth == 0:
            diagnostic = self.glsl_buffer_block_read_diagnostic(node, "load")
            if diagnostic is not None:
                return diagnostic
        raw_object_name = getattr(node.object, "name", None)
        if raw_object_name is not None:
            ray_builtin = self.hip_ray_builtin_expression(raw_object_name)
            if ray_builtin is not None:
                return f"{ray_builtin}.{node.member}"
            raw_member_access = f"{raw_object_name}.{node.member}"
            if raw_member_access in self.builtin_map:
                return self.builtin_map[raw_member_access]

        object_expr = self.visit(node.object)
        member_access = f"{object_expr}.{node.member}"
        if member_access in self.builtin_map:
            return self.builtin_map[member_access]

        swizzle = self.generate_vector_swizzle(node, object_expr)
        if swizzle is not None:
            return swizzle

        return member_access

    def generate_vector_swizzle(self, node, object_expr):
        object_node = getattr(node, "object_expr", getattr(node, "object", None))
        vector_info = self.vector_type_info(self.expression_result_type(object_node))
        if vector_info is None:
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
        member = getattr(node, "member", "")
        components = [component_aliases.get(component) for component in member]
        if not components or any(component is None for component in components):
            return None

        available_components = vector_info["components"]
        if any(component not in available_components for component in components):
            return None

        if len(components) == 1:
            return f"{object_expr}.{components[0]}"

        result_type = self.vector_type_for_components(
            vector_info["component_type"], len(components)
        )
        result_info = self.vector_type_info(result_type)
        if result_info is None:
            return None

        swizzle_call = self.generate_vector_swizzle_single_eval_call(
            result_info,
            vector_info,
            object_node,
            object_expr,
            components,
        )
        if swizzle_call is not None:
            return swizzle_call

        args = [f"{object_expr}.{component}" for component in components]
        return f"{result_info['constructor']}({', '.join(args)})"

    def visit_TernaryOpNode(self, node) -> str:
        condition = self.visit(node.condition)
        true_expr = self.visit(node.true_expr)
        false_expr = self.visit(node.false_expr)
        lowered = self.lower_vector_ternary_operation(
            node.condition,
            condition,
            node.true_expr,
            true_expr,
            node.false_expr,
            false_expr,
        )
        if lowered is not None:
            return lowered
        return f"({condition} ? {true_expr} : {false_expr})"

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
        if (
            literal_type == "uint"
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            return f"{value}u"
        if isinstance(value, str):
            escaped = self.escape_literal(value, quote='"')
            return f'"{escaped}"'
        return str(value)

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

    def visit_LiteralNode(self, node) -> str:
        literal_type = getattr(getattr(node, "literal_type", None), "name", None)
        return self.format_literal(node.value, literal_type)

    def visit_IdentifierNode(self, node) -> str:
        name = getattr(node, "name", str(node))
        ray_builtin = self.hip_ray_builtin_expression(name)
        if ray_builtin is not None and name not in self.variable_types:
            return ray_builtin
        # Handle built-in variables mapping
        return self.builtin_map.get(name, name)

    def hip_ray_builtin_expression(self, name):
        helper_name = {
            "gl_LaunchIDEXT": "cgl_ray_launch_id",
            "gl_LaunchSizeEXT": "cgl_ray_launch_size",
            "gl_HitTEXT": "cgl_ray_hit_t",
            "gl_HitKindEXT": "cgl_ray_hit_kind",
            "gl_WorldRayOriginEXT": "cgl_ray_world_origin",
            "gl_WorldRayDirectionEXT": "cgl_ray_world_direction",
            "gl_ObjectRayOriginEXT": "cgl_ray_object_origin",
            "gl_ObjectRayDirectionEXT": "cgl_ray_object_direction",
            "gl_RayTminEXT": "cgl_ray_t_min",
            "gl_IncomingRayFlagsEXT": "cgl_ray_incoming_flags",
            "gl_InstanceCustomIndexEXT": "cgl_ray_instance_custom_index",
            "gl_GeometryIndexEXT": "cgl_ray_geometry_index",
        }.get(str(name))
        if helper_name is None:
            return None
        self.require_hip_ray_runtime_helpers()
        return f"{helper_name}()"

    def visit_ExpressionStatementNode(self, node) -> str:
        expr = self.visit(node.expression)
        self.add_line(f"{expr};")
        return ""

    def visit_BlockNode(self, node) -> str:
        if hasattr(node, "statements"):
            self.emit_body(node.statements)
        return ""

    def visit_BreakNode(self, node) -> str:
        self.add_line("break;")
        return ""

    def visit_ContinueNode(self, node) -> str:
        self.add_line("continue;")
        return ""

    def visit_EnumNode(self, node) -> str:
        self.add_line(f"enum {node.name}")
        self.add_line("{")
        self.indent_level += 1

        if hasattr(node, "variants") and node.variants:
            for i, variant in enumerate(node.variants):
                if hasattr(variant, "value") and variant.value:
                    value = self.visit(variant.value)
                    if i == len(node.variants) - 1:
                        self.add_line(f"{variant.name} = {value}")
                    else:
                        self.add_line(f"{variant.name} = {value},")
                else:
                    if i == len(node.variants) - 1:
                        self.add_line(f"{variant.name}")
                    else:
                        self.add_line(f"{variant.name},")

        self.indent_level -= 1
        self.add_line("};")
        self.add_line()
        return ""

    def type_name_string(self, type_name):
        """Return a stable string spelling for TypeNode or legacy type values."""
        if type_name is None:
            return None
        if (
            hasattr(type_name, "name")
            or hasattr(type_name, "element_type")
            or hasattr(type_name, "pointee_type")
            or hasattr(type_name, "referenced_type")
        ):
            return self.convert_type_node_to_string(type_name)
        return str(type_name)

    def get_parameter_type(self, param):
        """Return a parameter type with HIP resource access metadata applied."""
        param_type = ResourceQueryMixin.get_parameter_type(self, param)
        return self.resource_type_with_access(param_type, param)

    def get_variable_node_type(self, node):
        """Return a variable type with HIP resource access metadata applied."""
        var_type = ResourceQueryMixin.get_variable_node_type(self, node)
        return self.resource_type_with_access(var_type, node)

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        if hasattr(type_node, "pointee_type"):
            pointee_type = self.convert_type_node_to_string(type_node.pointee_type)
            prefix = "" if getattr(type_node, "is_mutable", True) else "const "
            return f"{prefix}{pointee_type}*"
        if hasattr(type_node, "referenced_type"):
            referenced_type = self.convert_type_node_to_string(
                type_node.referenced_type
            )
            prefix = "" if getattr(type_node, "is_mutable", False) else "const "
            return f"{prefix}{referenced_type}&"
        if hasattr(type_node, "name"):
            generic_args = getattr(type_node, "generic_args", [])
            if generic_args:
                converted_args = []
                for arg in generic_args:
                    if (
                        hasattr(arg, "name")
                        or hasattr(arg, "element_type")
                        or hasattr(arg, "pointee_type")
                        or hasattr(arg, "referenced_type")
                    ):
                        converted = self.convert_type_node_to_string(arg)
                    elif hasattr(arg, "value"):
                        converted = str(arg.value)
                    else:
                        converted = str(arg)
                    if converted is None:
                        if hasattr(arg, "value"):
                            converted = str(arg.value)
                        else:
                            converted = str(arg)
                    converted_args.append(converted)
                args = ", ".join(converted_args)
                return f"{type_node.name}<{args}>"
            return type_node.name
        elif hasattr(type_node, "element_type"):
            if hasattr(type_node, "rows"):
                element_type = self.convert_type_node_to_string(type_node.element_type)
                prefix = "dmat" if element_type == "double" else "mat"
                return f"{prefix}{type_node.rows}x{type_node.cols}"
            elif not hasattr(type_node, "size"):
                return str(type_node)
            elif str(type(type_node)).find("ArrayType") != -1:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                if type_node.size is not None:
                    return f"{element_type}[{self.format_array_size(type_node.size)}]"
                else:
                    return f"{element_type}[]"
            else:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                size = type_node.size
                if element_type == "float":
                    return f"float{size}"
                elif element_type == "int":
                    return f"int{size}"
                else:
                    return f"{element_type}{size}"
        else:
            return str(type_node)

    def map_type(self, type_name) -> str:
        """Map a CrossGL type name or type node to a HIP type string."""
        if (
            hasattr(type_name, "name")
            or hasattr(type_name, "element_type")
            or hasattr(type_name, "pointee_type")
            or hasattr(type_name, "referenced_type")
        ):
            type_str = self.convert_type_node_to_string(type_name)
        else:
            type_str = str(type_name)

        geometry_stream_type = self.hip_geometry_stream_mapped_type(type_str)
        if geometry_stream_type is not None:
            return geometry_stream_type

        tessellation_patch_type = self.hip_tessellation_patch_mapped_type(type_str)
        if tessellation_patch_type is not None:
            return tessellation_patch_type

        # Handle array types
        if "[" in type_str and "]" in type_str:
            base_type = type_str.split("[")[0]
            array_part = type_str[type_str.find("[") :]
            canonical_base = self.canonical_resource_type(base_type)
            mapped_base = self.map_type(canonical_base or base_type)
            return f"{mapped_base}{array_part}"

        structured_buffer_type = self.hip_structured_buffer_type(type_str)
        if structured_buffer_type is not None:
            return structured_buffer_type

        byte_address_buffer_type = self.hip_byte_address_buffer_type(type_str)
        if byte_address_buffer_type is not None:
            return byte_address_buffer_type

        canonical_type = self.canonical_resource_type(type_str)
        mapped_type = self.type_map.get(canonical_type or type_str, type_str)
        return self.hip_mapped_type_result(mapped_type)

    def hip_geometry_stream_info(self, type_name):
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

        base_name, generic_arg = type_text.split("<", 1)
        base_name = base_name.strip()
        if base_name not in stream_names:
            return None
        generic_arg = generic_arg[:-1].strip()
        if not generic_arg:
            return None
        return base_name, self.map_type(generic_arg)

    def hip_geometry_stream_mapped_type(self, type_name):
        stream_info = self.hip_geometry_stream_info(type_name)
        if stream_info is None:
            return None
        stream_name, output_type = stream_info
        return f"CglHip{stream_name}<{output_type}>"

    def format_hip_geometry_stream_parameter(self, raw_param_type, name):
        stream_type = self.hip_geometry_stream_mapped_type(raw_param_type)
        if stream_type is None:
            return None
        return f"{stream_type}& {name}"

    def hip_tessellation_patch_type_info(self, type_name):
        parts = self.generic_type_parts(type_name)
        if parts is None:
            return None
        base_name, args = parts
        if base_name not in {"InputPatch", "OutputPatch"} or len(args) != 2:
            return None
        element_type, point_count = args
        if not element_type or not point_count:
            return None
        mapped_element_type = self.map_type(element_type)
        return base_name, mapped_element_type, point_count

    def hip_tessellation_patch_mapped_type(self, type_name):
        patch_info = self.hip_tessellation_patch_type_info(type_name)
        if patch_info is None:
            return None
        base_name, element_type, point_count = patch_info
        helper_name = {
            "InputPatch": "CglHipInputPatch",
            "OutputPatch": "CglHipOutputPatch",
        }[base_name]
        return f"{helper_name}<{element_type}, {point_count}>"

    def hip_mapped_type_result(self, mapped_type):
        base_type = str(mapped_type).split("[", 1)[0].strip()
        if base_type in {
            "CglRayTracingAccelerationStructure",
            "CglRayDesc",
            "CglRayQuery",
            "CglBuiltInTriangleIntersectionAttributes",
        }:
            self.require_hip_ray_runtime_helpers()
        return mapped_type

    def attribute_value_to_string(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if hasattr(value, "name") and value.name is not None:
            return str(value.name)
        if hasattr(value, "value") and value.value is not None:
            return str(value.value).strip('"')
        return str(value)

    def attribute_arguments(self, attr):
        return getattr(attr, "arguments", getattr(attr, "args", [])) or []

    def resource_binding_index_value(self, value, prefixes=()):
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

    def resource_register_space_value(self, value):
        raw_value = self.attribute_value_to_string(value)
        if raw_value is None:
            return None
        raw_value = str(raw_value).strip().lower()
        if raw_value.isdigit():
            return int(raw_value)
        if raw_value.startswith("space") and raw_value[5:].isdigit():
            return int(raw_value[5:])
        return None

    def explicit_hip_resource_binding_index(self, node):
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

    def explicit_hip_resource_set_index(self, node):
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

    def hip_resource_register_metadata(self, node):
        for attr in getattr(node, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() != "register":
                continue
            values = [
                self.attribute_value_to_string(argument)
                for argument in self.attribute_arguments(attr)
            ]
            values = [value for value in values if value]
            if values:
                return ",".join(values)
        return None

    def hip_resource_base_type_and_count(self, type_name):
        type_name = self.type_name_string(type_name)
        if not type_name:
            return type_name, 1
        if "[" not in type_name or "]" not in type_name:
            return type_name, 1
        base_type = type_name.split("[", 1)[0].strip()
        suffix = type_name[type_name.find("[") :]
        count = 1
        while suffix.startswith("["):
            close_bracket = suffix.find("]")
            if close_bracket < 0:
                break
            size_text = suffix[1:close_bracket].strip()
            if size_text.isdigit():
                count *= int(size_text)
            suffix = suffix[close_bracket + 1 :]
        return base_type, count

    def hip_resource_kind(self, type_name, node=None, forced_kind=None):
        if forced_kind is not None:
            return forced_kind
        if node is not None and self.is_glsl_buffer_block_node(node):
            return "glsl_buffer_block"

        base_type, _count = self.hip_resource_base_type_and_count(type_name)
        if not base_type:
            return None
        generic_parts = self.generic_type_parts(base_type)
        base_name = generic_parts[0] if generic_parts is not None else base_type
        base_name = base_name.rsplit("::", 1)[-1]

        if (
            self.structured_buffer_type_parts(base_type) is not None
            or self.byte_address_buffer_base_type(base_type) is not None
        ):
            return "buffer"

        canonical = self.canonical_resource_type(base_name) or base_name
        canonical = canonical.rsplit("::", 1)[-1]
        mapped_type = self.type_map.get(canonical, canonical)
        mapped_base = mapped_type.split("<", 1)[0].rsplit("::", 1)[-1]
        if mapped_base == "CglRayTracingAccelerationStructure":
            return "acceleration_structure"
        if canonical in {"sampler", "SamplerState"}:
            return "sampler"
        if canonical.startswith("sampler"):
            return "texture"
        if canonical.startswith(("image", "iimage", "uimage")):
            return "image"
        return None

    def hip_resource_binding_namespace(self, kind):
        if kind in {"cbuffer", "glsl_buffer_block"}:
            return "buffer"
        return kind

    def hip_resource_binding_range_conflicts(self, key, binding, count):
        end = binding + count - 1
        for used_start, used_end, _used_name in self.hip_used_resource_bindings.get(
            key, []
        ):
            if binding <= used_end and used_start <= end:
                return True
        return False

    def next_available_hip_resource_binding(self, namespace, set_index, count):
        key = (namespace, set_index)
        binding = self.hip_resource_binding_cursors.get(key, 0)
        while self.hip_resource_binding_range_conflicts(key, binding, count):
            binding += 1
        self.hip_resource_binding_cursors[key] = binding + count
        return binding

    def reserve_hip_resource_binding(self, namespace, set_index, binding, count, name):
        key = (namespace, set_index)
        end = binding + count - 1
        ranges = self.hip_used_resource_bindings.setdefault(key, [])
        for used_start, used_end, used_name in ranges:
            if binding <= used_end and used_start <= end:
                if used_start == binding and used_end == end and used_name == name:
                    return
                raise ValueError(
                    "Conflicting HIP resource binding for "
                    f"'{name}': {namespace} set {set_index} binding "
                    f"{binding}-{end} overlaps '{used_name}' binding "
                    f"{used_start}-{used_end}"
                )
        ranges.append((binding, end, name))
        self.hip_resource_binding_cursors[key] = max(
            self.hip_resource_binding_cursors.get(key, 0), end + 1
        )

    def hip_resource_metadata_comment(self, node, type_name, kind=None):
        name = getattr(node, "name", None)
        resource_kind = self.hip_resource_kind(type_name, node=node, forced_kind=kind)
        if not name or resource_kind is None:
            return ""

        _base_type, count = self.hip_resource_base_type_and_count(type_name)
        namespace = self.hip_resource_binding_namespace(resource_kind)
        set_index = self.explicit_hip_resource_set_index(node)
        binding = self.explicit_hip_resource_binding_index(node)
        if binding is None:
            binding = self.next_available_hip_resource_binding(
                namespace, set_index, count
            )
            binding_source = "automatic"
            self.reserve_hip_resource_binding(
                namespace, set_index, binding, count, name
            )
        else:
            binding_source = "explicit"
            self.reserve_hip_resource_binding(
                namespace, set_index, binding, count, name
            )

        parts = [
            "// CrossGL resource metadata:",
            f"name={name}",
            f"kind={resource_kind}",
        ]
        if resource_kind == "glsl_buffer_block":
            layout = self.glsl_buffer_block_layout(node)
            if layout:
                parts.append(f"layout={layout}")
            access = self.explicit_resource_access(node)
            if access:
                parts.append(f"access={access}")
        parts.extend(
            [
                f"set={set_index}",
                f"binding={binding}",
                f"binding_source={binding_source}",
            ]
        )
        if count != 1:
            parts.append(f"count={count}")
        register_metadata = self.hip_resource_register_metadata(node)
        if register_metadata:
            parts.append(f"register={register_metadata}")
        return " ".join(parts)

    def reserve_explicit_hip_resource_binding(self, node, type_name, kind=None):
        name = getattr(node, "name", None)
        resource_kind = self.hip_resource_kind(type_name, node=node, forced_kind=kind)
        binding = self.explicit_hip_resource_binding_index(node)
        if not name or resource_kind is None or binding is None:
            return
        _base_type, count = self.hip_resource_base_type_and_count(type_name)
        namespace = self.hip_resource_binding_namespace(resource_kind)
        set_index = self.explicit_hip_resource_set_index(node)
        self.reserve_hip_resource_binding(namespace, set_index, binding, count, name)

    def reserve_explicit_hip_resource_bindings(self, ast):
        for node in getattr(ast, "global_variables", []) or []:
            type_name = self.get_variable_node_type(node) or "float"
            self.reserve_explicit_hip_resource_binding(node, type_name)
        for cbuffer in getattr(ast, "cbuffers", []) or []:
            self.reserve_explicit_hip_resource_binding(
                cbuffer, getattr(cbuffer, "name", None), kind="cbuffer"
            )
        stages = getattr(ast, "stages", {}) or {}
        for stage in stages.values():
            for node in getattr(stage, "local_variables", []) or []:
                type_name = self.get_variable_node_type(node) or "float"
                self.reserve_explicit_hip_resource_binding(node, type_name)
            for cbuffer in getattr(stage, "local_cbuffers", []) or []:
                self.reserve_explicit_hip_resource_binding(
                    cbuffer, getattr(cbuffer, "name", None), kind="cbuffer"
                )

    def explicit_resource_access(self, node):
        if node is None:
            return None

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
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name == "access":
                arguments = getattr(attr, "arguments", []) or []
                if not arguments:
                    continue
                raw_access = self.attribute_value_to_string(arguments[0])
                access = access_names.get(str(raw_access).lower())
            else:
                access = access_names.get(attr_name)
            if access is not None:
                return access
        return None

    def is_glsl_buffer_block_node(self, node):
        attributes = {
            str(getattr(attr, "name", "")).lower()
            for attr in getattr(node, "attributes", []) or []
        }
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        if "glsl_buffer_block" in attributes:
            return True
        return "buffer" in qualifiers and bool(
            attributes & {"std140", "std430", "scalar"}
        )

    def glsl_buffer_block_layout(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name == "glsl_buffer_block":
                arguments = getattr(attr, "arguments", []) or []
                if arguments:
                    return self.attribute_value_to_string(arguments[0])
            if attr_name in {"std140", "std430", "scalar"}:
                return attr_name
        return None

    def glsl_buffer_block_declaration_type(self, type_name, node):
        type_name = self.type_name_string(type_name)
        if not self.is_glsl_buffer_block_node(node):
            return type_name
        if not type_name or "[" not in type_name or "]" not in type_name:
            return type_name
        open_bracket = type_name.find("[")
        base_type = type_name[:open_bracket].strip()
        array_suffix = type_name[open_bracket:]
        if array_suffix != "[]":
            return type_name
        if self.explicit_resource_access(node) == "readonly":
            return f"const {base_type}{array_suffix}"
        return type_name

    def glsl_buffer_block_metadata_comment(self, node, type_name):
        name = getattr(node, "name", None)
        if not name or not self.is_glsl_buffer_block_node(node):
            return ""
        parts = [
            "// CrossGL resource metadata:",
            f"name={name}",
            "kind=glsl_buffer_block",
        ]
        layout = self.glsl_buffer_block_layout(node)
        if layout:
            parts.append(f"layout={layout}")
        access = self.explicit_resource_access(node)
        if access:
            parts.append(f"access={access}")
        return " ".join(parts)

    def glsl_buffer_block_root_name(self, expr):
        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array", getattr(expr, "array_expr", None))
            return self.glsl_buffer_block_root_name(array_expr)
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object", getattr(expr, "object_expr", None))
            return self.glsl_buffer_block_root_name(object_expr)
        if isinstance(expr, IdentifierNode):
            return expr.name
        if isinstance(expr, VariableNode):
            return expr.name
        if isinstance(expr, str):
            return expr
        return getattr(expr, "name", None)

    def glsl_buffer_block_access(self, expr):
        root_name = self.glsl_buffer_block_root_name(expr)
        if root_name is None:
            return None, None
        return self.glsl_buffer_block_accesses.get(root_name), root_name

    def glsl_buffer_block_diagnostic(self, operation, expr, reason, fallback):
        _, resource_name = self.glsl_buffer_block_access(expr)
        resource_name = resource_name or "unknown"
        return (
            f"/* unsupported {self.resource_backend_name()} GLSL buffer block "
            f"{operation}: resource '{resource_name}' {reason} */ {fallback}"
        )

    def glsl_buffer_block_read_diagnostic(self, expr, operation):
        access, _ = self.glsl_buffer_block_access(expr)
        if access != "writeonly":
            return None
        fallback = self.diagnostic_zero_value_for_type(
            self.expression_result_type(expr)
        )
        return self.glsl_buffer_block_diagnostic(
            operation, expr, "is writeonly", fallback
        )

    def glsl_buffer_block_write_diagnostic(self, expr, operation):
        access, _ = self.glsl_buffer_block_access(expr)
        if access != "readonly":
            return None
        return self.glsl_buffer_block_diagnostic(
            operation, expr, "is readonly", "((void)0)"
        )

    def glsl_buffer_block_read_write_diagnostic(self, expr, operation, fallback):
        access, _ = self.glsl_buffer_block_access(expr)
        if access == "readonly":
            return self.glsl_buffer_block_diagnostic(
                operation, expr, "is readonly", fallback
            )
        if access == "writeonly":
            return self.glsl_buffer_block_diagnostic(
                operation, expr, "is writeonly", fallback
            )
        return None

    def resource_type_with_access(self, type_name, node):
        type_name = self.type_name_string(type_name)
        if not type_name:
            return type_name

        access = self.explicit_resource_access(node)
        if access is None:
            return type_name

        base_type = type_name
        array_suffix = ""
        if "[" in type_name and "]" in type_name:
            open_bracket = type_name.find("[")
            base_type = type_name[:open_bracket]
            array_suffix = type_name[open_bracket:]

        parts = self.structured_buffer_type_parts(base_type)
        if parts is not None:
            base_name, element_type = parts
            if base_name in {"StructuredBuffer", "RWStructuredBuffer"}:
                mapped_base = (
                    "StructuredBuffer" if access == "readonly" else "RWStructuredBuffer"
                )
                return f"{mapped_base}<{element_type}>{array_suffix}"

        byte_base = self.byte_address_buffer_base_type(base_type)
        if byte_base in {"ByteAddressBuffer", "RWByteAddressBuffer"}:
            mapped_base = (
                "ByteAddressBuffer" if access == "readonly" else "RWByteAddressBuffer"
            )
            return f"{mapped_base}{array_suffix}"

        return type_name

    def generic_type_parts(self, type_name):
        """Split a generic type name into base name and top-level arguments."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None

        base_type = type_name.split("[", 1)[0].strip()
        generic_start = base_type.find("<")
        generic_end = base_type.rfind(">")
        if generic_start == -1 or generic_end < generic_start:
            return None

        base_name = base_type[:generic_start].strip()
        args_text = base_type[generic_start + 1 : generic_end].strip()
        args = []
        depth = 0
        current = []
        for char in args_text:
            if char == "<":
                depth += 1
                current.append(char)
            elif char == ">":
                depth -= 1
                current.append(char)
            elif char == "," and depth == 0:
                args.append("".join(current).strip())
                current = []
            else:
                current.append(char)

        trailing_arg = "".join(current).strip()
        if trailing_arg:
            args.append(trailing_arg)
        return base_name, args

    def structured_buffer_type_parts(self, type_name):
        """Return structured-buffer base and element type, if applicable."""
        parts = self.generic_type_parts(type_name)
        if parts is None:
            return None

        base_name, args = parts
        if (
            base_name
            not in {
                "StructuredBuffer",
                "RWStructuredBuffer",
                "AppendStructuredBuffer",
                "ConsumeStructuredBuffer",
            }
            or not args
        ):
            return None
        return base_name, args[0]

    def structured_buffer_is_writable(self, type_name):
        """Return whether a structured-buffer type permits writes."""
        parts = self.structured_buffer_type_parts(type_name)
        return parts is not None and parts[0] == "RWStructuredBuffer"

    def structured_buffer_requires_counter(self, type_name):
        """Return whether a structured-buffer type needs an explicit counter."""
        parts = self.structured_buffer_type_parts(type_name)
        return parts is not None and parts[0] in {
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
        }

    def hip_structured_buffer_type(self, type_name):
        """Map structured-buffer resources to HIP pointer types."""
        parts = self.structured_buffer_type_parts(type_name)
        if parts is None:
            return None

        base_name, element_type = parts
        hip_element_type = self.map_type(element_type)
        if base_name in {"StructuredBuffer", "ConsumeStructuredBuffer"}:
            return f"const {hip_element_type}*"
        return f"{hip_element_type}*"

    def byte_address_buffer_base_type(self, type_name):
        """Return the byte-address buffer base type, if applicable."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None
        base_type = type_name.split("[", 1)[0].strip()
        if base_type in {"ByteAddressBuffer", "RWByteAddressBuffer"}:
            return base_type
        return None

    def byte_address_buffer_is_writable(self, type_name):
        """Return whether a byte-address buffer type permits writes."""
        return self.byte_address_buffer_base_type(type_name) == "RWByteAddressBuffer"

    def hip_byte_address_buffer_type(self, type_name):
        """Map ByteAddressBuffer and RWByteAddressBuffer to HIP byte pointers."""
        base_type = self.byte_address_buffer_base_type(type_name)
        if base_type == "ByteAddressBuffer":
            return "const unsigned char*"
        if base_type == "RWByteAddressBuffer":
            return "unsigned char*"
        return None

    def array_access_element_type(self, type_name):
        """Return the element type for HIP arrays and structured buffers."""
        array_element_type = super().array_access_element_type(type_name)
        if array_element_type is not None:
            return array_element_type

        parts = self.structured_buffer_type_parts(type_name)
        if parts is not None:
            return parts[1]
        return None

    def resource_call_result_type(self, func_name, raw_args):
        if func_name in {
            "textureProj",
            "textureProjLod",
            "textureProjGrad",
            "texelFetchOffset",
        }:
            resource_type = None
            if raw_args:
                resource_type = self.resource_base_type(
                    self.get_expression_type(raw_args[0])
                )
            if self.is_shadow_resource_type(resource_type):
                return "float"
            return "float4"
        return super().resource_call_result_type(func_name, raw_args)

    def expression_result_type(self, node):
        """Infer expression result types with HIP buffer operations."""
        if isinstance(node, WaveOpNode):
            return self.wave_result_type(
                getattr(node, "operation", ""), getattr(node, "arguments", []) or []
            )
        if isinstance(node, RayTracingOpNode):
            if getattr(node, "operation", "") == "ReportHit":
                return "bool"
            return None
        if isinstance(node, RayQueryOpNode):
            operation = getattr(node, "operation", "")
            if operation == "Proceed":
                return "bool"
            if operation in {"CandidateRayT", "CommittedRayT"}:
                return "float"
            return "uint"
        if isinstance(node, FunctionCallNode):
            function_expr = getattr(node, "function", getattr(node, "name", None))
            func_name = getattr(function_expr, "name", function_expr)
            if func_name in HIP_WAVE_OP_ARITIES:
                return self.wave_result_type(
                    func_name,
                    getattr(node, "arguments", getattr(node, "args", [])) or [],
                )
            if func_name == "ReportHit":
                return "bool"
            buffer_result_type = self.buffer_call_result_type(node)
            if buffer_result_type is not None:
                return buffer_result_type
        return super().expression_result_type(node)

    def buffer_call_result_type(self, node):
        """Infer result type for structured and byte-address buffer read calls."""
        function_expr = getattr(node, "function", getattr(node, "name", None))
        raw_args = getattr(node, "arguments", getattr(node, "args", []))

        byte_member_call = self.byte_address_buffer_member_call(function_expr)
        if byte_member_call is not None:
            _, operation, _ = byte_member_call
            if operation == "GetDimensions":
                return "uint"
            if operation in self.byte_address_buffer_atomic_operations():
                return "uint"
            if operation.startswith("Load") and raw_args:
                component_count = self.byte_address_buffer_component_count(operation)
                if component_count is not None:
                    return self.byte_address_buffer_value_type(component_count)
            return None

        member_call = self.structured_buffer_member_call(function_expr)
        if member_call is not None:
            _, operation, buffer_type = member_call
            if operation == "Load" and raw_args:
                return self.structured_buffer_type_parts(buffer_type)[1]
            if operation == "Consume":
                return self.structured_buffer_type_parts(buffer_type)[1]
            if operation == "GetDimensions" and not raw_args:
                return "uint"
            return None

        func_name = getattr(function_expr, "name", function_expr)
        if func_name == "buffer_load" and raw_args:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.byte_address_buffer_base_type(buffer_type) is not None:
                return "uint"
            parts = self.structured_buffer_type_parts(buffer_type)
            if parts is not None:
                return parts[1]
        if func_name == "buffer_consume" and raw_args:
            buffer_type = self.expression_result_type(raw_args[0])
            parts = self.structured_buffer_type_parts(buffer_type)
            if parts is not None:
                return parts[1]
        if func_name == "buffer_dimensions" and raw_args:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.buffer_type_supports_length(buffer_type):
                return "uint"
        if func_name in self.structured_buffer_atomic_operations() and raw_args:
            target = self.structured_buffer_atomic_target(raw_args[0])
            if target is not None:
                return target["target_type"]
            target_type = self.expression_result_type(raw_args[0])
            if self.hip_atomic_scalar_kind(target_type) is not None:
                return target_type
        return None

    def canonical_sampled_resource_type(self, type_name):
        """Return the sampler spelling for HLSL-style sampled resources."""
        if not isinstance(type_name, str):
            return None
        base_type = type_name.split("[", 1)[0].split("<", 1)[0].strip()
        return self.sampled_resource_type_aliases.get(base_type)

    def canonical_storage_resource_type(self, type_name):
        """Return the image spelling for HLSL-style writable resources."""
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
        """Return the canonical sampler/image resource type for an alias."""
        if isinstance(type_name, str):
            base_type = type_name.split("[", 1)[0].split("<", 1)[0].strip()
            if base_type in {
                "accelerationStructureEXT",
                "AccelerationStructure",
                "acceleration_structure",
                "RaytracingAccelerationStructure",
                "RayTracingAccelerationStructure",
            }:
                return "RayTracingAccelerationStructure"
            sampler_type = self.sampler_state_type_aliases.get(base_type)
            if sampler_type is not None:
                return sampler_type
        return self.canonical_sampled_resource_type(
            type_name
        ) or self.canonical_storage_resource_type(type_name)

    def dimension_query_spec(self, type_name):
        """Return HIP-specific resource metadata dimensions before shared specs."""
        base_type = self.resource_base_type(type_name)
        hip_specs = {
            "image1D": (("width",), False, False),
            "iimage1D": (("width",), False, False),
            "uimage1D": (("width",), False, False),
            "image1DArray": (("width", "elements"), False, False),
            "iimage1DArray": (("width", "elements"), False, False),
            "uimage1DArray": (("width", "elements"), False, False),
            "iimageCube": (("width", "height"), False, False),
            "uimageCube": (("width", "height"), False, False),
            "imageCubeArray": (("width", "height", "elements"), False, False),
            "iimageCubeArray": (("width", "height", "elements"), False, False),
            "uimageCubeArray": (("width", "height", "elements"), False, False),
        }
        spec = hip_specs.get(base_type)
        if spec is None:
            return super().dimension_query_spec(base_type)
        dimensions, mip, samples = spec
        return {"dimensions": dimensions, "mip": mip, "samples": samples}

    def resource_base_type(self, type_name):
        """Normalize resource aliases before resource dispatch decisions."""
        base_type = ResourceDiagnosticMixin.resource_base_type(self, type_name)
        return self.canonical_resource_type(base_type) or base_type

    def format_typed_declarator(self, type_name, name, dynamic_array_as_pointer=True):
        if (
            hasattr(type_name, "name")
            or hasattr(type_name, "element_type")
            or hasattr(type_name, "pointee_type")
            or hasattr(type_name, "referenced_type")
        ):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)

        if "[" not in type_name or "]" not in type_name:
            return f"{self.map_type(type_name)} {name}"

        open_bracket = type_name.find("[")
        base_type = type_name[:open_bracket]
        array_suffix = type_name[open_bracket:]
        mapped_base = self.map_type(base_type)

        return format_array_declarator(
            mapped_base,
            name,
            array_suffix,
            dynamic_array_as_pointer=dynamic_array_as_pointer,
        )

    def format_array_size(self, size):
        if size is None:
            return ""
        if isinstance(size, int):
            return str(size)
        return self.visit(size)

    def generate_kernel_wrapper(self, kernel_node: FunctionNode) -> str:
        """Generate a host-side HIP launch wrapper for a kernel node."""
        wrapper_lines = []

        # Generate wrapper function
        wrapper_name = f"launch_{kernel_node.name}"
        params = []
        args = []

        for param in kernel_node.parameters:
            param_type = self.map_type(param.param_type)
            params.append(f"{param_type} {param.name}")
            args.append(param.name)

        # Add grid and block size parameters
        params.extend(["dim3 gridSize", "dim3 blockSize", "hipStream_t stream = 0"])

        wrapper_lines.extend(
            [
                f"void {wrapper_name}({', '.join(params)})",
                "{",
                f"    hipLaunchKernelGGL({kernel_node.name}, gridSize, blockSize, 0, stream, {', '.join(args)});",
                "}",
            ]
        )

        return "\n".join(wrapper_lines)


def generate_hip_code(ast: ShaderNode) -> str:
    """Generate HIP source from a CrossGL shader AST."""
    generator = HipCodeGen()
    return generator.generate(ast)
