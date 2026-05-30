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
    LiteralPatternNode,
    MemberAccessNode,
    PrimitiveType,
    RangeNode,
    ReturnNode,
    ShaderNode,
    StructNode,
    VariableNode,
    WildcardPatternNode,
)
from .resource_arrays import format_array_declarator
from .resource_diagnostics import ResourceDiagnosticMixin
from .resource_query import ResourceQueryMixin
from .stage_utils import normalize_stage_name, stage_matches
from .vector_arithmetic import VectorArithmeticMixin


class HipCodeGen(VectorArithmeticMixin, ResourceQueryMixin, ResourceDiagnosticMixin):
    """Emit HIP source from the shared CrossGL translator AST."""

    resource_diagnostic_backend = "HIP"
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

    def __init__(self):
        """Initialize HIP type maps and per-generation visitor state."""
        self.indent_level = 0
        self.code_lines = []
        self.current_function = None
        self.variable_counter = 0
        self.variable_types = {}
        self.struct_member_types = {}
        self.function_return_types = {}
        self.helper_functions = {}
        self.query_resource_names = set()
        self.query_metadata_function_params = {}
        self.query_functions_by_name = {}
        self.structured_buffer_length_names = set()
        self.structured_buffer_length_function_params = {}
        self.current_structured_buffer_length_parameters = {}
        self.current_function_name = None
        self.resource_query_info_required = False

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
        self.struct_member_types = {}
        self.function_return_types = self.collect_function_return_types(node)
        self.helper_functions = {}
        self.resource_query_info_required = False
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

        self.add_includes()
        self.visit(node)
        self.insert_helper_functions()

        return "\n".join(self.code_lines)

    def unsupported_stage_types(self):
        return {
            "amplification",
            "geometry",
            "mesh",
            "object",
            "ray_any_hit",
            "ray_callable",
            "ray_closest_hit",
            "ray_generation",
            "ray_intersection",
            "ray_miss",
            "task",
            "tessellation_control",
            "tessellation_evaluation",
        }

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

                    self.visit(stage.entry_point)

        return ""

    def visit_FunctionNode(self, node: FunctionNode) -> str:
        """Render a CrossGL function or compute entry point as HIP code."""
        saved_variable_types = self.variable_types.copy()
        self.current_function = node.name
        saved_current_function_name = self.current_function_name
        saved_structured_buffer_length_parameters = (
            self.current_structured_buffer_length_parameters
        )
        self.current_function_name = node.name
        self.current_structured_buffer_length_parameters = {}

        qualifiers = []
        if hasattr(node, "qualifiers") and node.qualifiers:
            for qualifier in node.qualifiers:
                if "kernel" in qualifier or "compute" in qualifier:
                    qualifiers.append("__global__")
                elif "device" in qualifier:
                    qualifiers.append("__device__")
                else:
                    qualifiers.append("__device__")
        elif hasattr(node, "qualifier") and node.qualifier:
            if "kernel" in node.qualifier or "compute" in node.qualifier:
                qualifiers.append("__global__")
            elif "device" in node.qualifier:
                qualifiers.append("__device__")
            else:
                qualifiers.append("__device__")
        else:
            qualifiers.append("__device__")

        if hasattr(node, "return_type"):
            return_type = self.map_type(node.return_type)
        else:
            return_type = "void"

        param_list = getattr(node, "parameters", getattr(node, "params", []))
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
        signature = f"{qualifier_str} {return_type} {node.name}({params})"

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
        self.variable_types = saved_variable_types
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

        self.register_variable_type(param_name, param_type)
        return self.format_typed_declarator(param_type, param_name)

    def visit_StructNode(self, node: StructNode) -> str:
        self.add_line(f"struct {node.name}")
        self.add_line("{")
        self.indent_level += 1

        members = getattr(node, "members", [])
        member_types = {}
        for member in members:
            if hasattr(member, "member_type"):
                member_type = member.member_type
            elif hasattr(member, "vtype"):
                member_type = member.vtype
            elif hasattr(member, "var_type"):
                member_type = member.var_type
            else:
                member_type = "float"

            member_types[member.name] = member_type
            self.add_line(f"{self.format_typed_declarator(member_type, member.name)};")

        self.struct_member_types[node.name] = member_types
        self.indent_level -= 1
        self.add_line("};")
        self.add_line()
        return ""

    def visit_VariableNode(self, node: VariableNode) -> str:
        var_type = self.get_variable_node_type(node)
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
            self.register_variable_type(node.name, inferred_type)
            declaration = self.format_typed_declarator(
                inferred_type or "auto", node.name
            )
        else:
            var_type = var_type or "int"
            self.register_variable_type(node.name, var_type)
            declaration = self.format_typed_declarator(var_type, node.name)

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

    def visit_CbufferNode(self, node: CbufferNode) -> str:
        self.add_line(f"struct {node.name}")
        self.add_line("{")
        self.indent_level += 1

        for member in node.members:
            if isinstance(member, VariableNode):
                member_type = getattr(
                    member, "vtype", getattr(member, "var_type", "int")
                )
                declaration = self.format_typed_declarator(member_type, member.name)
                self.add_line(f"{declaration};")

        self.indent_level -= 1
        self.add_line("};")
        self.add_line()
        return ""

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

    def is_supported_match_arm(self, arm):
        if getattr(arm, "guard", None) is not None:
            return False
        pattern = getattr(arm, "pattern", None)
        return isinstance(pattern, (LiteralPatternNode, WildcardPatternNode))

    def validate_match_arms(self, arms):
        wildcard_index = None
        for index, arm in enumerate(arms):
            if not self.is_supported_match_arm(arm):
                return False
            if isinstance(getattr(arm, "pattern", None), WildcardPatternNode):
                if wildcard_index is not None:
                    return False
                wildcard_index = index
        return wildcard_index is None or wildcard_index == len(arms) - 1

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
        expression = self.visit(getattr(node, "expression", None))

        self.add_line(f"switch ({expression})")
        self.add_line("{")
        self.indent_level += 1

        arms = getattr(node, "arms", []) or []
        if not self.validate_match_arms(arms):
            raise ValueError(
                "Unsupported match arm for HIP codegen; only unguarded "
                "literal patterns and a final wildcard can be lowered to switch"
            )

        wildcard_body = None
        for arm in arms:
            pattern = getattr(arm, "pattern", None)
            if isinstance(pattern, WildcardPatternNode):
                wildcard_body = getattr(arm, "body", [])
                continue

            self.add_line(f"case {self.visit(pattern.literal)}:")
            self.indent_level += 1
            body = getattr(arm, "body", [])
            self.emit_body(body)
            if not self.statement_body_terminates(body):
                self.add_line("break;")
            self.indent_level -= 1

        if wildcard_body is not None:
            self.add_line("default:")
            self.indent_level += 1
            self.emit_body(wildcard_body)
            if not self.statement_body_terminates(wildcard_body):
                self.add_line("break;")
            self.indent_level -= 1

        self.indent_level -= 1
        self.add_line("}")

        return ""

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
            value = self.visit(node.value)
            self.add_line(f"return {value};")
        else:
            self.add_line("return;")
        return ""

    def visit_AssignmentNode(self, node) -> str:
        left = self.visit(node.left)
        right = self.visit(node.right)
        operator = getattr(node, "operator", getattr(node, "op", "="))
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
        if hasattr(func_expr, "name"):
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

        is_user_function = self.is_user_defined_function(func_name)
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

            resource_call = self.generate_resource_call(func_name, raw_args, args)
            if resource_call is not None:
                return resource_call

        if is_user_function:
            args = self.hip_user_function_call_arguments(func_name, raw_args, args)
            return f"{callee}({', '.join(args)})"

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
        elif func_name in {"barrier", "workgroupBarrier"}:
            return "__syncthreads()"
        elif func_name == "memoryBarrier":
            return "__threadfence()"

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

    def register_variable_type(self, name, type_name):
        if not name or type_name is None:
            return
        if not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        self.variable_types[name] = type_name

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

    def image_surface_x_offset(self, coord, value_type, image_type):
        if "1D" in image_type and "Array" not in image_type:
            return f"{coord} * sizeof({value_type})"
        return self.surface_x_offset(coord, value_type)

    def generate_resource_call(self, func_name, raw_args, args):
        if func_name in {"textureSize", "imageSize"}:
            return self.generate_dimension_query(func_name, raw_args, args)

        if func_name in {"textureSamples", "imageSamples"}:
            return self.generate_sample_count_query(func_name, raw_args, args)

        if func_name == "textureQueryLevels":
            return self.generate_texture_query_levels(raw_args)

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
                return access
            if operation == "Store":
                if self.structured_buffer_is_writable(buffer_type) and len(args) >= 2:
                    return f"{access} = {args[1]}"
                return self.structured_buffer_diagnostic_call(
                    "Store", buffer_type, "((void)0)"
                )
            return None

        if func_name == "buffer_load" and len(args) >= 2:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.structured_buffer_type_parts(buffer_type) is None:
                return None
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
                helper_name = self.require_byte_address_load_helper(component_count)
                return f"{helper_name}({buffer_name}, {args[0]})"

            if operation.startswith("Store") and len(args) >= 2:
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

    def visit_str(self, node) -> str:
        return str(node)

    def visit_int(self, node) -> str:
        return str(node)

    def visit_float(self, node) -> str:
        return str(node)

    def visit_ArrayAccessNode(self, node) -> str:
        array = self.visit(node.array)
        index = self.visit(node.index)
        return f"{array}[{index}]"

    def visit_ArrayLiteralNode(self, node: ArrayLiteralNode) -> str:
        elements = ", ".join(self.visit(element) for element in node.elements)
        return f"{{{elements}}}"

    def visit_MemberAccessNode(self, node) -> str:
        raw_object_name = getattr(node.object, "name", None)
        if raw_object_name is not None:
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
        # Handle built-in variables mapping
        return self.builtin_map.get(name, name)

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
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            return self.convert_type_node_to_string(type_name)
        return str(type_name)

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        if hasattr(type_node, "name"):
            generic_args = getattr(type_node, "generic_args", [])
            if generic_args:
                args = ", ".join(
                    self.convert_type_node_to_string(arg) for arg in generic_args
                )
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
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_str = self.convert_type_node_to_string(type_name)
        else:
            type_str = str(type_name)

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
        return self.type_map.get(canonical_type or type_str, type_str)

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

    def expression_result_type(self, node):
        """Infer expression result types with HIP buffer operations."""
        if isinstance(node, FunctionCallNode):
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
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
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
