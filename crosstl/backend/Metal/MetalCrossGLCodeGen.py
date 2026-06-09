"""Reverse code generator that emits CrossGL from Metal AST nodes."""

import re

from .MetalAst import *
from .MetalLexer import *
from .MetalParser import *


class MetalToCrossGLConverter:
    """Serialize Metal backend AST nodes back into CrossGL source."""

    crossgl_identifier_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    decimal_integer_literal_pattern = re.compile(r"^(?P<body>\d+)(?P<suffix>[uUlL]+)$")
    hex_integer_literal_pattern = re.compile(
        r"^(?P<body>0[xX][0-9a-fA-F]+)(?P<suffix>[uUlL]+)$"
    )
    binary_integer_literal_pattern = re.compile(
        r"^(?P<body>0[bB][01]+)(?P<suffix>[uUlL]+)$"
    )
    cast_literal_operand_pattern = re.compile(
        r"^(?:0[xX][0-9a-fA-F]+u?|0[bB][01]+u?|"
        r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fF]?|\d+u?)$"
    )
    binary_precedence = {
        "||": 1,
        "&&": 2,
        "|": 3,
        "^": 4,
        "&": 5,
        "==": 6,
        "!=": 6,
        "<": 7,
        ">": 7,
        "<=": 7,
        ">=": 7,
        "<<": 8,
        ">>": 8,
        "+": 9,
        "-": 9,
        "*": 10,
        "/": 10,
        "%": 10,
    }
    crossgl_reserved_identifiers = {
        "break",
        "case",
        "cbuffer",
        "compute",
        "continue",
        "default",
        "do",
        "else",
        "false",
        "for",
        "fragment",
        "if",
        "in",
        "out",
        "return",
        "sampler",
        "sampler1d",
        "sampler1darray",
        "sampler2d",
        "sampler2darray",
        "sampler2darrayshadow",
        "sampler2dms",
        "sampler2dmsarray",
        "sampler2dshadow",
        "sampler3d",
        "samplercube",
        "samplercubearray",
        "samplercubeshadow",
        "shader",
        "struct",
        "switch",
        "texture",
        "texture1d",
        "texture2d",
        "texture2darray",
        "texture3d",
        "texturecube",
        "true",
        "uniform",
        "vertex",
        "while",
    }
    storage_texture_accesses = {"access::read", "access::write", "access::read_write"}
    pixel_data_type_wrappers = {
        "r8unorm",
        "r8snorm",
        "r8uint",
        "r8sint",
        "r16unorm",
        "r16snorm",
        "r16uint",
        "r16sint",
        "r16float",
        "rg8unorm",
        "rg8snorm",
        "rg8uint",
        "rg8sint",
        "rg16unorm",
        "rg16snorm",
        "rg16uint",
        "rg16sint",
        "rg16float",
        "rgba8unorm",
        "rgba8snorm",
        "rgba8uint",
        "rgba8sint",
        "rgba16unorm",
        "rgba16snorm",
        "rgba16uint",
        "rgba16sint",
        "rgba16float",
        "rgba32uint",
        "rgba32sint",
        "rgba32float",
    }
    metal_math_namespace_prefixes = (
        "metal::fast::",
        "metal::precise::",
        "metal::",
        "fast::",
        "precise::",
    )
    metal_math_intrinsics = {
        "abs",
        "acos",
        "acosh",
        "all",
        "any",
        "asin",
        "asinh",
        "atan",
        "atan2",
        "atanh",
        "ceil",
        "clamp",
        "cos",
        "cosh",
        "cospi",
        "distance",
        "dot",
        "exp",
        "exp2",
        "fabs",
        "floor",
        "fma",
        "fmax",
        "fmin",
        "fmod",
        "fract",
        "isfinite",
        "isinf",
        "isnan",
        "isnormal",
        "isordered",
        "isunordered",
        "length",
        "log",
        "log2",
        "max",
        "min",
        "mix",
        "normalize",
        "pow",
        "reflect",
        "rsqrt",
        "select",
        "sign",
        "sin",
        "sincos",
        "sinh",
        "sinpi",
        "smoothstep",
        "sqrt",
        "step",
        "tan",
        "tanh",
    }
    metal_bit_intrinsics = {
        "popcount": "bitCount",
        "reverse_bits": "bitfieldReverse",
    }

    def __init__(self):
        self.rt_qualifiers = {
            "intersection",
            "anyhit",
            "closesthit",
            "miss",
            "callable",
            "mesh",
            "object",
            "amplification",
        }
        self.type_map = {
            # Scalar Types
            "void": "void",
            "bool": "bool",
            "char": "int8",
            "uchar": "uint8",
            "short": "int16",
            "ushort": "uint16",
            "int8_t": "int8",
            "uint8_t": "uint8",
            "int16_t": "int16",
            "uint16_t": "uint16",
            "int32_t": "int",
            "uint32_t": "uint",
            "int": "int",
            "uint": "uint",
            "long": "int64",
            "ulong": "uint64",
            "int64_t": "int64",
            "uint64_t": "uint64",
            "float": "float",
            "half": "float16",
            "bfloat": "f16",
            "double": "double",
            "size_t": "uint64",
            "ptrdiff_t": "int64",
            "atomic_int": "atomic_int",
            "atomic_uint": "atomic_uint",
            "atomic_bool": "atomic_bool",
            "atomic_ulong": "atomic_ulong",
            "atomic_float": "atomic_float",
            # Vector Types - float
            "float2": "vec2",
            "float3": "vec3",
            "float4": "vec4",
            # Vector Types - half
            "half2": "f16vec2",
            "half3": "f16vec3",
            "half4": "f16vec4",
            # Vector Types - int
            "int2": "ivec2",
            "int3": "ivec3",
            "int4": "ivec4",
            # Vector Types - uint
            "uint2": "uvec2",
            "uint3": "uvec3",
            "uint4": "uvec4",
            # Vector Types - short
            "short2": "i16vec2",
            "short3": "i16vec3",
            "short4": "i16vec4",
            # Vector Types - ushort
            "ushort2": "u16vec2",
            "ushort3": "u16vec3",
            "ushort4": "u16vec4",
            # Vector Types - char
            "char2": "i8vec2",
            "char3": "i8vec3",
            "char4": "i8vec4",
            # Vector Types - uchar
            "uchar2": "u8vec2",
            "uchar3": "u8vec3",
            "uchar4": "u8vec4",
            # Vector Types - bool
            "bool2": "bvec2",
            "bool3": "bvec3",
            "bool4": "bvec4",
            # Packed vector types
            "packed_float2": "vec2",
            "packed_float3": "vec3",
            "packed_float4": "vec4",
            "packed_half2": "f16vec2",
            "packed_half3": "f16vec3",
            "packed_half4": "f16vec4",
            "packed_char2": "i8vec2",
            "packed_char3": "i8vec3",
            "packed_char4": "i8vec4",
            "packed_uchar2": "u8vec2",
            "packed_uchar3": "u8vec3",
            "packed_uchar4": "u8vec4",
            "packed_short2": "i16vec2",
            "packed_short3": "i16vec3",
            "packed_short4": "i16vec4",
            "packed_ushort2": "u16vec2",
            "packed_ushort3": "u16vec3",
            "packed_ushort4": "u16vec4",
            "packed_int2": "ivec2",
            "packed_int3": "ivec3",
            "packed_int4": "ivec4",
            "packed_uint2": "uvec2",
            "packed_uint3": "uvec3",
            "packed_uint4": "uvec4",
            # SIMD types
            "simd_float2": "vec2",
            "simd_float3": "vec3",
            "simd_float4": "vec4",
            "simd_float2x2": "mat2",
            "simd_float3x3": "mat3",
            "simd_float4x4": "mat4",
            "simd_int2": "ivec2",
            "simd_int3": "ivec3",
            "simd_int4": "ivec4",
            "simd_uint2": "uvec2",
            "simd_uint3": "uvec3",
            "simd_uint4": "uvec4",
            # simd.h aliases commonly found in shared Metal headers
            "vector_float2": "vec2",
            "vector_float3": "vec3",
            "vector_float4": "vec4",
            "matrix_float3x3": "mat3",
            "matrix_float4x4": "mat4",
            # Matrix Types - float
            "float2x2": "mat2",
            "float2x3": "mat2x3",
            "float2x4": "mat2x4",
            "float3x2": "mat3x2",
            "float3x3": "mat3",
            "float3x4": "mat3x4",
            "float4x2": "mat4x2",
            "float4x3": "mat4x3",
            "float4x4": "mat4",
            # Matrix Types - half
            "half2x2": "f16mat2",
            "half2x3": "f16mat2x3",
            "half2x4": "f16mat2x4",
            "half3x2": "f16mat3x2",
            "half3x3": "f16mat3",
            "half3x4": "f16mat3x4",
            "half4x2": "f16mat4x2",
            "half4x3": "f16mat4x3",
            "half4x4": "f16mat4",
            # Texture Types
            "texture1d": "sampler1D",
            "texture1d<float>": "sampler1D",
            "texture1d<half>": "sampler1D",
            "texture1d<int>": "isampler1D",
            "texture1d<uint>": "usampler1D",
            "texture1d_array": "sampler1DArray",
            "texture1d_array<float>": "sampler1DArray",
            "texture1d_array<half>": "sampler1DArray",
            "texture1d_array<int>": "isampler1DArray",
            "texture1d_array<uint>": "usampler1DArray",
            "texture2d": "sampler2D",
            "texture2d<float>": "sampler2D",
            "texture2d<half>": "sampler2D",
            "texture2d<int>": "isampler2D",
            "texture2d<uint>": "usampler2D",
            "texture2d_ms": "sampler2DMS",
            "texture2d_ms<float>": "sampler2DMS",
            "texture2d_ms<half>": "sampler2DMS",
            "texture2d_ms<int>": "isampler2DMS",
            "texture2d_ms<uint>": "usampler2DMS",
            "texture2d_ms_array": "sampler2DMSArray",
            "texture2d_ms_array<float>": "sampler2DMSArray",
            "texture2d_ms_array<half>": "sampler2DMSArray",
            "texture2d_ms_array<int>": "isampler2DMSArray",
            "texture2d_ms_array<uint>": "usampler2DMSArray",
            "texture3d": "sampler3D",
            "texture3d<float>": "sampler3D",
            "texture3d<half>": "sampler3D",
            "texture3d<int>": "isampler3D",
            "texture3d<uint>": "usampler3D",
            "texturecube": "samplerCube",
            "texturecube<float>": "samplerCube",
            "texturecube<half>": "samplerCube",
            "texturecube<int>": "isamplerCube",
            "texturecube<uint>": "usamplerCube",
            "TextureCube": "samplerCube",
            "texturecube_array": "samplerCubeArray",
            "texturecube_array<float>": "samplerCubeArray",
            "texturecube_array<half>": "samplerCubeArray",
            "texturecube_array<int>": "isamplerCubeArray",
            "texturecube_array<uint>": "usamplerCubeArray",
            "texture2d_array": "sampler2DArray",
            "texture2d_array<float>": "sampler2DArray",
            "texture2d_array<half>": "sampler2DArray",
            "texture2d_array<int>": "isampler2DArray",
            "texture2d_array<uint>": "usampler2DArray",
            "texture_buffer": "samplerBuffer",
            "texture_buffer<float>": "samplerBuffer",
            "texture_buffer<half>": "samplerBuffer",
            "texture_buffer<int>": "isamplerBuffer",
            "texture_buffer<uint>": "usamplerBuffer",
            "depth2d": "sampler2DShadow",
            "depth2d<float>": "sampler2DShadow",
            "depth2d_array": "sampler2DArrayShadow",
            "depth2d_array<float>": "sampler2DArrayShadow",
            "depthcube": "samplerCubeShadow",
            "depthcube<float>": "samplerCubeShadow",
            "depthcube_array": "samplerCubeArrayShadow",
            "depthcube_array<float>": "samplerCubeArrayShadow",
            "depth2d_ms": "sampler2DMS",
            "depth2d_ms<float>": "sampler2DMS",
            "depth2d_ms_array": "sampler2DMSArray",
            "depth2d_ms_array<float>": "sampler2DMSArray",
            # SwiftUI layer-effect shaders expose the rendered layer as a
            # sampler-like object with sample(coord) syntax.
            "SwiftUI::Layer": "sampler2D",
            # RealityKit custom material shaders use framework-scoped opaque
            # parameter types with no direct CrossGL namespace syntax.
            "realitykit::surface_parameters": "surface_parameters",
            "realitykit::geometry_parameters": "geometry_parameters",
            "acceleration_structure": "acceleration_structure",
            "intersection_function_table": "intersection_function_table",
            "visible_function_table": "visible_function_table",
            "indirect_command_buffer": "indirect_command_buffer",
            "ray": "ray",
            "ray_data": "ray_data",
            "intersection_result": "intersection_result",
            "intersection_params": "intersection_params",
            "triangle_intersection_params": "triangle_intersection_params",
            "intersector": "intersector",
            # Sampler type
            "sampler": "sampler",
        }
        self.type_aliases = {}
        self.global_variable_types = {}
        self.current_variable_types = {}
        self.storage_texture_declaration_ids = set()
        self.global_storage_texture_names = set()
        self.current_storage_texture_names = set()
        self.global_structured_buffer_names = set()
        self.current_structured_buffer_names = set()
        self.global_sampler_names = set()
        self.suppress_structured_buffer_index_lowering = False
        self.struct_member_types = {}
        self.struct_name_map = {}
        self.identifier_maps = [{}]
        self.used_identifier_names = [set()]
        self.texture_method_functions = {
            "read": "textureLoad",
            "write": "textureStore",
            "sample_compare": "textureCompare",
            "sample_compare_level": "textureCompareLod",
            "gather": "textureGather",
            "gather_compare": "textureGatherCompare",
        }
        self.texture_method_storage_operations = {
            "read": "read",
            "write": "write",
        }
        self.sampled_texture_methods = {
            "sample_compare",
            "sample_compare_level",
            "gather",
            "gather_compare",
        }
        self.texture_method_operations = {
            "read": "load",
            "write": "store",
            "sample_compare": "sample_compare",
            "sample_compare_level": "sample_compare_lod",
            "gather": "gather",
            "gather_compare": "gather_compare",
        }
        self.texture_gather_components = {
            "x": "0",
            "y": "1",
            "z": "2",
            "w": "3",
        }
        self.texture_size_query_methods = {
            "get_width",
            "get_height",
            "get_depth",
            "get_array_size",
        }
        self.parameter_direction_qualifiers = ("inout", "out")
        self.fragment_execution_attribute_names = {
            "early_fragment_tests",
        }
        self.function_metadata_attribute_names = {
            "host_name",
        }

        self.map_semantics = {
            # Vertex attributes
            "attribute(0)": "POSITION",
            "attribute(1)": "NORMAL",
            "attribute(2)": "TANGENT",
            "attribute(3)": "BINORMAL",
            "attribute(4)": "TEXCOORD",
            "attribute(5)": "TEXCOORD0",
            "attribute(6)": "TEXCOORD1",
            "attribute(7)": "TEXCOORD2",
            "attribute(8)": "TEXCOORD3",
            "attribute(9)": "COLOR",
            "attribute(10)": "COLOR0",
            "attribute(11)": "COLOR1",
            "vertex_id": "gl_VertexID",
            "instance_id": "gl_InstanceID",
            "base_vertex": "gl_BaseVertex",
            "base_instance": "gl_BaseInstance",
            "position": "gl_Position",
            "point_size": "gl_PointSize",
            "clip_distance": "gl_ClipDistance",
            "front_facing": "gl_FrontFacing",
            "point_coord": "gl_PointCoord",
            "color(0)": "gl_FragColor",
            "color(1)": "gl_FragColor1",
            "color(2)": "gl_FragColor2",
            "color(3)": "gl_FragColor3",
            "color(4)": "gl_FragColor4",
            "depth(any)": "gl_FragDepth",
            "stencil": "gl_FragStencilRefEXT",
            "sample_id": "gl_SampleID",
            "sample_mask": "gl_SampleMask",
            "primitive_id": "gl_PrimitiveID",
            "viewport_array_index": "gl_ViewportIndex",
            "render_target_array_index": "gl_Layer",
            "thread_position_in_grid": "gl_GlobalInvocationID",
            "thread_position_in_threadgroup": "gl_LocalInvocationID",
            "threadgroup_position_in_grid": "gl_WorkGroupID",
            "thread_index_in_threadgroup": "gl_LocalInvocationIndex",
            "thread_index_in_simdgroup": "gl_SubgroupInvocationID",
            "simdgroup_index_in_threadgroup": "gl_SubgroupID",
            "threads_per_threadgroup": "gl_WorkGroupSize",
            "threadgroups_per_grid": "gl_NumWorkGroups",
            "stage_in": "",
        }

    def texture_method_descriptor(self, method):
        function = self.texture_method_functions.get(method)
        if function is None:
            return None
        return {
            "method": method,
            "function": function,
            "storage_operation": self.texture_method_storage_operations.get(method),
            "sampled_texture": method in self.sampled_texture_methods,
        }

    def resource_method_descriptor(self, method):
        descriptor = self.texture_method_descriptor(method)
        if descriptor is None:
            return None
        descriptor = dict(descriptor)
        descriptor["resource"] = (
            "texture_or_image" if descriptor["storage_operation"] else "texture"
        )
        descriptor["operation"] = self.texture_method_operations.get(method)
        return descriptor

    def push_identifier_scope(self):
        self.identifier_maps.append({})
        self.used_identifier_names.append(set())

    def pop_identifier_scope(self):
        self.identifier_maps.pop()
        self.used_identifier_names.pop()

    def sanitize_identifier(self, name):
        if not name:
            return name
        if (
            self.crossgl_identifier_pattern.match(name)
            and name not in self.crossgl_reserved_identifiers
        ):
            return name

        parts = []
        for index, char in enumerate(str(name)):
            valid = (
                char == "_"
                or ("A" <= char <= "Z")
                or ("a" <= char <= "z")
                or (index > 0 and "0" <= char <= "9")
            )
            parts.append(char if valid else f"_u{ord(char):x}")
        candidate = "".join(parts)
        if not candidate or not re.match(r"^[A-Za-z_]", candidate):
            candidate = f"_{candidate}"
        if candidate in self.crossgl_reserved_identifiers:
            candidate = f"{candidate}_"
        return candidate

    def declare_identifier(self, name):
        if not name:
            return name
        current_map = self.identifier_maps[-1]
        if name in current_map:
            return current_map[name]

        base = self.sanitize_identifier(name)
        candidate = base
        used = self.used_identifier_names[-1]
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        current_map[name] = candidate
        return candidate

    def render_identifier(self, name):
        for scope in reversed(self.identifier_maps):
            if name in scope:
                return scope[name]
        return self.sanitize_identifier(name)

    def reserve_generated_identifier(self, base):
        used = self.used_identifier_names[-1]
        sanitized_base = self.sanitize_identifier(base)
        candidate = sanitized_base
        suffix = 2
        while candidate in used:
            candidate = f"{sanitized_base}_{suffix}"
            suffix += 1
        used.add(candidate)
        return candidate

    def generate_sampler_constructor_arg(self, arg, is_main=False):
        if isinstance(arg, VariableNode) and self.is_scoped_identifier(arg.name):
            return arg.name
        return self.generate_expression(arg, is_main)

    def is_scoped_identifier(self, name):
        return (
            isinstance(name, str)
            and "::" in name
            and all(
                self.crossgl_identifier_pattern.match(part)
                and part not in self.crossgl_reserved_identifiers
                for part in name.split("::")
            )
        )

    def unwrap_texture_option_argument(self, expr, option_name):
        if (
            isinstance(expr, FunctionCallNode)
            and self.unscoped_function_name(expr.name) == option_name
            and len(expr.args) == 1
        ):
            return expr.args[0]
        return expr

    def unscoped_function_name(self, name):
        return str(name).split("::")[-1]

    def texture_sample_options_call(self, options, sample_args, is_main=False):
        if not options:
            return f"texture({', '.join(sample_args)})"
        if len(options) > 2:
            return None

        option = options[0]
        offset = options[1] if len(options) == 2 else None

        if (
            isinstance(option, FunctionCallNode)
            and self.unscoped_function_name(option.name) == "bias"
            and len(option.args) == 1
        ):
            bias = self.generate_expression(option.args[0], is_main)
            if offset is not None:
                rendered_offset = self.generate_expression(offset, is_main)
                args = sample_args + [rendered_offset, bias]
                return f"textureOffset({', '.join(args)})"
            return f"texture({', '.join(sample_args + [bias])})"

        gradient_args = self.texture_gradient_option_arguments(option)
        if gradient_args is not None:
            ddx = self.generate_expression(gradient_args[0], is_main)
            ddy = self.generate_expression(gradient_args[1], is_main)
            if offset is not None:
                rendered_offset = self.generate_expression(offset, is_main)
                return (
                    f"textureGradOffset("
                    f"{', '.join(sample_args + [ddx, ddy, rendered_offset])})"
                )
            return f"textureGrad({', '.join(sample_args + [ddx, ddy])})"

        min_lod_clamp_arg = self.unwrap_texture_option_argument(option, "min_lod_clamp")
        if min_lod_clamp_arg is not option:
            min_lod = self.generate_expression(min_lod_clamp_arg, is_main)
            if offset is not None:
                rendered_offset = self.generate_expression(offset, is_main)
                return (
                    "textureMinLodClampOffset("
                    f"{', '.join(sample_args + [min_lod, rendered_offset])})"
                )
            return f"textureMinLodClamp({', '.join(sample_args + [min_lod])})"

        level_arg = self.unwrap_texture_option_argument(option, "level")
        if level_arg is not option or not self.texture_sample_option_is_offset(option):
            lod = self.generate_expression(level_arg, is_main)
            if offset is not None:
                rendered_offset = self.generate_expression(offset, is_main)
                return (
                    f"textureLodOffset("
                    f"{', '.join(sample_args + [lod, rendered_offset])})"
                )
            return f"textureLod({', '.join(sample_args + [lod])})"

        rendered_offset = self.generate_expression(option, is_main)
        return f"textureOffset({', '.join(sample_args + [rendered_offset])})"

    def texture_sample_option_is_offset(self, option):
        mapped_type = self.expression_mapped_type(option)
        if self.is_integer_vector_type(mapped_type):
            return True
        constructor_type = getattr(option, "name", None) or getattr(
            option, "type_name", None
        )
        if constructor_type is None:
            return False
        return self.is_integer_vector_type(self.map_type(str(constructor_type)))

    def is_integer_vector_type(self, type_name):
        return type_name in {
            "ivec2",
            "ivec3",
            "ivec4",
            "uvec2",
            "uvec3",
            "uvec4",
            "i16vec2",
            "i16vec3",
            "i16vec4",
            "u16vec2",
            "u16vec3",
            "u16vec4",
            "i8vec2",
            "i8vec3",
            "i8vec4",
            "u8vec2",
            "u8vec3",
            "u8vec4",
        }

    def texture_gradient_option_arguments(self, option):
        if (
            isinstance(option, FunctionCallNode)
            and self.unscoped_function_name(option.name)
            in {"gradient2d", "gradient3d", "gradientcube"}
            and len(option.args) == 2
        ):
            return option.args
        return None

    def resource_classification_type(self, mapped_type):
        if not mapped_type:
            return mapped_type
        base = str(mapped_type).strip()
        while base.endswith("*") or base.endswith("&"):
            base = base[:-1].strip()
        return base

    def sampled_array_coordinate_constructor(self, texture_expr):
        mapped_type = self.resource_classification_type(
            self.expression_mapped_type(texture_expr)
        )
        return self.sampled_array_coordinate_constructor_for_type(mapped_type)

    def sampled_array_coordinate_constructor_for_type(self, mapped_type):
        return {
            "sampler1DArray": "vec2",
            "isampler1DArray": "vec2",
            "usampler1DArray": "vec2",
            "sampler2DArray": "vec3",
            "isampler2DArray": "vec3",
            "usampler2DArray": "vec3",
            "sampler2DArrayShadow": "vec3",
            "samplerCubeArray": "vec4",
            "samplerCubeArrayShadow": "vec4",
        }.get(mapped_type)

    def texture_sample_coordinate_and_options(
        self, texture_expr, coords_expr, options, is_main=False
    ):
        coords = self.generate_expression(coords_expr, is_main)
        remaining_options = list(options)
        constructor = self.sampled_array_coordinate_constructor(texture_expr)
        if constructor and remaining_options:
            layer = self.generate_expression(remaining_options.pop(0), is_main)
            coords = f"{constructor}({coords}, {layer})"
        return coords, remaining_options

    def texture_compare_method_base_arguments(
        self, obj_expr, method_args, is_main=False
    ):
        mapped_type = self.resource_classification_type(
            self.expression_mapped_type(obj_expr)
        )
        if mapped_type == "sampler2DArrayShadow" and len(method_args) >= 4:
            sampler = self.generate_expression(method_args[0], is_main)
            coord = self.generate_expression(method_args[1], is_main)
            layer = self.generate_expression(method_args[2], is_main)
            compare = self.generate_expression(method_args[3], is_main)
            return [sampler, f"vec3({coord}, {layer})", compare], 4
        if mapped_type == "samplerCubeArrayShadow" and len(method_args) >= 4:
            sampler = self.generate_expression(method_args[0], is_main)
            coord = self.generate_expression(method_args[1], is_main)
            layer = self.generate_expression(method_args[2], is_main)
            compare = self.generate_expression(method_args[3], is_main)
            return [sampler, f"vec4({coord}, {layer})", compare], 4
        return [self.generate_expression(arg, is_main) for arg in method_args[:3]], 3

    def texture_compare_option_method_call(
        self, obj, obj_expr, method_args, is_main=False
    ):
        compare_args, compare_arg_count = self.texture_compare_method_base_arguments(
            obj_expr, method_args, is_main
        )
        if len(method_args) == compare_arg_count and compare_arg_count == 4:
            return f"textureCompare({obj}, {', '.join(compare_args)})"
        if len(method_args) not in {compare_arg_count + 1, compare_arg_count + 2}:
            return None

        option = method_args[compare_arg_count]
        unsupported_option = self.unsupported_texture_compare_lod_option(obj, option)
        if unsupported_option is not None:
            return unsupported_option

        level_arg = self.unwrap_texture_option_argument(option, "level")
        if level_arg is not option:
            lod = self.generate_expression(level_arg, is_main)
            if len(method_args) == compare_arg_count + 1:
                return f"textureCompareLod({obj}, {', '.join(compare_args + [lod])})"
            offset = self.generate_expression(
                method_args[compare_arg_count + 1], is_main
            )
            return (
                f"textureCompareLodOffset("
                f"{obj}, {', '.join(compare_args + [lod, offset])})"
            )

        gradient_args = self.texture_gradient_option_arguments(option)
        if gradient_args is not None:
            ddx = self.generate_expression(gradient_args[0], is_main)
            ddy = self.generate_expression(gradient_args[1], is_main)
            if len(method_args) == compare_arg_count + 1:
                return (
                    f"textureCompareGrad({obj}, {', '.join(compare_args + [ddx, ddy])})"
                )
            offset = self.generate_expression(
                method_args[compare_arg_count + 1], is_main
            )
            return (
                f"textureCompareGradOffset("
                f"{obj}, {', '.join(compare_args + [ddx, ddy, offset])})"
            )

        if len(method_args) == compare_arg_count + 1 and not isinstance(
            option, FunctionCallNode
        ):
            offset = self.generate_expression(option, is_main)
            return f"textureCompareOffset({obj}, {', '.join(compare_args + [offset])})"

        return None

    def unsupported_texture_compare_lod_option(self, obj, option):
        if not isinstance(option, FunctionCallNode):
            return None
        option_name = self.unscoped_function_name(option.name)
        if option_name not in {"bias", "min_lod_clamp"}:
            return None
        return (
            "0.0 /* unsupported Metal depth compare lod option: "
            f"{option_name} on {obj} */"
        )

    def texture_gather_call(self, obj, texture_expr, method_args, is_main=False):
        args, offset = self.texture_gather_arguments(texture_expr, method_args, is_main)
        function = "textureGatherOffset" if offset else "textureGather"
        return f"{function}({obj}, {', '.join(args)})"

    def texture_gather_arguments(self, texture_expr, method_args, is_main=False):
        if len(method_args) < 2:
            return [
                self.texture_gather_argument(arg, is_main) for arg in method_args
            ], False

        sampler = self.generate_expression(method_args[0], is_main)
        coords = self.generate_expression(method_args[1], is_main)
        tail = list(method_args[2:])
        constructor = self.sampled_array_coordinate_constructor(texture_expr)
        if constructor and tail:
            layer = self.generate_expression(tail.pop(0), is_main)
            coords = f"{constructor}({coords}, {layer})"

        args = [sampler, coords]
        has_offset = False
        if tail:
            if self.texture_gather_component_index(tail[0]) is None:
                args.append(self.generate_expression(tail.pop(0), is_main))
                has_offset = True
            if tail:
                args.append(self.texture_gather_argument(tail.pop(0), is_main))

        args.extend(self.texture_gather_argument(arg, is_main) for arg in tail)
        return args, has_offset

    def texture_gather_compare_call(self, obj, obj_expr, method_args, is_main=False):
        compare_args, consumed = self.texture_gather_compare_base_arguments(
            obj_expr, method_args, is_main
        )
        tail = list(method_args[consumed:])
        if len(tail) > 1:
            return None
        if tail:
            offset = self.generate_expression(tail[0], is_main)
            return (
                "textureGatherCompareOffset("
                f"{obj}, {', '.join(compare_args + [offset])})"
            )
        return f"textureGatherCompare({obj}, {', '.join(compare_args)})"

    def texture_gather_compare_base_arguments(
        self, obj_expr, method_args, is_main=False
    ):
        if len(method_args) < 3:
            return [self.generate_expression(arg, is_main) for arg in method_args], len(
                method_args
            )

        mapped_type = self.resource_classification_type(
            self.expression_mapped_type(obj_expr)
        )
        if (
            mapped_type in {"sampler2DArrayShadow", "samplerCubeArrayShadow"}
            and len(method_args) >= 4
        ):
            constructor = self.sampled_array_coordinate_constructor_for_type(
                mapped_type
            )
            sampler = self.generate_expression(method_args[0], is_main)
            coord = self.generate_expression(method_args[1], is_main)
            layer = self.generate_expression(method_args[2], is_main)
            compare = self.generate_expression(method_args[3], is_main)
            return [sampler, f"{constructor}({coord}, {layer})", compare], 4

        return [self.generate_expression(arg, is_main) for arg in method_args[:3]], 3

    def texture_gather_argument(self, arg, is_main=False):
        component = self.texture_gather_component_index(arg)
        if component is not None:
            return component
        return self.generate_expression(arg, is_main)

    def texture_gather_component_index(self, arg):
        if not isinstance(arg, VariableNode):
            return None
        name = str(getattr(arg, "name", ""))
        parts = name.split("::")
        if len(parts) < 2 or parts[-2] != "component":
            return None
        return self.texture_gather_components.get(parts[-1])

    def is_multisample_resource_type(self, mapped_type):
        return bool(mapped_type and "MS" in str(mapped_type))

    def scalar_size_resource_type(self, mapped_type):
        return mapped_type in {
            "sampler1D",
            "isampler1D",
            "usampler1D",
            "samplerBuffer",
            "isamplerBuffer",
            "usamplerBuffer",
            "image1D",
            "iimage1D",
            "uimage1D",
        }

    def array_size_component(self, mapped_type):
        return "y" if mapped_type and "1DArray" in str(mapped_type) else "z"

    def resource_size_query_info(self, expr, is_main=False):
        if not isinstance(expr, MethodCallNode):
            return None
        method = expr.method
        if method not in self.texture_size_query_methods:
            return None

        obj = self.generate_expression(expr.object, is_main)
        mapped_type = self.expression_mapped_type(expr.object)
        query_function = (
            "imageSize"
            if self.is_storage_image_expression(expr.object)
            else "textureSize"
        )
        multisample = self.is_multisample_resource_type(mapped_type)

        lod = None
        if (
            query_function == "textureSize"
            and not multisample
            and method != "get_array_size"
        ):
            lod = self.generate_expression(expr.args[0], is_main) if expr.args else "0"

        components = {
            "get_width": "" if self.scalar_size_resource_type(mapped_type) else "x",
            "get_height": "y",
            "get_depth": "z",
            "get_array_size": self.array_size_component(mapped_type),
        }
        return {
            "object": obj,
            "function": query_function,
            "component": components[method],
            "lod": lod,
            "multisample": multisample,
        }

    def resource_size_query_call(self, info, lod=None):
        args = [info["object"]]
        if info["function"] == "textureSize" and not info["multisample"]:
            args.append(lod if lod is not None else info["lod"] or "0")
        return f"{info['function']}({', '.join(args)})"

    def resource_size_method_expression(self, expr, is_main=False):
        info = self.resource_size_query_info(expr, is_main)
        if info is None:
            return None
        size_call = self.resource_size_query_call(info)
        component = info["component"]
        return f"{size_call}.{component}" if component else size_call

    def texture_size_constructor_expression(self, expr, is_main=False):
        infos = [
            self.resource_size_query_info(arg, is_main)
            for arg in getattr(expr, "args", [])
        ]
        if not infos or any(info is None for info in infos):
            return None

        first = infos[0]
        if any(
            info["object"] != first["object"]
            or info["function"] != first["function"]
            or info["multisample"] != first["multisample"]
            for info in infos
        ):
            return None

        components = [info["component"] for info in infos]
        if components not in (["x", "y"], ["x", "y", "z"]):
            return None

        lods = {info["lod"] for info in infos if info["lod"] is not None}
        if len(lods) > 1:
            return None
        lod = next(iter(lods), None)
        return self.resource_size_query_call(first, lod)

    def generate(self, ast):
        typedefs = getattr(ast, "typedefs", []) or []
        self.type_aliases = {
            alias.name: alias.alias_type
            for alias in typedefs
            if isinstance(alias, TypeAliasNode)
        }
        self.prepare_texture_usage(ast)
        functions = getattr(ast, "functions", []) or []
        self.user_function_names = {
            function.name
            for function in functions
            if isinstance(function, FunctionNode) and function.name
        }
        code = ""
        includes = getattr(ast, "includes", []) or []
        for inc in includes:
            if isinstance(inc, PreprocessorNode):
                line = f"{inc.directive} {inc.content}".strip()
            else:
                line = str(inc).strip()
            if line:
                code += f"{line}\n"
        if includes:
            code += "\n"
        code += "shader main {\n"
        code += "\n"
        self.constant_struct_name = []

        # Get constants - support both 'constant' and 'constants' attributes
        constants = getattr(ast, "constant", []) or getattr(ast, "constants", []) or []
        for constant in constants:
            if isinstance(constant, ConstantBufferNode):
                self.process_constant_struct(ast)

        # Get structs - support both 'struct' and 'structs' attributes
        structs = getattr(ast, "structs", []) or getattr(ast, "struct", []) or []
        self.struct_name_map = self.build_struct_name_map(structs)
        self.struct_member_types = self.collect_struct_member_types(structs)
        enums = getattr(ast, "enums", []) or []
        emitted_typedefs = [
            declaration
            for alias in typedefs
            if isinstance(alias, TypeAliasNode)
            for declaration in [self.format_type_alias_declaration(alias)]
            if declaration is not None
        ]
        if emitted_typedefs:
            code += "    // Typedefs\n"
            for declaration in emitted_typedefs:
                code += f"    {declaration}\n"
            code += "\n"

        if enums:
            code += "    // Enums\n"
            used_enum_names = {
                enum.name for enum in enums if isinstance(enum, EnumNode) and enum.name
            }
            anonymous_enum_index = 0
            for enum in enums:
                if isinstance(enum, EnumNode):
                    enum_name = enum.name
                    if not enum_name:
                        while True:
                            candidate = f"MetalAnonymousEnum{anonymous_enum_index}"
                            anonymous_enum_index += 1
                            if candidate not in used_enum_names:
                                enum_name = candidate
                                used_enum_names.add(candidate)
                                break
                    code += f"    enum {enum_name} {{\n"
                    for member_name, member_value in enum.members:
                        if member_value is not None:
                            value = self.generate_expression(member_value, False)
                            code += f"        {member_name} = {value},\n"
                        else:
                            code += f"        {member_name},\n"
                    code += "    };\n\n"
        for struct_node in structs:
            if isinstance(struct_node, StructNode):
                if getattr(struct_node, "aggregate_kind", None) == "union":
                    union_name = self.map_struct_name(struct_node.name or "anonymous")
                    code += (
                        f"    // Metal union {union_name} represented as "
                        "struct-like layout; overlapping storage is not modeled\n"
                    )
                if struct_node.name in self.constant_struct_name:
                    code += "    // cbuffers\n"
                    code += f"    cbuffer {self.map_struct_name(struct_node.name)} {{\n"
                else:
                    code += "    // Structs\n"
                    struct_alignas = ""
                    if hasattr(struct_node, "alignas") and struct_node.alignas:
                        parts = []
                        for item in struct_node.alignas:
                            if isinstance(item, tuple) and item[0] == "type":
                                parts.append(f"alignas({self.map_type(item[1])})")
                            else:
                                parts.append(
                                    f"alignas({self.generate_expression(item, False)})"
                                )
                        struct_alignas = " ".join(parts) + " "
                    code += (
                        f"    {struct_alignas}struct "
                        f"{self.map_struct_name(struct_node.name)} {{\n"
                    )
                for member in struct_node.members:
                    if isinstance(member, StaticAssertNode):
                        cond = self.generate_expression(member.condition, False)
                        if member.message is not None:
                            msg = (
                                member.message
                                if isinstance(member.message, str)
                                else self.generate_expression(member.message, False)
                            )
                            code += f"        static_assert({cond}, {msg});\n"
                        else:
                            code += f"        static_assert({cond});\n"
                        continue
                    decl = self.format_decl(member, include_semantic=True)
                    code += f"        {decl};\n"
                code += "    }\n\n"

        globals_list = getattr(ast, "global_variables", []) or getattr(
            ast, "global_vars", []
        )
        if globals_list:
            code += "    // Globals\n"
            for glob in globals_list:
                if isinstance(glob, StaticAssertNode):
                    cond = self.generate_expression(glob.condition, False)
                    if glob.message is not None:
                        msg = (
                            glob.message
                            if isinstance(glob.message, str)
                            else self.generate_expression(glob.message, False)
                        )
                        code += f"    static_assert({cond}, {msg});\n"
                    else:
                        code += f"    static_assert({cond});\n"
                    continue
                if isinstance(glob, AssignmentNode):
                    if isinstance(glob.left, VariableNode):
                        if self.is_sampler_variable(glob.left):
                            self.global_sampler_names.add(glob.left.name)
                            continue
                        self.global_variable_types[glob.left.name] = glob.left.vtype
                    left = (
                        self.format_decl(glob.left, include_semantic=True)
                        if isinstance(glob.left, VariableNode)
                        else self.generate_expression(glob.left, False)
                    )
                    right = self.generate_initializer_value(
                        glob.right,
                        False,
                        (
                            getattr(glob.left, "vtype", None)
                            if isinstance(glob.left, VariableNode)
                            else None
                        ),
                        (
                            self.variable_has_array_initializer_shape(glob.left)
                            if isinstance(glob.left, VariableNode)
                            else False
                        ),
                    )
                    code += f"    {left} {glob.operator} {right};\n"
                elif isinstance(glob, VariableNode):
                    if self.is_sampler_variable(glob):
                        self.global_sampler_names.add(glob.name)
                        continue
                    self.global_variable_types[glob.name] = glob.vtype
                    if id(glob) in self.storage_texture_declaration_ids:
                        self.global_storage_texture_names.add(glob.name)
                    if self.structured_buffer_pointer_type(glob):
                        self.global_structured_buffer_names.add(glob.name)
                    decl = self.format_global_decl(glob, include_semantic=True)
                    code += f"    {decl};\n"
            code += "\n"

        for f in functions:
            qualifier = getattr(f, "qualifier", None)
            if qualifier == "vertex":
                code += "    // Vertex Shader\n"
                code += "    vertex {\n"
                code += self.generate_function(f, stage_entry=f.name != "main")
                code += "    }\n\n"
            elif qualifier == "fragment":
                code += "    // Fragment Shader\n"
                code += "    fragment {\n"
                code += self.generate_fragment_execution_layouts(f)
                code += self.generate_function(f, stage_entry=f.name != "main")
                code += "    }\n\n"
            elif qualifier == "kernel":
                code += "    // Compute Shader\n"
                code += "    compute {\n"
                code += self.generate_function(
                    f, stage_entry=self.should_emit_kernel_stage_entry(f)
                )
                code += "    }\n\n"
            elif qualifier in self.rt_qualifiers:
                code += f"    // {qualifier} function\n"
                code += self.generate_function(f)
            else:
                code += self.generate_function(f)

        code += "}\n"
        return code

    def process_constant_struct(self, node):
        constants = (
            getattr(node, "constant", []) or getattr(node, "constants", []) or []
        )
        structs = getattr(node, "structs", []) or getattr(node, "struct", []) or []
        for constant in constants:
            if isinstance(constant, ConstantBufferNode):
                self.constant_struct_name.extend(
                    struct.name for struct in structs if struct.name == constant.name
                )

    def collect_struct_member_types(self, structs):
        member_types = {}
        for struct_node in structs or []:
            struct_name = getattr(struct_node, "name", None)
            if not struct_name:
                continue
            members = {}
            for member in getattr(struct_node, "members", []) or []:
                member_name = getattr(member, "name", None)
                member_type = getattr(member, "vtype", None)
                if member_name and member_type:
                    members[member_name] = member_type
            member_types[struct_name] = members
        return member_types

    def build_struct_name_map(self, structs):
        mapped_names = {}
        used_names = set()
        for struct_node in structs or []:
            if not isinstance(struct_node, StructNode):
                continue
            raw_name = getattr(struct_node, "name", None)
            if not raw_name:
                continue
            base_name = self.sanitize_identifier(raw_name)
            candidate = base_name
            suffix = 2
            while candidate in used_names:
                candidate = f"{base_name}_{suffix}"
                suffix += 1
            used_names.add(candidate)
            mapped_names[raw_name] = candidate
        return mapped_names

    def map_struct_name(self, name):
        if not name:
            return name
        return self.struct_name_map.get(name, self.sanitize_identifier(name))

    def iter_ast_children(self, node):
        if node is None or isinstance(node, (str, int, float, bool)):
            return
        if isinstance(node, dict):
            yield from node.values()
            return
        if isinstance(node, (list, tuple, set)):
            yield from node
            return
        yield from getattr(node, "__dict__", {}).values()

    def collect_storage_texture_declaration_ids(self, root):
        storage_ids = set()

        def visit(node):
            if node is None or isinstance(node, (str, int, float, bool)):
                return
            if isinstance(
                node, VariableNode
            ) and self.is_access_qualified_storage_texture_type(node.vtype):
                storage_ids.add(id(node))
            for child in self.iter_ast_children(node):
                visit(child)

        visit(root)
        return storage_ids

    def prepare_texture_usage(self, ast):
        self.global_variable_types = {}
        self.current_variable_types = {}
        self.global_storage_texture_names = set()
        self.current_storage_texture_names = set()
        self.global_structured_buffer_names = set()
        self.current_structured_buffer_names = set()
        self.global_sampler_names = set()
        self.user_function_names = set()
        self.identifier_maps = [{}]
        self.used_identifier_names = [set()]
        self.storage_texture_declaration_ids = (
            self.collect_storage_texture_declaration_ids(ast)
        )
        self.struct_member_types = {}
        self.struct_name_map = {}

    def format_array_suffix(self, var, include_declarator_arrays=True):
        array_type = self.metal_array_type_parts(getattr(var, "vtype", None))
        suffix = f"[{array_type[1]}]" if array_type else ""
        if not include_declarator_arrays:
            return suffix
        return suffix + self.format_declarator_array_suffix(var)

    def format_declarator_array_suffix(self, var):
        if not hasattr(var, "array_sizes") or not var.array_sizes:
            return ""
        suffix = ""
        for size in var.array_sizes:
            if size is None:
                suffix += "[]"
            else:
                suffix += f"[{self.generate_expression(size, False)}]"
        return suffix

    def use_name_array_suffix(self, mapped_type, var):
        if not getattr(var, "array_sizes", None):
            return False
        return str(mapped_type).rstrip().endswith(("*", "&"))

    def variable_has_array_initializer_shape(self, var):
        return bool(getattr(var, "array_sizes", None)) or bool(
            self.metal_array_type_parts(getattr(var, "vtype", None))
        )

    def map_variable_type(self, var):
        raw_type = getattr(var, "vtype", None)
        structured_buffer_type = self.structured_buffer_pointer_type(var)
        if structured_buffer_type:
            return structured_buffer_type
        array_type = self.metal_array_type_parts(raw_type)
        type_to_map = array_type[0] if array_type else raw_type
        if (
            id(var) in self.storage_texture_declaration_ids
            or getattr(var, "name", None) in self.current_storage_texture_names
            or getattr(var, "name", None) in self.global_storage_texture_names
        ):
            storage_type = self.map_storage_texture_type(type_to_map)
            if storage_type:
                return storage_type
        resolved_type = self.resolve_type_alias(type_to_map)
        if resolved_type != type_to_map and self.is_metal_resource_type(resolved_type):
            return self.map_type(resolved_type)
        return self.map_type(type_to_map)

    def map_declared_variable_type(self, var):
        mapped_type = self.map_variable_type(var)
        semantics = {
            str(getattr(attr, "name", "")).lower()
            for attr in getattr(var, "attributes", []) or []
        }
        if semantics.intersection({"vertex_id", "instance_id", "primitive_id"}):
            return "int"
        return mapped_type

    def address_space_qualifier_prefix(self, var):
        if self.structured_buffer_pointer_type(var):
            return ""

        qualifiers = [
            str(qualifier).lower() for qualifier in getattr(var, "qualifiers", []) or []
        ]
        address_spaces = []
        for qualifier in (
            "threadgroup_imageblock",
            "threadgroup",
            "thread",
            "device",
            "constant",
        ):
            if qualifier in qualifiers and qualifier not in address_spaces:
                address_spaces.append(qualifier)
        return f"{' '.join(address_spaces)} " if address_spaces else ""

    def address_space_qualifier_annotations(self, var):
        qualifiers = {
            str(qualifier).lower() for qualifier in getattr(var, "qualifiers", []) or []
        }
        attributes = {
            str(getattr(attr, "name", "")).lower()
            for attr in getattr(var, "attributes", []) or []
        }
        annotations = []
        for qualifier in ("ray_data", "object_data"):
            compact = qualifier.replace("_", "")
            if (
                qualifier in qualifiers
                and qualifier not in attributes
                and compact not in attributes
            ):
                annotations.append(f"@{qualifier}")
        return " ".join(annotations)

    def is_sampler_variable(self, var):
        return self.is_sampler_type(getattr(var, "vtype", None))

    def is_sampler_type(self, metal_type):
        normalized = self.normalized_metal_type(metal_type)
        if normalized == "sampler":
            return True
        array_type = self.metal_array_type_parts(normalized)
        return bool(
            array_type and self.normalized_metal_type(array_type[0]) == "sampler"
        )

    def format_decl(
        self, var, include_semantic=False, declare_name=True, semantic_context=None
    ):
        alignas_prefix = ""
        if hasattr(var, "alignas") and var.alignas:
            parts = []
            for item in var.alignas:
                if isinstance(item, tuple) and item[0] == "type":
                    parts.append(f"alignas({self.map_type(item[1])})")
                else:
                    parts.append(f"alignas({self.generate_expression(item, False)})")
            alignas_prefix = " ".join(parts) + " "
        mapped_type = self.map_declared_variable_type(var)
        name_array_suffix = ""
        include_declarator_arrays = True
        grouped_type_suffix = (
            getattr(var, "declarator_type_suffix", "")
            if getattr(var, "declarator_type_suffix_grouped", False)
            else ""
        )
        if grouped_type_suffix and getattr(var, "array_sizes", None):
            include_declarator_arrays = False
            base_type = mapped_type
            if base_type.endswith(grouped_type_suffix):
                base_type = base_type[: -len(grouped_type_suffix)].rstrip()
            type_array_suffix = self.format_declarator_array_suffix(var)
            type_str = f"{base_type}{type_array_suffix}{grouped_type_suffix}"
        else:
            if self.use_name_array_suffix(mapped_type, var):
                include_declarator_arrays = False
                name_array_suffix = self.format_declarator_array_suffix(var)
            type_array_suffix = self.format_array_suffix(var, include_declarator_arrays)
            type_str = f"{mapped_type}{type_array_suffix}"
        address_space = self.address_space_qualifier_prefix(var)
        const_str = (
            "const "
            if hasattr(var, "is_const")
            and var.is_const
            and address_space.strip() != "constant"
            else ""
        )
        semantic = (
            self.map_semantic(
                getattr(var, "attributes", None), context=semantic_context
            )
            if include_semantic
            else ""
        )
        address_space_annotations = self.address_space_qualifier_annotations(var)
        semantic = " ".join(
            part for part in [address_space_annotations, semantic] if part
        )
        access = self.storage_texture_access_attribute(var)
        storage_format = self.storage_texture_format_attributes(var)
        name = (
            self.declare_identifier(var.name)
            if declare_name
            else self.sanitize_identifier(var.name)
        )
        if name_array_suffix:
            name = f"{name}{name_array_suffix}"
        parts = [alignas_prefix + const_str + address_space + type_str, name]
        if semantic:
            parts.append(semantic)
        if access:
            parts.append(access)
        if storage_format:
            parts.append(storage_format)
        return " ".join(part for part in parts if part)

    def format_parameter_decl(self, var, index, semantic_context=None):
        if getattr(var, "name", None):
            declaration = self.format_decl(
                var,
                include_semantic=True,
                semantic_context=semantic_context,
            )
            return self.with_parameter_direction_qualifier(var, declaration)

        generated_name = self.reserve_generated_identifier(f"_unnamed_param_{index}")
        original_name = var.name
        var.name = generated_name
        try:
            declaration = self.format_decl(
                var,
                include_semantic=True,
                declare_name=False,
                semantic_context=semantic_context,
            )
            return self.with_parameter_direction_qualifier(var, declaration)
        finally:
            var.name = original_name

    def with_parameter_direction_qualifier(self, var, declaration):
        qualifiers = [
            str(qualifier).lower() for qualifier in getattr(var, "qualifiers", []) or []
        ]
        for qualifier in self.parameter_direction_qualifiers:
            if qualifier in qualifiers:
                return f"{qualifier} {declaration}"
        return declaration

    def format_global_decl(self, var, include_semantic=False):
        declaration = self.format_decl(var, include_semantic=include_semantic)
        attributes = getattr(var, "attributes", []) or []
        if any(
            isinstance(attr, AttributeNode) and attr.name == "argument_buffer"
            for attr in attributes
        ):
            declaration = re.sub(r"(?<=\S)&\s+(\w+)", r" \1", declaration, count=1)
        return declaration

    def generate_function(self, func, indent=2, stage_entry=False):
        """Render one Metal function node as a CrossGL function block."""
        code = ""
        if stage_entry:
            code += "    " * indent
            code += "@ stage_entry\n"
        code += "    " * indent
        previous_variable_types = self.current_variable_types
        self.current_variable_types = dict(self.global_variable_types)
        previous_storage_texture_names = self.current_storage_texture_names
        self.current_storage_texture_names = set(self.global_storage_texture_names)
        previous_structured_buffer_names = self.current_structured_buffer_names
        self.current_structured_buffer_names = set(self.global_structured_buffer_names)
        self.push_identifier_scope()
        try:
            for param in func.params:
                self.current_variable_types[param.name] = param.vtype
                if id(param) in self.storage_texture_declaration_ids:
                    self.current_storage_texture_names.add(param.name)
                if self.structured_buffer_pointer_type(param):
                    self.current_structured_buffer_names.add(param.name)
            semantic_context = {
                "kind": "parameter",
                "function_qualifier": getattr(func, "qualifier", None),
            }
            params = ", ".join(
                self.format_parameter_decl(p, index, semantic_context=semantic_context)
                for index, p in enumerate(func.params)
            )
            fn_semantic = self.map_semantic(self.function_semantic_attributes(func))
            suffix = f" {fn_semantic}" if fn_semantic else ""
            function_name = self.sanitize_identifier(self.function_output_name(func))
            return_type = self.map_function_return_type(func.return_type)
            code += f"{return_type} {function_name}({params}){suffix} {{\n"
            code += self.generate_function_body(func.body, indent=indent + 1)
            code += "    }\n\n"
        finally:
            self.pop_identifier_scope()
            self.current_variable_types = previous_variable_types
            self.current_storage_texture_names = previous_storage_texture_names
            self.current_structured_buffer_names = previous_structured_buffer_names
        return code

    def map_function_return_type(self, return_type):
        mapped_type = self.map_type(return_type)
        if str(mapped_type).rstrip().endswith("&"):
            return str(mapped_type).rstrip()[:-1].rstrip()
        return mapped_type

    def function_output_name(self, func):
        host_name = self.function_host_name(func)
        if host_name and getattr(func, "qualifier", None) in {
            "vertex",
            "fragment",
            "kernel",
        }:
            return host_name
        return func.name

    def should_emit_kernel_stage_entry(self, func):
        if getattr(func, "name", None) == "main":
            return False
        return self.kernel_stage_entry_builtin_types_supported(func)

    def kernel_stage_entry_builtin_types_supported(self, func):
        # Keep imported kernels on the explicit stage-entry path only when the
        # current Metal generator can validate their builtin parameter shapes.
        expected_types = {
            "thread_position_in_grid": {"uint", "uint2", "uint3"},
            "thread_position_in_threadgroup": {"uint3"},
            "threadgroup_position_in_grid": {"uint3"},
            "thread_index_in_threadgroup": {"uint"},
            "threads_per_threadgroup": {"uint3"},
            "threadgroups_per_grid": {"uint3"},
            "thread_index_in_simdgroup": {"uint"},
            "threads_per_simdgroup": {"uint"},
        }
        for param in getattr(func, "params", []) or []:
            semantic = self.compute_builtin_attribute_name(param)
            if semantic is None:
                continue
            allowed_types = expected_types.get(semantic)
            if allowed_types is None:
                continue
            param_type = self.normalized_metal_type(getattr(param, "vtype", None))
            if param_type not in allowed_types:
                return False
        return True

    def compute_builtin_attribute_name(self, var):
        compute_builtins = {
            "thread_position_in_grid",
            "thread_position_in_threadgroup",
            "threadgroup_position_in_grid",
            "thread_index_in_threadgroup",
            "threads_per_threadgroup",
            "threadgroups_per_grid",
            "thread_index_in_simdgroup",
            "threads_per_simdgroup",
        }
        for attr in getattr(var, "attributes", []) or []:
            name = str(getattr(attr, "name", "")).lower()
            if name in compute_builtins:
                return name
        return None

    def function_host_name(self, func):
        for attr in getattr(func, "attributes", []) or []:
            if getattr(attr, "name", None) != "host_name":
                continue
            args = getattr(attr, "args", []) or []
            if not args:
                return None
            raw_name = str(args[0]).strip()
            if (
                len(raw_name) >= 2
                and raw_name[0] in {'"', "'"}
                and raw_name[-1] == raw_name[0]
            ):
                return raw_name[1:-1]
            return raw_name
        return None

    def function_semantic_attributes(self, func):
        return [
            attr
            for attr in getattr(func, "attributes", []) or []
            if getattr(attr, "name", None)
            not in (
                self.fragment_execution_attribute_names
                | self.function_metadata_attribute_names
            )
            and "::" not in str(getattr(attr, "name", ""))
        ]

    def generate_fragment_execution_layouts(self, func):
        layouts = []
        for attr in getattr(func, "attributes", []) or []:
            name = getattr(attr, "name", None)
            if name not in self.fragment_execution_attribute_names:
                continue
            if getattr(attr, "args", None):
                continue
            if name not in layouts:
                layouts.append(name)
        if not layouts:
            return ""
        return f"        layout({', '.join(layouts)}) in;\n"

    def generate_function_body(self, body, indent=0, is_main=False):
        code = ""
        for stmt in body:
            code += "    " * indent
            if isinstance(stmt, VariableNode):
                self.current_variable_types[stmt.name] = stmt.vtype
                if id(stmt) in self.storage_texture_declaration_ids:
                    self.current_storage_texture_names.add(stmt.name)
                if self.structured_buffer_pointer_type(stmt):
                    self.current_structured_buffer_names.add(stmt.name)
                decl = self.format_decl(stmt, include_semantic=False)
                code += f"{decl};\n"
            elif isinstance(stmt, AssignmentNode):
                code += self.generate_assignment(stmt, is_main) + ";\n"
            elif isinstance(stmt, ReturnNode):
                if not is_main:
                    if stmt.value is None:
                        code += "return;\n"
                    else:
                        code += (
                            f"return {self.generate_expression(stmt.value, is_main)};\n"
                        )
            elif isinstance(stmt, BinaryOpNode):
                code += f"{self.generate_expression(stmt.left, is_main)} {stmt.op} {self.generate_expression(stmt.right, is_main)};\n"
            elif isinstance(stmt, BlockNode):
                code += "{\n"
                code += self.generate_function_body(
                    stmt.statements, indent + 1, is_main
                )
                code += "    " * indent + "}\n"
            elif isinstance(stmt, RangeForNode):
                code += self.generate_range_for_loop(stmt, indent, is_main)
            elif isinstance(stmt, ForNode):
                code += self.generate_for_loop(stmt, indent, is_main)
            elif isinstance(stmt, WhileNode):
                code += self.generate_while_loop(stmt, indent, is_main)
            elif isinstance(stmt, DoWhileNode):
                code += self.generate_do_while_loop(stmt, indent, is_main)
            elif isinstance(stmt, IfNode):
                code += self.generate_if_statement(stmt, indent, is_main)
            elif isinstance(stmt, SwitchNode):
                code += self.generate_switch_statement(stmt, indent, is_main)
            elif (
                isinstance(stmt, FunctionCallNode)
                or isinstance(stmt, MethodCallNode)
                or isinstance(stmt, CallNode)
            ):
                code += f"{self.generate_expression(stmt, is_main)};\n"
            elif isinstance(stmt, PostfixOpNode):
                code += f"{self.generate_expression(stmt, is_main)};\n"
            elif isinstance(stmt, ContinueNode):
                code += "continue;\n"
            elif isinstance(stmt, BreakNode):
                code += "break;\n"
            elif isinstance(stmt, DiscardNode):
                code += "discard;\n"
            elif isinstance(stmt, StaticAssertNode):
                cond = self.generate_expression(stmt.condition, is_main)
                if stmt.message is not None:
                    msg = (
                        stmt.message
                        if isinstance(stmt.message, str)
                        else self.generate_expression(stmt.message, is_main)
                    )
                    code += f"static_assert({cond}, {msg});\n"
                else:
                    code += f"static_assert({cond});\n"
            elif isinstance(stmt, EnumNode):
                code += self.generate_local_enum(stmt, indent, is_main)
            elif isinstance(stmt, str):
                code += f"{stmt};\n"
            else:
                expr = self.generate_expression(stmt, is_main)
                if expr:
                    code += f"{expr};\n"
                else:
                    code += f"// Unhandled statement type: {type(stmt).__name__}\n"
        return code

    def generate_local_enum(self, enum, indent, is_main):
        enum_name = enum.name or "MetalAnonymousEnum"
        code = f"enum {enum_name} {{\n"
        for member_name, member_value in enum.members:
            code += "    " * (indent + 1) + member_name
            if member_value is not None:
                code += f" = {self.generate_expression(member_value, is_main)}"
            code += ",\n"
        code += "    " * indent + "};\n"
        return code

    def generate_for_loop(self, node, indent, is_main):
        init = self.generate_for_clause(node.init, is_main)
        condition = self.generate_for_clause(node.condition, is_main)
        update = self.generate_for_clause(node.update, is_main)

        code = f"for ({init}; {condition}; {update}) {{\n"
        code += self.generate_function_body(node.body, indent + 1, is_main)
        code += "    " * indent + "}\n"
        return code

    def generate_for_clause(self, expr, is_main):
        if isinstance(expr, list):
            return ", ".join(self.generate_for_clause(item, is_main) for item in expr)
        return self.generate_expression(expr, is_main)

    def generate_range_for_loop(self, node, indent, is_main):
        iterable = self.generate_expression(node.iterable, is_main)

        code = f"for {node.name} in {iterable} {{\n"
        code += self.generate_function_body(node.body, indent + 1, is_main)
        code += "    " * indent + "}\n"
        return code

    def generate_while_loop(self, node, indent, is_main):
        condition = self.generate_expression(node.condition, is_main)
        code = f"while ({condition}) {{\n"
        code += self.generate_function_body(node.body, indent + 1, is_main)
        code += "    " * indent + "}\n"
        return code

    def generate_do_while_loop(self, node, indent, is_main):
        condition = self.generate_expression(node.condition, is_main)
        code = "do {\n"
        code += self.generate_function_body(node.body, indent + 1, is_main)
        code += "    " * indent + f"}} while ({condition});\n"
        return code

    def generate_if_statement(self, node, indent, is_main):
        code = ""
        if node.if_chain:
            for condition, body in node.if_chain:
                code += f"if ({self.generate_expression(condition, is_main)}) {{\n"
                code += self.generate_function_body(body, indent + 1, is_main)
                code += "    " * indent + "}"
        if node.else_if_chain:
            for condition, body in node.else_if_chain:
                code += (
                    f" else if ({self.generate_expression(condition, is_main)}) {{\n"
                )
                code += self.generate_function_body(body, indent + 1, is_main)
                code += "    " * indent + "}"

        if node.else_body:
            code += " else {\n"
            code += self.generate_function_body(node.else_body, indent + 1, is_main)
            code += "    " * indent + "}"

        code += "\n"
        return code

    def generate_assignment(self, node, is_main):
        if self.is_structured_buffer_element_access(node.left):
            structured_store = self.generate_structured_buffer_store(
                node.left, node.right, node.operator, is_main
            )
            if structured_store is not None:
                return structured_store
        lhs = self.generate_expression(node.left, is_main)
        rhs = self.generate_initializer_value(
            node.right,
            is_main,
            (
                getattr(node.left, "vtype", None)
                if isinstance(node.left, VariableNode)
                else None
            ),
            (
                self.variable_has_array_initializer_shape(node.left)
                if isinstance(node.left, VariableNode)
                else False
            ),
        )
        op = node.operator
        return f"{lhs} {op} {rhs}"

    def generate_initializer_value(
        self, expr, is_main=False, expected_type=None, expected_array=False
    ):
        if isinstance(expr, InitializerListNode):
            return self.generate_initializer_list(
                expr, is_main, expected_type, expected_array
            )
        return self.generate_expression(expr, is_main)

    def generate_initializer_list(
        self, node, is_main=False, expected_type=None, expected_array=False
    ):
        mapped_type = self.map_type(expected_type) if expected_type else None
        member_types = {}
        if expected_type:
            member_types = self.struct_member_types.get(
                self.normalized_metal_type(expected_type), {}
            )

        named_elements = []
        positional_elements = []
        for element in node.elements:
            if isinstance(element, DesignatedInitializerNode):
                named_elements.append(
                    self.generate_designated_initializer(element, is_main, member_types)
                )
            else:
                positional_elements.append(
                    self.generate_initializer_value(element, is_main)
                )

        elements = named_elements + positional_elements
        if expected_array:
            return "{" + ", ".join(elements) + "}"
        if mapped_type and mapped_type.startswith(("vec", "ivec", "uvec", "bvec")):
            return f"{mapped_type}({', '.join(elements)})"
        if mapped_type and named_elements:
            return f"{mapped_type}{{{', '.join(elements)}}}"
        if mapped_type and positional_elements and not named_elements:
            return f"{mapped_type}({', '.join(elements)})"
        return "{" + ", ".join(elements) + "}"

    def generate_designated_initializer(self, node, is_main=False, member_types=None):
        member_types = member_types or {}
        if len(node.designators) == 1 and node.designators[0][0] == "field":
            field_name = node.designators[0][1]
            value = self.generate_initializer_value(
                node.value, is_main, member_types.get(field_name)
            )
            return f"{field_name}: {value}"

        designators = []
        for designator in node.designators:
            kind = designator[0]
            if kind == "range":
                start = self.generate_expression(designator[1], is_main)
                end = self.generate_expression(designator[2], is_main)
                designators.append(f"[{start} ... {end}]")
                continue
            target = designator[1]
            if kind == "index":
                designators.append(f"[{self.generate_expression(target, is_main)}]")
            else:
                designators.append(f".{target}")
        value = self.generate_initializer_value(node.value, is_main)
        return f"{''.join(designators)} = {value}"

    def normalize_literal_string(self, value):
        if re.fullmatch(r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[hH]", value):
            return value[:-1]
        for pattern in (
            self.hex_integer_literal_pattern,
            self.binary_integer_literal_pattern,
            self.decimal_integer_literal_pattern,
        ):
            match = pattern.fullmatch(value)
            if match:
                suffix = match.group("suffix")
                if "u" in suffix.lower():
                    return f"{match.group('body')}u"
                return match.group("body")
        return value

    def generate_expression(self, expr, is_main=False):
        """Render a Metal backend expression node as CrossGL syntax."""
        if expr is None:
            return ""
        elif isinstance(expr, str):
            return self.normalize_literal_string(expr)
        elif isinstance(expr, VariableNode):
            if expr.vtype:
                return self.format_decl(expr, include_semantic=False)
            else:
                return self.render_identifier(expr.name)
        elif isinstance(expr, AssignmentNode):
            return self.generate_assignment(expr, is_main)
        elif isinstance(expr, BinaryOpNode):
            left = self.generate_binary_operand(expr.left, expr.op, False, is_main)
            right = self.generate_binary_operand(expr.right, expr.op, True, is_main)
            return f"{left} {expr.op} {right}"
        elif isinstance(expr, FunctionCallNode):
            if getattr(expr, "is_braced_constructor", False) and expr.args:
                initializer = expr.args[0]
                if isinstance(initializer, InitializerListNode):
                    return self.generate_initializer_list(
                        initializer, is_main, expr.name
                    )
            sync_call = self.metal_synchronization_function_call(expr.name, expr.args)
            if sync_call is not None:
                return sync_call
            function_name = self.map_function_call_name(expr.name)
            if function_name == "sampler":
                args = ", ".join(
                    self.generate_sampler_constructor_arg(arg, is_main)
                    for arg in expr.args
                )
            else:
                args = ", ".join(
                    self.generate_expression(arg, is_main) for arg in expr.args
                )
            return f"{function_name}({args})"
        elif isinstance(expr, LambdaNode):
            return self.generate_lambda_expression(expr, is_main)
        elif isinstance(expr, CallNode):
            callee = self.generate_expression(expr.callee, is_main)
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            return f"{callee}({args})"
        elif isinstance(expr, MethodCallNode):
            obj = self.generate_expression(expr.object, is_main)
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            method = expr.method
            resource_size_expr = self.resource_size_method_expression(expr, is_main)
            if resource_size_expr is not None:
                return resource_size_expr
            if method == "get_num_mip_levels":
                return f"textureQueryLevels({obj})"
            if method == "get_num_samples":
                samples_function = (
                    "imageSamples"
                    if self.is_storage_image_expression(expr.object)
                    else "textureSamples"
                )
                return f"{samples_function}({obj})"
            descriptor = self.resource_method_descriptor(method)
            if descriptor and descriptor["sampled_texture"]:
                diagnostic = self.unsupported_storage_texture_sampled_method(
                    expr.object, method, is_main
                )
                if diagnostic is not None:
                    return diagnostic
            if method == "sample_compare":
                option_call = self.texture_compare_option_method_call(
                    obj,
                    expr.object,
                    expr.args,
                    is_main,
                )
                if option_call is not None:
                    return option_call
            if descriptor and descriptor["storage_operation"] == "read":
                if self.is_storage_image_expression(expr.object):
                    coord = self.storage_image_coordinate_expression(
                        expr.object, expr.args
                    )
                    return f"imageLoad({obj}, {coord})" if coord else f"{obj}.read()"
                sampled_read = self.sampled_texture_read_expression(
                    expr.object, expr.args, is_main
                )
                if sampled_read is not None:
                    return sampled_read
                return (
                    f"{descriptor['function']}({obj}, {args})"
                    if args
                    else f"{obj}.read()"
                )
            if descriptor and descriptor["storage_operation"] == "write":
                if self.is_storage_image_expression(expr.object):
                    coord = self.storage_image_coordinate_expression(
                        expr.object, expr.args[1:]
                    )
                    value = (
                        self.generate_expression(expr.args[0], is_main)
                        if expr.args
                        else ""
                    )
                    if coord and value:
                        return f"imageStore({obj}, {coord}, {value})"
                    return f"{obj}.write({args})"
                return self.unsupported_sampled_texture_write(
                    expr.object, method, is_main
                )
            if descriptor and method == "gather":
                return self.texture_gather_call(obj, expr.object, expr.args, is_main)
            if descriptor and method == "gather_compare":
                gather_compare_call = self.texture_gather_compare_call(
                    obj, expr.object, expr.args, is_main
                )
                if gather_compare_call is not None:
                    return gather_compare_call
            if descriptor:
                return f"{descriptor['function']}({obj}, {args})"
            return f"{obj}.{method}({args})"
        elif isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object, is_main)
            return f"{obj}.{expr.member}"
        elif isinstance(expr, ArrayAccessNode):
            if (
                not self.suppress_structured_buffer_index_lowering
                and self.is_structured_buffer_element_access(expr)
            ):
                return self.generate_structured_buffer_load(expr, is_main)
            array = self.generate_expression(expr.array, is_main)
            index = self.generate_expression(expr.index, is_main)
            return f"{array}[{index}]"
        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_expression(expr.operand, is_main)
            if expr.op == "post...":
                return operand
            if isinstance(expr.operand, (AssignmentNode, BinaryOpNode, TernaryOpNode)):
                operand = f"({operand})"
            return f"({expr.op}{operand})"
        elif isinstance(expr, PostfixOpNode):
            operand = self.generate_expression(expr.operand, is_main)
            return f"{operand}{expr.op}"
        elif isinstance(expr, TernaryOpNode):
            return f"{self.generate_expression(expr.condition, is_main)} ? {self.generate_expression(expr.true_expr, is_main)} : {self.generate_expression(expr.false_expr, is_main)}"
        elif isinstance(expr, CastNode):
            mapped_type = self.map_type(expr.target_type)
            value = self.generate_expression(expr.expression, is_main)
            if mapped_type == expr.target_type and "::" not in str(mapped_type):
                return f"{self.sanitize_identifier(mapped_type)}({value})"
            return (
                f"({mapped_type}){self.generate_cast_operand(expr.expression, value)}"
            )
        elif isinstance(expr, VectorConstructorNode):
            size_query = self.texture_size_constructor_expression(expr, is_main)
            if size_query is not None:
                return size_query
            mapped_type = self.map_type(expr.type_name)
            if mapped_type == "sampler":
                args = ", ".join(
                    self.generate_sampler_constructor_arg(arg, is_main)
                    for arg in expr.args
                )
            else:
                args = ", ".join(
                    self.generate_expression(arg, is_main) for arg in expr.args
                )
            return f"{mapped_type}({args})"
        elif isinstance(expr, InitializerListNode):
            return self.generate_initializer_list(expr, is_main)
        elif isinstance(expr, DesignatedInitializerNode):
            return self.generate_designated_initializer(expr, is_main)
        elif isinstance(expr, TextureSampleNode):
            diagnostic = self.unsupported_storage_texture_sampled_method(
                expr.texture, "sample", is_main
            )
            if diagnostic is not None:
                return diagnostic
            texture = self.generate_expression(expr.texture, is_main)
            sampler_expr = expr.sampler
            coords_expr = expr.coordinates
            if coords_expr is None and self.is_samplerless_sample_expression(
                expr.texture
            ):
                coords_expr = sampler_expr
                sampler_expr = None
            sampler_name = self.expression_base_name(sampler_expr)
            sampler = (
                ""
                if sampler_name in self.global_sampler_names
                else self.generate_expression(sampler_expr, is_main)
            )
            sample_args = [texture]
            if sampler:
                sample_args.append(sampler)
            options = getattr(expr, "options", None)
            if options is None:
                options = [expr.lod] if getattr(expr, "lod", None) is not None else []
            coords, options = self.texture_sample_coordinate_and_options(
                expr.texture, coords_expr, options, is_main
            )
            sample_args.append(coords)

            if options:
                option_call = self.texture_sample_options_call(
                    options, sample_args, is_main
                )
                if option_call is not None:
                    return option_call

                lod_expr = self.unwrap_texture_option_argument(expr.lod, "level")
                lod = self.generate_expression(lod_expr, is_main)
                return f"textureLod({', '.join(sample_args + [lod])})"

            return f"texture({', '.join(sample_args)})"
        elif isinstance(expr, float) or isinstance(expr, int) or isinstance(expr, bool):
            return str(expr)
        else:
            return f"/* Unhandled expression: {type(expr).__name__} */"

    def generate_lambda_expression(self, expr, is_main=False):
        previous_variable_types = self.current_variable_types
        self.current_variable_types = dict(previous_variable_types)
        self.push_identifier_scope()
        try:
            for param in expr.params:
                self.current_variable_types[param.name] = param.vtype
            params = ", ".join(
                self.format_decl(param, include_semantic=False) for param in expr.params
            )
            specifiers = getattr(expr, "specifiers", []) or []
            specifier_text = f" {' '.join(specifiers)}" if specifiers else ""
            return_type = getattr(expr, "return_type", None)
            return_text = f" -> {self.map_type(return_type)}" if return_type else ""
            body = self.generate_function_body(expr.body, indent=1, is_main=is_main)
            if body:
                return (
                    f"[{expr.capture}]({params}){specifier_text}{return_text} {{\n"
                    f"{body}"
                    "}"
                )
            return f"[{expr.capture}]({params}){specifier_text}{return_text} {{}}"
        finally:
            self.pop_identifier_scope()
            self.current_variable_types = previous_variable_types

    def metal_synchronization_function_call(self, name, args):
        unscoped_name = str(name).split("::")[-1]

        if unscoped_name in {"threadgroup_barrier", "simdgroup_barrier"}:
            flags = self.metal_mem_flag_names(args)
            if flags == {"mem_threadgroup"}:
                return "workgroupBarrier()"
            if flags == {"mem_device"}:
                return "memoryBarrierBuffer()"
            if flags == {"mem_texture"}:
                return "memoryBarrierImage()"
            if flags == {"mem_device", "mem_threadgroup", "mem_texture"}:
                return "allMemoryBarrier()"
            return None

        if unscoped_name == "atomic_thread_fence":
            flags = self.metal_mem_flag_names(args[:1])
            if flags == {"mem_device"}:
                return "memoryBarrier()"
            if flags == {"mem_threadgroup"}:
                return "memoryBarrierShared()"
            if flags == {"mem_texture"}:
                return "memoryBarrierImage()"
            return None

        return None

    def metal_mem_flag_names(self, args):
        if len(args) != 1:
            return None
        flags = self.collect_metal_mem_flags(args[0])
        return flags or None

    def collect_metal_mem_flags(self, expr):
        if isinstance(expr, BinaryOpNode) and expr.op == "|":
            left = self.collect_metal_mem_flags(expr.left)
            right = self.collect_metal_mem_flags(expr.right)
            if left is None or right is None:
                return None
            return left | right
        if isinstance(expr, VariableNode):
            name = str(expr.name).split("::")[-1]
            if name.startswith("mem_"):
                return {name}
        return None

    def map_function_call_name(self, name):
        match = re.fullmatch(r"(?:metal::)?as_type<(.+)>", name)
        if not match:
            metal_type_constructor = self.map_metal_type_constructor_name(name)
            if metal_type_constructor is not None:
                return metal_type_constructor
            metal_bit_name = self.map_metal_bit_function_name(name)
            if metal_bit_name is not None:
                return metal_bit_name
            metal_math_name = self.map_metal_math_function_name(name)
            if metal_math_name is not None:
                return metal_math_name
            return self.sanitize_identifier(name)

        target_type = self.normalized_metal_type(match.group(1))
        mapped_type = self.map_type(target_type)
        if target_type.startswith("float") or mapped_type in {"float", "double"}:
            return "asfloat"
        if target_type.startswith("uint") or mapped_type.startswith("uvec"):
            return "asuint"
        if target_type.startswith("int") or mapped_type.startswith("ivec"):
            return "asint"
        return name

    def map_metal_bit_function_name(self, name):
        text = str(name)
        if text.startswith("metal::"):
            return self.metal_bit_intrinsics.get(text.split("::")[-1])
        if text in self.user_function_names:
            return None
        return self.metal_bit_intrinsics.get(text)

    def map_metal_type_constructor_name(self, name):
        text = str(name)
        if "::" not in text:
            return None
        normalized = self.normalized_metal_type(text)
        mapped = self.map_type(normalized)
        if normalized in self.type_map or mapped != normalized:
            return mapped
        return None

    def map_metal_math_function_name(self, name):
        for prefix in self.metal_math_namespace_prefixes:
            if not str(name).startswith(prefix):
                continue
            unscoped = str(name)[len(prefix) :]
            if unscoped in self.metal_math_intrinsics:
                return unscoped
        return None

    def generate_binary_operand(self, operand, parent_op, is_right, is_main=False):
        text = self.generate_expression(operand, is_main)
        if not isinstance(operand, BinaryOpNode):
            return text

        parent_precedence = self.binary_precedence.get(parent_op, 0)
        operand_precedence = self.binary_precedence.get(operand.op, 0)
        if operand_precedence < parent_precedence or (
            is_right
            and operand_precedence == parent_precedence
            and (
                parent_op not in {"+", "*", "&&", "||", "&", "|", "^"}
                or operand.op != parent_op
            )
        ):
            return f"({text})"
        return text

    def generate_cast_operand(self, operand, rendered_operand):
        if isinstance(operand, (AssignmentNode, BinaryOpNode, TernaryOpNode)):
            return f"({rendered_operand})"
        if isinstance(operand, str) and self.cast_literal_operand_pattern.fullmatch(
            rendered_operand
        ):
            return f"({rendered_operand})"
        return rendered_operand

    def map_type(self, metal_type):
        """Map a Metal type name to the closest CrossGL type name."""
        if not metal_type:
            return metal_type

        array_type = self.metal_array_type_parts(metal_type)
        if array_type:
            element_type, size = array_type
            return f"{self.map_type(element_type)}[{size}]"

        base = metal_type.strip()
        if base.endswith("..."):
            base = base[:-3].strip()
        if base.startswith("metal::"):
            base = base.split("metal::", 1)[1]
        if base.startswith("raytracing::"):
            base = base.split("raytracing::", 1)[1]
        suffix = ""
        while base.endswith("*") or base.endswith("&"):
            suffix = base[-1] + suffix
            base = base[:-1].strip()
        for tag_prefix in ("struct ", "enum "):
            if base.startswith(tag_prefix):
                base = base[len(tag_prefix) :].strip()
                break

        atomic_alias = self.atomic_type_alias(base)
        if atomic_alias:
            return f"{atomic_alias}{suffix}"

        function_table_alias = self.function_table_type_alias(base)
        if function_table_alias:
            return f"{function_table_alias}{suffix}"

        pixel_payload_type = self.pixel_data_payload_type(base)
        if pixel_payload_type:
            return f"{self.map_type(pixel_payload_type)}{suffix}"

        vector_type = self.metal_vector_type_parts(base)
        if vector_type:
            element_type, size = vector_type
            return f"{self.map_generic_vector_type(element_type, size)}{suffix}"

        # Normalize Metal resource access qualifiers without dropping dimensions or
        # other non-resource generic arguments, e.g. matrix<bfloat, 4, 4>.
        if "<" in base and ">" in base:
            base_name, inner = base.split("<", 1)
            inner = inner.rstrip(">")
            generic_args = self.split_generic_arguments(inner)
            if self.should_elide_resource_access_qualifier(base_name, generic_args):
                base = f"{base_name}<{generic_args[0].strip()}>"

        sampled_resource_type = self.map_sampled_texture_type(base)
        if sampled_resource_type:
            return f"{sampled_resource_type}{suffix}"

        mapped = self.type_map.get(base, base)
        mapped = self.struct_name_map.get(mapped, mapped)
        if mapped == base:
            mapped = self.map_scoped_type_name(mapped)
        return f"{mapped}{suffix}"

    def format_type_alias_declaration(self, alias):
        if getattr(alias, "is_function_type", False) or self.is_resource_type_alias(
            alias
        ):
            return None

        mapped_type = self.crossgl_typedef_source_type(self.map_type_alias(alias))
        alias_name = self.sanitize_identifier(alias.name)
        if not mapped_type or not alias_name or mapped_type == alias_name:
            return None
        return f"typedef {mapped_type} {alias_name};"

    def map_type_alias(self, alias):
        storage_type = self.map_storage_texture_type(alias.alias_type)
        if storage_type:
            return storage_type
        if self.normalized_metal_type(alias.alias_type) == "half":
            return "f16"
        return self.map_type(alias.alias_type)

    def crossgl_typedef_source_type(self, mapped_type):
        mapped_type = str(mapped_type).strip()
        if not mapped_type or "::" in mapped_type:
            return None
        if any(char in mapped_type for char in "*&[]"):
            return None

        normalized_type = self.normalize_crossgl_typedef_source_type(mapped_type)
        if normalized_type is not None:
            return normalized_type
        if re.fullmatch(r"(?:matrix|vec|vector)<[^{};]+>", mapped_type):
            return mapped_type
        if mapped_type in self.crossgl_typedef_source_types():
            return mapped_type
        return None

    def normalize_crossgl_typedef_source_type(self, mapped_type):
        scalar_aliases = {
            "int8": "i8",
            "uint8": "u8",
            "int16": "i16",
            "uint16": "u16",
            "int64": "i64",
            "uint64": "u64",
            "float16": "f16",
            "float32": "f32",
            "float64": "f64",
        }
        if mapped_type in scalar_aliases:
            return scalar_aliases[mapped_type]

        vector_match = re.fullmatch(r"(f16|i8|u8|i16|u16)vec([2-4])", mapped_type)
        if vector_match:
            element_type, size = vector_match.groups()
            return f"vector<{element_type},{size}>"

        matrix_match = re.fullmatch(r"f16mat([2-4])(?:x([2-4]))?", mapped_type)
        if matrix_match:
            columns, rows = matrix_match.groups()
            rows = rows or columns
            return f"matrix<f16,{columns},{rows}>"
        return None

    def crossgl_typedef_source_types(self):
        return {
            "bool",
            "int",
            "uint",
            "float",
            "double",
            "f16",
            "bfloat",
            "i8",
            "i16",
            "i32",
            "i64",
            "u8",
            "u16",
            "u32",
            "u64",
            "f32",
            "f64",
            "half",
            "char",
            "string",
            "void",
            "vec2",
            "vec3",
            "vec4",
            "ivec2",
            "ivec3",
            "ivec4",
            "uvec2",
            "uvec3",
            "uvec4",
            "dvec2",
            "dvec3",
            "dvec4",
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
        }

    def map_scoped_type_name(self, type_name):
        if "::" not in str(type_name):
            return type_name

        base_name, _generic_args = self.generic_type_parts(type_name)
        if base_name and "::" in base_name:
            return type_name

        return self.sanitize_identifier(str(type_name).split("::")[-1])

    def is_resource_type_alias(self, alias):
        return self.is_metal_resource_type(alias.alias_type)

    def resolve_type_alias(self, metal_type):
        if not metal_type:
            return metal_type

        base = str(metal_type).strip()
        suffix = ""
        while base.endswith("*") or base.endswith("&"):
            suffix = base[-1] + suffix
            base = base[:-1].strip()

        seen = set()
        while base in self.type_aliases and base not in seen:
            seen.add(base)
            aliased = str(self.type_aliases[base]).strip()
            alias_suffix = ""
            while aliased.endswith("*") or aliased.endswith("&"):
                alias_suffix = aliased[-1] + alias_suffix
                aliased = aliased[:-1].strip()
            base = aliased
            suffix = alias_suffix + suffix
        return f"{base}{suffix}"

    def atomic_type_alias(self, metal_type):
        base_name, generic_args = self.generic_type_parts(metal_type)
        if base_name != "atomic" or len(generic_args) != 1:
            return None

        element_type = self.map_type(generic_args[0].strip())
        return {
            "int": "atomic_int",
            "uint": "atomic_uint",
            "bool": "atomic_bool",
            "uint64": "atomic_ulong",
            "float": "atomic_float",
        }.get(element_type)

    def function_table_type_alias(self, metal_type):
        base_name, generic_args = self.generic_type_parts(metal_type)
        if (
            base_name in {"visible_function_table", "intersection_function_table"}
            and generic_args
        ):
            return base_name
        return None

    def pixel_data_payload_type(self, metal_type):
        base_name, generic_args = self.generic_type_parts(metal_type)
        if base_name in self.pixel_data_type_wrappers and len(generic_args) == 1:
            return generic_args[0].strip()
        return None

    def map_sampled_texture_type(self, metal_type):
        base_name, generic_args = self.generic_type_parts(metal_type)
        if not base_name or not generic_args:
            return None

        resource_type = {
            "texture1d": "sampler1D",
            "texture1d_array": "sampler1DArray",
            "texture2d": "sampler2D",
            "texture2d_array": "sampler2DArray",
            "texture2d_ms": "sampler2DMS",
            "texture2d_ms_array": "sampler2DMSArray",
            "texture3d": "sampler3D",
            "texturecube": "samplerCube",
            "texturecube_array": "samplerCubeArray",
            "texture_buffer": "samplerBuffer",
            "depth2d": "sampler2DShadow",
            "depth2d_array": "sampler2DArrayShadow",
            "depthcube": "samplerCubeShadow",
            "depthcube_array": "samplerCubeArrayShadow",
            "depth2d_ms": "sampler2DMS",
            "depth2d_ms_array": "sampler2DMSArray",
        }.get(base_name)
        if resource_type is None:
            return None
        if base_name.startswith("depth"):
            return resource_type

        integer_prefix = self.integer_resource_prefix(generic_args[0])
        if integer_prefix is None:
            if self.is_float_resource_element(generic_args[0]):
                return resource_type
            return None
        return f"{integer_prefix}{resource_type}"

    def integer_resource_prefix(self, metal_type):
        mapped = self.map_type(str(metal_type).strip())
        if mapped in {"uint", "uint8", "uint16", "uint64"}:
            return "u"
        if mapped in {"int", "int8", "int16", "int64"}:
            return "i"
        return None

    def is_float_resource_element(self, metal_type):
        return self.map_type(str(metal_type).strip()) in {
            "float",
            "float16",
            "double",
        }

    def should_elide_resource_access_qualifier(self, base_name, generic_args):
        if len(generic_args) < 2 or not self.is_metal_resource_type_name(base_name):
            return False
        return self.normalized_access_qualifier(generic_args[1]).startswith("access::")

    def is_metal_resource_type_name(self, base_name):
        base = str(base_name).strip()
        while base.startswith("metal::"):
            base = base.split("metal::", 1)[1]
        return base in {
            "texture1d",
            "texture1d_array",
            "texture2d",
            "texture2d_array",
            "texture2d_ms",
            "texture2d_ms_array",
            "texture3d",
            "texturecube",
            "texturecube_array",
            "texture_buffer",
            "depth2d",
            "depth2d_array",
            "depth2d_ms",
            "depth2d_ms_array",
            "depthcube",
            "depthcube_array",
        }

    def is_metal_resource_type(self, metal_type):
        resolved_type = self.resolve_type_alias(metal_type)
        base_name, _generic_args = self.generic_type_parts(resolved_type)
        if not base_name:
            base_name = self.normalized_metal_type(resolved_type)
        return self.is_metal_resource_type_name(base_name)

    def generic_type_parts(self, metal_type):
        base = self.normalized_metal_type(metal_type)
        if "<" not in base or not base.endswith(">"):
            return None, []
        base_name, inner = base.split("<", 1)
        return base_name.strip(), self.split_generic_arguments(inner[:-1])

    def metal_array_type_parts(self, metal_type):
        base_name, generic_args = self.generic_type_parts(metal_type)
        if base_name != "array" or len(generic_args) < 2:
            return None
        return generic_args[0].strip(), generic_args[1].strip()

    def metal_vector_type_parts(self, metal_type):
        base_name, generic_args = self.generic_type_parts(metal_type)
        if base_name not in {"vec", "vector"} or len(generic_args) < 2:
            return None
        return generic_args[0].strip(), generic_args[1].strip()

    def map_generic_vector_type(self, element_type, size):
        size = str(size).strip()
        mapped_element = self.map_type(element_type)
        prefixes = {
            "float": "vec",
            "float16": "f16vec",
            "double": "dvec",
            "int": "ivec",
            "uint": "uvec",
            "int16": "i16vec",
            "uint16": "u16vec",
            "int8": "i8vec",
            "uint8": "u8vec",
            "bool": "bvec",
        }
        prefix = prefixes.get(mapped_element)
        if prefix and size in {"2", "3", "4"}:
            return f"{prefix}{size}"
        return f"vec<{mapped_element}, {size}>"

    def normalized_metal_type(self, metal_type):
        if not metal_type:
            return ""
        base = str(metal_type).strip()
        if base.startswith("metal::"):
            base = base.split("metal::", 1)[1]
        while base.endswith("*") or base.endswith("&"):
            base = base[:-1].strip()
        return base

    def access_qualified_texture_parts(self, metal_type):
        metal_type = self.resolve_type_alias(metal_type)
        array_type = self.metal_array_type_parts(metal_type)
        type_to_check = array_type[0] if array_type else metal_type
        base_name, generic_args = self.generic_type_parts(type_to_check)
        if len(generic_args) < 2:
            return None, generic_args
        access = self.normalized_access_qualifier(generic_args[1])
        if access not in self.storage_texture_accesses:
            return None, generic_args
        generic_args = [*generic_args]
        generic_args[1] = access
        return base_name, generic_args

    def normalized_access_qualifier(self, access):
        access = str(access).replace(" ", "")
        while access.startswith("metal::"):
            access = access.split("metal::", 1)[1]
        return access

    def is_access_qualified_storage_texture_type(self, metal_type):
        base_name, generic_args = self.access_qualified_texture_parts(metal_type)
        return bool(
            generic_args
            and base_name
            in {
                "texture1d",
                "texture1d_array",
                "texture2d",
                "texture2d_array",
                "texture_buffer",
                "texture3d",
            }
        )

    def map_storage_texture_type(self, metal_type):
        base_name, generic_args = self.access_qualified_texture_parts(metal_type)
        if not generic_args:
            return None
        return self.map_access_qualified_texture_type(base_name, generic_args)

    def storage_texture_access_attribute(self, var):
        if not (
            id(var) in self.storage_texture_declaration_ids
            or getattr(var, "name", None) in self.current_storage_texture_names
            or getattr(var, "name", None) in self.global_storage_texture_names
        ):
            return ""

        _, generic_args = self.access_qualified_texture_parts(
            getattr(var, "vtype", None)
        )
        if len(generic_args) < 2:
            return ""

        access = self.normalized_access_qualifier(generic_args[1])
        access_attributes = {
            "access::read": "@readonly",
            "access::write": "@writeonly",
            "access::read_write": "@readwrite",
        }
        return access_attributes.get(access, "")

    def storage_texture_format_attributes(self, var):
        if not (
            id(var) in self.storage_texture_declaration_ids
            or getattr(var, "name", None) in self.current_storage_texture_names
            or getattr(var, "name", None) in self.global_storage_texture_names
        ):
            return ""

        _, generic_args = self.access_qualified_texture_parts(
            getattr(var, "vtype", None)
        )
        if not generic_args:
            return ""

        element_type = self.normalized_metal_type(generic_args[0])
        if element_type == "half":
            return "@rgba16f @metal_texture_element_half"
        return ""

    def split_generic_arguments(self, inner):
        args = []
        current = []
        depth = 0
        for char in inner:
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            if char == "," and depth == 0:
                args.append("".join(current).strip())
                current = []
                continue
            current.append(char)
        if current:
            args.append("".join(current).strip())
        return args

    def map_access_qualified_texture_type(self, base_name, generic_args):
        if len(generic_args) < 2:
            return None

        access = self.normalized_access_qualifier(generic_args[1])
        if access not in self.storage_texture_accesses:
            return None

        image_type = {
            "texture1d": "image1D",
            "texture1d_array": "image1DArray",
            "texture2d": "image2D",
            "texture2d_array": "image2DArray",
            "texture_buffer": "imageBuffer",
            "texture3d": "image3D",
        }.get(base_name)
        if image_type is None:
            return None

        element_type = generic_args[0].strip()
        integer_prefix = self.integer_resource_prefix(element_type)
        if integer_prefix == "u":
            return f"u{image_type}"
        if integer_prefix == "i":
            return f"i{image_type}"
        return image_type

    def expression_base_name(self, expr):
        if isinstance(expr, str):
            return expr
        if isinstance(expr, VariableNode):
            return expr.name
        if isinstance(expr, ArrayAccessNode):
            return self.expression_base_name(expr.array)
        if isinstance(expr, MemberAccessNode):
            return self.expression_base_name(expr.object)
        return None

    def expression_metal_type(self, expr):
        if expr is None:
            return None
        if isinstance(expr, str):
            return self.current_variable_types.get(
                expr, self.global_variable_types.get(expr)
            )
        if isinstance(expr, VariableNode):
            name = getattr(expr, "name", None)
            if not name:
                return None
            return self.current_variable_types.get(
                name, self.global_variable_types.get(name)
            )
        if isinstance(expr, ArrayAccessNode):
            array_type = self.expression_metal_type(expr.array)
            array_parts = self.metal_array_type_parts(array_type)
            if array_parts:
                return array_parts[0]
            return array_type
        if isinstance(expr, MemberAccessNode):
            object_type = self.expression_metal_type(expr.object)
            if object_type is None:
                return None
            object_type = self.normalized_metal_type(object_type)
            member_types = self.struct_member_types.get(object_type)
            if not member_types:
                return None
            return member_types.get(str(expr.member))
        return None

    def expression_mapped_type(self, expr):
        metal_type = self.expression_metal_type(expr)
        storage_type = self.map_storage_texture_type(metal_type)
        if storage_type:
            return storage_type
        return (
            self.map_type(self.resolve_type_alias(metal_type)) if metal_type else None
        )

    def is_samplerless_sample_expression(self, expr):
        metal_type = self.resolve_type_alias(self.expression_metal_type(expr))
        return self.normalized_metal_type(metal_type) in {"SwiftUI::Layer"}

    def has_attribute(self, node, name):
        return any(
            getattr(attr, "name", None) == name
            for attr in getattr(node, "attributes", []) or []
        )

    def pointer_element_type(self, metal_type):
        if not metal_type:
            return None
        base = str(metal_type).strip()
        if base.startswith("metal::"):
            base = base.split("metal::", 1)[1]

        pointer_depth = 0
        while base.endswith("*") or base.endswith("&"):
            if base.endswith("*"):
                pointer_depth += 1
            base = base[:-1].strip()
        return base if pointer_depth else None

    def structured_buffer_pointer_type(self, var):
        if not self.has_attribute(var, "buffer"):
            return None

        qualifiers = {
            str(qualifier).lower() for qualifier in getattr(var, "qualifiers", []) or []
        }
        if not qualifiers.intersection({"device", "constant"}):
            return None

        element_type = self.pointer_element_type(getattr(var, "vtype", None))
        if not element_type:
            return None
        element_type = self.resolve_type_alias(element_type)

        buffer_type = (
            "StructuredBuffer"
            if qualifiers.intersection({"constant", "const"})
            else "RWStructuredBuffer"
        )
        return f"{buffer_type}<{self.map_type(element_type)}>"

    def is_structured_buffer_expression(self, expr):
        name = self.expression_base_name(expr)
        return bool(
            name
            and (
                name in self.current_structured_buffer_names
                or name in self.global_structured_buffer_names
            )
        )

    def is_structured_buffer_element_access(self, expr):
        return isinstance(
            expr, ArrayAccessNode
        ) and self.is_structured_buffer_expression(expr.array)

    def generate_without_structured_buffer_index_lowering(self, expr, is_main=False):
        previous = self.suppress_structured_buffer_index_lowering
        self.suppress_structured_buffer_index_lowering = True
        try:
            return self.generate_expression(expr, is_main)
        finally:
            self.suppress_structured_buffer_index_lowering = previous

    def generate_structured_buffer_load(self, access, is_main=False):
        buffer = self.generate_without_structured_buffer_index_lowering(
            access.array, is_main
        )
        index = self.generate_expression(access.index, is_main)
        return f"buffer_load({buffer}, {index})"

    def generate_structured_buffer_store(self, access, value, operator, is_main=False):
        buffer = self.generate_without_structured_buffer_index_lowering(
            access.array, is_main
        )
        index = self.generate_expression(access.index, is_main)
        rendered_value = self.generate_expression(value, is_main)
        if operator != "=":
            compound_ops = {
                "+=": "+",
                "-=": "-",
                "*=": "*",
                "/=": "/",
                "%=": "%",
                "&=": "&",
                "|=": "|",
                "^=": "^",
                "<<=": "<<",
                ">>=": ">>",
            }
            binary_op = compound_ops.get(operator)
            if binary_op is None:
                return None
            current_value = f"buffer_load({buffer}, {index})"
            rendered_value = f"{current_value} {binary_op} {rendered_value}"
        return f"buffer_store({buffer}, {index}, {rendered_value})"

    def is_storage_image_expression(self, expr):
        mapped_type = self.expression_mapped_type(expr)
        return bool(
            mapped_type and str(mapped_type).startswith(("image", "iimage", "uimage"))
        )

    def unsupported_storage_texture_sampled_method(self, texture_expr, method, is_main):
        if not self.is_storage_image_expression(texture_expr):
            return None
        texture = self.generate_expression(texture_expr, is_main)
        fallback = (
            "0.0"
            if method in {"sample_compare", "sample_compare_level"}
            else "vec4(0.0)"
        )
        return (
            f"{fallback} /* unsupported Metal storage texture sampled method: "
            f"{method} on {texture} */"
        )

    def unsupported_sampled_texture_write(self, texture_expr, method, is_main):
        texture = self.generate_expression(texture_expr, is_main)
        return f"/* unsupported Metal sampled texture write: {method} on {texture} */"

    def storage_image_coordinate_expression(self, image_expr, args):
        if not args:
            return ""
        rendered_args = [self.generate_expression(arg) for arg in args]
        image_type = self.expression_mapped_type(image_expr)
        if (
            image_type in {"image1DArray", "iimage1DArray", "uimage1DArray"}
            and len(rendered_args) >= 2
        ):
            return f"uvec2({rendered_args[0]}, {rendered_args[1]})"
        if (
            image_type in {"image2DArray", "iimage2DArray", "uimage2DArray"}
            and len(rendered_args) >= 2
        ):
            return f"uvec3({rendered_args[0]}, {rendered_args[1]})"
        return rendered_args[0]

    def sampled_texture_read_expression(self, texture_expr, args, is_main=False):
        mapped_type = self.expression_mapped_type(texture_expr)
        if not self.is_sampled_texture_type(mapped_type) or not args:
            return None

        texture = self.generate_expression(texture_expr, is_main)
        cube_read = self.unsupported_sampled_cube_texture_read_expression(
            texture, mapped_type
        )
        if cube_read is not None:
            return cube_read

        rendered_args = [self.generate_expression(arg, is_main) for arg in args]
        coord, tail = self.sampled_texture_read_coordinate_and_tail(
            mapped_type, rendered_args
        )
        if coord is None:
            return None

        if self.is_multisample_resource_type(mapped_type):
            if not tail:
                return None
            return f"texelFetch({texture}, {coord}, {tail[0]})"

        if self.is_sampled_buffer_texture_type(mapped_type):
            return f"texelFetch({texture}, {coord})"

        lod = tail[0] if tail else "0"
        return f"texelFetch({texture}, {coord}, {lod})"

    def unsupported_sampled_cube_texture_read_expression(self, texture, mapped_type):
        if not self.is_sampled_cube_texture_type(mapped_type):
            return None
        fallback = self.sampled_texture_read_zero_value(mapped_type)
        return (
            f"{fallback} /* unsupported Metal sampled cube texture read: "
            f"read on {texture} requires face-aware texel fetch */"
        )

    def is_sampled_cube_texture_type(self, mapped_type):
        mapped_type = self.resource_classification_type(mapped_type)
        return bool(mapped_type and "Cube" in str(mapped_type))

    def sampled_texture_read_zero_value(self, mapped_type):
        mapped_type = str(self.resource_classification_type(mapped_type) or "")
        if mapped_type.startswith("usampler"):
            return "uvec4(0u)"
        if mapped_type.startswith("isampler"):
            return "ivec4(0)"
        return "vec4(0.0)"

    def is_sampled_buffer_texture_type(self, mapped_type):
        mapped_type = self.resource_classification_type(mapped_type)
        return mapped_type in {"samplerBuffer", "isamplerBuffer", "usamplerBuffer"}

    def is_sampled_texture_type(self, mapped_type):
        if mapped_type == "sampler":
            return False
        return bool(
            mapped_type
            and str(mapped_type).startswith(("sampler", "isampler", "usampler"))
        )

    def sampled_texture_read_coordinate_and_tail(self, mapped_type, rendered_args):
        if not rendered_args:
            return None, []

        coord = rendered_args[0]
        tail = rendered_args[1:]
        constructor = self.sampled_texture_array_coordinate_constructor(mapped_type)
        if constructor is not None:
            if not tail:
                return None, []
            coord = f"{constructor}({coord}, {tail[0]})"
            tail = tail[1:]

        return coord, tail

    def sampled_texture_array_coordinate_constructor(self, mapped_type):
        mapped_type = str(mapped_type)
        if "1DArray" in mapped_type:
            return "uvec2"
        if "2DArray" in mapped_type or "2DMSArray" in mapped_type:
            return "uvec3"
        if "CubeArray" in mapped_type:
            return "vec4"
        return None

    def map_semantic(self, semantic, *, context=None):
        """Map Metal attributes to CrossGL semantic annotation syntax."""
        if not semantic:
            return ""

        if isinstance(context, dict):
            context_kind = context.get("kind")
            function_qualifier = str(context.get("function_qualifier") or "").lower()
        else:
            context_kind = context
            function_qualifier = ""

        outputs = []
        attr_names = {
            str(getattr(attr, "name", "")).lower()
            for attr in semantic
            if isinstance(attr, AttributeNode)
        }
        has_barycentric_coord = "barycentric_coord" in attr_names
        no_perspective_barycentric = has_barycentric_coord and any(
            "no_perspective" in attr_name for attr_name in attr_names
        )
        for attr in semantic:
            if not isinstance(attr, AttributeNode):
                continue
            name = attr.name
            args = (
                [self.format_metadata_argument(a) for a in attr.args]
                if attr.args
                else []
            )
            key = f"{name}({args[0]})" if args else name
            if (
                context_kind == "parameter"
                and function_qualifier == "fragment"
                and name == "position"
            ):
                out = "gl_FragCoord"
            elif name == "barycentric_coord":
                out = (
                    "gl_BaryCoordNoPerspEXT"
                    if no_perspective_barycentric
                    else "gl_BaryCoordEXT"
                )
            elif has_barycentric_coord and "no_perspective" in str(name).lower():
                continue
            elif context_kind == "parameter" and name == "sample_mask":
                out = "gl_SampleMaskIn"
            else:
                out = self.map_semantics.get(key, self.map_semantics.get(name, None))
            if out is None:
                out = self.dynamic_fragment_output_semantic(name, args)
            if out is None:
                if args:
                    out = f"{name}({', '.join(args)})"
                else:
                    out = name
            if out:
                outputs.append(f"@{out}")
        return " ".join(outputs)

    def format_metadata_argument(self, arg):
        """Render a Metal attribute argument as CrossGL metadata syntax."""
        text = str(arg).strip()
        while text.startswith("::"):
            text = text[2:].lstrip()
        return text

    def dynamic_fragment_output_semantic(self, name, args):
        if name == "color" and args:
            if re.fullmatch(r"\d+", args[0]):
                index = int(args[0])
                return "gl_FragColor" if index == 0 else f"gl_FragColor{index}"
            return "gl_FragColor"
        if name == "depth" and args and args[0].lower() in {"any", "less", "greater"}:
            return "gl_FragDepth"
        return None

    def generate_switch_statement(self, node, indent, is_main):
        expression = self.generate_expression(node.expression, is_main)
        code = f"switch ({expression}) {{\n"

        for case in node.cases:
            case_value = self.generate_expression(case.value, is_main)
            code += "    " * (indent + 1) + f"case {case_value}:\n"

            for stmt in case.statements:
                code += "    " * (indent + 2)
                if isinstance(stmt, SwitchNode):
                    code += self.generate_switch_statement(stmt, indent + 2, is_main)
                elif isinstance(stmt, IfNode):
                    code += self.generate_if_statement(stmt, indent + 2, is_main)
                elif isinstance(stmt, ForNode):
                    code += self.generate_for_loop(stmt, indent + 2, is_main)
                else:
                    code += self.generate_expression(stmt, is_main) + ";\n"

            code += "    " * (indent + 2) + "break;\n"

        if node.default:
            code += "    " * (indent + 1) + "default:\n"

            for stmt in node.default:
                code += "    " * (indent + 2)
                if isinstance(stmt, SwitchNode):
                    code += self.generate_switch_statement(stmt, indent + 2, is_main)
                elif isinstance(stmt, IfNode):
                    code += self.generate_if_statement(stmt, indent + 2, is_main)
                elif isinstance(stmt, ForNode):
                    code += self.generate_for_loop(stmt, indent + 2, is_main)
                else:
                    code += self.generate_expression(stmt, is_main) + ";\n"

            code += "    " * (indent + 2) + "break;\n"

        code += "    " * indent + "}\n"
        return code
