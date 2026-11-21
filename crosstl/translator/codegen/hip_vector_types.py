"""Helper insertion, the ray-tracing runtime, the variable registry and vector type information of the HIP generator, plus half, narrow-integer and surface-read helpers."""

from ..ast import (
    ArrayAccessNode,
    IdentifierNode,
    VariableNode,
)
from .c_narrow_integer_vectors import generic_narrow_vector_spelling
from .hip_tables import HIP_FP16_VECTOR_TYPES


class HipVectorTypeMixin:
    """Vector type members of ``HipCodeGen``."""

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

    def swizzle_result_vector_info(self, result_type):
        result_info = self.vector_type_info(result_type)
        if result_info is not None:
            return result_info
        if self.native_vector_component_type(result_type) == "half":
            return {
                "type": "half2",
                "constructor": "__floats2half2_rn",
                "component_type": "half",
                "components": ("x", "y"),
            }
        return None

    def native_vector_component_type(self, type_name):
        # half2 maps to the native half2 type, which vector_type_info leaves
        # undescribed; its components are still half
        type_text = self.type_name_string(type_name)
        if type_text is None:
            return None
        compact_type = "".join(str(type_text).split())
        if compact_type in {
            "half2",
            "f16vec2",
            "float16vec2",
            "vec2<f16>",
            "vec2<float16>",
            "vec2<half>",
        }:
            return "half"
        return None

    def hip_half_vector_type_info(self, type_name):
        type_text = self.type_name_string(type_name)
        if type_text is None:
            return None
        compact_type = "".join(str(type_text).split())
        component_count = HIP_FP16_VECTOR_TYPES.get(compact_type)
        if component_count is None:
            return None
        self.require_hip_half_vector_helper(component_count)
        vector_type = f"cgl_half{component_count}"
        return {
            "type": vector_type,
            "constructor": f"cgl_make_half{component_count}",
            "component_type": "half",
            "components": ("x", "y", "z", "w")[:component_count],
        }

    def hip_narrow_integer_vector_type_info(self, type_name):
        type_text = self.type_name_string(type_name)
        if type_text is None:
            return None

        compact_type = "".join(str(type_text).split())
        compact_type = generic_narrow_vector_spelling(compact_type) or compact_type
        if (
            len(compact_type) < 8
            or not compact_type.startswith("vec")
            or compact_type[3] not in {"2", "3", "4"}
            or compact_type[4] != "<"
            or compact_type[-1] != ">"
        ):
            return None

        scalar_type = compact_type[5:-1]
        native_prefix, component_type = {
            "i8": ("char", "int"),
            "u8": ("uchar", "uint"),
            "i16": ("short", "int"),
            "u16": ("ushort", "uint"),
        }.get(scalar_type, (None, None))
        if native_prefix is None:
            return None

        size = int(compact_type[3])
        components = ("x", "y", "z", "w")[:size]
        vector_type = f"{native_prefix}{size}"
        return {
            "type": vector_type,
            "constructor": f"make_{vector_type}",
            "component_type": component_type,
            "components": components,
            "vector_element": scalar_type,
        }

    def require_hip_half3_helper(self):
        self.require_hip_half_vector_helper(3)

    def require_hip_half4_helper(self):
        self.require_hip_half_vector_helper(4)

    def require_hip_half_conversion_helper(self):
        helper_name = "cgl_half_conversion"
        if helper_name in self.helper_functions:
            return
        self.helper_functions[helper_name] = (
            "__host__ __device__ inline half cgl_to_half(half value)\n"
            "{\n"
            "    return value;\n"
            "}\n"
            "\n"
            "template <typename T>\n"
            "__host__ __device__ inline half cgl_to_half(T value)\n"
            "{\n"
            "    return __float2half(static_cast<float>(value));\n"
            "}"
        )

    def require_hip_half_vector_helper(self, component_count):
        self.require_hip_half_conversion_helper()
        helper_name = f"cgl_half{component_count}_type"
        if helper_name in self.helper_functions:
            return

        vector_type = f"cgl_half{component_count}"
        components = ("x", "y", "z", "w")[:component_count]
        fields = "\n".join(f"    half {component};" for component in components)
        zero_initializers = ", ".join(
            f"{component}(__float2half(0.0f))" for component in components
        )
        parameters = ", ".join(f"half {component}_value" for component in components)
        value_initializers = ", ".join(
            f"{component}({component}_value)" for component in components
        )
        template_parameters = ", ".join(
            f"typename {component.upper()}" for component in components
        )
        make_parameters = ", ".join(
            f"{component.upper()} {component}" for component in components
        )
        make_arguments = ", ".join(
            f"cgl_to_half({component})" for component in components
        )
        alignment = " alignas(8)" if component_count == 4 else ""
        self.helper_functions[helper_name] = (
            f"struct{alignment} {vector_type}\n"
            "{\n"
            f"{fields}\n"
            "\n"
            f"    __host__ __device__ {vector_type}()\n"
            f"        : {zero_initializers}\n"
            "    {\n"
            "    }\n"
            "\n"
            f"    __host__ __device__ {vector_type}({parameters})\n"
            f"        : {value_initializers}\n"
            "    {\n"
            "    }\n"
            "};\n"
            "\n"
            f"__host__ __device__ inline {vector_type} "
            f"cgl_make_half{component_count}()\n"
            "{\n"
            f"    return {vector_type}();\n"
            "}\n"
            "\n"
            f"template <{template_parameters}>\n"
            f"__host__ __device__ inline {vector_type} "
            f"cgl_make_half{component_count}({make_parameters})\n"
            "{\n"
            f"    return {vector_type}({make_arguments});\n"
            "}"
        )

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
