"""Helper functions the Mojo generator emits for resources, ray tracing, stage streams and buffers."""

import re

from .mojo_tables import (
    MOJO_CONCRETE_RESOURCE_TYPES,
    MOJO_CONCRETE_SAMPLED_TEXTURE_TYPES,
    MOJO_DTYPE_INFO,
    MOJO_TYPED_BUFFER_RESOURCE_TYPES,
)


class MojoResourceHelperMixin:
    """Resource-helper members of ``MojoCodeGen``."""

    def expanded_math_helpers(self, helpers):
        expanded = set(helpers)
        if {"magnitude", "normalize", "reflect", "refract"} & expanded:
            expanded.add("dot_product")
        if "normalize" in expanded:
            expanded.add("magnitude")
        if "smoothstep" in expanded:
            expanded.add("clamp")
        if "clamp" in expanded:
            expanded.update({"min", "max"})
        return expanded

    def generate_builtin_placeholder(self, name, type_name):
        return self.generate_zero_initialized_variable(name, type_name)

    def generate_gpu_builtin_placeholder(self, name, type_name, emitted_types):
        # Every function that reads a GPU index declares it as a local (mojo_kernel_arguments); one still missing
        # would be a zero-valued module-scope variable, which Mojo rejects.
        self.refuse_mojo_module_variable(name, "a GPU index built-in")

    def generate_ray_helpers(self):
        code = "# CrossGL ray tracing placeholders\n"
        if "RayDesc" not in self.struct_types:
            code += (
                "@value\n"
                "struct RayDesc:\n"
                "    var origin: SIMD[DType.float32, 4]\n"
                "    var t_min: Float32\n"
                "    var direction: SIMD[DType.float32, 4]\n"
                "    var t_max: Float32\n\n"
            )
        if "RayQuery" not in self.struct_types:
            code += "@value\nstruct RayQuery:\n    pass\n\n"
        if "BuiltInTriangleIntersectionAttributes" not in self.struct_types:
            code += (
                "@value\n"
                "struct BuiltInTriangleIntersectionAttributes:\n"
                "    var barycentrics: SIMD[DType.float32, 2]\n\n"
            )

        if "ray_launch_id" in self.required_ray_helpers:
            code += (
                "def ray_launch_id() -> SIMD[DType.uint32, 4]:\n"
                "    return SIMD[DType.uint32, 4](0, 0, 0, 0)\n\n"
            )
        if "ray_launch_size" in self.required_ray_helpers:
            code += (
                "def ray_launch_size() -> SIMD[DType.uint32, 4]:\n"
                "    return SIMD[DType.uint32, 4](0, 0, 0, 0)\n\n"
            )
        for helper_name, return_type, return_value in [
            ("ray_hit_t", "Float32", "0.0"),
            ("ray_hit_kind", "UInt32", "0"),
            ("ray_t_min", "Float32", "0.0"),
            ("ray_incoming_flags", "UInt32", "0"),
            ("ray_instance_custom_index", "UInt32", "0"),
            ("ray_geometry_index", "UInt32", "0"),
        ]:
            if helper_name in self.required_ray_helpers:
                code += f"def {helper_name}() -> {return_type}:\n"
                code += f"    return {return_value}\n\n"
        for helper_name in [
            "ray_world_origin",
            "ray_world_direction",
            "ray_object_origin",
            "ray_object_direction",
        ]:
            if helper_name in self.required_ray_helpers:
                code += f"def {helper_name}() -> SIMD[DType.float32, 4]:\n"
                code += "    return SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)\n\n"

        if "trace_ray" in self.required_ray_helpers:
            code += (
                "def trace_ray[AS: AnyType, Flags: AnyType, Mask: AnyType, "
                "HitGroup: AnyType, Multiplier: AnyType, Miss: AnyType, "
                "Ray: AnyType, Payload: AnyType](scene: AS, flags: Flags, "
                "mask: Mask, hit_group: HitGroup, multiplier: Multiplier, "
                "miss: Miss, ray: Ray, payload: Payload):\n"
                "    pass\n\n"
            )
        if "call_shader" in self.required_ray_helpers:
            code += (
                "def call_shader[Index: AnyType, Data: AnyType](index: Index, "
                "data: Data):\n"
                "    pass\n\n"
            )
        if "report_hit" in self.required_ray_helpers:
            code += (
                "def report_hit[Distance: AnyType, Kind: AnyType, Attributes: AnyType]"
                "(distance: Distance, kind: Kind, attributes: Attributes) -> Bool:\n"
                "    return False\n\n"
            )
        if "ignore_hit" in self.required_ray_helpers:
            code += "def ignore_hit():\n    pass\n\n"
        if "accept_hit_and_end_search" in self.required_ray_helpers:
            code += "def accept_hit_and_end_search():\n    pass\n\n"

        for helper_name in sorted(self.required_ray_helpers):
            if not helper_name.startswith("ray_query_"):
                continue
            return_type = "Bool" if helper_name == "ray_query_proceed" else "UInt32"
            return_value = "False" if return_type == "Bool" else "0"
            code += (
                f"def {helper_name}[Query: AnyType](query: Query) -> {return_type}:\n"
                f"    return {return_value}\n\n"
            )
        return code + "\n"

    def resource_requirements_use_concrete_bindings(self, resource_types):
        if not set(resource_types).issubset(MOJO_CONCRETE_RESOURCE_TYPES):
            return False
        if not self.required_resource_sample_sampler_types.issubset(
            MOJO_CONCRETE_SAMPLED_TEXTURE_TYPES
        ):
            return False
        unsupported_resource_requirements = (
            self.required_resource_sample_types
            or self.required_resource_lod_types
            or self.required_resource_lod_sampler_types
            or self.required_resource_grad_types
            or self.required_resource_grad_sampler_types
            or self.required_resource_size_types
            or self.required_resource_query_level_types
            or self.required_resource_sample_count_helpers
            or self.required_resource_texel_fetch_types
            or self.required_resource_texel_fetch_offset_types
            or self.required_image_load_types
            or self.required_image_store_types
            or self.required_resource_builtin_helpers
            or self.required_buffer_resource_types
            or self.required_buffer_load_helpers
            or self.required_buffer_store_helpers
            or self.required_buffer_append_helpers
            or self.required_buffer_consume_helpers
            or self.required_buffer_dimensions_helpers
            or self.required_byte_address_vector_load_helpers
            or self.required_byte_address_vector_store_helpers
        )
        return not unsupported_resource_requirements

    def generate_concrete_resource_type(self, resource_type):
        if resource_type == "Sampler":
            return (
                "@value\n"
                "struct Sampler:\n"
                "    var set: Int32\n"
                "    var binding: Int32\n"
                "    var linear_filter: Bool\n"
                "    def __init__(out self, set: Int32 = 0, "
                "binding: Int32 = 0, linear_filter: Bool = False):\n"
                "        self.set = set\n"
                "        self.binding = binding\n"
                "        self.linear_filter = linear_filter\n\n"
            )
        if resource_type == "Texture2D":
            return (
                "def _crossgl_texture_coord_index(coord: Float32, extent: Int32) "
                "-> Int:\n"
                "    if Int(extent) <= 1:\n"
                "        return 0\n"
                "    var clamped = coord\n"
                "    if clamped < 0.0:\n"
                "        clamped = 0.0\n"
                "    if clamped >= 1.0:\n"
                "        clamped = 0.999999\n"
                "    var index = Int(Int32(floor(clamped * Float32(Int(extent)))))\n"
                "    if index < 0:\n"
                "        return 0\n"
                "    if index >= Int(extent):\n"
                "        return Int(extent) - 1\n"
                "    return index\n\n"
                "@value\n"
                "struct Texture2D:\n"
                "    var set: Int32\n"
                "    var binding: Int32\n"
                "    var width: Int32\n"
                "    var height: Int32\n"
                "    var texels: List[SIMD[DType.float32, 4]]\n"
                "    def __init__(out self, set: Int32 = 0, "
                "binding: Int32 = 0, width: Int32 = 0, height: Int32 = 0, "
                "var texels: List[SIMD[DType.float32, 4]] = "
                "List[SIMD[DType.float32, 4]]()):\n"
                "        self.set = set\n"
                "        self.binding = binding\n"
                "        self.width = width\n"
                "        self.height = height\n"
                "        self.texels = texels\n\n"
                "    def sample(self, coord: SIMD[DType.float32, 2], "
                "sampler: Sampler) -> SIMD[DType.float32, 4]:\n"
                "        if Int(self.width) <= 0 or Int(self.height) <= 0 "
                "or self.texels.size == 0:\n"
                "            return SIMD[DType.float32, 4](0.0, 0.0, 0.0, 1.0)\n"
                "        var x = _crossgl_texture_coord_index(coord[0], self.width)\n"
                "        var y = _crossgl_texture_coord_index(coord[1], self.height)\n"
                "        var index = y * Int(self.width) + x\n"
                "        if index < 0 or index >= self.texels.size:\n"
                "            return SIMD[DType.float32, 4](0.0, 0.0, 0.0, 1.0)\n"
                "        return self.texels[index]\n\n"
            )
        return self.generate_resource_type(resource_type)

    def generate_concrete_resource_sample_sampler_helper(self, resource_type):
        coord_type = self.resource_sample_coord_type(resource_type)
        return_type = self.resource_sample_return_type(resource_type)
        code = (
            f"def sample_sampler(tex: {resource_type}, sampler: Sampler, "
            f"coord: {coord_type}) -> {return_type}:\n"
        )
        code += "    return tex.sample(coord, sampler)\n\n"
        return code

    def generate_resource_type(self, resource_type):
        code = f"@value\nstruct {resource_type}:\n"
        if resource_type == "Sampler":
            code += "    var set: Int32\n"
            code += "    var binding: Int32\n"
            code += (
                "    def __init__(out self, set: Int32 = 0, " "binding: Int32 = 0):\n"
            )
            code += "        self.set = set\n"
            code += "        self.binding = binding\n\n"
            return code
        if self.resource_size_return_type(resource_type) is not None:
            code += "    var set: Int32\n"
            code += "    var binding: Int32\n"
            code += "    var width: Int32\n"
            code += "    var height: Int32\n"
            code += "    var depth_or_layers: Int32\n"
            code += "    var levels: Int32\n"
            code += "    var samples: Int32\n"
            code += (
                "    def __init__(out self, set: Int32 = 0, "
                "binding: Int32 = 0, width: Int32 = 0, height: Int32 = 0, "
                "depth_or_layers: Int32 = 0, levels: Int32 = 1, "
                "samples: Int32 = 1):\n"
            )
            code += "        self.set = set\n"
            code += "        self.binding = binding\n"
            code += "        self.width = width\n"
            code += "        self.height = height\n"
            code += "        self.depth_or_layers = depth_or_layers\n"
            code += "        self.levels = levels\n"
            code += "        self.samples = samples\n\n"
            return code
        code += "    pass\n\n"
        return code

    def generate_buffer_resource_type(self, buffer_type):
        if buffer_type in MOJO_TYPED_BUFFER_RESOURCE_TYPES:
            return f"@value\nstruct {buffer_type}[T: AnyType]:\n    pass\n\n"
        return f"@value\nstruct {buffer_type}:\n    pass\n\n"

    def generate_geometry_stream_helper(self, stream_name):
        return (
            f"@value\n"
            f"struct CrossGLMojo{stream_name}[T: AnyType]:\n"
            f"    def Append(self, value: T):\n"
            f"        pass\n"
            f"    def RestartStrip(self):\n"
            f"        pass\n\n"
        )

    def generate_tessellation_patch_helper(self, patch_name):
        return (
            f"@value\n"
            f"struct CrossGLMojo{patch_name}[T: CollectionElement, N: Int]:\n"
            f"    var data: Array[T, N]\n"
            f"    def __getitem__(self, index: Int) -> T:\n"
            f"        return self.data[index]\n"
            f"    def __getitem__(self, index: Int32) -> T:\n"
            f"        return self.data[Int(index)]\n"
            f"    def __getitem__(self, index: UInt32) -> T:\n"
            f"        return self.data[Int(index)]\n"
            f"    def __setitem__(mut self, index: Int, value: T):\n"
            f"        self.data[index] = value\n\n"
            f"    def __setitem__(mut self, index: Int32, value: T):\n"
            f"        self.data[Int(index)] = value\n"
            f"    def __setitem__(mut self, index: UInt32, value: T):\n"
            f"        self.data[Int(index)] = value\n\n"
        )

    def generate_set_mesh_output_counts_helper(self):
        return (
            "def _crossgl_set_mesh_output_counts[Vertices: AnyType, "
            "Primitives: AnyType](vertices: Vertices, primitives: Primitives):\n"
            "    pass\n\n"
        )

    def generate_buffer_load_helper(self, buffer_type, element_type):
        element_type = self.buffer_helper_element_type(buffer_type, element_type)
        element_mojo_type = self.map_type(element_type)
        buffer_mojo_type = self.mapped_buffer_type(buffer_type, element_type)
        zero_value = self.zero_value_for_type(element_type)
        code = ""
        for index_type in ("Int32", "UInt32"):
            code += (
                f"def buffer_load(buffer: {buffer_mojo_type}, "
                f"index: {index_type}) -> {element_mojo_type}:\n"
            )
            code += f"    return {zero_value}\n\n"
        return code

    def generate_buffer_store_helper(self, buffer_type, element_type):
        element_type = self.buffer_helper_element_type(buffer_type, element_type)
        element_mojo_type = self.map_type(element_type)
        buffer_mojo_type = self.mapped_buffer_type(buffer_type, element_type)
        code = ""
        for index_type in ("Int32", "UInt32"):
            code += (
                f"def buffer_store(buffer: {buffer_mojo_type}, "
                f"index: {index_type}, value: {element_mojo_type}):\n"
            )
            code += "    pass\n\n"
        return code

    def generate_buffer_append_helper(self, buffer_type, element_type):
        element_mojo_type = self.map_type(element_type)
        buffer_mojo_type = self.mapped_buffer_type(buffer_type, element_type)
        code = f"def buffer_append(buffer: {buffer_mojo_type}, value: {element_mojo_type}):\n"
        code += "    pass\n\n"
        return code

    def generate_buffer_consume_helper(self, buffer_type, element_type):
        element_mojo_type = self.map_type(element_type)
        buffer_mojo_type = self.mapped_buffer_type(buffer_type, element_type)
        code = (
            f"def buffer_consume(buffer: {buffer_mojo_type}) -> {element_mojo_type}:\n"
        )
        code += f"    return {self.zero_value_for_type(element_type)}\n\n"
        return code

    def generate_buffer_dimensions_helper(self, buffer_type, element_type):
        buffer_mojo_type = self.mapped_buffer_type(buffer_type, element_type)
        code = f"def buffer_dimensions(buffer: {buffer_mojo_type}) -> SIMD[DType.int32, 2]:\n"
        code += "    return SIMD[DType.int32, 2](0, 0)\n\n"
        for dimensions_type in ("Int32", "UInt32"):
            code += (
                f"def buffer_dimensions(buffer: {buffer_mojo_type}, "
                f"dimensions: {dimensions_type}):\n"
            )
            code += "    pass\n\n"
            code += (
                f"def buffer_dimensions(buffer: {buffer_mojo_type}, "
                f"dimensions: {dimensions_type}, stride: {dimensions_type}):\n"
            )
            code += "    pass\n\n"
        return code

    def generate_byte_address_vector_load_helper(self, buffer_type, width):
        vector_type = self.byte_address_vector_mojo_type(width)
        zero_value = self.byte_address_vector_zero_value(width)
        code = ""
        for index_type in ("Int32", "UInt32"):
            code += (
                f"def buffer_load{width}(buffer: {buffer_type}, "
                f"index: {index_type}) -> {vector_type}:\n"
            )
            code += f"    return {zero_value}\n\n"
        return code

    def generate_byte_address_vector_store_helper(self, buffer_type, width):
        vector_type = self.byte_address_vector_mojo_type(width)
        code = ""
        for index_type in ("Int32", "UInt32"):
            code += (
                f"def buffer_store{width}(buffer: {buffer_type}, "
                f"index: {index_type}, value: {vector_type}):\n"
            )
            code += "    pass\n\n"
        return code

    def generate_reinterpret_helper(self, func_name, arg_type, return_type):
        code = f"def {func_name}(value: {arg_type}) -> {return_type}:\n"
        return_match = re.fullmatch(r"SIMD\[(DType\.\w+), (\d+)\]", return_type)
        if arg_type.startswith("SIMD[") and return_match is not None:
            code += (
                f"    return bitcast[{return_match.group(1)}, "
                f"{return_match.group(2)}](value)\n\n"
            )
            return code
        scalar_dtype = {
            "Float32": "DType.float32",
            "Int32": "DType.int32",
            "UInt32": "DType.uint32",
        }.get(return_type)
        if scalar_dtype is not None:
            code += f"    return bitcast[{scalar_dtype}](value)\n\n"
            return code
        code += f"    return {return_type}(value)\n\n"
        return code

    def generate_resource_sample_helper(self, resource_type):
        coord_type = self.resource_sample_coord_type(resource_type)
        return_type = self.resource_sample_return_type(resource_type)
        code = (
            f"def sample(tex: {resource_type}, coord: {coord_type}) -> {return_type}:\n"
        )
        code += f"    return {self.zero_mojo_value(return_type)}\n\n"
        return code

    def generate_resource_sample_sampler_helper(self, resource_type):
        coord_type = self.resource_sample_coord_type(resource_type)
        return_type = self.resource_sample_return_type(resource_type)
        code = (
            f"def sample_sampler(tex: {resource_type}, sampler: Sampler, "
            f"coord: {coord_type}) -> {return_type}:\n"
        )
        code += f"    return {self.zero_mojo_value(return_type)}\n\n"
        return code

    def generate_resource_lod_helper(self, resource_type):
        coord_type = self.resource_sample_coord_type(resource_type)
        return_type = self.resource_sample_return_type(resource_type)
        code = (
            f"def sample_lod(tex: {resource_type}, coord: {coord_type}, "
            f"lod: Float32) -> {return_type}:\n"
        )
        if return_type == "SIMD[DType.float32, 4]":
            code += "    return SIMD[DType.float32, 4](0.0, 0.0, lod, 1.0)\n\n"
        else:
            code += f"    return {self.zero_mojo_value(return_type)}\n\n"
        return code

    def generate_resource_lod_sampler_helper(self, resource_type):
        coord_type = self.resource_sample_coord_type(resource_type)
        return_type = self.resource_sample_return_type(resource_type)
        code = (
            f"def sample_lod_sampler(tex: {resource_type}, sampler: Sampler, "
            f"coord: {coord_type}, lod: Float32) -> {return_type}:\n"
        )
        if return_type == "SIMD[DType.float32, 4]":
            code += "    return SIMD[DType.float32, 4](0.0, 0.0, lod, 1.0)\n\n"
        else:
            code += f"    return {self.zero_mojo_value(return_type)}\n\n"
        return code

    def generate_resource_grad_helper(self, resource_type):
        coord_type = self.resource_sample_coord_type(resource_type)
        return_type = self.resource_sample_return_type(resource_type)
        code = (
            f"def sample_grad(tex: {resource_type}, coord: {coord_type}, "
            f"ddx: {coord_type}, ddy: {coord_type}) -> {return_type}:\n"
        )
        code += f"    return {self.zero_mojo_value(return_type)}\n\n"
        return code

    def generate_resource_grad_sampler_helper(self, resource_type):
        coord_type = self.resource_sample_coord_type(resource_type)
        return_type = self.resource_sample_return_type(resource_type)
        code = (
            f"def sample_grad_sampler(tex: {resource_type}, sampler: Sampler, "
            f"coord: {coord_type}, ddx: {coord_type}, ddy: {coord_type}) "
            f"-> {return_type}:\n"
        )
        code += f"    return {self.zero_mojo_value(return_type)}\n\n"
        return code

    def generate_resource_size_helper(self, helper_name, resource_type):
        return_type = self.resource_size_return_type(resource_type)
        size_value = self.resource_size_expression(resource_type)
        if helper_name == "texture_size":
            lod_size_value = self.resource_size_expression(resource_type, lod="lod")
            code = f"def texture_size(tex: {resource_type}) -> {return_type}:\n"
            code += f"    return {size_value}\n\n"
            code += f"def texture_size(tex: {resource_type}, lod: Int32) -> {return_type}:\n"
            code += f"    return {lod_size_value}\n\n"
            return code

        code = f"def image_size(image: {resource_type}) -> {return_type}:\n"
        code += (
            f"    return {self.resource_size_expression(resource_type, 'image')}\n\n"
        )
        return code

    def generate_mip_dimension_helper(self):
        return (
            "def _crossgl_mip_dimension(value: Int32, lod: Int32) -> Int32:\n"
            "    if value <= 0:\n"
            "        return 0\n"
            "    var scaled = value\n"
            "    var level = lod\n"
            "    while level > 0:\n"
            "        if scaled > 1:\n"
            "            scaled = scaled >> 1\n"
            "        level -= 1\n"
            "    return scaled\n\n"
        )

    def resource_size_expression(self, resource_type, parameter_name="tex", lod=None):
        return_type = self.resource_size_return_type(resource_type)
        base_type = self.image_resource_base_type(resource_type)

        def component(field_name, mip_scaled=False):
            value = f"{parameter_name}.{field_name}"
            if lod is not None and mip_scaled:
                return f"_crossgl_mip_dimension({value}, {lod})"
            return value

        if return_type == "Int32":
            return component("width", mip_scaled=True)

        vector_match = re.fullmatch(r"SIMD\[DType\.int32, (\d+)\]", return_type)
        if not vector_match:
            return self.zero_mojo_value(return_type)

        width = int(vector_match.group(1))
        if width == 2:
            if "1DArray" in base_type:
                components = [
                    component("width", mip_scaled=True),
                    component("depth_or_layers"),
                ]
            else:
                components = [
                    component("width", mip_scaled=True),
                    component("height", mip_scaled=True),
                ]
        else:
            depth_is_mip_dimension = base_type == "Texture3D"
            components = [
                component("width", mip_scaled=True),
                component("height", mip_scaled=True),
                component("depth_or_layers", mip_scaled=depth_is_mip_dimension),
                "0",
            ]
        return f"{return_type}({', '.join(components)})"

    def generate_resource_query_levels_helper(self, resource_type):
        code = f"def texture_query_levels(tex: {resource_type}) -> Int32:\n"
        code += "    return tex.levels\n\n"
        return code

    def generate_resource_sample_count_helper(self, helper_name, resource_type):
        parameter_name = "image" if helper_name == "image_samples" else "tex"
        code = f"def {helper_name}({parameter_name}: {resource_type}) -> Int32:\n"
        code += f"    return {parameter_name}.samples\n\n"
        return code

    def generate_texel_fetch_helper(self, resource_type):
        coord_type = self.resource_texel_coord_type(resource_type)
        return_type = self.resource_sample_return_type(resource_type)
        code = (
            f"def texel_fetch(tex: {resource_type}, coord: {coord_type}, "
            f"lod: Int32) -> {return_type}:\n"
        )
        code += f"    return {self.zero_mojo_value(return_type)}\n\n"
        return code

    def generate_texel_fetch_offset_helper(self, resource_type):
        coord_type = self.resource_texel_coord_type(resource_type)
        offset_type = self.resource_offset_coord_type(resource_type)
        return_type = self.resource_sample_return_type(resource_type)
        code = (
            f"def texel_fetch_offset(tex: {resource_type}, coord: {coord_type}, "
            f"lod: Int32, offset: {offset_type}) -> {return_type}:\n"
        )
        code += f"    return {self.zero_mojo_value(return_type)}\n\n"
        return code

    def generate_image_load_helper(self, resource_type):
        coord_type = self.resource_texel_coord_type(resource_type)
        value_type = self.image_value_type(resource_type)
        if self.is_multisample_resource_type(resource_type):
            code = (
                f"def image_load(image: {resource_type}, coord: {coord_type}, "
                f"sample: Int32) -> {value_type}:\n"
            )
        else:
            code = (
                f"def image_load(image: {resource_type}, coord: {coord_type}) -> "
                f"{value_type}:\n"
            )
        code += f"    return {self.zero_mojo_value(value_type)}\n\n"
        return code

    def generate_image_store_helper(self, resource_type):
        coord_type = self.resource_texel_coord_type(resource_type)
        value_type = self.image_value_type(resource_type)
        if self.is_multisample_resource_type(resource_type):
            code = (
                f"def image_store(image: {resource_type}, coord: {coord_type}, "
                f"sample: Int32, value: {value_type}):\n"
            )
        else:
            code = (
                f"def image_store(image: {resource_type}, coord: {coord_type}, "
                f"value: {value_type}):\n"
            )
        code += "    pass\n\n"
        return code

    def generate_resource_builtin_helper(self, helper):
        params = [
            f"arg{index}: {arg_type}"
            for index, arg_type in enumerate(helper["arg_types"])
        ]
        code = (
            f"def {helper['name']}({', '.join(params)}) -> {helper['return_type']}:\n"
        )
        code += f"    return {self.zero_mojo_value(helper['return_type'])}\n\n"
        return code

    def is_multisample_resource_type(self, resource_type):
        return "MS" in resource_type

    def is_shadow_resource_type(self, resource_type):
        return isinstance(resource_type, str) and resource_type.endswith("Shadow")

    def is_texture_resource_type(self, resource_type):
        return isinstance(resource_type, str) and resource_type.startswith("Texture")

    def is_image_resource_type(self, resource_type):
        return isinstance(resource_type, str) and resource_type.startswith(
            ("Image", "IImage", "UImage")
        )

    def resource_sample_return_type(self, resource_type):
        return self.map_type(self.texture_result_type_name(resource_type))

    def texture_result_type_name(self, resource_type):
        if self.is_shadow_resource_type(resource_type):
            return "float"
        typed_info = self.typed_texture_resource_info(resource_type)
        if typed_info is not None:
            _, dtype, source_width = typed_info
            if source_width == 1:
                return MOJO_DTYPE_INFO[dtype][0]
            return self.vector_type_name_for_dtype_width(dtype, source_width)
        return "vec4"

    def image_value_type(self, resource_type):
        typed_info = self.typed_image_resource_info(resource_type)
        if typed_info is not None:
            _, dtype, source_width = typed_info
            if source_width == 1:
                return self.map_type(MOJO_DTYPE_INFO[dtype][0])
            return self.map_type(
                self.vector_type_name_for_dtype_width(dtype, source_width)
            )
        if resource_type.startswith("IImage"):
            return "Int32"
        if resource_type.startswith("UImage"):
            return "UInt32"
        return "SIMD[DType.float32, 4]"

    def image_atomic_value_type(self, resource_type):
        typed_info = self.typed_image_resource_info(resource_type)
        if typed_info is not None:
            _, dtype, source_width = typed_info
            if source_width in {1, 4} and dtype in {"DType.int32", "DType.uint32"}:
                return self.map_type(MOJO_DTYPE_INFO[dtype][0])
            return self.image_value_type(resource_type)
        return self.image_value_type(resource_type)

    def zero_mojo_value(self, mojo_type):
        if mojo_type in {
            "Int",
            "Int16",
            "Int32",
            "Int64",
            "UInt16",
            "UInt32",
            "UInt64",
        }:
            return "0"
        if mojo_type in {"Float16", "Float32", "Float64"}:
            return "0.0"
        if mojo_type == "Bool":
            return "False"

        vector_match = re.fullmatch(r"SIMD\[(DType\.\w+), (\d+)\]", mojo_type)
        if vector_match:
            dtype = vector_match.group(1)
            width = int(vector_match.group(2))
            zero = MOJO_DTYPE_INFO.get(dtype, MOJO_DTYPE_INFO["DType.float32"])[2]
            return f"{mojo_type}({', '.join([zero] * width)})"

        return f"{mojo_type}()"
