"""Result types that ``RustCodeGen.builtin_function_result_type`` infers for shader intrinsics, by family: resource calls, geometric and elementwise math, and relational, bitwise, packing and numeric calls."""

from ..unhandled import (
    UNHANDLED,
)


class RustBuiltinResultTypeMixin:
    """Intrinsic result-type members of ``RustCodeGen.builtin_function_result_type``."""

    def rust_resource_builtin_result_type(
        self,
        func_name,
        arg_types,
        mapped_name,
    ):
        """Result type of a texture sample, gather or query, an image or buffer access, or a resource store."""
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
        }:
            if arg_types:
                return self.texture_sample_result_type(arg_types[0])
            return "vec4"

        if mapped_name in {
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
            if arg_types:
                return self.texture_gather_result_type(arg_types[0])
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
        return UNHANDLED

    def rust_math_builtin_result_type(
        self,
        arg_types,
        mapped_name,
    ):
        """Result type of a geometric, matrix, elementwise float or vector test intrinsic."""
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
            mapped_name in {
                "sqrt",
                "rcp",
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
        return UNHANDLED

    def rust_bitwise_numeric_builtin_result_type(
        self,
        arg_types,
        mapped_name,
    ):
        """Result type of a relational, bit, bit-cast, packing or numeric intrinsic."""
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
            "double_bits_to_long": "ulong",
            "float_bits_to_int": "int",
            "float_bits_to_uint": "uint",
            "int_bits_to_float": "float",
            "long_bits_to_double": "double",
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
            mapped_name in {
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
        return UNHANDLED
