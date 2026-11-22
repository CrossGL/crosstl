"""Resource calls of the Slang generator: image atomics, resource queries and sampled-texture arguments."""

from .slang_byte_address_words import slang_byte_address_vector_call
from .slang_projected_compare import (
    generate_slang_projected_compare,
    is_slang_projected_compare,
)


class SlangResourceCallsMixin:
    """Resource-call members of ``SlangCodeGen``."""

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

    def image_atomic_operation_from_atomic_op(self, operation):
        return {
            "imageAtomicAdd": "imageAtomicAdd",
            "imageAtomicMin": "imageAtomicMin",
            "imageAtomicMax": "imageAtomicMax",
            "imageAtomicAnd": "imageAtomicAnd",
            "imageAtomicOr": "imageAtomicOr",
            "imageAtomicXor": "imageAtomicXor",
            "imageAtomicExchange": "imageAtomicExchange",
            "imageAtomicCompSwap": "imageAtomicCompSwap",
            "imageAtomicCompareExchange": "imageAtomicCompSwap",
            "atomicAdd": "imageAtomicAdd",
            "atomicMin": "imageAtomicMin",
            "atomicMax": "imageAtomicMax",
            "atomicAnd": "imageAtomicAnd",
            "atomicOr": "imageAtomicOr",
            "atomicXor": "imageAtomicXor",
            "atomicExchange": "imageAtomicExchange",
            "atomicCompSwap": "imageAtomicCompSwap",
            "atomicCompareExchange": "imageAtomicCompSwap",
            "Add": "imageAtomicAdd",
            "Min": "imageAtomicMin",
            "Max": "imageAtomicMax",
            "And": "imageAtomicAnd",
            "Or": "imageAtomicOr",
            "Xor": "imageAtomicXor",
            "Exchange": "imageAtomicExchange",
            "CompSwap": "imageAtomicCompSwap",
            "CompareExchange": "imageAtomicCompSwap",
        }.get(operation)

    def generate_slang_atomic_op_expression(self, node):
        operation = self.image_atomic_operation_from_atomic_op(node.operation)
        if operation is None:
            return self.unsupported_image_atomic_call(
                node.operation, "is not recognized by the Slang backend"
            )

        result = self.image_atomic_expression(operation, [node.target, *node.arguments])
        if result is not None:
            return result

        return self.unsupported_image_atomic_call(
            node.operation, "requires image resource target"
        )

    def image_atomic_helper_suffix(self, image_type):
        return {
            "RWTexture1D<int>": "iimage1D",
            "RWTexture1D<uint>": "uimage1D",
            "RWTexture2D<int>": "iimage2D",
            "RWTexture2D<uint>": "uimage2D",
            "RWTexture3D<int>": "iimage3D",
            "RWTexture3D<uint>": "uimage3D",
            "RWTexture1DArray<int>": "iimage1DArray",
            "RWTexture1DArray<uint>": "uimage1DArray",
            "RWTexture2DArray<int>": "iimage2DArray",
            "RWTexture2DArray<uint>": "uimage2DArray",
        }.get(image_type)

    def image_atomic_return_type(self, image_type):
        element_type = self.image_resource_element_type(image_type)
        if element_type in {"int", "uint"}:
            return element_type
        return None

    def image_atomic_coord_type(self, image_type):
        if image_type in {"RWTexture1D<int>", "RWTexture1D<uint>"}:
            return "int"
        if image_type in {"RWTexture1DArray<int>", "RWTexture1DArray<uint>"}:
            return "int2"
        if image_type in {"RWTexture2D<int>", "RWTexture2D<uint>"}:
            return "int2"
        if image_type in {
            "RWTexture3D<int>",
            "RWTexture3D<uint>",
            "RWTexture2DArray<int>",
            "RWTexture2DArray<uint>",
        }:
            return "int3"
        return None

    def image_atomic_helper_name(self, operation, image_type):
        suffix = self.image_atomic_helper_suffix(image_type)
        if not suffix:
            return None
        return f"cgl_{operation}_{suffix}"

    def image_atomic_zero_value(self, image_type=None, result_type=None):
        expected_type = self.atomic_result_expected_type()
        if expected_type:
            return self.zero_value_for_type(expected_type)
        if result_type:
            return self.zero_value_for_type(result_type)

        element_type = self.image_resource_element_type(image_type)
        if isinstance(element_type, str) and element_type.startswith("uint"):
            return "0u"

        expected_type = self.convert_type(self.current_expression_expected_type)
        if expected_type == "uint":
            return "0u"
        return "0"

    def unsupported_image_atomic_call(
        self, operation, reason, image_type=None, result_type=None
    ):
        return self.refuse_lowering(
            f"Slang image atomic: {operation} {reason}",
            self.image_atomic_zero_value(image_type, result_type),
        )

    def image_atomic_required_args_reason(self, operation):
        if operation == "imageAtomicCompSwap":
            return "requires image, coordinate, compare, and value arguments"
        return "requires image, coordinate, and value arguments"

    def image_atomic_value_expression(
        self, operation, args, image_type, intrinsic, return_type
    ):
        if not self.expression_prelude_active():
            if self.atomic_value_context_stack:
                context = self.atomic_value_context_stack[-1]
                return self.unsupported_image_atomic_call(
                    operation,
                    "expression-valued atomic cannot be used in "
                    f"{context}; assign the atomic result before the loop",
                    image_type,
                )
            return None

        original_name = self.unique_expression_temp_name(f"cgl_{operation}_original")
        image_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[1])
        target_expr = f"{image_name}[{coord}]"
        if operation == "imageAtomicCompSwap":
            compare = self.generate_expression_with_expected(args[2], return_type)
            value = self.generate_expression_with_expected(args[3], return_type)
            atomic_statement = (
                f"InterlockedCompareExchange({target_expr}, "
                f"{compare}, {value}, {original_name});"
            )
        else:
            value = self.generate_expression_with_expected(args[2], return_type)
            atomic_statement = f"{intrinsic}({target_expr}, {value}, {original_name});"

        self.add_expression_prelude(
            [
                f"{return_type} {original_name};",
                atomic_statement,
            ],
            result_name=original_name,
        )
        return original_name

    def image_atomic_expression(self, operation, args):
        if not self.image_atomic_intrinsic(operation):
            return None

        required_args = 4 if operation == "imageAtomicCompSwap" else 3
        if len(args) < required_args:
            return self.unsupported_image_atomic_call(
                operation, self.image_atomic_required_args_reason(operation)
            )

        image_type = self.resource_base_type(self.image_resource_type(args[0]))
        base_helper_name = self.image_atomic_helper_name(operation, image_type)
        if self.image_resource_access(args[0]) in {"readonly", "writeonly"}:
            return self.unsupported_image_atomic_call(
                operation,
                "requires readwrite image resource",
                image_type,
            )

        if not base_helper_name:
            reason = (
                "requires scalar int or uint "
                "image1D/image1DArray/image2D/image3D/image2DArray resource"
            )
            return self.unsupported_image_atomic_call(
                operation,
                reason,
                image_type,
            )

        return_type = self.image_atomic_return_type(image_type)
        target_reason = self.atomic_result_expected_type_unsupported_reason(
            operation, return_type
        )
        if target_reason:
            return self.unsupported_image_atomic_call(
                operation,
                target_reason,
                image_type,
                self.atomic_result_diagnostic_fallback_type(return_type),
            )

        intrinsic = self.image_atomic_intrinsic(operation)
        value_expr = self.image_atomic_value_expression(
            operation,
            args,
            image_type,
            intrinsic,
            return_type,
        )
        if value_expr is not None:
            return value_expr

        image_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[1])
        helper_name = self.helper_function_name(base_helper_name)
        self.register_helper_function(
            helper_name,
            self.build_image_atomic_helper(helper_name, operation, image_type),
        )
        if operation == "imageAtomicCompSwap":
            compare = self.generate_expression(args[2])
            value = self.generate_expression(args[3])
            return f"{helper_name}({image_name}, {coord}, {compare}, {value})"

        value = self.generate_expression(args[2])
        return f"{helper_name}({image_name}, {coord}, {value})"

    def build_image_atomic_helper(self, helper_name, operation, image_type):
        return_type = self.image_atomic_return_type(image_type)
        coord_type = self.image_atomic_coord_type(image_type)
        intrinsic = self.image_atomic_intrinsic(operation)
        if not return_type or not coord_type or not intrinsic:
            return ""

        if operation == "imageAtomicCompSwap":
            return (
                f"{return_type} {helper_name}({image_type} image, "
                f"{coord_type} coord, {return_type} compareValue, "
                f"{return_type} value)\n"
                "{\n"
                f"    {return_type} original;\n"
                "    InterlockedCompareExchange(image[coord], compareValue, value, original);\n"
                "    return original;\n"
                "}"
            )

        return (
            f"{return_type} {helper_name}({image_type} image, "
            f"{coord_type} coord, {return_type} value)\n"
            "{\n"
            f"    {return_type} original;\n"
            f"    {intrinsic}(image[coord], value, original);\n"
            "    return original;\n"
            "}"
        )

    def resource_query_slang_type(self, resource_arg, resource_type):
        if self.is_storage_image_type(resource_type):
            image_type = self.resource_base_type(self.image_resource_type(resource_arg))
            if image_type:
                return image_type
        resource_name = self.expression_root_identifier_name(resource_arg)
        if resource_name in self.explicit_sampler_texture_names:
            separated_type = self.separated_slang_texture_type(
                self.resource_base_type(resource_type)
            )
            if separated_type is not None:
                return separated_type
        return self.convert_type(resource_type)

    def resource_query_helper_name(self, func_name, resource_type, resource_slang_type):
        base_name = f"cgl_{func_name}_{resource_type}"
        if resource_slang_type == self.convert_type(resource_type):
            return base_name
        return f"{base_name}_{self.resource_helper_type_suffix(resource_slang_type)}"

    def resource_helper_type_suffix(self, resource_slang_type):
        return "".join(
            char if char.isalnum() else "_"
            for char in str(resource_slang_type).strip("_")
        ).strip("_")

    def generate_resource_call(self, func_name, args, statement_context=False):
        if func_name == "imageLoad":
            return self.image_load_expression(args)

        if func_name == "imageStore":
            return self.image_store_expression(args)

        if func_name == "buffer_load":
            return self.buffer_load_expression(args)

        if func_name == "buffer_store":
            return self.buffer_store_expression(args, statement_context)

        vector_call = slang_byte_address_vector_call(
            self, func_name, args, statement_context
        )
        if vector_call is not None:
            return vector_call

        if func_name == "buffer_append":
            return self.buffer_append_expression(args)

        if func_name == "buffer_consume":
            return self.buffer_consume_expression(args)

        if func_name in {"buffer_increment_counter", "buffer_decrement_counter"}:
            return self.buffer_counter_expression(func_name, args)

        if func_name == "buffer_dimensions":
            return self.buffer_dimensions_expression(args)

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
            return self.image_atomic_expression(func_name, args)

        structured_buffer_atomic = self.structured_buffer_atomic_expression(
            func_name, args, statement_context=statement_context
        )
        if structured_buffer_atomic is not None:
            return structured_buffer_atomic

        workgroup_atomic = self.workgroup_atomics.expression(
            func_name, args, statement_context
        )
        if workgroup_atomic is not None:
            return workgroup_atomic

        if func_name in {"texture", "textureLod", "textureGrad"}:
            sample_args = self.sampled_texture_operation_args(func_name, args)
            if sample_args is None:
                return self.unsupported_sampled_texture_call(
                    func_name,
                    self.sampled_texture_operation_unsupported_reason(func_name, args),
                )

            texture_name, coord, extra_args = sample_args
            coord_node = args[self.sampled_texture_coord_index(args)]
            coord_reason = self.sampled_texture_coordinate_rank_unsupported_reason(
                args[0], coord_node
            )
            if coord_reason:
                return self.unsupported_sampled_texture_call(func_name, coord_reason)

            result_type = self.sampled_texture_value_type(
                self.get_expression_type(args[0])
            )
            if func_name == "texture":
                if extra_args:
                    bias_reason = self.scalar_texture_bias_unsupported_reason(
                        extra_args[0]
                    )
                    if bias_reason:
                        return self.unsupported_sampled_texture_call(
                            func_name, bias_reason
                        )
                    expected_reason = (
                        self.texture_result_expected_type_unsupported_reason(
                            func_name, result_type
                        )
                    )
                    if expected_reason:
                        return self.unsupported_sampled_texture_call(
                            func_name, expected_reason
                        )
                    bias = self.generate_expression(extra_args[0])
                    return self.slang_texture_method_call(
                        texture_name, "SampleBias", args, coord, bias
                    )
                expected_reason = self.texture_result_expected_type_unsupported_reason(
                    func_name, result_type
                )
                if expected_reason:
                    return self.unsupported_sampled_texture_call(
                        func_name, expected_reason
                    )
                return self.slang_texture_method_call(
                    texture_name, "Sample", args, coord
                )

            if func_name == "textureLod" and extra_args:
                lod_reason = self.scalar_texture_lod_unsupported_reason(extra_args[0])
                if lod_reason:
                    return self.unsupported_sampled_texture_call(func_name, lod_reason)
                expected_reason = self.texture_result_expected_type_unsupported_reason(
                    func_name, result_type
                )
                if expected_reason:
                    return self.unsupported_sampled_texture_call(
                        func_name, expected_reason
                    )
                lod = self.generate_expression(extra_args[0])
                return self.slang_texture_method_call(
                    texture_name, "SampleLevel", args, coord, lod
                )

            if func_name == "textureGrad" and len(extra_args) >= 2:
                grad_reason = self.texture_gradient_rank_unsupported_reason(
                    args[0], extra_args[0]
                ) or self.texture_gradient_rank_unsupported_reason(
                    args[0], extra_args[1]
                )
                if grad_reason:
                    return self.unsupported_sampled_texture_call(func_name, grad_reason)
                expected_reason = self.texture_result_expected_type_unsupported_reason(
                    func_name, result_type
                )
                if expected_reason:
                    return self.unsupported_sampled_texture_call(
                        func_name, expected_reason
                    )
                ddx = self.generate_expression(extra_args[0])
                ddy = self.generate_expression(extra_args[1])
                return self.slang_texture_method_call(
                    texture_name, "SampleGrad", args, coord, ddx, ddy
                )

            return self.unsupported_sampled_texture_call(
                func_name,
                self.sampled_texture_operation_arity_requirement(func_name),
            )

        if func_name in {"textureOffset", "textureLodOffset", "textureGradOffset"}:
            return self.generate_texture_offset(func_name, args)

        if func_name in {
            "textureProj",
            "textureProjOffset",
            "textureProjLod",
            "textureProjLodOffset",
            "textureProjGrad",
            "textureProjGradOffset",
        }:
            return self.generate_texture_projected(func_name, args)

        if func_name in {
            "textureGather",
            "textureGatherOffset",
            "textureGatherOffsets",
        }:
            return self.generate_texture_gather(func_name, args)

        if func_name in {
            "textureCompare",
            "textureCompareLod",
            "textureCompareLodOffset",
            "textureCompareGrad",
            "textureCompareOffset",
        }:
            return self.generate_texture_compare(func_name, args)

        if is_slang_projected_compare(func_name):
            return generate_slang_projected_compare(self, func_name, args)

        if func_name in {"textureGatherCompare", "textureGatherCompareOffset"}:
            return self.generate_texture_gather_compare(func_name, args)

        if func_name in {"texelFetch", "texelFetchOffset"}:
            fetch_args = self.sampled_texture_operation_args(func_name, args)
            if fetch_args is None:
                return self.unsupported_sampled_texture_call(
                    func_name,
                    self.sampled_texture_operation_unsupported_reason(func_name, args),
                )
            texture_name, coord, extra_args = fetch_args

            coord_node = args[self.sampled_texture_coord_index(args)]
            coord_reason = self.texel_fetch_coordinate_rank_unsupported_reason(
                args[0], coord_node
            )
            if coord_reason:
                return self.unsupported_sampled_texture_call(func_name, coord_reason)
            fetch_index_reason = (
                self.scalar_integer_texture_argument_unsupported_reason(
                    extra_args[0], "fetch index argument"
                )
            )
            if fetch_index_reason:
                return self.unsupported_sampled_texture_call(
                    func_name, fetch_index_reason
                )
            if func_name == "texelFetchOffset":
                offset_reason = self.texel_fetch_offset_rank_unsupported_reason(
                    args[0], extra_args[1]
                )
                if offset_reason:
                    return self.unsupported_sampled_texture_call(
                        func_name, offset_reason
                    )
            texture_type = self.get_expression_type(args[0])
            result_type = self.sampled_texture_value_type(texture_type)
            expected_reason = self.texture_result_expected_type_unsupported_reason(
                func_name, result_type
            )
            if expected_reason:
                return self.unsupported_sampled_texture_call(func_name, expected_reason)
            lod_or_sample = self.generate_expression(extra_args[0])
            if self.is_multisample_sampler_type(texture_type):
                return f"{texture_name}[{coord}, {lod_or_sample}]"
            coord_constructor = self.texel_fetch_coord_constructor(texture_type)
            load_coord = f"{coord_constructor}({coord}, {lod_or_sample})"
            if func_name == "texelFetchOffset":
                offset = self.generate_expression(extra_args[1])
                return f"{texture_name}.Load({load_coord}, {offset})"
            return f"{texture_name}.Load({load_coord})"

        if func_name in {"textureSize", "imageSize"}:
            return self.generate_dimension_query(func_name, args)

        if func_name in {"textureSamples", "imageSamples"}:
            return self.generate_sample_count_query(func_name, args)

        if func_name == "textureQueryLevels":
            return self.generate_texture_query_levels(args)

        if func_name == "textureQueryLod":
            return self.generate_texture_query_lod(args)

        return None

    def generate_dimension_query(self, func_name, args):
        if not args:
            return self.unsupported_resource_query_call(
                func_name, "requires a resource argument"
            )

        resource_type = self.resource_base_type(self.get_expression_type(args[0]))
        if not self.dimension_query_accepts_resource(func_name, resource_type):
            return self.unsupported_resource_query_call(
                func_name, self.dimension_query_requirement(func_name)
            )

        spec = self.dimension_query_spec(resource_type)
        if spec is None:
            return self.unsupported_resource_query_call(
                func_name, self.dimension_query_requirement(func_name)
            )

        arity_reason = self.dimension_query_arity_unsupported_reason(
            func_name, args, spec
        )
        if arity_reason:
            return self.unsupported_resource_query_call(func_name, arity_reason)

        result_type = self.query_return_type(spec["dimensions"])
        expected_reason = self.resource_query_expected_type_unsupported_reason(
            func_name, result_type
        )
        if expected_reason:
            return self.unsupported_resource_query_call(
                func_name,
                expected_reason,
                self.zero_value_for_type(self.current_expression_expected_type),
            )

        resource_name = self.generate_expression(args[0])
        resource_slang_type = self.resource_query_slang_type(args[0], resource_type)
        base_helper_name = self.resource_query_helper_name(
            func_name, resource_type, resource_slang_type
        )
        helper_name = self.helper_function_name(base_helper_name)
        self.register_helper_function(
            helper_name,
            self.build_dimension_query_helper(
                helper_name, resource_type, spec, resource_slang_type
            ),
        )

        if spec["mip"]:
            lod = self.generate_expression(args[1]) if len(args) > 1 else "0"
            query = f"{helper_name}({resource_name}, {lod})"
        else:
            query = f"{helper_name}({resource_name})"
        return self.coerce_resource_query_result(result_type, query)

    def dimension_query_accepts_resource(self, func_name, resource_type):
        if func_name == "textureSize":
            return self.is_sampled_texture_resource_type(resource_type)
        if func_name == "imageSize":
            return self.is_storage_image_type(resource_type)
        return False

    def dimension_query_requirement(self, func_name):
        if func_name == "textureSize":
            return "requires a sampled texture resource"
        if func_name == "imageSize":
            return "requires an image resource"
        return "requires a resource"

    def dimension_query_arity_unsupported_reason(self, func_name, args, spec):
        if func_name == "imageSize":
            if len(args) != 1:
                return "accepts only a resource argument"
            return None

        if len(args) > 2:
            return "accepts resource and optional mip argument"
        if len(args) == 2:
            return self.scalar_texture_mip_unsupported_reason(args[1])
        return None

    def resource_query_expected_type_unsupported_reason(self, func_name, result_type):
        expected_type = self.convert_type(self.current_expression_expected_type)
        if not expected_type or expected_type == "auto":
            return None
        if not (
            self.is_scalar_value_type(expected_type)
            or self.is_vector_value_type(expected_type)
        ):
            return None
        if expected_type == result_type:
            return None
        if self.resource_query_result_cast_type(result_type) is not None:
            return None
        return f"returns {result_type} but target expects {expected_type}"

    def resource_query_result_cast_type(self, result_type):
        expected_type = self.convert_type(self.current_expression_expected_type)
        result_info = self.vector_value_info(result_type)
        expected_info = self.vector_value_info(expected_type)
        if result_info is None or expected_info is None:
            return None
        if result_info["component_type"] != "int":
            return None
        if expected_info["component_type"] not in {"half", "float", "double"}:
            return None
        if result_info["size"] != expected_info["size"]:
            return None
        return expected_info["type"]

    def coerce_resource_query_result(self, result_type, query):
        cast_type = self.resource_query_result_cast_type(result_type)
        if cast_type is None:
            return query
        return f"{cast_type}({query})"

    def generate_sample_count_query(self, func_name, args):
        if not args:
            return self.unsupported_resource_query_call(
                func_name, "requires a resource argument"
            )
        if len(args) != 1:
            return self.unsupported_resource_query_call(
                func_name, "accepts only a resource argument"
            )

        resource_type = self.resource_base_type(self.get_expression_type(args[0]))
        spec = self.dimension_query_spec(resource_type)
        if not self.sample_count_query_accepts_resource(func_name, resource_type):
            return self.unsupported_resource_query_call(
                func_name, self.sample_count_query_requirement(func_name)
            )
        if spec is None or not spec["samples"]:
            return self.unsupported_resource_query_call(
                func_name, self.sample_count_query_requirement(func_name)
            )

        expected_reason = self.resource_query_expected_type_unsupported_reason(
            func_name, "int"
        )
        if expected_reason:
            return self.unsupported_resource_query_call(
                func_name,
                expected_reason,
                self.zero_value_for_type(self.current_expression_expected_type),
            )

        resource_name = self.generate_expression(args[0])
        resource_slang_type = self.resource_query_slang_type(args[0], resource_type)
        base_helper_name = self.resource_query_helper_name(
            func_name, resource_type, resource_slang_type
        )
        helper_name = self.helper_function_name(base_helper_name)
        self.register_helper_function(
            helper_name,
            self.build_sample_count_query_helper(
                helper_name, resource_type, spec, resource_slang_type
            ),
        )
        return f"{helper_name}({resource_name})"

    def sampled_texture_args(self, args):
        coord_index = self.sampled_texture_coord_index(args)
        if len(args) <= coord_index:
            return None

        texture_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[coord_index])
        return texture_name, coord, args[coord_index + 1 :]

    def sampled_texture_coord_index(self, args):
        return 2 if self.is_explicit_sampler_argument(args) else 1

    def sampled_texture_operation_args(self, func_name, args):
        sample_args = self.sampled_texture_args(args)
        if sample_args is None:
            return None

        if not self.sampled_texture_operation_accepts_resource(func_name, args[0]):
            return None

        _texture_name, _coord, extra_args = sample_args
        if not self.sampled_texture_operation_accepts_extra_args(func_name, extra_args):
            return None

        return sample_args

    def sampled_texture_operation_accepts_extra_args(self, func_name, extra_args):
        if func_name == "texture":
            return len(extra_args) <= 1
        if func_name in {"textureLod", "texelFetch"}:
            return len(extra_args) == 1
        if func_name == "texelFetchOffset":
            return len(extra_args) == 2
        if func_name == "textureGrad":
            return len(extra_args) == 2
        return False

    def sampled_texture_operation_accepts_resource(self, func_name, texture_arg):
        resource_type = self.resource_base_type(self.get_expression_type(texture_arg))
        if resource_type is None:
            return True
        if func_name == "texelFetch":
            return self.is_texel_fetch_sampler_type(resource_type)
        if func_name == "texelFetchOffset":
            return self.is_texel_fetch_offset_sampler_type(resource_type)
        return self.sampled_texture_sampling_accepts_resource(texture_arg)

    def sampled_texture_sampling_accepts_resource(self, texture_arg):
        resource_type = self.resource_base_type(self.get_expression_type(texture_arg))
        if resource_type is None:
            return True
        return self.is_lod_query_sampler_type(resource_type)

    def sampled_texture_operation_unsupported_reason(self, func_name, args):
        if not args:
            return "requires texture and coordinate arguments"

        coord_index = self.sampled_texture_coord_index(args)
        if len(args) <= coord_index:
            return "requires texture and coordinate arguments"

        if not self.sampled_texture_operation_accepts_resource(func_name, args[0]):
            return self.sampled_texture_operation_resource_requirement(func_name)

        return self.sampled_texture_operation_arity_requirement(func_name)

    def sampled_texture_operation_resource_requirement(self, func_name):
        if func_name == "texelFetch":
            return "requires a non-shadow texel-fetchable sampled texture resource"
        if func_name == "texelFetchOffset":
            return (
                "requires an offset-capable non-shadow non-multisampled sampled "
                "texture resource"
            )
        return "requires a non-shadow non-multisampled sampled texture resource"

    def sampled_texture_operation_arity_requirement(self, func_name):
        if func_name == "texture":
            return "accepts coordinate and optional bias arguments"
        if func_name == "textureLod":
            return "requires one lod argument"
        if func_name == "textureGrad":
            return "requires gradient x and gradient y arguments"
        if func_name == "texelFetch":
            return "requires one lod/sample argument"
        if func_name == "texelFetchOffset":
            return "requires lod/sample and offset arguments"
        return "has unsupported arguments"

    def explicit_sampler_expression(self, args):
        if not self.is_explicit_sampler_argument(args):
            return None
        if not self.explicit_sampler_call_uses_separated_texture(args[0]):
            return None
        return self.generate_expression(args[1])

    def explicit_sampler_call_uses_separated_texture(self, texture_arg):
        texture_name = self.expression_root_identifier_name(texture_arg)
        if texture_name in self.explicit_sampler_texture_names:
            return True
        mapped_type = self.convert_type(self.get_expression_type(texture_arg))
        base_type = self.resource_base_type(mapped_type)
        return isinstance(base_type, str) and base_type.startswith("Texture")

    def slang_texture_method_args(self, args, *method_args):
        sampler = self.explicit_sampler_expression(args)
        if sampler is None:
            return list(method_args)
        return [sampler, *method_args]

    def slang_texture_method_call(self, texture_name, method, args, *method_args):
        rendered_args = ", ".join(self.slang_texture_method_args(args, *method_args))
        return f"{texture_name}.{method}({rendered_args})"

    def unsupported_sampled_texture_call(self, func_name, reason):
        fallback = self.texture_result_diagnostic_fallback("float4")
        return self.refuse_lowering(
            f"Slang sampled texture: {func_name} {reason}", fallback
        )
