"""Resource calls in the Mojo generator: size and level queries, texel fetches, image and buffer loads and stores, member functions, reinterpretation and image atomics."""

import re

from ..refusals import MojoRefusal
from .mojo_tables import (
    MOJO_BUFFER_COUNTER_METHODS,
    MOJO_BUFFER_LOAD_RESOURCE_TYPES,
    MOJO_BUFFER_OP_ALIASES,
    MOJO_BUFFER_STORE_RESOURCE_TYPES,
    MOJO_BYTE_ADDRESS_ATOMIC_METHODS,
    MOJO_BYTE_ADDRESS_BUFFER_TYPES,
    MOJO_BYTE_ADDRESS_LOAD_METHODS,
    MOJO_BYTE_ADDRESS_STORE_METHODS,
    MOJO_DTYPE_INFO,
    MOJO_REINTERPRET_ALIAS_MAP,
    MOJO_REINTERPRET_BUILTINS,
    MOJO_REINTERPRET_TARGET_TYPES,
    MOJO_RESOURCE_OPERATION_MEMBER_REQUIREMENTS,
    MOJO_RESOURCE_QUERY_MEMBER_REQUIREMENTS,
    MOJO_TYPED_BUFFER_LOAD_RESOURCE_TYPES,
)


class MojoResourceCallMixin:
    """Resource-call members of ``MojoCodeGen``."""

    def generate_resource_size_call(self, args, helper_name):
        if helper_name == "texture_size":
            required_kind = "texture"
            is_valid_kind = self.is_texture_resource_type
            expected_args = {1, 2}
        else:
            required_kind = "image"
            is_valid_kind = self.is_image_resource_type
            expected_args = {1}

        if not args:
            raise MojoRefusal(
                f"Unsupported {helper_name} for Mojo codegen; "
                f"{required_kind} resource required: <missing>"
            )

        resource_type = self.map_type(self.expression_result_type(args[0]))

        if not is_valid_kind(resource_type):
            raise MojoRefusal(
                f"Unsupported {helper_name} for Mojo codegen; "
                f"{required_kind} resource required: {resource_type}"
            )
        if len(args) not in expected_args:
            expected_counts = sorted(count - 1 for count in expected_args)
            expected_text = " or ".join(str(count) for count in expected_counts)
            raise MojoRefusal(
                f"Unsupported {helper_name} for Mojo codegen; expected "
                f"{expected_text} argument(s) after {required_kind} resource, "
                f"got {len(args) - 1}"
            )
        if len(args) == 2:
            self.validate_resource_argument_type(
                args[1], "Int32", helper_name, resource_type, "lod"
            )

        if self.resource_size_return_type(resource_type) is None:
            raise MojoRefusal(
                f"Unsupported {helper_name} for Mojo codegen; "
                f"sizable {required_kind} resource required: {resource_type}"
            )
        self.required_resource_size_types.add(resource_type)
        self.required_resource_size_helpers.add((helper_name, resource_type))

        generated_args = ", ".join(self.generate_expression(arg) for arg in args)
        return f"{helper_name}({generated_args})"

    def generate_resource_query_levels_call(self, args):
        if not args:
            raise MojoRefusal(
                "Unsupported texture_query_levels for Mojo codegen; "
                "texture resource required: <missing>"
            )
        if len(args) != 1:
            raise MojoRefusal(
                "Unsupported texture_query_levels for Mojo codegen; expected "
                f"0 argument(s) after texture resource, got {len(args) - 1}"
            )

        resource_type = self.map_type(self.expression_result_type(args[0]))
        if not self.is_texture_resource_type(resource_type):
            raise MojoRefusal(
                "Unsupported texture_query_levels for Mojo codegen; "
                f"texture resource required: {resource_type}"
            )
        if self.is_multisample_resource_type(resource_type):
            raise MojoRefusal(
                "Unsupported texture_query_levels for Mojo codegen; "
                f"non-multisample texture required: {resource_type}"
            )
        self.required_resource_query_level_types.add(resource_type)

        generated_args = ", ".join(self.generate_expression(arg) for arg in args)
        return f"texture_query_levels({generated_args})"

    def generate_resource_sample_count_call(self, args, helper_name):
        if not args:
            required_kind = "texture" if helper_name == "texture_samples" else "image"
            raise MojoRefusal(
                f"Unsupported {helper_name} for Mojo codegen; "
                f"{required_kind} resource required: <missing>"
            )
        if len(args) != 1:
            required_kind = "texture" if helper_name == "texture_samples" else "image"
            raise MojoRefusal(
                f"Unsupported {helper_name} for Mojo codegen; expected "
                f"0 argument(s) after {required_kind} resource, got {len(args) - 1}"
            )

        resource_type = self.map_type(self.expression_result_type(args[0]))
        if helper_name == "texture_samples":
            required_kind = "texture"
            is_valid_kind = self.is_texture_resource_type(resource_type)
        else:
            required_kind = "image"
            is_valid_kind = self.is_image_resource_type(resource_type)

        if not is_valid_kind:
            raise MojoRefusal(
                f"Unsupported {helper_name} for Mojo codegen; "
                f"{required_kind} resource required: {resource_type}"
            )
        if not self.is_multisample_resource_type(resource_type):
            raise MojoRefusal(
                f"Unsupported {helper_name} for Mojo codegen; "
                f"multisample {required_kind} required: {resource_type}"
            )

        self.required_resource_sample_count_helpers.add((helper_name, resource_type))
        generated_args = ", ".join(self.generate_expression(arg) for arg in args)
        return f"{helper_name}({generated_args})"

    def generate_texel_fetch_call(self, args, *, with_offset=False):
        helper_name = "texel_fetch_offset" if with_offset else "texel_fetch"
        if args:
            resource_type = self.map_type(self.expression_result_type(args[0]))
            self.validate_texel_fetch_args(args, resource_type, with_offset=with_offset)
            if (
                with_offset
                and self.resource_offset_coord_type(resource_type) is not None
            ):
                self.required_resource_texel_fetch_offset_types.add(resource_type)
            elif self.resource_texel_coord_type(resource_type) is not None:
                self.required_resource_texel_fetch_types.add(resource_type)

        generated_args = ", ".join(self.generate_expression(arg) for arg in args)
        return f"{helper_name}({generated_args})"

    def validate_texel_fetch_args(self, args, resource_type, *, with_offset=False):
        helper_name = "texel_fetch_offset" if with_offset else "texel_fetch"
        if not self.is_texture_resource_type(resource_type):
            raise MojoRefusal(
                f"Unsupported {helper_name} for Mojo codegen; "
                f"texture resource required: {resource_type}"
            )
        if self.is_shadow_resource_type(resource_type):
            raise MojoRefusal(
                f"Unsupported {helper_name} for Mojo codegen; "
                f"non-shadow texture required: {resource_type}"
            )
        if self.resource_texel_coord_type(resource_type) is None or (
            with_offset and self.resource_offset_coord_type(resource_type) is None
        ):
            raise MojoRefusal(
                f"Unsupported {helper_name} for Mojo codegen; "
                f"texel-fetch{'-offset' if with_offset else ''} texture required: "
                f"{resource_type}"
            )
        expected_count = 4 if with_offset else 3
        if len(args) != expected_count:
            raise MojoRefusal(
                f"Unsupported {helper_name} for Mojo codegen; expected "
                f"{expected_count - 1} argument(s) after texture resource, "
                f"got {len(args) - 1}"
            )

        self.validate_resource_argument_type(
            args[1],
            self.resource_texel_coord_type(resource_type),
            helper_name,
            resource_type,
            "coordinate",
        )
        self.validate_resource_argument_type(
            args[2], "Int32", helper_name, resource_type, "lod_or_sample"
        )
        if with_offset:
            self.validate_resource_argument_type(
                args[3],
                self.resource_offset_coord_type(resource_type),
                helper_name,
                resource_type,
                "offset",
            )

    def generate_image_load_call(self, args):
        if args:
            self.validate_resource_read_access(args[0], "imageLoad")
            resource_type = self.map_type(self.expression_result_type(args[0]))
            self.validate_image_load_args(args, resource_type)
            if self.resource_texel_coord_type(resource_type) is not None:
                self.required_image_load_types.add(resource_type)

        generated_args = ", ".join(self.generate_expression(arg) for arg in args)
        return f"image_load({generated_args})"

    def generate_image_store_call(self, args):
        if args:
            self.validate_resource_write_access(args[0], "imageStore")
            resource_type = self.map_type(self.expression_result_type(args[0]))
            self.validate_image_store_args(args, resource_type)
            if self.resource_texel_coord_type(resource_type) is not None:
                self.required_image_store_types.add(resource_type)

        generated_args = ", ".join(self.generate_expression(arg) for arg in args)
        return f"image_store({generated_args})"

    def validate_image_load_args(self, args, resource_type):
        if not self.is_image_resource_type(resource_type):
            raise MojoRefusal(
                "Unsupported image_load for Mojo codegen; "
                f"image resource required: {resource_type}"
            )
        coord_type = self.resource_texel_coord_type(resource_type)
        if coord_type is None:
            raise MojoRefusal(
                "Unsupported image_load for Mojo codegen; "
                f"loadable image resource required: {resource_type}"
            )

        expected_count = 3 if self.is_multisample_resource_type(resource_type) else 2
        if len(args) != expected_count:
            raise MojoRefusal(
                "Unsupported image_load for Mojo codegen; expected "
                f"{expected_count - 1} argument(s) after image resource, got "
                f"{len(args) - 1}"
            )

        self.validate_resource_argument_type(
            args[1],
            coord_type,
            "image_load",
            resource_type,
            "coordinate",
        )
        if self.is_multisample_resource_type(resource_type):
            self.validate_resource_argument_type(
                args[2], "Int32", "image_load", resource_type, "sample"
            )

    def validate_image_store_args(self, args, resource_type):
        if not self.is_image_resource_type(resource_type):
            raise MojoRefusal(
                "Unsupported image_store for Mojo codegen; "
                f"image resource required: {resource_type}"
            )
        coord_type = self.resource_texel_coord_type(resource_type)
        if coord_type is None:
            raise MojoRefusal(
                "Unsupported image_store for Mojo codegen; "
                f"storable image resource required: {resource_type}"
            )

        expected_count = 4 if self.is_multisample_resource_type(resource_type) else 3
        if len(args) != expected_count:
            raise MojoRefusal(
                "Unsupported image_store for Mojo codegen; expected "
                f"{expected_count - 1} argument(s) after image resource, got "
                f"{len(args) - 1}"
            )

        self.validate_resource_argument_type(
            args[1],
            coord_type,
            "image_store",
            resource_type,
            "coordinate",
        )
        value_index = 2
        if self.is_multisample_resource_type(resource_type):
            self.validate_resource_argument_type(
                args[2], "Int32", "image_store", resource_type, "sample"
            )
            value_index = 3
        self.validate_image_store_value_type(args[value_index], resource_type)

    def validate_image_store_value_type(self, value_arg, resource_type):
        expected_type = self.image_value_type(resource_type)
        actual_type = self.resource_builtin_arg_type(value_arg)
        if actual_type != expected_type:
            raise MojoRefusal(
                "Unsupported image_store for Mojo codegen; value for "
                f"{resource_type} must be {expected_type}, got {actual_type}"
            )

    def generate_buffer_load_call(self, args):
        if args:
            self.validate_resource_read_access(args[0], "buffer_load")
            info = self.buffer_resource_info(self.expression_result_type(args[0]))
            self.validate_buffer_operation(
                "buffer_load", info, MOJO_BUFFER_LOAD_RESOURCE_TYPES
            )
            if info is not None and self.buffer_helper_element_type(*info) is not None:
                self.register_buffer_resource_type(info[0])
                self.required_buffer_load_helpers.add(info)

        generated_args = ", ".join(self.generate_expression(arg) for arg in args)
        return f"buffer_load({generated_args})"

    def generate_buffer_store_call(self, args):
        if args:
            self.validate_resource_write_access(args[0], "buffer_store")
            info = self.buffer_resource_info(self.expression_result_type(args[0]))
            self.validate_buffer_operation(
                "buffer_store", info, MOJO_BUFFER_STORE_RESOURCE_TYPES
            )
            if info is not None and self.buffer_helper_element_type(*info) is not None:
                self.register_buffer_resource_type(info[0])
                self.required_buffer_store_helpers.add(info)

        generated_args = ", ".join(self.generate_expression(arg) for arg in args)
        return f"buffer_store({generated_args})"

    def generate_buffer_append_call(self, args):
        if args:
            self.validate_resource_write_access(args[0], "buffer_append")
            info = self.buffer_resource_info(self.expression_result_type(args[0]))
            if info is not None and info[0] != "AppendStructuredBuffer":
                raise MojoRefusal(
                    "Unsupported buffer_append for Mojo codegen; "
                    f"{info[0]} is not an append buffer"
                )
            if info is not None and info[1] is not None:
                self.register_buffer_resource_type(info[0])
                self.required_buffer_append_helpers.add(info)

        generated_args = ", ".join(self.generate_expression(arg) for arg in args)
        return f"buffer_append({generated_args})"

    def generate_buffer_consume_call(self, args):
        if args:
            self.validate_resource_read_access(args[0], "buffer_consume")
            info = self.buffer_resource_info(self.expression_result_type(args[0]))
            if info is not None and info[0] != "ConsumeStructuredBuffer":
                raise MojoRefusal(
                    "Unsupported buffer_consume for Mojo codegen; "
                    f"{info[0]} is not a consume buffer"
                )
            if info is not None and info[1] is not None:
                self.register_buffer_resource_type(info[0])
                self.required_buffer_consume_helpers.add(info)

        generated_args = ", ".join(self.generate_expression(arg) for arg in args)
        return f"buffer_consume({generated_args})"

    def generate_buffer_dimensions_call(self, args):
        if args:
            info = self.buffer_resource_info(self.expression_result_type(args[0]))
            if info is not None:
                if len(args) not in {2, 3}:
                    raise MojoRefusal(
                        "Unsupported buffer_dimensions for Mojo codegen; "
                        f"expected 1 or 2 dimension argument(s), got {len(args) - 1}"
                    )
                self.register_buffer_resource_type(info[0])
                self.required_buffer_dimensions_helpers.add(info)

        generated_args = ", ".join(self.generate_expression(arg) for arg in args)
        return f"buffer_dimensions({generated_args})"

    def generate_buffer_op_node(self, expr):
        raw_operation = getattr(expr, "operation", "")
        operation = MOJO_BUFFER_OP_ALIASES.get(raw_operation)
        if operation is None:
            raise MojoRefusal(f"Unsupported Mojo buffer operation {raw_operation}")

        buffer_expr = getattr(expr, "buffer_expr", None)
        if buffer_expr is None:
            raise MojoRefusal(
                f"Invalid Mojo buffer operation {raw_operation}; "
                "missing buffer expression"
            )

        arguments = getattr(expr, "arguments", [])
        args = [buffer_expr, *arguments]
        if operation == "Load":
            return self.generate_buffer_load_call(args)
        if operation == "Store":
            return self.generate_buffer_store_call(args)
        if operation == "Append":
            return self.generate_buffer_append_call(args)
        if operation == "Consume":
            return self.generate_buffer_consume_call(args)
        if operation == "GetDimensions":
            return self.generate_buffer_dimensions_call(args)
        if operation in MOJO_BYTE_ADDRESS_LOAD_METHODS:
            return self.generate_byte_address_vector_buffer_op(
                buffer_expr, arguments, operation
            )
        if operation in MOJO_BYTE_ADDRESS_STORE_METHODS:
            return self.generate_byte_address_vector_buffer_op(
                buffer_expr, arguments, operation
            )
        raise MojoRefusal(f"Invalid Mojo buffer operation {raw_operation}")

    def generate_byte_address_vector_buffer_op(self, buffer_expr, args, operation):
        info = self.buffer_resource_info(self.expression_result_type(buffer_expr))
        if info is None or info[0] not in MOJO_BYTE_ADDRESS_BUFFER_TYPES:
            raise MojoRefusal(
                f"Invalid {operation} for Mojo codegen; byte-address buffer required"
            )

        obj = self.generate_expression(buffer_expr)
        buffer_type, _ = info
        width = (
            MOJO_BYTE_ADDRESS_LOAD_METHODS.get(operation)
            or MOJO_BYTE_ADDRESS_STORE_METHODS[operation]
        )
        self.register_buffer_resource_type(buffer_type)
        if operation in MOJO_BYTE_ADDRESS_LOAD_METHODS:
            self.validate_resource_read_access(buffer_expr, operation)
            self.required_byte_address_vector_load_helpers.add((buffer_type, width))
            generated_args = ", ".join(
                [obj, *[self.generate_expression(arg) for arg in args]]
            )
            return f"buffer_load{width}({generated_args})"

        self.validate_resource_write_access(buffer_expr, operation)
        self.validate_buffer_operation(
            operation, info, MOJO_BUFFER_STORE_RESOURCE_TYPES
        )
        self.required_byte_address_vector_store_helpers.add((buffer_type, width))
        generated_args = ", ".join(
            [obj, *[self.generate_expression(arg) for arg in args]]
        )
        return f"buffer_store{width}({generated_args})"

    def generate_member_function_call(self, func_expr, args):
        texture_call = self.generate_texture_member_function_call(func_expr, args)
        if texture_call is not None:
            return texture_call

        member = getattr(func_expr, "member", None)
        obj_expr = getattr(func_expr, "object", getattr(func_expr, "object_expr", None))
        obj_type = self.expression_result_type(obj_expr)
        info = self.buffer_resource_info(obj_type)
        if info is None:
            self.reject_invalid_resource_query_member_receiver(member, obj_type)
            self.reject_invalid_resource_operation_member_receiver(member, obj_type)
            return None

        obj = self.generate_expression(obj_expr)
        buffer_type, element_type = info

        if buffer_type in MOJO_BYTE_ADDRESS_BUFFER_TYPES:
            if member in MOJO_BYTE_ADDRESS_LOAD_METHODS:
                self.validate_resource_read_access(obj_expr, member)
                width = MOJO_BYTE_ADDRESS_LOAD_METHODS[member]
                self.register_buffer_resource_type(buffer_type)
                if width == 1:
                    self.required_buffer_load_helpers.add(info)
                    generated_args = ", ".join(
                        [obj, *[self.generate_expression(arg) for arg in args]]
                    )
                    return f"buffer_load({generated_args})"
                self.required_byte_address_vector_load_helpers.add((buffer_type, width))
                generated_args = ", ".join(
                    [obj, *[self.generate_expression(arg) for arg in args]]
                )
                return f"buffer_load{width}({generated_args})"

            if member in MOJO_BYTE_ADDRESS_STORE_METHODS:
                self.validate_resource_write_access(obj_expr, member)
                self.validate_buffer_operation(
                    member, info, MOJO_BUFFER_STORE_RESOURCE_TYPES
                )
                width = MOJO_BYTE_ADDRESS_STORE_METHODS[member]
                self.register_buffer_resource_type(buffer_type)
                if width == 1:
                    self.required_buffer_store_helpers.add(info)
                    generated_args = ", ".join(
                        [obj, *[self.generate_expression(arg) for arg in args]]
                    )
                    return f"buffer_store({generated_args})"
                self.required_byte_address_vector_store_helpers.add(
                    (buffer_type, width)
                )
                generated_args = ", ".join(
                    [obj, *[self.generate_expression(arg) for arg in args]]
                )
                return f"buffer_store{width}({generated_args})"

            if member in MOJO_BYTE_ADDRESS_ATOMIC_METHODS:
                self.validate_resource_read_write_access(obj_expr, member)
                raise MojoRefusal(
                    "Unsupported Mojo byte-address buffer atomic method "
                    f"{member}; byte-address atomics are not available in "
                    "the Mojo placeholder backend"
                )

        if member == "Load" and element_type is not None:
            self.validate_resource_read_access(obj_expr, "Load")
            self.validate_buffer_operation(
                "Load", info, MOJO_TYPED_BUFFER_LOAD_RESOURCE_TYPES
            )
            self.register_buffer_resource_type(buffer_type)
            self.required_buffer_load_helpers.add(info)
            generated_args = ", ".join(
                [obj, *[self.generate_expression(arg) for arg in args]]
            )
            return f"buffer_load({generated_args})"

        if member == "Store" and element_type is not None:
            self.validate_resource_write_access(obj_expr, "Store")
            self.validate_buffer_operation(
                "Store", info, MOJO_BUFFER_STORE_RESOURCE_TYPES
            )
            self.register_buffer_resource_type(buffer_type)
            self.required_buffer_store_helpers.add(info)
            generated_args = ", ".join(
                [obj, *[self.generate_expression(arg) for arg in args]]
            )
            return f"buffer_store({generated_args})"

        if member == "Append":
            if buffer_type != "AppendStructuredBuffer":
                raise MojoRefusal(
                    "Unsupported Append for Mojo codegen; "
                    f"{buffer_type} is not an append buffer"
                )
            self.validate_resource_write_access(obj_expr, "Append")
            self.register_buffer_resource_type(buffer_type)
            self.required_buffer_append_helpers.add(info)
            generated_args = ", ".join(
                [obj, *[self.generate_expression(arg) for arg in args]]
            )
            return f"buffer_append({generated_args})"

        if member == "Consume":
            if buffer_type != "ConsumeStructuredBuffer":
                raise MojoRefusal(
                    "Unsupported Consume for Mojo codegen; "
                    f"{buffer_type} is not a consume buffer"
                )
            self.validate_resource_read_access(obj_expr, "Consume")
            self.register_buffer_resource_type(buffer_type)
            self.required_buffer_consume_helpers.add(info)
            return f"buffer_consume({obj})"

        if member == "GetDimensions":
            self.register_buffer_resource_type(buffer_type)
            self.required_buffer_dimensions_helpers.add(info)
            generated_args = ", ".join(
                [obj, *[self.generate_expression(arg) for arg in args]]
            )
            return f"buffer_dimensions({generated_args})"

        if member in MOJO_BUFFER_COUNTER_METHODS:
            self.validate_resource_read_write_access(obj_expr, member)
            raise MojoRefusal(
                "Unsupported Mojo buffer counter method "
                f"{member}; structured-buffer counters are not available "
                "in the Mojo placeholder backend"
            )

        return None

    def reject_invalid_resource_query_member_receiver(self, member, obj_type):
        requirement = MOJO_RESOURCE_QUERY_MEMBER_REQUIREMENTS.get(member)
        if requirement is None:
            return

        helper_name, required_receiver = requirement
        mapped_type = self.map_type(obj_type) if obj_type is not None else "<unknown>"
        raise MojoRefusal(
            f"Unsupported {helper_name} for Mojo codegen; "
            f"{required_receiver} required: {mapped_type}"
        )

    def reject_invalid_resource_operation_member_receiver(self, member, obj_type):
        requirement = MOJO_RESOURCE_OPERATION_MEMBER_REQUIREMENTS.get(member)
        if requirement is None:
            return

        helper_name, required_receiver = requirement
        mapped_type = self.map_type(obj_type) if obj_type is not None else "<unknown>"
        raise MojoRefusal(
            f"Unsupported {helper_name} for Mojo codegen; "
            f"{required_receiver} required: {mapped_type}"
        )

    def generate_texture_member_function_call(self, func_expr, args):
        member = getattr(func_expr, "member", None)
        obj_expr = getattr(func_expr, "object", getattr(func_expr, "object_expr", None))
        resource_type = self.map_type(self.expression_result_type(obj_expr))
        if not (
            self.is_texture_resource_type(resource_type)
            or self.is_image_resource_type(resource_type)
        ):
            return None

        call_args = [obj_expr, *args]
        stripped_arg_count = max(len(self.strip_split_sampler_arg(call_args)) - 1, 0)
        return self.generate_texture_operation_call(
            member, call_args, stripped_arg_count, fetch_args=[obj_expr, *args]
        )

    def generate_reinterpret_call(self, func_name, args):
        if not args:
            return f"{func_name}()"

        source_type = self.expression_result_type(args[0])
        return_type_name = self.reinterpret_return_type_name(func_name, source_type)
        return_type = self.map_type(return_type_name)
        arg_type = self.reinterpret_argument_type(source_type)
        self.required_reinterpret_helpers.add((func_name, arg_type, return_type))
        generated_args = ", ".join(self.generate_expression(arg) for arg in args)
        return f"{func_name}({generated_args})"

    def mojo_reinterpret_builtin_name(self, func_name):
        if not func_name or self.is_user_defined_function(func_name):
            return None
        if func_name in MOJO_REINTERPRET_BUILTINS:
            return func_name
        return MOJO_REINTERPRET_ALIAS_MAP.get(func_name)

    def reinterpret_argument_type(self, source_type):
        if source_type is None:
            return "UInt32"
        arg_type = self.map_type(source_type)
        if arg_type in {"Float32", "Int32", "UInt32"}:
            return arg_type
        if self.vector_type_info(source_type) is not None:
            return arg_type
        return "UInt32"

    def reinterpret_return_type_name(self, func_name, source_type):
        scalar_type, vector_prefix = MOJO_REINTERPRET_TARGET_TYPES[func_name]
        vector_info = self.vector_type_info(source_type)
        if vector_info is not None:
            return f"{vector_prefix}{vector_info[1]}"
        return scalar_type

    def generate_resource_builtin_call(self, args, helper_base, return_kind):
        if not args:
            return f"{helper_base}()"

        validation_args, sampler_arg = self.split_sampler_arg(args)
        self.validate_texture_builtin_resource(validation_args, helper_base)
        helper_base = (
            f"{helper_base}_sampler" if sampler_arg is not None else helper_base
        )
        arg_types = tuple(self.resource_builtin_arg_type(arg) for arg in args)
        helper_name = self.resource_builtin_helper_name(helper_base, arg_types)
        self.required_resource_builtin_helpers[(helper_name, arg_types)] = {
            "name": helper_name,
            "arg_types": arg_types,
            "return_type": self.resource_builtin_return_type(return_kind, args),
        }

        generated_args = ", ".join(self.generate_expression(arg) for arg in args)
        return f"{helper_name}({generated_args})"

    def validate_texture_builtin_resource(self, args, helper_base):
        if not args:
            return

        resource_type = self.map_type(self.expression_result_type(args[0]))
        if not self.is_texture_resource_type(resource_type):
            raise MojoRefusal(
                f"Unsupported {helper_base} for Mojo codegen; "
                f"texture resource required: {resource_type}"
            )
        if self.is_multisample_resource_type(resource_type):
            raise MojoRefusal(
                f"Unsupported {helper_base} for Mojo codegen; "
                f"non-multisample texture required: {resource_type}"
            )
        if (
            helper_base.startswith("texture_compare")
            or helper_base.startswith("texture_gather_compare")
        ) and not self.is_shadow_resource_type(resource_type):
            raise MojoRefusal(
                f"Unsupported {helper_base} for Mojo codegen; "
                f"shadow texture required: {resource_type}"
            )
        if helper_base in {
            "texture_query_lod",
            "calculate_level_of_detail",
            "calculate_level_of_detail_unclamped",
        } and self.is_shadow_resource_type(resource_type):
            raise MojoRefusal(
                f"Unsupported {helper_base} for Mojo codegen; "
                f"non-shadow texture required: {resource_type}"
            )
        self.validate_texture_builtin_args(args, helper_base, resource_type)

    def validate_texture_builtin_args(self, args, helper_base, resource_type):
        argument_specs = self.texture_builtin_argument_specs(helper_base, resource_type)
        if argument_specs is None:
            return

        allowed_counts = {
            len(specs)
            for specs in argument_specs
            if self.texture_arg_specs_known(specs)
        }
        if allowed_counts and len(args) - 1 not in allowed_counts:
            expected = " or ".join(str(count) for count in sorted(allowed_counts))
            raise MojoRefusal(
                f"Unsupported {helper_base} for Mojo codegen; expected {expected} "
                f"argument(s) after texture resource, got {len(args) - 1}"
            )

        for specs in argument_specs:
            if len(args) - 1 != len(specs):
                continue
            if not self.texture_arg_specs_known(specs):
                return
            for index, (label, expected_type) in enumerate(specs, start=1):
                self.validate_resource_argument_type(
                    args[index], expected_type, helper_base, resource_type, label
                )
            return

    def texture_builtin_argument_specs(self, helper_base, resource_type):
        sample_coord = self.resource_sample_coord_type(resource_type)
        compare_coord = self.resource_compare_coord_type(resource_type)
        offset_coord = self.resource_offset_coord_type(resource_type)
        projected_coord = self.resource_projected_coord_type(resource_type)

        if helper_base in {
            "sample_offset",
            "sample_lod_offset",
            "sample_grad_offset",
        } and None in {sample_coord, offset_coord}:
            return None
        if helper_base in {"sample_proj", "sample_proj_offset"} and (
            projected_coord is None
        ):
            return None
        if (
            helper_base in {
                "sample_proj_lod",
                "sample_proj_lod_offset",
                "sample_proj_grad",
                "sample_proj_grad_offset",
            }
            and projected_coord is None
        ):
            return None
        if (
            helper_base in {
                "texture_compare",
                "texture_compare_offset",
                "texture_compare_lod",
                "texture_compare_lod_offset",
                "texture_compare_grad",
                "texture_compare_grad_offset",
                "texture_compare_proj",
                "texture_compare_proj_offset",
                "texture_compare_proj_lod",
                "texture_compare_proj_lod_offset",
                "texture_compare_proj_grad",
                "texture_compare_proj_grad_offset",
                "texture_gather_compare",
                "texture_gather_compare_offset",
            }
            and compare_coord is None
        ):
            return None

        specs = {
            "sample_offset": [[("coordinate", sample_coord), ("offset", offset_coord)]],
            "sample_lod_offset": [
                [
                    ("coordinate", sample_coord),
                    ("lod", "Float32"),
                    ("offset", offset_coord),
                ]
            ],
            "sample_grad_offset": [
                [
                    ("coordinate", sample_coord),
                    ("ddx", sample_coord),
                    ("ddy", sample_coord),
                    ("offset", offset_coord),
                ]
            ],
            "sample_bias": [[("coordinate", sample_coord), ("bias", "Float32")]],
            "sample_bias_offset": [
                [
                    ("coordinate", sample_coord),
                    ("bias", "Float32"),
                    ("offset", offset_coord),
                ]
            ],
            "sample_proj": [[("coordinate", projected_coord)]],
            "sample_proj_offset": [
                [("coordinate", projected_coord), ("offset", offset_coord)]
            ],
            "sample_proj_lod": [[("coordinate", projected_coord), ("lod", "Float32")]],
            "sample_proj_lod_offset": [
                [
                    ("coordinate", projected_coord),
                    ("lod", "Float32"),
                    ("offset", offset_coord),
                ]
            ],
            "sample_proj_grad": [
                [
                    ("coordinate", projected_coord),
                    ("ddx", sample_coord),
                    ("ddy", sample_coord),
                ]
            ],
            "sample_proj_grad_offset": [
                [
                    ("coordinate", projected_coord),
                    ("ddx", sample_coord),
                    ("ddy", sample_coord),
                    ("offset", offset_coord),
                ]
            ],
            "texture_query_lod": [[("coordinate", sample_coord)]],
            "calculate_level_of_detail": [[("coordinate", sample_coord)]],
            "calculate_level_of_detail_unclamped": [[("coordinate", sample_coord)]],
            "texture_gather": [
                [("coordinate", sample_coord)],
                [("coordinate", sample_coord), ("component", "Int32")],
            ],
            "texture_gather_offset": [
                [("coordinate", sample_coord), ("offset", offset_coord)],
                [
                    ("coordinate", sample_coord),
                    ("offset", offset_coord),
                    ("component", "Int32"),
                ],
            ],
            "texture_gather_offsets": [
                [
                    ("coordinate", sample_coord),
                    ("offset0", offset_coord),
                    ("offset1", offset_coord),
                    ("offset2", offset_coord),
                    ("offset3", offset_coord),
                ],
                [
                    ("coordinate", sample_coord),
                    ("offset0", offset_coord),
                    ("offset1", offset_coord),
                    ("offset2", offset_coord),
                    ("offset3", offset_coord),
                    ("component", "Int32"),
                ],
            ],
            "texture_compare": [[("coordinate", compare_coord), ("depth", "Float32")]],
            "texture_compare_offset": [
                [
                    ("coordinate", compare_coord),
                    ("depth", "Float32"),
                    ("offset", offset_coord),
                ]
            ],
            "texture_compare_lod": [
                [
                    ("coordinate", compare_coord),
                    ("depth", "Float32"),
                    ("lod", "Float32"),
                ]
            ],
            "texture_compare_lod_offset": [
                [
                    ("coordinate", compare_coord),
                    ("depth", "Float32"),
                    ("lod", "Float32"),
                    ("offset", offset_coord),
                ]
            ],
            "texture_compare_grad": [
                [
                    ("coordinate", compare_coord),
                    ("depth", "Float32"),
                    ("ddx", compare_coord),
                    ("ddy", compare_coord),
                ]
            ],
            "texture_compare_grad_offset": [
                [
                    ("coordinate", compare_coord),
                    ("depth", "Float32"),
                    ("ddx", compare_coord),
                    ("ddy", compare_coord),
                    ("offset", offset_coord),
                ]
            ],
            "texture_compare_proj": [
                [("coordinate", compare_coord), ("depth", "Float32")]
            ],
            "texture_compare_proj_offset": [
                [
                    ("coordinate", compare_coord),
                    ("depth", "Float32"),
                    ("offset", offset_coord),
                ]
            ],
            "texture_compare_proj_lod": [
                [
                    ("coordinate", compare_coord),
                    ("depth", "Float32"),
                    ("lod", "Float32"),
                ]
            ],
            "texture_compare_proj_lod_offset": [
                [
                    ("coordinate", compare_coord),
                    ("depth", "Float32"),
                    ("lod", "Float32"),
                    ("offset", offset_coord),
                ]
            ],
            "texture_compare_proj_grad": [
                [
                    ("coordinate", compare_coord),
                    ("depth", "Float32"),
                    ("ddx", compare_coord),
                    ("ddy", compare_coord),
                ]
            ],
            "texture_compare_proj_grad_offset": [
                [
                    ("coordinate", compare_coord),
                    ("depth", "Float32"),
                    ("ddx", compare_coord),
                    ("ddy", compare_coord),
                    ("offset", offset_coord),
                ]
            ],
            "texture_gather_compare": [
                [("coordinate", compare_coord), ("depth", "Float32")]
            ],
            "texture_gather_compare_offset": [
                [
                    ("coordinate", compare_coord),
                    ("depth", "Float32"),
                    ("offset", offset_coord),
                ]
            ],
        }
        return specs.get(helper_base)

    def texture_arg_specs_known(self, specs):
        return all(expected_type is not None for _, expected_type in specs)

    def generate_image_atomic_call(self, args, helper_base):
        if not args:
            return f"{helper_base}()"

        self.validate_resource_read_write_access(args[0], helper_base)
        resource_type = self.map_type(self.expression_result_type(args[0]))
        if not self.is_image_resource_type(resource_type):
            raise MojoRefusal(
                f"Unsupported {helper_base} for Mojo codegen; "
                f"image resource required: {resource_type}"
            )
        return_type = self.image_atomic_value_type(resource_type)
        if return_type not in {"Int32", "UInt32"}:
            raise MojoRefusal(
                "Unsupported image atomic for Mojo codegen; "
                f"scalar integer image required: {resource_type}"
            )
        if self.typed_image_atomic_has_vector_value_arg(
            args, helper_base, resource_type
        ):
            raise MojoRefusal(
                "Unsup"
                "ported image atomic for Mojo codegen; "
                f"scalar integer image required: {resource_type}"
            )
        self.validate_image_atomic_args(args, helper_base, resource_type, return_type)

        arg_types = tuple(self.resource_builtin_arg_type(arg) for arg in args)
        helper_name = self.resource_builtin_helper_name(helper_base, arg_types)
        self.required_resource_builtin_helpers[(helper_name, arg_types)] = {
            "name": helper_name,
            "arg_types": arg_types,
            "return_type": return_type,
        }

        generated_args = ", ".join(self.generate_expression(arg) for arg in args)
        return f"{helper_name}({generated_args})"

    def validate_image_atomic_args(self, args, helper_base, resource_type, value_type):
        coord_type = self.resource_texel_coord_type(resource_type)
        if coord_type is None:
            raise MojoRefusal(
                f"Unsupported {helper_base} for Mojo codegen; "
                f"atomic image resource required: {resource_type}"
            )

        value_count = 2 if helper_base == "image_atomic_comp_swap" else 1
        sample_count = 1 if self.is_multisample_resource_type(resource_type) else 0
        expected_count = 2 + sample_count + value_count
        if len(args) != expected_count:
            raise MojoRefusal(
                f"Unsupported {helper_base} for Mojo codegen; expected "
                f"{expected_count - 1} argument(s) after image resource, "
                f"got {len(args) - 1}"
            )

        self.validate_resource_argument_type(
            args[1], coord_type, helper_base, resource_type, "coordinate"
        )
        value_index = 2
        if sample_count:
            self.validate_resource_argument_type(
                args[2], "Int32", helper_base, resource_type, "sample"
            )
            value_index = 3

        labels = ("compare", "value") if value_count == 2 else ("value",)
        for label, arg in zip(labels, args[value_index:]):
            self.validate_resource_argument_type(
                arg, value_type, helper_base, resource_type, label
            )

    def typed_image_atomic_has_vector_value_arg(self, args, helper_base, resource_type):
        typed_info = self.typed_image_resource_info(resource_type)
        if typed_info is None:
            return False
        _, dtype, source_width = typed_info
        if source_width != 4 or dtype not in {"DType.int32", "DType.uint32"}:
            return False
        value_count = 2 if helper_base == "image_atomic_comp_swap" else 1
        sample_count = 1 if self.is_multisample_resource_type(resource_type) else 0
        value_index = 2 + sample_count
        for arg in args[value_index : value_index + value_count]:
            if self.resource_builtin_arg_type(arg) not in {"Int32", "UInt32"}:
                return True
        return False

    def resource_builtin_arg_type(self, arg):
        if isinstance(arg, bool):
            return "Bool"
        if isinstance(arg, int):
            return "Int32"
        if isinstance(arg, float):
            return "Float32"

        type_name = self.expression_result_type(arg)
        mapped_type = self.map_type(type_name)
        if mapped_type not in {"Float32", None}:
            return mapped_type

        dtype = self.expression_mojo_dtype(arg)
        if dtype is not None:
            scalar_type = MOJO_DTYPE_INFO.get(dtype, MOJO_DTYPE_INFO["DType.float32"])[
                0
            ]
            return self.map_type(scalar_type)

        return "Float32"

    def validate_resource_argument_type(
        self, arg, expected_type, helper_name, resource_type, label
    ):
        actual_type = self.resource_builtin_arg_type(arg)
        if actual_type != expected_type:
            raise MojoRefusal(
                f"Unsupported {helper_name} for Mojo codegen; {label} for "
                f"{resource_type} must be {expected_type}, got {actual_type}"
            )

    def resource_builtin_return_type(self, return_kind, args=None):
        if return_kind == "float":
            return "Float32"
        if return_kind == "vec2":
            return "SIMD[DType.float32, 2]"
        if args:
            resource_type = self.map_type(self.expression_result_type(args[0]))
            if self.is_texture_resource_type(resource_type):
                return self.resource_sample_return_type(resource_type)
        return "SIMD[DType.float32, 4]"

    def resource_builtin_helper_name(self, helper_base, arg_types):
        parts = [helper_base]
        for arg_type in arg_types:
            safe_type = re.sub(r"[^0-9A-Za-z]+", "_", arg_type).strip("_")
            parts.append(safe_type)
        return "_crossgl_" + "_".join(parts)
