"""Result-shape checks of the Mojo generator: target shapes of resource-size, image, buffer and reinterpret results, and scalar conversions."""

from ..ast import (
    ArrayLiteralNode,
    AtomicOpNode,
    BufferOpNode,
    ConstructorNode,
    FunctionCallNode,
    MatchNode,
    MemberAccessNode,
    TernaryOpNode,
    TextureOpNode,
)
from ..refusals import MojoRefusal
from .mojo_tables import (
    MOJO_ATOMIC_OP_ALIASES,
    MOJO_BUFFER_OP_ALIASES,
    MOJO_BYTE_ADDRESS_LOAD_METHODS,
    MOJO_IMAGE_ATOMIC_BUILTINS,
    MojoScalarConversionError,
)


class MojoTargetShapeMixin:
    """Target-shape members of ``MojoCodeGen``."""

    def generate_user_function_call_arguments(self, func_name, args):
        parameter_types = self.selected_function_parameter_types(func_name, args)
        generated_args = []
        for index, arg in enumerate(args):
            target_type = (
                parameter_types[index] if index < len(parameter_types) else None
            )
            self.validate_expression_target_shape(
                arg,
                target_type,
                f"argument {index + 1} for {func_name}",
            )
            generated_args.append(
                self.generate_expression(
                    arg,
                    target_type,
                    f"argument {index + 1} for {func_name}",
                )
            )
        for extra_name, _extra_type in self.stage_local_function_extra_params.get(
            func_name, []
        ):
            if extra_name in self.variable_types:
                generated_args.append(extra_name)
        return ", ".join(generated_args)

    def validate_expression_target_shape(self, expr, target_type, context):
        self.validate_resource_size_target_shape(expr, target_type, context)
        self.validate_image_result_target_shape(expr, target_type, context)
        self.validate_buffer_result_target_shape(expr, target_type, context)
        self.validate_reinterpret_result_target_shape(expr, target_type, context)
        self.validate_aggregate_target_shape(expr, target_type, context)

    def validate_scalar_target_conversion(self, expr, target_type, context):
        """Refuse a source vector where the target needs a scalar.

        Mojo does not convert ``SIMD[dtype, N]`` to a scalar implicitly, and a
        comparison of two vectors is a lane-wise ``SIMD[DType.bool, N]`` mask
        rather than a ``Bool``.
        """
        if expr is None or target_type is None:
            return

        if isinstance(expr, TernaryOpNode):
            self.validate_scalar_target_conversion(
                expr.true_expr, target_type, f"{context} true branch"
            )
            self.validate_scalar_target_conversion(
                expr.false_expr, target_type, f"{context} false branch"
            )
            return

        target_name = self.type_name(target_type)
        if not self.is_scalar_type_name(target_name):
            return

        source_type = self.expression_result_type(expr)
        if self.vector_type_info(source_type) is None:
            return

        source_name = self.type_name(source_type)
        raise MojoScalarConversionError(
            "Mojo cannot preserve source conversion from "
            f"'{source_name}' to '{target_name}' in {context}: vector to scalar",
            source_type=source_name,
            target_type=target_name,
            reason="vector-to-scalar",
            source_location=getattr(expr, "source_location", None),
        )

    def validate_condition_scalar(self, condition, context):
        self.validate_scalar_target_conversion(condition, "bool", context)

    def validate_aggregate_target_shape(self, expr, target_type, context):
        if expr is None or target_type is None:
            return

        if isinstance(expr, TernaryOpNode):
            self.validate_aggregate_target_shape(
                expr.true_expr, target_type, f"{context} true branch"
            )
            self.validate_aggregate_target_shape(
                expr.false_expr, target_type, f"{context} false branch"
            )
            return

        if isinstance(expr, MatchNode):
            arms = getattr(expr, "arms", []) or []
            self.validate_match_expression_arms(
                arms, self.expression_result_type(getattr(expr, "expression", None))
            )
            for arm_index, arm in enumerate(arms, start=1):
                arm_expr, _ = self.match_arm_value_expression(arm)
                self.validate_aggregate_target_shape(
                    arm_expr, target_type, f"{context} match arm {arm_index}"
                )
            return

        if isinstance(expr, ArrayLiteralNode) and self.is_array_type_name(target_type):
            element_type, _ = self.parse_array_type_name(target_type)
            for index, element in enumerate(expr.elements, start=1):
                self.validate_aggregate_target_shape(
                    element, element_type, f"{context} array literal element {index}"
                )
            return

        target_name = self.type_name(target_type)
        source_type = self.expression_result_type(expr)
        if source_type is None:
            return

        target_vector = self.vector_type_info(target_name)
        target_matrix = self.matrix_type_info(target_name)
        if target_vector is None and target_matrix is None:
            return

        source_vector = self.vector_type_info(source_type)
        source_matrix = self.matrix_type_info(source_type)
        if source_vector is None and source_matrix is None:
            return

        if target_vector is not None and source_vector is not None:
            if source_vector[:2] == target_vector[:2]:
                return
            if self.compatible_numeric_vector_target(source_vector, target_vector):
                return
        elif target_matrix is not None and source_matrix is not None:
            if source_matrix == target_matrix:
                return

        raise MojoRefusal(
            f"Invalid aggregate target for Mojo codegen; {context} "
            f"expects {target_name}, got {self.type_name(source_type)}"
        )

    def validate_resource_size_target_shape(self, expr, target_type, context):
        if expr is None:
            return

        if target_type is None:
            if isinstance(expr, ConstructorNode):
                self.validate_constructor_resource_size_target_shapes(
                    expr, None, context
                )
            return

        if isinstance(expr, TernaryOpNode):
            self.validate_resource_size_target_shape(
                expr.true_expr, target_type, f"{context} true branch"
            )
            self.validate_resource_size_target_shape(
                expr.false_expr, target_type, f"{context} false branch"
            )
            return

        if isinstance(expr, MatchNode):
            arms = getattr(expr, "arms", []) or []
            self.validate_match_expression_arms(
                arms, self.expression_result_type(getattr(expr, "expression", None))
            )
            for arm_index, arm in enumerate(arms, start=1):
                arm_expr, _ = self.match_arm_value_expression(arm)
                self.validate_resource_size_target_shape(
                    arm_expr, target_type, f"{context} match arm {arm_index}"
                )
            return

        if isinstance(expr, ArrayLiteralNode) and self.is_array_type_name(target_type):
            element_type, _ = self.parse_array_type_name(target_type)
            for index, element in enumerate(expr.elements, start=1):
                self.validate_resource_size_target_shape(
                    element, element_type, f"{context} array literal element {index}"
                )
            return

        if isinstance(expr, ConstructorNode):
            self.validate_constructor_resource_size_target_shapes(
                expr, target_type, context
            )
            return

        size_info = self.resource_size_expression_info(expr)
        if size_info is None:
            return

        helper_name, resource_type, expected_type = size_info
        if expected_type is None:
            return
        if self.resource_size_target_matches(expected_type, target_type):
            return

        raise MojoRefusal(
            f"Unsupported {helper_name} target for Mojo codegen; {context} "
            f"expects {expected_type} from {resource_type}, got "
            f"{self.type_name(target_type)}"
        )

    def validate_image_result_target_shape(self, expr, target_type, context):
        if expr is None or target_type is None:
            return

        if isinstance(expr, TernaryOpNode):
            self.validate_image_result_target_shape(
                expr.true_expr, target_type, f"{context} true branch"
            )
            self.validate_image_result_target_shape(
                expr.false_expr, target_type, f"{context} false branch"
            )
            return

        if isinstance(expr, MatchNode):
            arms = getattr(expr, "arms", []) or []
            self.validate_match_expression_arms(
                arms, self.expression_result_type(getattr(expr, "expression", None))
            )
            for arm_index, arm in enumerate(arms, start=1):
                arm_expr, _ = self.match_arm_value_expression(arm)
                self.validate_image_result_target_shape(
                    arm_expr, target_type, f"{context} match arm {arm_index}"
                )
            return

        if isinstance(expr, ArrayLiteralNode) and self.is_array_type_name(target_type):
            element_type, _ = self.parse_array_type_name(target_type)
            for index, element in enumerate(expr.elements, start=1):
                self.validate_image_result_target_shape(
                    element, element_type, f"{context} array literal element {index}"
                )
            return

        if isinstance(expr, ConstructorNode):
            self.validate_constructor_image_result_target_shapes(
                expr, target_type, context
            )
            return

        image_info = self.image_result_expression_info(expr)
        if image_info is None:
            return

        helper_name, resource_type, expected_type = image_info
        if expected_type is None:
            return
        if self.image_result_target_matches(expected_type, target_type):
            return
        if self.explicit_scalar_conversion_context(context) and (
            self.is_scalar_type_name(expected_type)
            and self.is_scalar_type_name(target_type)
        ):
            return

        raise MojoRefusal(
            f"Unsupported {helper_name} target for Mojo codegen; {context} "
            f"expects {expected_type} from {resource_type}, got "
            f"{self.type_name(target_type)}"
        )

    def validate_buffer_result_target_shape(self, expr, target_type, context):
        if expr is None or target_type is None:
            return

        if isinstance(expr, TernaryOpNode):
            self.validate_buffer_result_target_shape(
                expr.true_expr, target_type, f"{context} true branch"
            )
            self.validate_buffer_result_target_shape(
                expr.false_expr, target_type, f"{context} false branch"
            )
            return

        if isinstance(expr, MatchNode):
            arms = getattr(expr, "arms", []) or []
            self.validate_match_expression_arms(
                arms, self.expression_result_type(getattr(expr, "expression", None))
            )
            for arm_index, arm in enumerate(arms, start=1):
                arm_expr, _ = self.match_arm_value_expression(arm)
                self.validate_buffer_result_target_shape(
                    arm_expr, target_type, f"{context} match arm {arm_index}"
                )
            return

        if isinstance(expr, ArrayLiteralNode) and self.is_array_type_name(target_type):
            element_type, _ = self.parse_array_type_name(target_type)
            for index, element in enumerate(expr.elements, start=1):
                self.validate_buffer_result_target_shape(
                    element, element_type, f"{context} array literal element {index}"
                )
            return

        if isinstance(expr, ConstructorNode):
            self.validate_constructor_buffer_result_target_shapes(
                expr, target_type, context
            )
            return

        buffer_info = self.buffer_result_expression_info(expr)
        if buffer_info is None:
            return

        helper_name, resource_type, expected_type = buffer_info
        if expected_type is None or self.result_shape_target_matches(
            expected_type, target_type
        ):
            return

        raise MojoRefusal(
            f"Unsupported {helper_name} target for Mojo codegen; {context} "
            f"expects {self.type_name(expected_type)} from {resource_type}, got "
            f"{self.type_name(target_type)}"
        )

    def validate_reinterpret_result_target_shape(self, expr, target_type, context):
        if expr is None or target_type is None:
            return

        if isinstance(expr, TernaryOpNode):
            self.validate_reinterpret_result_target_shape(
                expr.true_expr, target_type, f"{context} true branch"
            )
            self.validate_reinterpret_result_target_shape(
                expr.false_expr, target_type, f"{context} false branch"
            )
            return

        if isinstance(expr, MatchNode):
            arms = getattr(expr, "arms", []) or []
            self.validate_match_expression_arms(
                arms, self.expression_result_type(getattr(expr, "expression", None))
            )
            for arm_index, arm in enumerate(arms, start=1):
                arm_expr, _ = self.match_arm_value_expression(arm)
                self.validate_reinterpret_result_target_shape(
                    arm_expr, target_type, f"{context} match arm {arm_index}"
                )
            return

        if isinstance(expr, ArrayLiteralNode) and self.is_array_type_name(target_type):
            element_type, _ = self.parse_array_type_name(target_type)
            for index, element in enumerate(expr.elements, start=1):
                self.validate_reinterpret_result_target_shape(
                    element, element_type, f"{context} array literal element {index}"
                )
            return

        if isinstance(expr, ConstructorNode):
            self.validate_constructor_reinterpret_result_target_shapes(
                expr, target_type, context
            )
            return

        reinterpret_info = self.reinterpret_result_expression_info(expr)
        if reinterpret_info is None:
            return

        helper_name, expected_type = reinterpret_info
        if self.result_shape_target_matches(expected_type, target_type):
            return

        raise MojoRefusal(
            f"Unsupported {helper_name} target for Mojo codegen; {context} "
            f"expects {self.type_name(expected_type)}, got "
            f"{self.type_name(target_type)}"
        )

    def validate_constructor_image_result_target_shapes(
        self, expr, target_type, context
    ):
        struct_type, fields = self.constructor_node_struct_type_and_fields(
            expr, target_type
        )
        if struct_type is None:
            return

        for field_name, field_type, argument in self.constructor_node_field_bindings(
            expr, struct_type, fields
        ):
            self.validate_image_result_target_shape(
                argument,
                field_type,
                f"{context} field {struct_type}.{field_name}",
            )

    def validate_constructor_buffer_result_target_shapes(
        self, expr, target_type, context
    ):
        struct_type, fields = self.constructor_node_struct_type_and_fields(
            expr, target_type
        )
        if struct_type is None:
            return

        for field_name, field_type, argument in self.constructor_node_field_bindings(
            expr, struct_type, fields
        ):
            self.validate_buffer_result_target_shape(
                argument,
                field_type,
                f"{context} field {struct_type}.{field_name}",
            )

    def validate_constructor_reinterpret_result_target_shapes(
        self, expr, target_type, context
    ):
        struct_type, fields = self.constructor_node_struct_type_and_fields(
            expr, target_type
        )
        if struct_type is None:
            return

        for field_name, field_type, argument in self.constructor_node_field_bindings(
            expr, struct_type, fields
        ):
            self.validate_reinterpret_result_target_shape(
                argument,
                field_type,
                f"{context} field {struct_type}.{field_name}",
            )

    def validate_constructor_resource_size_target_shapes(
        self, expr, target_type, context
    ):
        struct_type, fields = self.constructor_node_struct_type_and_fields(
            expr, target_type
        )
        if struct_type is None:
            return

        for field_name, field_type, argument in self.constructor_node_field_bindings(
            expr, struct_type, fields
        ):
            self.validate_resource_size_target_shape(
                argument,
                field_type,
                f"{context} field {struct_type}.{field_name}",
            )

    def resource_size_expression_info(self, expr):
        if isinstance(expr, FunctionCallNode):
            func_expr = getattr(expr, "function", None)
            if func_expr is None:
                func_expr = getattr(expr, "name", None)
            args = list(getattr(expr, "args", []) or [])

            if isinstance(func_expr, MemberAccessNode) and getattr(
                func_expr, "member", None
            ) in {"GetDimensions", "textureSize"}:
                resource_expr = getattr(
                    func_expr, "object", getattr(func_expr, "object_expr", None)
                )
                return self.resource_size_expression_info_for_resource(
                    resource_expr, "resource_size"
                )

            func_name = self.function_call_name(expr)
            if func_name == "textureSize" and args:
                return self.resource_size_expression_info_for_resource(
                    args[0], "texture_size"
                )
            if func_name == "imageSize" and args:
                return self.resource_size_expression_info_for_resource(
                    args[0], "image_size"
                )

        if isinstance(expr, TextureOpNode) and getattr(expr, "operation", "") in {
            "GetDimensions",
            "textureSize",
        }:
            texture_expr = getattr(expr, "texture_expr", None)
            resource_type = self.map_type(self.expression_result_type(texture_expr))
            helper_name = (
                "image_size"
                if self.is_image_resource_type(resource_type)
                else "texture_size"
            )
            expected_type = self.resource_size_result_type_name(resource_type)
            return helper_name, resource_type, expected_type

        return None

    def resource_size_expression_info_for_resource(self, resource_expr, helper_name):
        resource_type = self.map_type(self.expression_result_type(resource_expr))
        if helper_name == "resource_size":
            helper_name = (
                "image_size"
                if self.is_image_resource_type(resource_type)
                else "texture_size"
            )
        expected_type = self.resource_size_result_type_name(resource_type)
        return helper_name, resource_type, expected_type

    def image_result_expression_info(self, expr):
        if isinstance(expr, FunctionCallNode):
            func_expr = getattr(expr, "function", None)
            if func_expr is None:
                func_expr = getattr(expr, "name", None)
            args = list(getattr(expr, "args", []) or [])

            if isinstance(func_expr, MemberAccessNode):
                member = getattr(func_expr, "member", None)
                resource_expr = getattr(
                    func_expr, "object", getattr(func_expr, "object_expr", None)
                )
                atomic_helper = MOJO_ATOMIC_OP_ALIASES.get(member)
                if member == "Load":
                    return self.image_result_expression_info_for_resource(
                        resource_expr, "image_load"
                    )
                if atomic_helper is not None:
                    return self.image_result_expression_info_for_resource(
                        resource_expr, atomic_helper
                    )

            func_name = self.function_call_name(expr)
            if func_name == "imageLoad" and args:
                return self.image_result_expression_info_for_resource(
                    args[0], "image_load"
                )
            atomic_helper = MOJO_IMAGE_ATOMIC_BUILTINS.get(func_name)
            if atomic_helper is not None and args:
                return self.image_result_expression_info_for_resource(
                    args[0], atomic_helper
                )

        if isinstance(expr, TextureOpNode):
            operation = getattr(expr, "operation", "")
            args = self.texture_op_args(expr, include_sampler=False)
            if args:
                if operation in {"Load", "texelFetch"}:
                    return self.image_result_expression_info_for_resource(
                        args[0], "image_load"
                    )
                atomic_helper = MOJO_ATOMIC_OP_ALIASES.get(operation)
                if atomic_helper is not None:
                    return self.image_result_expression_info_for_resource(
                        args[0], atomic_helper
                    )

        if isinstance(expr, AtomicOpNode):
            helper_base = MOJO_ATOMIC_OP_ALIASES.get(getattr(expr, "operation", ""))
            if helper_base is not None:
                return self.image_result_expression_info_for_resource(
                    expr.target, helper_base
                )

        return None

    def buffer_result_expression_info(self, expr):
        buffer_object_info = self.mojo_buffer_object_result_info(expr)
        if buffer_object_info is not None:
            return buffer_object_info
        if isinstance(expr, FunctionCallNode):
            func_expr = getattr(expr, "function", None)
            if func_expr is None:
                func_expr = getattr(expr, "name", None)
            args = list(getattr(expr, "args", []) or [])

            if isinstance(func_expr, MemberAccessNode):
                member = getattr(func_expr, "member", None)
                resource_expr = getattr(
                    func_expr, "object", getattr(func_expr, "object_expr", None)
                )
                info = self.buffer_resource_info(
                    self.expression_result_type(resource_expr)
                )
                return self.buffer_result_info_for_operation(member, info)

            func_name = self.function_call_name(expr)
            if func_name == "buffer_load" and args:
                info = self.buffer_resource_info(self.expression_result_type(args[0]))
                return self.buffer_result_info_for_operation("Load", info)
            if func_name == "buffer_consume" and args:
                info = self.buffer_resource_info(self.expression_result_type(args[0]))
                return self.buffer_result_info_for_operation("Consume", info)

        if isinstance(expr, BufferOpNode):
            operation = MOJO_BUFFER_OP_ALIASES.get(getattr(expr, "operation", ""))
            info = self.buffer_resource_info(
                self.expression_result_type(getattr(expr, "buffer_expr", None))
            )
            return self.buffer_result_info_for_operation(operation, info)

        return None

    def buffer_result_info_for_operation(self, operation, info):
        if operation is None or info is None:
            return None

        buffer_type, element_type = info
        if operation == "Load":
            expected_type = self.buffer_helper_element_type(buffer_type, element_type)
            return "buffer_load", self.mapped_buffer_type(*info), expected_type

        if operation in MOJO_BYTE_ADDRESS_LOAD_METHODS:
            width = MOJO_BYTE_ADDRESS_LOAD_METHODS[operation]
            helper_name = "buffer_load" if width == 1 else f"buffer_load{width}"
            return helper_name, buffer_type, self.byte_address_load_result_type(width)

        if operation == "Consume":
            return "buffer_consume", self.mapped_buffer_type(*info), element_type

        return None

    def reinterpret_result_expression_info(self, expr):
        if not isinstance(expr, FunctionCallNode):
            return None

        func_name = self.function_call_name(expr)
        reinterpret_name = self.mojo_reinterpret_builtin_name(func_name)
        if reinterpret_name is None:
            return None

        source_type = self.expression_result_type(expr.args[0]) if expr.args else None
        return (
            reinterpret_name,
            self.reinterpret_return_type_name(reinterpret_name, source_type),
        )

    def image_result_expression_info_for_resource(self, resource_expr, helper_name):
        resource_type = self.map_type(self.expression_result_type(resource_expr))
        if not self.is_image_resource_type(resource_type):
            return None

        if helper_name == "image_load":
            return (
                helper_name,
                resource_type,
                self.image_result_type_name(resource_type),
            )

        expected_type = self.image_atomic_result_type_name(resource_type)
        return helper_name, resource_type, expected_type

    def image_atomic_result_type_name(self, resource_type):
        value_type = self.image_value_type(resource_type)
        if value_type == "Int32":
            return "int"
        if value_type == "UInt32":
            return "uint"
        return None

    def resource_size_target_matches(self, expected_type, target_type):
        if expected_type == "int":
            return self.map_type(target_type) == "Int32"

        expected_info = self.vector_type_info(expected_type)
        target_info = self.vector_type_info(target_type)
        if expected_info is None or target_info is None:
            return False
        return expected_info[0] == target_info[0] and expected_info[1] == target_info[1]

    def image_result_target_matches(self, expected_type, target_type):
        if self.map_type(expected_type) == self.map_type(target_type):
            return True

        expected_info = self.vector_type_info(expected_type)
        target_info = self.vector_type_info(target_type)
        if expected_info is not None or target_info is not None:
            if expected_info is None or target_info is None:
                return False
            return (
                expected_info[0] == target_info[0]
                and expected_info[1] == target_info[1]
            )

        return False

    def result_shape_target_matches(self, expected_type, target_type):
        if self.map_type(expected_type) == self.map_type(target_type):
            return True

        expected_info = self.vector_type_info(expected_type)
        target_info = self.vector_type_info(target_type)
        if expected_info is not None or target_info is not None:
            return (
                expected_info is not None
                and target_info is not None
                and expected_info[1] == target_info[1]
            )

        return self.is_scalar_type_name(expected_type) and self.is_scalar_type_name(
            target_type
        )

    def explicit_scalar_conversion_context(self, context):
        if context is None:
            return False
        return context.startswith("cast to ") or context.startswith(
            "scalar constructor "
        )

    def is_scalar_type_name(self, type_name):
        if type_name is None:
            return False
        if self.vector_type_info(type_name) is not None:
            return False
        return self.map_type(type_name) in {
            "Float16",
            "Float32",
            "Float64",
            "Int16",
            "Int32",
            "Int64",
            "UInt16",
            "UInt32",
            "UInt64",
            "Bool",
        }
