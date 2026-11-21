"""Result types of CrossGL expressions, built-in functions and resource operations as the Rust generator infers them."""

import re

from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    BlockNode,
    ConstructorNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MatchNode,
    MemberAccessNode,
    ReturnNode,
    TernaryOpNode,
    TextureNode,
    TextureOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
)


class RustResultTypeMixin:
    """Result-type members of ``RustCodeGen``."""

    def is_generated_struct_type(self, type_name):
        if type_name is None:
            return False
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)
        return type_name in self.struct_member_types

    def get_expression_name(self, expr):
        if isinstance(expr, IdentifierNode):
            return expr.name
        if isinstance(expr, VariableNode):
            return expr.name
        if isinstance(expr, str):
            return expr
        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
            return self.get_expression_name(array_expr)
        return None

    def block_expression_result_type(self, expr):
        explicit_tail = getattr(expr, "expression", None)
        if explicit_tail is not None:
            return self.expression_result_type(explicit_tail)

        statements = self.statement_list(expr)
        if not statements:
            return None

        tail = statements[-1]
        tail_expression = self.block_tail_expression(tail)
        if tail_expression is not None:
            return self.expression_result_type(tail_expression)
        if isinstance(tail, ReturnNode):
            return self.expression_result_type(getattr(tail, "value", None))
        return None

    def expression_result_type(self, expr):
        if expr is None:
            return None
        if isinstance(expr, bool):
            return "bool"
        if isinstance(expr, int):
            return "int"
        if isinstance(expr, float):
            return "float"
        if isinstance(expr, ArrayAccessNode):
            return self.array_access_element_type(expr)
        if isinstance(expr, (IdentifierNode, VariableNode)):
            name = self.get_expression_name(expr)
            variable_type = self.variable_types.get(name)
            if variable_type is not None:
                return variable_type
            return self.rust_builtin_value_type(name)
        if isinstance(expr, LiteralNode):
            literal_type = getattr(getattr(expr, "literal_type", None), "name", None)
            if literal_type:
                return literal_type
            if isinstance(expr.value, bool):
                return "bool"
            if isinstance(expr.value, float):
                return "float"
            if isinstance(expr.value, int):
                return "int"
            return None
        if isinstance(expr, BlockNode):
            return self.block_expression_result_type(expr)
        if isinstance(expr, ConstructorNode):
            return self.constructor_result_type(expr)
        if isinstance(expr, TextureNode):
            texture_type = self.expression_result_type(expr.texture_expr)
            return (
                "float"
                if self.is_shadow_texture_type(texture_type)
                else self.texture_sample_result_type(texture_type)
            )
        if isinstance(expr, TextureOpNode):
            operation = getattr(expr, "operation", "")
            if operation in {
                "SampleCmp",
                "SampleCmpLevel",
                "SampleCmpLevelZero",
                "SampleCmpGrad",
                "CalculateLevelOfDetail",
                "CalculateLevelOfDetailUnclamped",
                "textureCompare",
                "textureCompareOffset",
                "textureCompareLod",
                "textureCompareLodOffset",
                "textureCompareGrad",
                "textureCompareGradOffset",
                "textureCompareProj",
                "textureCompareProjOffset",
                "textureCompareProjLod",
                "textureCompareProjLodOffset",
                "textureCompareProjGrad",
                "textureCompareProjGradOffset",
            }:
                return "float"
            if operation in {"textureQueryLod"}:
                return "vec2"
            if operation in {"textureQueryLevels", "textureSamples"}:
                return "int"
            if operation in {"GetDimensions"}:
                return "void"
            if operation in {"textureDimensions"}:
                texture_type = self.expression_result_type(expr.texture_expr)
                return self.texture_dimensions_result_type(texture_type)
            if operation in {"textureSize"}:
                texture_type = self.expression_result_type(expr.texture_expr)
                return self.texture_size_result_type(texture_type)
            texture_type = self.expression_result_type(expr.texture_expr)
            if operation in {"Sample", "sample", "texture"} and (
                self.is_shadow_texture_type(texture_type)
            ):
                return "float"
            if str(operation).startswith("textureGather") or str(operation).startswith(
                "Gather"
            ):
                return self.texture_gather_result_type(texture_type)
            return self.texture_sample_result_type(texture_type)
        if isinstance(expr, WaveOpNode):
            return self.wave_op_result_type(
                getattr(expr, "operation", ""),
                getattr(expr, "arguments", []),
            )
        if isinstance(expr, MatchNode):
            return self.match_expression_result_type(expr)
        if isinstance(expr, FunctionCallNode):
            func_expr = getattr(expr, "function", getattr(expr, "name", None))
            arguments = getattr(expr, "arguments", getattr(expr, "args", [])) or []
            if isinstance(func_expr, MemberAccessNode):
                texture_member_type = self.texture_member_function_result_type(
                    func_expr,
                    arguments,
                )
                if texture_member_type is not None:
                    return texture_member_type
                receiver_type = self.expression_result_type(
                    getattr(
                        func_expr, "object_expr", getattr(func_expr, "object", None)
                    )
                )
                if receiver_type is not None and func_expr.member in {
                    "add",
                    "sub",
                    "mul",
                    "div",
                    "rem",
                }:
                    return receiver_type
            func_name = getattr(func_expr, "name", func_expr)
            if isinstance(func_name, str) and self.vector_type_info(func_name):
                return func_name
            if isinstance(func_name, str) and self.matrix_type_info(func_name):
                return func_name
            enum_result_type = self.enum_variant_call_result_type(func_name)
            if enum_result_type is not None:
                return enum_result_type
            if isinstance(func_name, str) and "::" in func_name:
                base_name = func_name.split("::", 1)[0]
                if base_name in self.current_generic_param_names:
                    return base_name
            scalar_type = self.scalar_constructor_type(func_name)
            if scalar_type is not None:
                return scalar_type
            return_type = self.user_function_return_types.get(func_name)
            if return_type and return_type != "void":
                return return_type
            builtin_type = self.builtin_function_result_type(func_name, arguments)
            if builtin_type is not None:
                return builtin_type
            return None
        if isinstance(expr, BinaryOpNode):
            left_type = self.expression_result_type(expr.left)
            right_type = self.expression_result_type(expr.right)
            operator = self.map_operator(
                getattr(expr, "operator", getattr(expr, "op", None))
            )
            bool_vector_logical_plan = self.bool_vector_logical_plan(
                left_type, right_type, operator
            )
            if bool_vector_logical_plan is not None:
                return bool_vector_logical_plan["result_type"]
            vector_comparison_plan = self.vector_comparison_plan(
                left_type, right_type, operator
            )
            if vector_comparison_plan is not None:
                return vector_comparison_plan["result_type"]
            matrix_multiply_plan = self.matrix_multiply_plan(
                left_type, right_type, operator
            )
            if matrix_multiply_plan is not None:
                return matrix_multiply_plan["result_type"]
            matrix_vector_plan = self.binary_matrix_vector_plan(
                left_type, right_type, operator
            )
            if matrix_vector_plan is not None:
                return matrix_vector_plan["result_type"]
            composite_type = self.binary_composite_operand_type(
                left_type, right_type, operator
            )
            if composite_type is not None:
                return composite_type
            if self.vector_type_info(left_type):
                return left_type
            if self.vector_type_info(right_type):
                return right_type
            if self.matrix_type_info(left_type):
                return left_type
            if self.matrix_type_info(right_type):
                return right_type
            scalar_type = self.binary_scalar_result_type(
                left_type,
                right_type,
                operator,
            )
            if scalar_type is not None:
                return scalar_type
            return left_type or right_type
        if isinstance(expr, UnaryOpNode):
            operator = self.map_operator(
                getattr(expr, "operator", getattr(expr, "op", None))
            )
            if operator == "!":
                operand_type = self.expression_result_type(expr.operand)
                operand_vector = self.vector_type_info(operand_type)
                if (
                    operand_vector is not None
                    and operand_vector["component_type"] == "bool"
                ):
                    return operand_type
                return "bool"
            return self.expression_result_type(expr.operand)
        if isinstance(expr, TernaryOpNode):
            vector_ternary_plan = self.bool_vector_ternary_plan(
                expr.condition, expr.true_expr, expr.false_expr
            )
            if vector_ternary_plan is not None:
                return vector_ternary_plan["result_type"]
            true_type = self.expression_result_type(expr.true_expr)
            false_type = self.expression_result_type(expr.false_expr)
            vector_type = self.promoted_vector_type(true_type, false_type)
            if vector_type is not None:
                return vector_type
            if self.vector_type_info(true_type):
                return true_type
            if self.vector_type_info(false_type):
                return false_type
            matrix_type = self.promoted_matrix_type(true_type, false_type)
            if matrix_type is not None:
                return matrix_type
            if self.matrix_type_info(true_type):
                return true_type
            if self.matrix_type_info(false_type):
                return false_type
            scalar_type = self.promoted_scalar_type(true_type, false_type)
            if scalar_type is not None:
                return scalar_type
            return true_type or false_type
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
            object_type = self.expression_result_type(object_expr)
            object_type_name = (
                self.convert_type_node_to_string(object_type)
                if object_type is not None
                and (
                    hasattr(object_type, "name") or hasattr(object_type, "element_type")
                )
                else object_type
            )
            member = getattr(expr, "member", "")
            member_type = self.resolve_struct_member_type(object_type_name, member)
            if member_type is not None:
                return member_type

            vector_info = self.vector_type_info(object_type)
            if not vector_info:
                return None
            if len(member) == 1:
                return vector_info["component_type"]
            if all(component in "xyzwrgba" for component in member):
                return self.vector_type_for_components(
                    vector_info["component_type"], len(member)
                )
        return None

    def texture_size_result_type(self, texture_type):
        texture_name = str(texture_type or "")
        if texture_name in {"sampler1D", "Texture1D<f32>"}:
            return "int"
        if texture_name in {
            "sampler1DArray",
            "sampler2D",
            "samplerCube",
            "Texture1DArray<f32>",
            "Texture2D<f32>",
            "TextureCube<f32>",
            "sampler2DMS",
            "Texture2DMS<f32>",
            "sampler2DShadow",
            "samplerCubeShadow",
            "DepthTexture2D<f32>",
            "DepthTextureCube<f32>",
        }:
            return "ivec2"
        if texture_name in {
            "sampler2DArray",
            "sampler3D",
            "samplerCubeArray",
            "Texture2DArray<f32>",
            "Texture3D<f32>",
            "TextureCubeArray<f32>",
            "sampler2DMSArray",
            "Texture2DMSArray<f32>",
            "sampler2DArrayShadow",
            "samplerCubeArrayShadow",
            "DepthTexture2DArray<f32>",
            "DepthTextureCubeArray<f32>",
        }:
            return "ivec3"
        return "ivec2"

    def texture_sample_result_type(self, texture_type):
        payload_type = self.texture_payload_type(texture_type)
        if payload_type is None:
            return "vec4"
        return self.texture_payload_sample_result_type(payload_type)

    def texture_gather_result_type(self, texture_type):
        payload_type = self.texture_payload_type(texture_type)
        if payload_type is None:
            return "vec4"
        return self.texture_payload_component_vector_result_type(payload_type)

    def texture_payload_type(self, texture_type):
        texture_type = str(texture_type or "")
        mapped_type = self.unqualify_runtime_type_name(self.map_type(texture_type))
        texture_base, texture_args = self.generic_type_parts(mapped_type)
        texture_base = texture_base.rsplit("::", 1)[-1]
        if texture_base not in {
            "Texture1D",
            "Texture1DArray",
            "Texture2D",
            "Texture2DArray",
            "Texture3D",
            "TextureCube",
            "TextureCubeArray",
            "Texture2DMS",
            "Texture2DMSArray",
            "DepthTexture2D",
            "DepthTexture2DArray",
            "DepthTextureCube",
            "DepthTextureCubeArray",
        }:
            return None
        if not texture_args:
            return None
        return texture_args[0]

    def texture_payload_sample_result_type(self, payload_type):
        payload_type = self.unqualify_runtime_type_name(self.map_type(payload_type))
        vector_info = self.vector_type_info(payload_type)
        if vector_info is not None:
            return (
                self.vector_type_for_components(
                    vector_info["component_type"],
                    vector_info["size"],
                )
                or "vec4"
            )

        scalar_type = self.normalize_scalar_type(payload_type)
        if scalar_type is None:
            return "vec4"
        return self.vector_type_for_components(scalar_type, 4) or "vec4"

    def texture_payload_component_vector_result_type(self, payload_type):
        payload_type = self.unqualify_runtime_type_name(self.map_type(payload_type))
        vector_info = self.vector_type_info(payload_type)
        if vector_info is not None:
            return (
                self.vector_type_for_components(vector_info["component_type"], 4)
                or "vec4"
            )

        scalar_type = self.normalize_scalar_type(payload_type)
        if scalar_type is None:
            return "vec4"
        return self.vector_type_for_components(scalar_type, 4) or "vec4"

    def texture_dimensions_result_type(self, texture_type):
        size_type = self.texture_size_result_type(texture_type)
        size_info = self.vector_type_info(size_type)
        if size_info is not None:
            return self.vector_type_for_components("uint", size_info["size"])
        if self.normalize_scalar_type(size_type) is not None:
            return "uint"
        return size_type

    def storage_image_value_result_type(self, image_type):
        image_type = str(image_type or "")
        element_type = self.storage_image_runtime_element_type(image_type)
        if element_type is not None:
            return self.rust_storage_image_element_crossgl_type(element_type)
        if image_type.startswith("uimage") or "Vec4<u32>" in image_type:
            return "uvec4"
        if image_type.startswith("iimage") or "Vec4<i32>" in image_type:
            return "ivec4"
        return "vec4"

    def storage_image_size_result_type(self, image_type):
        image_base, _ = self.generic_type_parts(image_type)
        image_base = image_base.rsplit("::", 1)[-1]
        if image_base in {"image1D", "iimage1D", "uimage1D", "Image1D"}:
            return "int"
        if image_base in {
            "image1DArray",
            "iimage1DArray",
            "uimage1DArray",
            "image2D",
            "iimage2D",
            "uimage2D",
            "imageCube",
            "Image1DArray",
            "Image2D",
            "ImageCube",
            "image2DMS",
            "iimage2DMS",
            "uimage2DMS",
            "Image2DMS",
        }:
            return "ivec2"
        if image_base in {
            "image3D",
            "iimage3D",
            "uimage3D",
            "image2DArray",
            "iimage2DArray",
            "uimage2DArray",
            "Image3D",
            "Image2DArray",
            "image2DMSArray",
            "iimage2DMSArray",
            "uimage2DMSArray",
            "Image2DMSArray",
        }:
            return "ivec3"
        return "ivec2"

    def storage_image_atomic_result_type(self, image_type):
        image_type = str(image_type or "")
        element_type = self.storage_image_runtime_element_type(image_type)
        if element_type is not None and "i32" in element_type:
            return "int"
        if image_type.startswith("iimage") or "Vec4<i32>" in image_type:
            return "int"
        return "uint"

    def storage_image_format_element_type(self, format_name):
        format_name = str(format_name or "").lower()
        match = re.fullmatch(
            r"(rgba|rgb|rg|r)(?:8|16|32|64)?(ui|i|f|unorm|snorm)?",
            format_name,
        )
        if match is None:
            return None

        lanes = {"r": 1, "rg": 2, "rgb": 3, "rgba": 4}[match.group(1)]
        suffix = match.group(2) or "f"
        scalar_type = "u32" if suffix == "ui" else "i32" if suffix == "i" else "f32"
        if lanes == 1:
            return scalar_type
        return f"Vec{lanes}<{scalar_type}>"

    def storage_image_runtime_element_type(self, image_type):
        image_base, image_args = self.generic_type_parts(image_type)
        if not image_args or not image_base.rsplit("::", 1)[-1].startswith("Image"):
            return None
        return image_args[0]

    def rust_storage_image_element_crossgl_type(self, element_type):
        element_type = str(element_type or "")
        element_base, element_args = self.generic_type_parts(element_type)
        if element_args:
            component = element_args[0]
            vector_type_name = element_base.rsplit("::", 1)[-1]
            lane_count = (
                vector_type_name[3:]
                if vector_type_name.startswith("Vec")
                else vector_type_name
            )
            if component == "u32":
                return f"uvec{lane_count}"
            if component == "i32":
                return f"ivec{lane_count}"
            return f"vec{lane_count}"
        if element_type == "u32":
            return "uint"
        if element_type == "i32":
            return "int"
        return "float"

    def buffer_element_result_type(self, buffer_type):
        base_type, generic_args = self.generic_type_parts(buffer_type)
        if generic_args and base_type in {
            "StructuredBuffer",
            "RWStructuredBuffer",
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
            "Buffer",
            "RwBuffer",
            "AppendBuffer",
            "ConsumeBuffer",
        }:
            return generic_args[0]
        if base_type in {
            "ByteAddressBuffer",
            "RWByteAddressBuffer",
            "RwByteAddressBuffer",
        }:
            return "uint"
        return None

    def vector_or_scalar_component_type(self, type_name):
        vector_info = self.vector_type_info(type_name)
        if vector_info is not None:
            return vector_info["component_type"]
        return self.normalize_scalar_type(type_name)

    def boolean_value_type(self, type_name):
        vector_info = self.vector_type_info(type_name)
        if vector_info is not None:
            return self.vector_type_for_components("bool", vector_info["size"])
        if self.normalize_scalar_type(type_name) is not None:
            return "bool"
        return None

    def outer_product_result_type(self, column_type, row_type):
        column_info = self.vector_type_info(column_type)
        row_info = self.vector_type_info(row_type)
        if column_info is None or row_info is None:
            return None

        component_type = self.promoted_scalar_type(
            column_info["component_type"], row_info["component_type"]
        )
        if component_type is None:
            return None

        return self.matrix_type_for_dimensions(
            component_type, row_info["size"], column_info["size"]
        )

    def relational_function_result_type(self, left_type, right_type, operator):
        vector_plan = self.vector_comparison_plan(left_type, right_type, operator)
        if vector_plan is not None:
            return vector_plan["result_type"]

        left_scalar = self.normalize_scalar_type(left_type)
        right_scalar = self.normalize_scalar_type(right_type)
        if left_scalar is None or right_scalar is None:
            return None

        component_type = self.promoted_scalar_type(left_scalar, right_scalar)
        if component_type is None:
            return None
        if self.normalize_scalar_type(component_type) == "bool" and operator not in {
            "==",
            "!=",
        }:
            return None
        return "bool"

    def integer_index_value_type(self, type_name):
        vector_info = self.vector_type_info(type_name)
        if vector_info is not None:
            return self.vector_type_for_components("int", vector_info["size"])
        if self.normalize_scalar_type(type_name) is not None:
            return "int"
        return None

    def bitcast_value_type(self, type_name, component_type):
        vector_info = self.vector_type_info(type_name)
        if vector_info is not None:
            return self.vector_type_for_components(component_type, vector_info["size"])
        if self.normalize_scalar_type(type_name) is not None:
            return component_type
        return None

    def promoted_argument_type(self, arg_types):
        result_type = None
        for arg_type in arg_types:
            if arg_type is None:
                continue
            if result_type is None:
                result_type = arg_type
                continue
            result_type = self.promoted_value_type(result_type, arg_type)
        return result_type

    def promoted_value_type(self, left_type, right_type):
        for promoted in (
            self.promoted_vector_type(left_type, right_type),
            self.promoted_vector_scalar_type(left_type, right_type),
            self.promoted_matrix_type(left_type, right_type),
            self.promoted_matrix_scalar_type(left_type, right_type),
            self.promoted_scalar_type(left_type, right_type),
        ):
            if promoted is not None:
                return promoted
        return left_type or right_type

    def match_expression_result_type(self, expr):
        subject_type = self.expression_result_type(getattr(expr, "expression", None))
        result_type = None

        for arm in getattr(expr, "arms", []) or []:
            saved_variable_types = self.variable_types.copy()
            saved_local_variable_names = self.local_variable_names.copy()
            try:
                self.register_match_pattern_bindings(
                    getattr(arm, "pattern", None),
                    subject_type,
                )
                arm_type = self.match_arm_body_result_type(getattr(arm, "body", None))
            finally:
                self.variable_types = saved_variable_types
                self.local_variable_names = saved_local_variable_names

            if arm_type is None:
                continue
            if result_type is None:
                result_type = arm_type
                continue

            promoted_vector = self.promoted_vector_type(result_type, arm_type)
            if promoted_vector is not None:
                result_type = promoted_vector
                continue

            promoted_scalar = self.promoted_scalar_type(result_type, arm_type)
            if promoted_scalar is not None:
                result_type = promoted_scalar

        return result_type

    def match_arm_body_result_type(self, body):
        statements = self.statement_list(body)
        if not statements:
            return None

        tail = statements[-1]
        if hasattr(tail, "expression"):
            return self.expression_result_type(tail.expression)
        if isinstance(tail, ReturnNode):
            return self.expression_result_type(getattr(tail, "value", None))
        if isinstance(tail, MatchNode):
            return self.expression_result_type(tail)
        return None

    def constructor_result_type(self, expr):
        type_name = self.normalize_turbofish_type_name(
            self.convert_type_node_to_string(expr.constructor_type)
        )
        variant_info = self.enum_variant_struct_constructor_info(type_name)
        if variant_info is not None:
            enum_type, _variant_name, _field_types = variant_info
            return enum_type

        base_type, existing_args = self.generic_type_parts(type_name)
        if existing_args or base_type not in self.struct_generic_params:
            return type_name

        generic_params = self.struct_generic_params.get(base_type, [])
        if not generic_params:
            return type_name

        substitutions = {}
        named_arguments = getattr(expr, "named_arguments", {}) or {}
        for field_name, value in named_arguments.items():
            member_type = self.resolve_struct_member_type(base_type, field_name)
            value_type = self.expression_result_type(value)
            self.collect_type_parameter_bindings(
                member_type,
                value_type,
                substitutions,
                set(generic_params),
            )

        if not all(param in substitutions for param in generic_params):
            return type_name

        args = ", ".join(substitutions[param] for param in generic_params)
        return f"{base_type}<{args}>"

    def resolve_struct_member_type(self, object_type, member):
        if object_type is None:
            return None
        if hasattr(object_type, "name") or hasattr(object_type, "element_type"):
            object_type = self.convert_type_node_to_string(object_type)
        else:
            object_type = str(object_type)

        exact_members = self.struct_member_types.get(object_type, {})
        if member in exact_members:
            return exact_members[member]

        base_type, generic_args = self.generic_type_parts(object_type)
        member_types = self.struct_member_types.get(base_type, {})
        if member not in member_types:
            return None

        member_type = member_types[member]
        generic_params = self.struct_generic_params.get(base_type, [])
        substitutions = dict(zip(generic_params, generic_args))
        return self.substitute_type_parameters(member_type, substitutions)

    def resolve_struct_positional_member_types(self, object_type):
        if object_type is None:
            return []
        if hasattr(object_type, "name") or hasattr(object_type, "element_type"):
            object_type = self.convert_type_node_to_string(object_type)
        else:
            object_type = str(object_type)

        exact_members = self.struct_member_types.get(object_type)
        if exact_members is not None:
            return list(exact_members.values())

        base_type, generic_args = self.generic_type_parts(object_type)
        member_types = self.struct_member_types.get(base_type)
        if not member_types:
            return []

        generic_params = self.struct_generic_params.get(base_type, [])
        substitutions = dict(zip(generic_params, generic_args))
        return [
            self.substitute_type_parameters(member_type, substitutions)
            for member_type in member_types.values()
        ]

    def generic_type_parts(self, type_name):
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)

        if "<" not in type_name or not type_name.endswith(">"):
            return type_name, []

        base_type, args_text = type_name.split("<", 1)
        args_text = args_text[:-1]
        args = [arg.strip() for arg in self.split_top_level_generic_args(args_text)]
        return base_type, [arg for arg in args if arg]

    def substitute_type_parameters(self, type_name, substitutions):
        if type_name is None:
            return None
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)

        if type_name in substitutions:
            return substitutions[type_name]

        if self.is_array_type_name(type_name):
            base_type, sizes = self.c_array_type_parts(type_name)
            mapped_base = self.substitute_type_parameters(base_type, substitutions)
            return self.format_c_array_type(mapped_base, sizes)

        base_type, generic_args = self.generic_type_parts(type_name)
        if not generic_args:
            return type_name

        mapped_args = [
            self.substitute_type_parameters(arg, substitutions) for arg in generic_args
        ]
        return f"{base_type}<{', '.join(mapped_args)}>"

    def collect_type_parameter_bindings(
        self,
        expected_type,
        actual_type,
        substitutions,
        generic_params,
    ):
        if expected_type is None or actual_type is None:
            return

        expected_type = str(expected_type)
        actual_type = str(actual_type)
        if expected_type in generic_params:
            substitutions.setdefault(expected_type, actual_type)
            return

        expected_base, expected_args = self.generic_type_parts(expected_type)
        actual_base, actual_args = self.generic_type_parts(actual_type)
        if expected_base != actual_base or len(expected_args) != len(actual_args):
            return

        for expected_arg, actual_arg in zip(expected_args, actual_args):
            self.collect_type_parameter_bindings(
                expected_arg,
                actual_arg,
                substitutions,
                generic_params,
            )

    def array_access_element_type(self, expr):
        array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
        array_type = self.expression_result_type(array_expr)
        if array_type is None:
            array_name = self.get_expression_name(array_expr)
            array_type = self.variable_types.get(array_name)
            if array_type is None:
                return None
        if hasattr(array_type, "name") or hasattr(array_type, "element_type"):
            array_type = self.convert_type_node_to_string(array_type)
        else:
            array_type = str(array_type)
        reference_parts = self.reference_type_parts(array_type)
        if reference_parts is not None:
            # A storage slice or workgroup array parameter indexes like the array it refers to.
            array_type = reference_parts[1]
        pointer_parts = self.pointer_type_parts(self.map_type(array_type))
        if pointer_parts is not None:
            return pointer_parts[1]
        if "[" not in array_type or "]" not in array_type:
            matrix_info = self.matrix_type_info(array_type)
            if matrix_info is not None:
                # m[i] is column i of a matrix.
                return self.vector_type_for_components(
                    matrix_info["component_type"], matrix_info["rows"]
                )
            vector_info = self.vector_type_info(array_type)
            if vector_info is not None:
                return vector_info["component_type"]
            patch_info = self.rust_tessellation_patch_info(array_type)
            if patch_info is not None and patch_info["valid"]:
                return patch_info["element_type"]
            return None
        element_type, _size = self.c_array_outer_element_type_and_size(array_type)
        return element_type or None

    def vector_type_info(self, type_name):
        if type_name is None:
            return None
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)

        mapped_type = self.map_type(type_name)
        mapped_type = self.unqualify_runtime_type_name(mapped_type)
        vector_details = {
            "Vec2<f16>": ("f16", 2),
            "Vec3<f16>": ("f16", 3),
            "Vec4<f16>": ("f16", 4),
            "Vec2<f32>": ("float", 2),
            "Vec3<f32>": ("float", 3),
            "Vec4<f32>": ("float", 4),
            "Vec2<f64>": ("double", 2),
            "Vec3<f64>": ("double", 3),
            "Vec4<f64>": ("double", 4),
            "Vec2<i8>": ("i8", 2),
            "Vec3<i8>": ("i8", 3),
            "Vec4<i8>": ("i8", 4),
            "Vec2<u8>": ("u8", 2),
            "Vec3<u8>": ("u8", 3),
            "Vec4<u8>": ("u8", 4),
            "Vec2<i16>": ("i16", 2),
            "Vec3<i16>": ("i16", 3),
            "Vec4<i16>": ("i16", 4),
            "Vec2<u16>": ("u16", 2),
            "Vec3<u16>": ("u16", 3),
            "Vec4<u16>": ("u16", 4),
            "Vec2<i32>": ("int", 2),
            "Vec3<i32>": ("int", 3),
            "Vec4<i32>": ("int", 4),
            "Vec2<u32>": ("uint", 2),
            "Vec3<u32>": ("uint", 3),
            "Vec4<u32>": ("uint", 4),
            "Vec2<bool>": ("bool", 2),
            "Vec3<bool>": ("bool", 3),
            "Vec4<bool>": ("bool", 4),
        }
        details = vector_details.get(mapped_type)
        if details is None:
            return None
        component_type, size = details
        return {"component_type": component_type, "size": size}

    def unqualify_runtime_type_name(self, type_name):
        type_name = str(type_name)
        if type_name.startswith("math::"):
            return type_name[len("math::") :]
        return type_name

    def matrix_type_info(self, type_name):
        if type_name is None:
            return None
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)
        type_name = self.normalize_hlsl_matrix_type_name(type_name)

        matrix_details = {
            "mat2": (2, 2),
            "mat3": (3, 3),
            "mat4": (4, 4),
            "mat2x2": (2, 2),
            "mat2x3": (2, 3),
            "mat2x4": (2, 4),
            "mat3x2": (3, 2),
            "mat3x3": (3, 3),
            "mat3x4": (3, 4),
            "mat4x2": (4, 2),
            "mat4x3": (4, 3),
            "mat4x4": (4, 4),
            "dmat2": (2, 2),
            "dmat3": (3, 3),
            "dmat4": (4, 4),
            "dmat2x2": (2, 2),
            "dmat2x3": (2, 3),
            "dmat2x4": (2, 4),
            "dmat3x2": (3, 2),
            "dmat3x3": (3, 3),
            "dmat3x4": (3, 4),
            "dmat4x2": (4, 2),
            "dmat4x3": (4, 3),
            "dmat4x4": (4, 4),
        }
        details = matrix_details.get(type_name)
        if details is None:
            return None
        columns, rows = details
        component_type = "double" if type_name.startswith("dmat") else "float"
        return {"columns": columns, "rows": rows, "component_type": component_type}

    def normalize_hlsl_matrix_type_name(self, type_name):
        if type_name is None:
            return None

        compact = re.sub(r"\s+", "", str(type_name))
        match = re.fullmatch(
            r"(float|double|half|min16float|float16_t)([234])x([234])",
            compact,
        )
        if match is None:
            return str(type_name)

        component_type, rows, columns = match.groups()
        prefix = "dmat" if component_type == "double" else "mat"
        rows = int(rows)
        columns = int(columns)
        if rows == columns:
            return f"{prefix}{columns}"
        return f"{prefix}{columns}x{rows}"

    def scalar_type_for_type_constructor(self, scalar_type):
        scalar_type = self.normalize_scalar_type(scalar_type)
        aliases = {
            "bool": "bool",
            "i8": "i8",
            "u8": "u8",
            "i16": "short",
            "u16": "ushort",
            "i32": "int",
            "u32": "uint",
            "i64": "long",
            "u64": "ulong",
            "f16": "half",
            "f32": "float",
            "f64": "double",
        }
        return aliases.get(scalar_type)

    def matrix_type_for_dimensions(self, component_type, columns, rows):
        component_type = self.scalar_type_for_type_constructor(component_type)
        if component_type == "double":
            prefix = "dmat"
        elif component_type == "float":
            prefix = "mat"
        else:
            return None

        if columns == rows:
            return f"{prefix}{columns}"
        return f"{prefix}{columns}x{rows}"

    def vector_type_for_components(self, component_type, component_count):
        normalized_component = self.normalize_scalar_type(component_type)
        if (
            normalized_component in {"i8", "u8", "i16", "u16", "f16"}
            and 2 <= component_count <= 4
        ):
            return f"vec{component_count}<{normalized_component}>"

        component_type = self.scalar_type_for_type_constructor(component_type)
        if component_type is None:
            return None
        if component_count < 2 or component_count > 4:
            return component_type
        prefixes = {
            "float": "vec",
            "double": "dvec",
            "int": "ivec",
            "uint": "uvec",
            "bool": "bvec",
        }
        prefix = prefixes.get(component_type)
        if prefix is None:
            return None
        return f"{prefix}{component_count}"
