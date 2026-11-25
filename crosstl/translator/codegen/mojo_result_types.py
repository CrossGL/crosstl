"""Result types that ``MojoCodeGen.expression_result_type`` infers, by kind of expression: values and operators, function calls, and texture operations."""

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    BinaryOpNode,
    FunctionCallNode,
    MatchNode,
    TernaryOpNode,
    TextureOpNode,
    UnaryOpNode,
    VariableNode,
)
from ..unhandled import (
    UNHANDLED,
)
from .generic_function_utils import (
    generic_function_call_name,
    numeric_trait_method_result_type,
)
from .mojo_numeric_functions import mojo_numeric_function_result_type
from .mojo_tables import (
    MOJO_ATOMIC_OP_ALIASES,
    MOJO_BUILTIN_PLACEHOLDERS,
    MOJO_DTYPE_INFO,
    MOJO_GENERIC_TEXTURE_BUILTINS,
    MOJO_IMAGE_ATOMIC_BUILTINS,
    MOJO_MATRIX_TYPES,
    MOJO_SCALAR_ATOMIC_OPS,
    MOJO_SCALAR_ATOMIC_TYPES,
)

# CrossGL built-ins that apply per lane and return the type of their operands (a vector operand gives the vector
# type), so ``nextafter(abs(x), y)`` and ``isinf(abs(x))`` see a float operand.
_PER_LANE_SAME_TYPE_BUILTINS = frozenset(
    {
        "abs",
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atanh",
        "ceil",
        "cos",
        "cosh",
        "exp",
        "exp2",
        "floor",
        "fma",
        "log",
        "log2",
        "round",
        "sign",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
        "trunc",
    }
)


class MojoResultTypeMixin:
    """Result-type members of ``MojoCodeGen.expression_result_type``."""

    def mojo_value_expression_result_type(
        self,
        expr,
    ):
        """Result type of a raw string, variable, array literal, match, array access, unary, binary or ternary expression."""
        if isinstance(expr, str):
            if expr in MOJO_BUILTIN_PLACEHOLDERS and expr not in self.variable_types:
                return MOJO_BUILTIN_PLACEHOLDERS[expr][1]
            return self.identifier_replacement_result_type(
                expr
            ) or self.variable_types.get(expr)
        if isinstance(expr, VariableNode) and hasattr(expr, "name"):
            if (
                expr.name in MOJO_BUILTIN_PLACEHOLDERS
                and expr.name not in self.variable_types
            ):
                return MOJO_BUILTIN_PLACEHOLDERS[expr.name][1]
            return self.identifier_replacement_result_type(
                expr.name
            ) or self.variable_types.get(expr.name)
        if isinstance(expr, ArrayLiteralNode):
            element_type = self.infer_array_literal_element_type(expr)
            return self.array_type_name(element_type, len(expr.elements))
        if isinstance(expr, MatchNode):
            arms = getattr(expr, "arms", []) or []
            self.validate_match_expression_arms(
                arms, self.expression_result_type(getattr(expr, "expression", None))
            )
            for arm in arms:
                arm_expr, _ = self.match_arm_value_expression(arm)
                arm_type = self.expression_result_type(arm_expr)
                if arm_type is not None:
                    return arm_type
            return None
        if isinstance(expr, ArrayAccessNode):
            array_type = self.expression_result_type(expr.array)
            patch_info = self.tessellation_patch_type_info(array_type)
            if patch_info is not None:
                _patch_name, element_type, _point_count = patch_info
                return element_type
            array_element_type = self.array_element_type(array_type)
            if array_element_type is not None:
                return array_element_type
            pointer_element_type = self.pointer_element_type(array_type)
            if pointer_element_type is not None:
                return pointer_element_type
            matrix_info = self.matrix_type_info(array_type)
            if matrix_info is not None:
                dtype, _, rows = matrix_info
                return self.vector_type_name_for_dtype_width(dtype, rows)
            vector_info = self.vector_type_info(array_type)
            if vector_info is not None:
                return MOJO_DTYPE_INFO[vector_info[0]][0]
            return None
        if isinstance(expr, UnaryOpNode):
            op = self.map_operator(expr.op)
            operand_type = self.expression_result_type(expr.operand)
            if op == "not":
                info = self.vector_type_info(operand_type)
                if info is not None and info[0] == "DType.bool":
                    return operand_type
                return "bool"
            if op in {"+", "-", "~"}:
                return operand_type
            return None
        if isinstance(expr, BinaryOpNode):
            left_type = self.expression_result_type(expr.left)
            right_type = self.expression_result_type(expr.right)
            return self.binary_expression_result_type(expr.op, left_type, right_type)
        if isinstance(expr, TernaryOpNode):
            true_type = self.expression_result_type(expr.true_expr)
            false_type = self.expression_result_type(expr.false_expr)
            if true_type == false_type:
                return true_type
            if true_type is None:
                return false_type
            if false_type is None:
                return true_type
            return None
        return UNHANDLED

    def mojo_function_call_result_type(
        self,
        expr,
    ):
        """Result type of a function call."""
        buffer_object_type = self.mojo_buffer_object_result_type(expr)
        if buffer_object_type is not None:
            return buffer_object_type
        if isinstance(expr, FunctionCallNode):
            member_result_type = self.member_function_result_type(expr)
            if member_result_type is not None:
                return member_result_type

            func_name = self.function_call_name(expr)
            if func_name in {"any", "all"} and not self.is_user_defined_function(
                func_name
            ):
                return "bool"
            numeric_result_type = numeric_trait_method_result_type(self, expr)
            if numeric_result_type is not None:
                return numeric_result_type
            specialized_func_name = generic_function_call_name(
                self,
                func_name,
                expr.args,
            )
            if specialized_func_name in self.function_return_types:
                return self.selected_function_return_type(
                    specialized_func_name,
                    expr.args,
                )
            if func_name in self.enum_variant_result_types:
                return self.enum_variant_result_types[func_name]
            if func_name in self.struct_types:
                return func_name
            if func_name in self.vector_constructor_info:
                return func_name
            vector_func_name = self.normalize_generic_vector_type_name(func_name)
            if vector_func_name in self.vector_constructor_info:
                return vector_func_name
            if func_name in MOJO_MATRIX_TYPES:
                return func_name
            if func_name in self.function_return_types:
                return self.selected_function_return_type(func_name, expr.args)
            if func_name in self.scalar_constructor_map:
                return func_name
            if func_name in {"fract", "frac"} and expr.args:
                return self.expression_result_type(expr.args[0]) or "float"
            if func_name == "roundEven" and len(expr.args) == 1:
                return self.round_even_result_type(expr.args[0])
            if func_name in {"inverse", "transpose"} and expr.args:
                return self.expression_result_type(expr.args[0])
            if func_name in {"length", "dot"}:
                return "float"
            if func_name == "normalize" and expr.args:
                return self.expression_result_type(expr.args[0]) or "float"
            if func_name in {"inversesqrt", "inverseSqrt", "rsqrt"} and expr.args:
                return self.expression_result_type(expr.args[0]) or "float"
            if func_name in {"mix", "lerp"} and expr.args:
                return self.expression_result_type(expr.args[0]) or "float"
            if func_name in {"degrees", "radians"} and expr.args:
                return self.expression_result_type(expr.args[0]) or "float"
            if func_name in {"atan", "atan2"} and expr.args:
                for arg in expr.args:
                    arg_type = self.expression_result_type(arg)
                    if self.vector_type_info(arg_type) is not None:
                        return arg_type
                return self.expression_result_type(expr.args[0]) or "float"
            if func_name == "pow" and expr.args:
                return self.expression_result_type(expr.args[0]) or "float"
            if func_name == "mod" and expr.args:
                for arg in expr.args:
                    arg_type = self.expression_result_type(arg)
                    if self.vector_type_info(arg_type) is not None:
                        return arg_type
                return self.expression_result_type(expr.args[0]) or "float"
            if func_name in {"min", "max", "clamp"} and expr.args:
                for arg in expr.args:
                    arg_type = self.expression_result_type(arg)
                    if self.vector_type_info(arg_type) is not None:
                        return arg_type
                return self.expression_result_type(expr.args[0]) or "float"
            if func_name == "saturate" and expr.args:
                return self.expression_result_type(expr.args[0]) or "float"
            if func_name in {"texture", "textureLod", "textureGrad"}:
                args = self.strip_split_sampler_arg(expr.args)
                if args:
                    resource_type = self.map_type(self.expression_result_type(args[0]))
                    return self.texture_result_type_name(resource_type)
                return "vec4"
            if func_name in {"texelFetch", "texelFetchOffset"}:
                if expr.args:
                    resource_type = self.map_type(
                        self.expression_result_type(expr.args[0])
                    )
                    return self.texture_result_type_name(resource_type)
                return "vec4"
            if func_name == "textureQueryLevels":
                return "int"
            if func_name in {"imageSamples", "textureSamples"}:
                return "int"
            if func_name in {"textureSize", "imageSize"} and expr.args:
                resource_type = self.map_type(self.expression_result_type(expr.args[0]))
                size_type = self.resource_size_return_type(resource_type)
                if size_type == "Int32":
                    return "int"
                if size_type == "SIMD[DType.int32, 2]":
                    return "ivec2"
                if size_type == "SIMD[DType.int32, 4]":
                    return "ivec3"
            if func_name == "imageLoad" and expr.args:
                resource_type = self.map_type(self.expression_result_type(expr.args[0]))
                return self.image_result_type_name(resource_type)
            if func_name == "imageStore":
                return "void"
            if func_name == "buffer_load" and expr.args:
                info = self.buffer_resource_info(
                    self.expression_result_type(expr.args[0])
                )
                if info is not None:
                    return self.buffer_helper_element_type(*info)
            if func_name == "buffer_consume" and expr.args:
                info = self.buffer_resource_info(
                    self.expression_result_type(expr.args[0])
                )
                if info is not None:
                    return info[1]
            if func_name in {"buffer_store", "buffer_append", "buffer_dimensions"}:
                return "void"
            reinterpret_name = self.mojo_reinterpret_builtin_name(func_name)
            if reinterpret_name is not None:
                source_type = (
                    self.expression_result_type(expr.args[0]) if expr.args else None
                )
                return self.reinterpret_return_type_name(reinterpret_name, source_type)
            if func_name in MOJO_GENERIC_TEXTURE_BUILTINS:
                _, return_kind = MOJO_GENERIC_TEXTURE_BUILTINS[func_name]
                if return_kind == "float":
                    return "float"
                if return_kind == "vec2":
                    return "vec2"
                if expr.args:
                    resource_type = self.map_type(
                        self.expression_result_type(expr.args[0])
                    )
                    if self.is_texture_resource_type(resource_type):
                        return self.texture_result_type_name(resource_type)
                return "vec4"
            if func_name == "atomicStore" and not self.is_user_defined_function(
                func_name
            ):
                return "void"
            if func_name in MOJO_SCALAR_ATOMIC_OPS and expr.args:
                target_type = self.type_name(self.expression_result_type(expr.args[0]))
                if target_type in MOJO_SCALAR_ATOMIC_TYPES:
                    return target_type
            if func_name in MOJO_IMAGE_ATOMIC_BUILTINS and expr.args:
                resource_type = self.map_type(self.expression_result_type(expr.args[0]))
                value_type = self.image_value_type(resource_type)
                if value_type == "UInt32":
                    return "uint"
                return "int"
            if func_name == "__shfl_down_sync" and len(expr.args) >= 2:
                return self.expression_result_type(expr.args[1])
            numeric_type = mojo_numeric_function_result_type(self, func_name, expr.args)
            if numeric_type is not None:
                return numeric_type
            if (
                func_name in _PER_LANE_SAME_TYPE_BUILTINS
                and expr.args
                and func_name not in self.function_return_types
                and not self.is_user_defined_function(func_name)
            ):
                return self.per_lane_builtin_result_type(expr.args)
            return self.function_return_types.get(func_name)
        return UNHANDLED

    def mojo_texture_operation_result_type(
        self,
        expr,
    ):
        """Result type of a texture operation."""
        if isinstance(expr, TextureOpNode):
            operation = getattr(expr, "operation", "")
            if operation in {
                "SampleCmp",
                "SampleCmpLevel",
                "SampleCmpLevelZero",
                "SampleCmpGrad",
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
            if operation == "textureQueryLod":
                return "vec2"
            if operation in MOJO_GENERIC_TEXTURE_BUILTINS:
                _, return_kind = MOJO_GENERIC_TEXTURE_BUILTINS[operation]
                if return_kind == "float":
                    return "float"
                if return_kind == "vec2":
                    return "vec2"
                resource_type = self.map_type(
                    self.expression_result_type(expr.texture_expr)
                )
                if self.is_texture_resource_type(resource_type):
                    return self.texture_result_type_name(resource_type)
                return "vec4"
            if operation == "textureQueryLevels":
                return "int"
            if operation in {"imageSamples", "textureSamples"}:
                return "int"
            if operation in {"GetDimensions", "textureSize"}:
                resource_type = self.map_type(
                    self.expression_result_type(expr.texture_expr)
                )
                size_type = self.resource_size_return_type(resource_type)
                if size_type == "Int32":
                    return "int"
                if size_type == "SIMD[DType.int32, 2]":
                    return "ivec2"
                if size_type == "SIMD[DType.int32, 4]":
                    return "ivec3"
            if operation in MOJO_ATOMIC_OP_ALIASES:
                resource_type = self.map_type(
                    self.expression_result_type(expr.texture_expr)
                )
                value_type = self.image_value_type(resource_type)
                if value_type == "UInt32":
                    return "uint"
                return "int"
            resource_type = self.map_type(
                self.expression_result_type(expr.texture_expr)
            )
            if self.is_texture_resource_type(resource_type):
                return self.texture_result_type_name(resource_type)
            return "vec4"
        return UNHANDLED

    def per_lane_builtin_result_type(self, args):
        """The type of a per-lane built-in call: its first vector operand's type, else its first operand's."""
        operand_types = [self.expression_result_type(arg) for arg in args]
        for operand_type in operand_types:
            if self.vector_type_info(operand_type) is not None:
                return operand_type
        return operand_types[0]
