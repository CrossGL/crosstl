"""Utilities for lowering vector arithmetic expressions during code generation."""

from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)


class VectorArithmeticMixin:
    def collect_function_return_types(self, ast_node):
        function_return_types = {}

        for func in getattr(ast_node, "functions", []):
            function_return_types[func.name] = getattr(func, "return_type", None)

        for stage in getattr(ast_node, "stages", {}).values():
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                function_return_types[entry_point.name] = getattr(
                    entry_point, "return_type", None
                )
            for func in getattr(stage, "local_functions", []):
                function_return_types[func.name] = getattr(func, "return_type", None)

        return function_return_types

    def resource_call_result_type(self, func_name, raw_args):
        if not isinstance(func_name, str):
            return None

        resource_type = None
        if raw_args:
            resource_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
            )

        if func_name in {"textureSize", "imageSize"} and resource_type:
            spec = self.dimension_query_spec(resource_type)
            if spec is None:
                return None
            dimensions = spec[0]
            return self.query_return_type(dimensions)

        if func_name in {"textureSamples", "imageSamples", "textureQueryLevels"}:
            return "int"

        if func_name == "textureQueryLod":
            return "float2"

        if func_name in {
            "textureGather",
            "textureGatherOffset",
            "textureGatherOffsets",
            "textureGatherCompare",
            "textureGatherCompareOffset",
        }:
            return "float4"

        if func_name in {
            "textureCompare",
            "textureCompareLod",
            "textureCompareGrad",
            "textureCompareOffset",
            "textureCompareLodOffset",
            "textureCompareGradOffset",
        }:
            return "float"

        if func_name in {"texture", "textureLod", "textureGrad", "texelFetch"}:
            if self.is_shadow_resource_type(resource_type):
                return "float"
            return "float4"

        if func_name == "imageLoad" and resource_type:
            return self.image_value_type(resource_type)

        return None

    def expression_result_type(self, node):
        if node is None:
            return None
        if isinstance(node, (IdentifierNode, VariableNode, ArrayAccessNode)):
            return self.get_expression_type(node)
        if isinstance(node, LiteralNode):
            literal_type = getattr(getattr(node, "literal_type", None), "name", None)
            if literal_type:
                return literal_type
            if isinstance(node.value, bool):
                return "bool"
            if isinstance(node.value, float):
                return "float"
            if isinstance(node.value, int):
                return "int"
            return None
        if isinstance(node, FunctionCallNode):
            func_expr = getattr(node, "function", getattr(node, "name", None))
            func_name = getattr(func_expr, "name", func_expr)
            if isinstance(func_name, str) and self.vector_type_info(func_name):
                return func_name
            resource_result_type = self.resource_call_result_type(
                func_name, getattr(node, "arguments", getattr(node, "args", []))
            )
            if resource_result_type is not None:
                return resource_result_type
            if isinstance(func_name, str):
                return self.function_return_types.get(func_name)
            return None
        if isinstance(node, BinaryOpNode):
            left_type = self.expression_result_type(node.left)
            right_type = self.expression_result_type(node.right)
            if self.vector_type_info(left_type):
                return left_type
            if self.vector_type_info(right_type):
                return right_type
            return left_type or right_type
        if isinstance(node, UnaryOpNode):
            return self.expression_result_type(node.operand)
        if isinstance(node, TernaryOpNode):
            return self.expression_result_type(
                node.true_expr
            ) or self.expression_result_type(node.false_expr)
        if isinstance(node, MemberAccessNode):
            object_expr = getattr(node, "object_expr", getattr(node, "object", None))
            object_type = self.expression_result_type(object_expr)
            object_type_name = (
                self.convert_type_node_to_string(object_type)
                if object_type is not None and not isinstance(object_type, str)
                else object_type
            )
            member = getattr(node, "member", "")
            struct_members = self.struct_member_types.get(object_type_name, {})
            if member in struct_members:
                return struct_members[member]
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

    def vector_type_info(self, type_name):
        if type_name is None:
            return None
        if not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        mapped_type = self.map_vector_arithmetic_type(type_name)
        vector_details = {
            "float2": ("make_float2", "float", ("x", "y")),
            "float3": ("make_float3", "float", ("x", "y", "z")),
            "float4": ("make_float4", "float", ("x", "y", "z", "w")),
            "double2": ("make_double2", "double", ("x", "y")),
            "double3": ("make_double3", "double", ("x", "y", "z")),
            "double4": ("make_double4", "double", ("x", "y", "z", "w")),
            "int2": ("make_int2", "int", ("x", "y")),
            "int3": ("make_int3", "int", ("x", "y", "z")),
            "int4": ("make_int4", "int", ("x", "y", "z", "w")),
            "uint2": ("make_uint2", "uint", ("x", "y")),
            "uint3": ("make_uint3", "uint", ("x", "y", "z")),
            "uint4": ("make_uint4", "uint", ("x", "y", "z", "w")),
            "uchar2": ("make_uchar2", "bool", ("x", "y")),
            "uchar3": ("make_uchar3", "bool", ("x", "y", "z")),
            "uchar4": ("make_uchar4", "bool", ("x", "y", "z", "w")),
        }
        details = vector_details.get(mapped_type)
        if details is None:
            return None
        constructor, component_type, components = details
        return {
            "type": mapped_type,
            "constructor": constructor,
            "component_type": component_type,
            "components": components,
        }

    def vector_type_for_components(self, component_type, component_count):
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

    def lower_vector_binary_operation(
        self,
        left_node,
        left_expr,
        right_node,
        right_expr,
        operator,
    ):
        if operator not in {"+", "-", "*", "/"}:
            return None
        left_type = self.expression_result_type(left_node)
        right_type = self.expression_result_type(right_node)
        left_info = self.vector_type_info(left_type)
        right_info = self.vector_type_info(right_type)
        if not left_info and not right_info:
            return None
        if left_info and right_info:
            if len(left_info["components"]) != len(right_info["components"]):
                return None
            helper_name = self.require_vector_binary_helper(
                left_info, operator, "vector"
            )
            if helper_name is None:
                return None
            return f"{helper_name}({left_expr}, {right_expr})"
        if left_info and right_type is not None:
            helper_name = self.require_vector_binary_helper(
                left_info, operator, "scalar_right"
            )
            if helper_name is None:
                return None
            return f"{helper_name}({left_expr}, {right_expr})"
        if right_info and left_type is not None:
            helper_name = self.require_vector_binary_helper(
                right_info, operator, "scalar_left"
            )
            if helper_name is None:
                return None
            return f"{helper_name}({left_expr}, {right_expr})"
        return None

    def require_vector_binary_helper(self, vector_info, operator, operand_shape):
        if vector_info["component_type"] == "bool":
            return None

        operator_names = {
            "+": "add",
            "-": "sub",
            "*": "mul",
            "/": "div",
        }
        operation_name = operator_names[operator]
        helper_name = f"cgl_{vector_info['type']}_{operation_name}"
        if operand_shape == "scalar_right":
            helper_name += "_scalar"
        elif operand_shape == "scalar_left":
            helper_name = f"cgl_scalar_{operation_name}_{vector_info['type']}"

        if helper_name in self.helper_functions:
            return helper_name

        vector_type = vector_info["type"]
        scalar_type = self.vector_scalar_parameter_type(vector_info)
        components = vector_info["components"]
        constructor = vector_info["constructor"]

        if operand_shape == "vector":
            params = f"{vector_type} lhs, {vector_type} rhs"
            args = [
                f"(lhs.{component} {operator} rhs.{component})"
                for component in components
            ]
        elif operand_shape == "scalar_right":
            params = f"{vector_type} lhs, {scalar_type} rhs"
            args = [f"(lhs.{component} {operator} rhs)" for component in components]
        else:
            params = f"{scalar_type} lhs, {vector_type} rhs"
            args = [f"(lhs {operator} rhs.{component})" for component in components]

        helper = (
            f"__device__ inline {vector_type} {helper_name}({params})\n"
            "{\n"
            f"    return {constructor}({', '.join(args)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def vector_scalar_parameter_type(self, vector_info):
        if vector_info["component_type"] == "uint":
            return "unsigned int"
        return vector_info["component_type"]
