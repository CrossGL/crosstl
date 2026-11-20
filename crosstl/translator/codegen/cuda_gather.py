"""Texture gather and the remaining node visitors of the CUDA generator."""

from ..ast import (
    LiteralNode,
    UnaryOpNode,
)
from ..matrix_shape import matrix_type_spelling
from .array_utils import prepend_array_dimension
from .c_wide_integer_vectors import wide_integer_vector_type


class CudaGatherMixin:
    """Gather members of ``CudaCodeGen``."""

    def generate_texture_gather_call(self, func_name, raw_args, args):
        texture_type = self.resource_base_type(
            self.resource_expression_type(raw_args[0])
        )
        if texture_type != "sampler2D":
            return None

        coordinate_index = 1
        offset_index = None
        component = None
        component_arg = None
        if func_name == "textureGather":
            if len(args) == 2:
                pass
            elif len(args) == 3:
                if self.is_sampler_state_argument(raw_args[1]):
                    coordinate_index = 2
                else:
                    component = args[2]
                    component_arg = raw_args[2]
            elif len(args) == 4 and self.is_sampler_state_argument(raw_args[1]):
                coordinate_index = 2
                component = args[3]
                component_arg = raw_args[3]
            else:
                return None
        elif func_name == "textureGatherOffset":
            sampler_arg_offset = self.sampled_texture_sampler_argument_offset(raw_args)
            coordinate_index = 1 + sampler_arg_offset
            offset_index = coordinate_index + 1
            if len(args) <= offset_index:
                return None
            if len(args) > offset_index + 1:
                component = args[offset_index + 1]
                component_arg = raw_args[offset_index + 1]
        else:
            return None

        coordinate_diagnostic = self.texture_coordinate_rank_diagnostic(
            func_name,
            texture_type,
            raw_args[coordinate_index],
        )
        if coordinate_diagnostic is not None:
            return coordinate_diagnostic

        if offset_index is not None:
            offset_diagnostic = self.texture_offset_rank_diagnostic(
                func_name,
                texture_type,
                raw_args[offset_index],
            )
            if offset_diagnostic is not None:
                return offset_diagnostic

        component_diagnostic = self.texture_gather_component_diagnostic(
            texture_type, component_arg
        )
        if component_diagnostic is not None:
            return component_diagnostic

        texture_name = args[0]
        coord = args[coordinate_index]
        x = self.coord_component(coord, "x")
        y = self.coord_component(coord, "y")
        if offset_index is not None:
            offset = args[offset_index]
            x = f"({x} + {self.coord_component(offset, 'x')})"
            y = f"({y} + {self.coord_component(offset, 'y')})"
        gather_args = f"{texture_name}, " f"{x}, " f"{y}"
        if component is not None:
            return f"tex2Dgather<float4>({gather_args}, {component})"
        return f"tex2Dgather<float4>({gather_args})"

    def texture_gather_component_diagnostic(self, texture_type, component_arg):
        if not self.is_texture_gather_literal_component(component_arg):
            return None
        component_value = self.literal_int_value(component_arg)
        if component_value in {0, 1, 2, 3}:
            return None
        return self.unsupported_sampled_resource_call(
            "textureGather component literal must be 0, 1, 2, or 3",
            texture_type,
            [],
        )

    def is_texture_gather_literal_component(self, node):
        if isinstance(node, LiteralNode):
            return True
        if isinstance(node, UnaryOpNode) and not getattr(node, "is_postfix", False):
            operator = getattr(node, "operator", getattr(node, "op", None))
            if operator in {"+", "-"}:
                return self.is_texture_gather_literal_component(node.operand)
        return False

    def literal_int_value(self, node):
        if isinstance(node, LiteralNode):
            value = node.value
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                literal_type = getattr(
                    getattr(node, "literal_type", None), "name", None
                )
                if literal_type in {"char", "string"}:
                    return None
                try:
                    return int(value, 0)
                except ValueError:
                    return None
            return None

        if isinstance(node, UnaryOpNode) and not getattr(node, "is_postfix", False):
            operator = getattr(node, "operator", getattr(node, "op", None))
            if operator not in {"+", "-"}:
                return None
            value = self.literal_int_value(node.operand)
            if value is None:
                return None
            if operator == "-":
                return -value
            return value

        return None

    def is_sampler_state_argument(self, raw_arg):
        return (
            self.resource_base_type(self.resource_expression_type(raw_arg)) == "sampler"
        )

    def visit_ArrayNode(self, node):
        """Visit array declaration"""
        if hasattr(node, "element_type"):
            element_type = self.convert_crossgl_type_to_cuda(node.element_type)
        else:
            element_type = self.convert_crossgl_type_to_cuda(node.vtype)

        if node.size:
            self.emit(
                f"{element_type} {node.name}[{self.format_array_size(node.size)}];"
            )
        else:
            # Dynamic array - use pointer in CUDA
            self.emit(f"{element_type}* {node.name};")

    def visit_TernaryOpNode(self, node):
        """Visit ternary conditional operator"""
        condition = self.visit(node.condition)
        true_expr = self.visit(node.true_expr)
        false_expr = self.visit(node.false_expr)
        lowered = self.lower_vector_ternary_operation(
            node.condition,
            condition,
            node.true_expr,
            true_expr,
            node.false_expr,
            false_expr,
        )
        if lowered is not None:
            return lowered
        return f"({condition} ? {true_expr} : {false_expr})"

    def visit_list(self, node_list):
        """Visit a list of nodes"""
        results = []
        for node in node_list:
            result = self.visit(node)
            if result:
                results.append(result)
        return results

    def visit_str(self, node):
        """Visit string literals"""
        return node

    def visit_int(self, node):
        """Visit integer literals"""
        return str(node)

    def visit_float(self, node):
        """Visit float literals"""
        return str(node)

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        if hasattr(type_node, "pointee_type"):
            pointee_type = self.convert_type_node_to_string(type_node.pointee_type)
            prefix = "" if getattr(type_node, "is_mutable", True) else "const "
            return f"{prefix}{pointee_type}*"
        if hasattr(type_node, "referenced_type"):
            referenced_type = self.convert_type_node_to_string(
                type_node.referenced_type
            )
            prefix = "" if getattr(type_node, "is_mutable", False) else "const "
            return f"{prefix}{referenced_type}&"
        if hasattr(type_node, "name"):
            generic_args = getattr(type_node, "generic_args", [])
            if generic_args:
                converted_args = []
                for arg in generic_args:
                    if (
                        hasattr(arg, "name")
                        or hasattr(arg, "element_type")
                        or hasattr(arg, "pointee_type")
                        or hasattr(arg, "referenced_type")
                    ):
                        converted = self.convert_type_node_to_string(arg)
                    elif hasattr(arg, "value"):
                        converted = str(arg.value)
                    else:
                        converted = str(arg)
                    if converted is None:
                        if hasattr(arg, "value"):
                            converted = str(arg.value)
                        else:
                            converted = str(arg)
                    converted_args.append(converted)
                args = ", ".join(converted_args)
                return f"{type_node.name}<{args}>"
            return type_node.name
        elif hasattr(type_node, "element_type"):
            if hasattr(type_node, "rows"):
                element_type = self.convert_type_node_to_string(type_node.element_type)
                return matrix_type_spelling(type_node, element_type, short_square=False)
            elif not hasattr(type_node, "size"):
                return str(type_node)
            elif str(type(type_node)).find("ArrayType") != -1:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                if type_node.size is not None:
                    return prepend_array_dimension(
                        element_type, self.format_array_size(type_node.size)
                    )
                else:
                    return prepend_array_dimension(element_type)
            else:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                size = type_node.size
                if element_type == "float":
                    return f"vec{size}"
                elif element_type in {"f16", "float16", "half"}:
                    return f"vec{size}<f16>"
                elif element_type == "int":
                    return f"ivec{size}"
                elif element_type == "uint":
                    return f"uvec{size}"
                elif element_type in {"i8", "u8", "i16", "u16"}:
                    return f"vec{size}<{element_type}>"
                elif wide_integer_vector_type(f"vec{size}<{element_type}>"):
                    return f"vec{size}<{element_type}>"
                else:
                    return f"{element_type}{size}"
        else:
            return str(type_node)
