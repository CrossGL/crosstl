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
    """Helpers for inferring and lowering vector arithmetic expressions."""

    def collect_function_return_types(self, ast_node):
        """Collect function return types from global and stage-local functions."""
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

    def is_user_defined_function(self, func_name):
        """Return whether a call target names a CrossGL-defined function."""
        return isinstance(func_name, str) and func_name in getattr(
            self,
            "function_return_types",
            {},
        )

    def resource_call_result_type(self, func_name, raw_args):
        """Infer the result type of resource-related intrinsic calls."""
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
        """Infer the best-effort result type for an expression node."""
        if node is None:
            return None
        if isinstance(node, (IdentifierNode, VariableNode)):
            return self.get_expression_type(node)
        if isinstance(node, ArrayAccessNode):
            array_expr = getattr(node, "array_expr", getattr(node, "array", None))
            array_type = self.expression_result_type(array_expr)
            element_type = self.array_access_element_type(array_type)
            if element_type is not None:
                return element_type
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
            raw_args = getattr(node, "arguments", getattr(node, "args", []))
            if func_name == "normalize" and raw_args:
                return self.expression_result_type(raw_args[0])
            if func_name == "cross" and raw_args:
                return self.expression_result_type(raw_args[0])
            if func_name in {"dot", "length"} and raw_args:
                vector_info = self.vector_type_info(
                    self.expression_result_type(raw_args[0])
                )
                if vector_info is not None:
                    return vector_info["component_type"]
            if (
                func_name
                in {
                    "abs",
                    "sign",
                    "fract",
                    "frac",
                    "floor",
                    "ceil",
                    "round",
                    "trunc",
                    "saturate",
                }
                and raw_args
            ):
                return self.expression_result_type(raw_args[0])
            if func_name in {"mod", "min", "max", "atan2", "pow"} and raw_args:
                return self.expression_result_type(raw_args[0]) or (
                    self.expression_result_type(raw_args[1])
                    if len(raw_args) > 1
                    else None
                )
            if func_name == "clamp" and raw_args:
                return self.expression_result_type(raw_args[0])
            if func_name == "mix" and raw_args:
                return self.expression_result_type(raw_args[0]) or (
                    self.expression_result_type(raw_args[1])
                    if len(raw_args) > 1
                    else None
                )
            if func_name in self.scalar_float_math_functions() and raw_args:
                return self.expression_result_type(raw_args[0])
            if isinstance(func_name, str):
                return self.function_return_types.get(func_name)
            return None
        if isinstance(node, BinaryOpNode):
            left_type = self.expression_result_type(node.left)
            right_type = self.expression_result_type(node.right)
            left_info = self.vector_type_info(left_type)
            right_info = self.vector_type_info(right_type)
            operator = getattr(node, "operator", getattr(node, "op", "+"))
            if operator in {"<", "<=", ">", ">=", "==", "!="}:
                vector_info = left_info or right_info
                if vector_info is not None:
                    return self.vector_type_for_components(
                        "bool",
                        len(vector_info["components"]),
                    )
                return "bool"
            if left_info:
                return left_type
            if right_info:
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

    def array_access_element_type(self, type_name):
        """Return the element type yielded by one indexing operation."""
        if type_name is None:
            return None

        vector_info = self.vector_type_info(type_name)
        if vector_info is not None:
            return vector_info["component_type"]

        if not isinstance(type_name, str):
            if type_name.__class__.__name__ == "ArrayType":
                return getattr(type_name, "element_type", None)
            type_name = self.convert_type_node_to_string(type_name)

        if "[" not in type_name or "]" not in type_name:
            return None

        open_bracket = type_name.find("[")
        close_bracket = type_name.find("]", open_bracket)
        if close_bracket == -1:
            return None

        base_type = type_name[:open_bracket]
        remaining_suffix = type_name[close_bracket + 1 :]
        if remaining_suffix:
            return f"{base_type}{remaining_suffix}"
        return base_type

    def vector_type_info(self, type_name):
        """Return constructor and component metadata for a vector type."""
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

    def is_repeat_safe_expression(self, expr):
        """Return whether an expression can be safely emitted more than once."""
        if isinstance(expr, (IdentifierNode, VariableNode, LiteralNode)):
            return True
        if isinstance(expr, str):
            return True
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
            return self.is_repeat_safe_expression(object_expr)
        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
            index_expr = getattr(expr, "index_expr", getattr(expr, "index", None))
            return self.is_repeat_safe_expression(
                array_expr
            ) and self.is_repeat_safe_expression(index_expr)
        return False

    def generate_vector_scalar_splat_call(self, vector_info, raw_args, args):
        """Return a single-evaluation helper call for complex scalar splats."""
        if len(args) != 1:
            return None

        arg_type = self.expression_result_type(raw_args[0])
        if arg_type is None or self.vector_type_info(arg_type):
            return None
        if self.is_repeat_safe_expression(raw_args[0]):
            return None

        helper_name = self.require_vector_splat_helper(vector_info)
        return f"{helper_name}({args[0]})"

    def require_vector_splat_helper(self, vector_info):
        """Register a helper that splats one scalar into a CUDA/HIP vector."""
        vector_type = vector_info["type"]
        helper_name = f"cgl_{vector_type}_splat"
        if helper_name in self.helper_functions:
            return helper_name

        scalar_type = self.vector_scalar_parameter_type(vector_info)
        constructor = vector_info["constructor"]
        args = ["value"] * len(vector_info["components"])
        helper = (
            f"__device__ inline {vector_type} {helper_name}({scalar_type} value)\n"
            "{\n"
            f"    return {constructor}({', '.join(args)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def generate_vector_constructor_single_eval_call(self, vector_info, raw_args, args):
        """Return a helper call for constructors that flatten complex vector inputs."""
        pieces, helper_needed = self.vector_constructor_helper_pieces(
            vector_info, raw_args, args
        )
        if not helper_needed:
            return None

        helper_name = self.require_vector_constructor_helper(vector_info, pieces)
        return f"{helper_name}({', '.join(piece['arg_expr'] for piece in pieces)})"

    def vector_constructor_helper_pieces(self, vector_info, raw_args, args):
        target_size = len(vector_info["components"])
        pieces = []
        helper_needed = False
        emitted_lanes = 0

        for raw_arg, arg_expr in zip(raw_args, args):
            if emitted_lanes >= target_size:
                break

            swizzle_components = self.member_swizzle_components(raw_arg)
            if swizzle_components is not None:
                object_node = getattr(
                    raw_arg,
                    "object_expr",
                    getattr(raw_arg, "object", None),
                )
                object_info = self.vector_type_info(
                    self.expression_result_type(object_node)
                )
                if object_info is not None:
                    components = swizzle_components[: target_size - emitted_lanes]
                    pieces.append(
                        {
                            "kind": "vector",
                            "param_type": object_info["type"],
                            "arg_expr": self.visit(object_node),
                            "components": components,
                        }
                    )
                    helper_needed = helper_needed or not self.is_repeat_safe_expression(
                        object_node
                    )
                    emitted_lanes += len(components)
                    continue

            arg_info = self.vector_type_info(self.expression_result_type(raw_arg))
            if arg_info is not None:
                components = arg_info["components"][: target_size - emitted_lanes]
                pieces.append(
                    {
                        "kind": "vector",
                        "param_type": arg_info["type"],
                        "arg_expr": arg_expr,
                        "components": components,
                    }
                )
                helper_needed = helper_needed or not self.is_repeat_safe_expression(
                    raw_arg
                )
                emitted_lanes += len(components)
                continue

            pieces.append(
                {
                    "kind": "scalar",
                    "param_type": self.vector_constructor_scalar_parameter_type(
                        self.expression_result_type(raw_arg),
                        vector_info["component_type"],
                    ),
                    "arg_expr": arg_expr,
                    "components": (),
                }
            )
            emitted_lanes += 1

        return pieces, helper_needed

    def vector_constructor_scalar_parameter_type(self, type_name, fallback_type):
        component_type = self.scalar_component_type(type_name) or fallback_type
        if component_type == "uint":
            return "unsigned int"
        return component_type

    def require_vector_constructor_helper(self, vector_info, pieces):
        vector_type = vector_info["type"]
        signature = "_".join(
            self.vector_constructor_piece_signature(piece) for piece in pieces
        )
        helper_name = self.sanitize_helper_name(
            f"cgl_{vector_type}_construct_{signature}"
        )
        if helper_name in self.helper_functions:
            return helper_name

        constructor = vector_info["constructor"]
        params = [
            f"{piece['param_type']} arg{index}" for index, piece in enumerate(pieces)
        ]
        lanes = []
        for index, piece in enumerate(pieces):
            if piece["kind"] == "vector":
                lanes.extend(
                    f"arg{index}.{component}" for component in piece["components"]
                )
            else:
                lanes.append(f"arg{index}")

        helper = (
            f"__device__ inline {vector_type} {helper_name}({', '.join(params)})\n"
            "{\n"
            f"    return {constructor}({', '.join(lanes)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def generate_vector_swizzle_single_eval_call(
        self, result_info, source_info, object_node, object_expr, components
    ):
        """Return a helper call for multi-lane swizzles of complex expressions."""
        if self.is_repeat_safe_expression(object_node):
            return None

        pieces = [
            {
                "kind": "vector",
                "param_type": source_info["type"],
                "arg_expr": object_expr,
                "components": components,
            }
        ]
        helper_name = self.require_vector_constructor_helper(result_info, pieces)
        return f"{helper_name}({object_expr})"

    def vector_operation_piece(
        self,
        node,
        arg_expr,
        vector_info,
        component_count,
        fallback_component_type,
    ):
        if vector_info is not None:
            return {
                "kind": "vector",
                "param_type": vector_info["type"],
                "arg_expr": arg_expr,
                "components": vector_info["components"][:component_count],
                "repeat_safe": self.is_repeat_safe_expression(node),
            }

        return {
            "kind": "scalar",
            "param_type": self.vector_constructor_scalar_parameter_type(
                self.expression_result_type(node),
                fallback_component_type,
            ),
            "arg_expr": arg_expr,
            "components": (),
            "repeat_safe": self.is_repeat_safe_expression(node),
        }

    def vector_operation_piece_expr(self, piece, component):
        if piece["kind"] == "vector":
            return f"{piece['arg_expr']}.{component}"
        return piece["arg_expr"]

    def vector_operation_piece_param_expr(self, piece, index, component):
        if piece["kind"] == "vector":
            return f"arg{index}.{component}"
        return f"arg{index}"

    def generate_vector_binary_single_eval_call(
        self,
        result_info,
        operation_name,
        operator,
        pieces,
    ):
        if all(piece["repeat_safe"] for piece in pieces):
            return None

        helper_name = self.require_vector_binary_component_helper(
            result_info,
            operation_name,
            operator,
            pieces,
        )
        return f"{helper_name}({', '.join(piece['arg_expr'] for piece in pieces)})"

    def require_vector_binary_component_helper(
        self,
        result_info,
        operation_name,
        operator,
        pieces,
    ):
        signature = "_".join(
            self.vector_constructor_piece_signature(piece) for piece in pieces
        )
        helper_name = self.sanitize_helper_name(
            f"cgl_{result_info['type']}_{operation_name}_{signature}"
        )
        if helper_name in self.helper_functions:
            return helper_name

        params = [
            f"{piece['param_type']} arg{index}" for index, piece in enumerate(pieces)
        ]
        component_args = []
        for component in result_info["components"]:
            left_value = self.vector_operation_piece_param_expr(pieces[0], 0, component)
            right_value = self.vector_operation_piece_param_expr(
                pieces[1], 1, component
            )
            component_args.append(f"({left_value} {operator} {right_value})")

        helper = (
            f"__device__ inline {result_info['type']} {helper_name}"
            f"({', '.join(params)})\n"
            "{\n"
            f"    return {result_info['constructor']}({', '.join(component_args)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def generate_vector_ternary_single_eval_call(self, result_info, pieces):
        if all(piece["repeat_safe"] for piece in pieces):
            return None

        helper_name = self.require_vector_ternary_helper(result_info, pieces)
        return f"{helper_name}({', '.join(piece['arg_expr'] for piece in pieces)})"

    def generate_scalar_clamp_single_eval_call(self, scalar_type, raw_args, args):
        if scalar_type in {"float", "double"}:
            return None
        if all(self.is_repeat_safe_expression(raw_arg) for raw_arg in raw_args):
            return None

        helper_name = self.require_scalar_clamp_helper(scalar_type)
        return f"{helper_name}({', '.join(args)})"

    def require_scalar_clamp_helper(self, scalar_type):
        helper_name = self.sanitize_helper_name(f"cgl_{scalar_type}_clamp")
        if helper_name in self.helper_functions:
            return helper_name

        helper = (
            f"__device__ inline {scalar_type} {helper_name}"
            f"({scalar_type} value, {scalar_type} min_value, "
            f"{scalar_type} max_value)\n"
            "{\n"
            "    return "
            f"{self.format_clamp_component(scalar_type, 'value', 'min_value', 'max_value')};\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def generate_scalar_mix_single_eval_call(self, raw_args, args):
        if all(self.is_repeat_safe_expression(raw_arg) for raw_arg in raw_args):
            return None

        scalar_type = self.scalar_mix_type(raw_args)
        if scalar_type is None:
            return None

        helper_name = self.require_scalar_mix_helper(scalar_type)
        return f"{helper_name}({', '.join(args)})"

    def scalar_mix_type(self, raw_args):
        component_types = []
        for raw_arg in raw_args:
            arg_type = self.expression_result_type(raw_arg)
            if self.vector_type_info(arg_type) is not None:
                return None
            component_type = self.scalar_component_type(arg_type)
            if component_type not in {"float", "double", None}:
                return None
            component_types.append(component_type)
        return "double" if "double" in component_types else "float"

    def require_scalar_mix_helper(self, scalar_type):
        helper_name = self.sanitize_helper_name(f"cgl_{scalar_type}_mix")
        if helper_name in self.helper_functions:
            return helper_name

        helper = (
            f"__device__ inline {scalar_type} {helper_name}"
            f"({scalar_type} x, {scalar_type} y, {scalar_type} a)\n"
            "{\n"
            f"    return {self.format_mix_component('x', 'y', 'a')};\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def lower_bool_scalar_mix_operation(
        self,
        x_node,
        x_expr,
        y_node,
        y_expr,
        factor_node,
        factor_expr,
    ):
        """Lower mix(x, y, bool) using scalar selection semantics."""
        factor_type = self.expression_result_type(factor_node)
        if self.scalar_component_type(factor_type) != "bool":
            return None

        scalar_type = self.scalar_select_type(x_node, y_node)
        if scalar_type is None:
            return None

        if all(
            self.is_repeat_safe_expression(raw_arg)
            for raw_arg in (x_node, y_node, factor_node)
        ):
            return f"({factor_expr} ? {y_expr} : {x_expr})"

        helper_name = self.require_scalar_select_helper(scalar_type)
        return f"{helper_name}({factor_expr}, {y_expr}, {x_expr})"

    def scalar_select_type(self, true_or_false_node, other_node):
        component_types = []
        mapped_types = []
        for raw_arg in (true_or_false_node, other_node):
            arg_type = self.expression_result_type(raw_arg)
            if self.vector_type_info(arg_type) is not None:
                return None
            component_type = self.scalar_component_type(arg_type)
            if component_type is None:
                return None
            component_types.append(component_type)
            mapped_types.append(self.map_vector_arithmetic_type(arg_type))

        if set(component_types) <= {"float", "double"}:
            return "double" if "double" in component_types else "float"
        if len(set(component_types)) == 1:
            if mapped_types[0] == mapped_types[1]:
                return mapped_types[0]
            return self.map_vector_arithmetic_type(component_types[0])
        return None

    def require_scalar_select_helper(self, scalar_type):
        helper_name = self.sanitize_helper_name(f"cgl_{scalar_type}_select")
        if helper_name in self.helper_functions:
            return helper_name

        helper = (
            f"__device__ inline {scalar_type} {helper_name}"
            f"(bool condition, {scalar_type} true_value, "
            f"{scalar_type} false_value)\n"
            "{\n"
            "    return condition ? true_value : false_value;\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def require_vector_ternary_helper(self, result_info, pieces):
        signature = "_".join(
            self.vector_constructor_piece_signature(piece) for piece in pieces
        )
        helper_name = self.sanitize_helper_name(
            f"cgl_{result_info['type']}_select_{signature}"
        )
        if helper_name in self.helper_functions:
            return helper_name

        params = [
            f"{piece['param_type']} arg{index}" for index, piece in enumerate(pieces)
        ]
        component_args = []
        for component in result_info["components"]:
            condition_value = self.vector_operation_piece_param_expr(
                pieces[0], 0, component
            )
            true_value = self.vector_operation_piece_param_expr(pieces[1], 1, component)
            false_value = self.vector_operation_piece_param_expr(
                pieces[2], 2, component
            )
            component_args.append(f"({condition_value} ? {true_value} : {false_value})")

        helper = (
            f"__device__ inline {result_info['type']} {helper_name}"
            f"({', '.join(params)})\n"
            "{\n"
            f"    return {result_info['constructor']}({', '.join(component_args)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def vector_constructor_piece_signature(self, piece):
        if piece["kind"] == "vector":
            return f"{piece['param_type']}_{''.join(piece['components'])}"
        return piece["param_type"]

    def sanitize_helper_name(self, value):
        return "".join(character if character.isalnum() else "_" for character in value)

    def vector_type_for_components(self, component_type, component_count):
        """Return a vector type name for a component type/count pair."""
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

    def lower_vector_unary_operation(self, operand_node, operand_expr, operator):
        """Lower vector unary arithmetic into a helper call when required."""
        if operator not in {"-", "!", "not"}:
            return None

        operand_type = self.expression_result_type(operand_node)
        vector_info = self.vector_type_info(operand_type)
        if vector_info is None:
            return None

        helper_name = (
            self.require_vector_logical_not_helper(vector_info)
            if operator in {"!", "not"}
            else self.require_vector_negation_helper(vector_info)
        )
        if helper_name is None:
            return None
        return f"{helper_name}({operand_expr})"

    def lower_vector_binary_operation(
        self,
        left_node,
        left_expr,
        right_node,
        right_expr,
        operator,
    ):
        """Lower vector binary arithmetic into a helper call when required."""
        if operator not in {"+", "-", "*", "/", "%"}:
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
        if left_info and self.scalar_component_type(right_type) is not None:
            helper_name = self.require_vector_binary_helper(
                left_info, operator, "scalar_right"
            )
            if helper_name is None:
                return None
            return f"{helper_name}({left_expr}, {right_expr})"
        if right_info and self.scalar_component_type(left_type) is not None:
            helper_name = self.require_vector_binary_helper(
                right_info, operator, "scalar_left"
            )
            if helper_name is None:
                return None
            return f"{helper_name}({left_expr}, {right_expr})"
        return None

    def lower_scalar_modulo_operation(
        self,
        left_node,
        left_expr,
        right_node,
        right_expr,
    ):
        """Lower floating-point scalar modulo to CUDA/HIP math functions."""
        left_type = self.expression_result_type(left_node)
        right_type = self.expression_result_type(right_node)
        left_component = self.scalar_component_type(left_type)
        right_component = self.scalar_component_type(right_type)
        if left_component == "double" or right_component == "double":
            return f"fmod({left_expr}, {right_expr})"
        if left_component == "float" or right_component == "float":
            return f"fmodf({left_expr}, {right_expr})"
        return None

    def generate_min_max_call(self, func_name, raw_args, args):
        left_info = self.vector_type_info(self.expression_result_type(raw_args[0]))
        right_info = self.vector_type_info(self.expression_result_type(raw_args[1]))
        if not left_info and not right_info:
            return None

        if left_info and right_info:
            if len(left_info["components"]) != len(right_info["components"]):
                return None
            helper_name = self.require_vector_min_max_helper(
                left_info,
                func_name,
                "vector",
            )
        elif left_info:
            helper_name = self.require_vector_min_max_helper(
                left_info,
                func_name,
                "scalar_right",
            )
        else:
            helper_name = self.require_vector_min_max_helper(
                right_info,
                func_name,
                "scalar_left",
            )

        if helper_name is None:
            return None
        return f"{helper_name}({args[0]}, {args[1]})"

    def lower_vector_logical_operation(
        self,
        left_node,
        left_expr,
        right_node,
        right_expr,
        operator,
    ):
        """Lower bool-vector logical operations into constructor expressions."""
        logical_operators = {
            "&&": "&&",
            "and": "&&",
            "||": "||",
            "or": "||",
        }
        if operator not in logical_operators:
            return None

        lowered_operator = logical_operators[operator]
        left_type = self.expression_result_type(left_node)
        right_type = self.expression_result_type(right_node)
        left_info = self.vector_type_info(left_type)
        right_info = self.vector_type_info(right_type)
        if not left_info and not right_info:
            return None

        vector_info = left_info or right_info
        if vector_info["component_type"] != "bool":
            return None

        component_count = len(vector_info["components"])
        left_piece = self.vector_operation_piece(
            left_node, left_expr, left_info, component_count, "bool"
        )
        right_piece = self.vector_operation_piece(
            right_node, right_expr, right_info, component_count, "bool"
        )

        if left_info and right_info:
            if (
                len(left_info["components"]) != len(right_info["components"])
                or right_info["component_type"] != "bool"
            ):
                return None
        elif left_info and self.scalar_component_type(right_type) == "bool":
            pass
        elif right_info and self.scalar_component_type(left_type) == "bool":
            pass
        else:
            return None

        helper_call = self.generate_vector_binary_single_eval_call(
            vector_info,
            f"logical_{'and' if lowered_operator == '&&' else 'or'}",
            lowered_operator,
            (left_piece, right_piece),
        )
        if helper_call is not None:
            return helper_call

        component_args = []
        for component in vector_info["components"]:
            left_value = self.vector_operation_piece_expr(left_piece, component)
            right_value = self.vector_operation_piece_expr(right_piece, component)
            component_args.append(f"({left_value} {lowered_operator} {right_value})")

        return f"{vector_info['constructor']}({', '.join(component_args)})"

    def lower_vector_bitwise_operation(
        self,
        left_node,
        left_expr,
        right_node,
        right_expr,
        operator,
    ):
        """Lower integer-vector bitwise operations into constructor expressions."""
        if operator not in {"&", "|", "^", "<<", ">>"}:
            return None

        left_type = self.expression_result_type(left_node)
        right_type = self.expression_result_type(right_node)
        left_info = self.vector_type_info(left_type)
        right_info = self.vector_type_info(right_type)
        if not left_info and not right_info:
            return None

        result_info = self.vector_bitwise_result_info(
            left_type,
            left_info,
            right_type,
            right_info,
            operator,
        )
        if result_info is None:
            return None

        component_args = []
        for component in result_info["components"]:
            left_value = f"{left_expr}.{component}" if left_info else left_expr
            right_value = f"{right_expr}.{component}" if right_info else right_expr
            component_args.append(f"({left_value} {operator} {right_value})")

        return f"{result_info['constructor']}({', '.join(component_args)})"

    def vector_bitwise_result_info(
        self,
        left_type,
        left_info,
        right_type,
        right_info,
        operator,
    ):
        integer_components = {"int", "uint"}
        left_component = (
            left_info["component_type"]
            if left_info
            else self.scalar_component_type(left_type)
        )
        right_component = (
            right_info["component_type"]
            if right_info
            else self.scalar_component_type(right_type)
        )
        if left_component not in integer_components:
            return None
        if right_component not in {None, "int", "uint"}:
            return None

        if left_info is not None:
            result_info = left_info
        elif right_info is not None and operator in {"&", "|", "^"}:
            result_info = right_info
        else:
            return None

        if result_info["component_type"] not in integer_components:
            return None

        if right_info is not None:
            if len(right_info["components"]) != len(result_info["components"]):
                return None
            if operator in {"&", "|", "^"} and (
                right_info["component_type"] != result_info["component_type"]
            ):
                return None

        if left_info is not None and right_info is None:
            if operator in {"&", "|", "^"} and right_component not in {
                None,
                result_info["component_type"],
            }:
                return None
        elif left_info is None and right_info is not None:
            if left_component != result_info["component_type"]:
                return None

        return result_info

    def lower_vector_ternary_operation(
        self,
        condition_node,
        condition_expr,
        true_node,
        true_expr,
        false_node,
        false_expr,
    ):
        """Lower bool-vector ternary selection into a component-wise constructor."""
        condition_info = self.vector_type_info(
            self.expression_result_type(condition_node)
        )
        if condition_info is None or condition_info["component_type"] != "bool":
            return None

        true_type = self.expression_result_type(true_node)
        false_type = self.expression_result_type(false_node)
        true_info = self.vector_type_info(true_type)
        false_info = self.vector_type_info(false_type)
        result_info = self.vector_ternary_result_info(
            condition_info,
            true_type,
            true_info,
            false_type,
            false_info,
        )
        if result_info is None:
            return None

        component_count = len(condition_info["components"])
        pieces = (
            self.vector_operation_piece(
                condition_node,
                condition_expr,
                condition_info,
                component_count,
                "bool",
            ),
            self.vector_operation_piece(
                true_node,
                true_expr,
                true_info,
                component_count,
                result_info["component_type"],
            ),
            self.vector_operation_piece(
                false_node,
                false_expr,
                false_info,
                component_count,
                result_info["component_type"],
            ),
        )
        helper_call = self.generate_vector_ternary_single_eval_call(result_info, pieces)
        if helper_call is not None:
            return helper_call

        component_args = []
        for component in condition_info["components"]:
            true_value = self.vector_operation_piece_expr(pieces[1], component)
            false_value = self.vector_operation_piece_expr(pieces[2], component)
            condition_value = self.vector_operation_piece_expr(pieces[0], component)
            component_args.append(f"({condition_value} ? {true_value} : {false_value})")

        return f"{result_info['constructor']}({', '.join(component_args)})"

    def lower_bool_vector_mix_operation(
        self,
        x_node,
        x_expr,
        y_node,
        y_expr,
        factor_node,
        factor_expr,
        x_info,
        y_info,
        factor_info,
    ):
        """Lower mix(x, y, bvec) using component-wise boolean selection."""
        if factor_info is None or factor_info["component_type"] != "bool":
            return None
        if x_info is None or y_info is None:
            return None
        if (
            len(x_info["components"]) != len(y_info["components"])
            or len(factor_info["components"]) != len(x_info["components"])
            or x_info["component_type"] != y_info["component_type"]
        ):
            return None

        return self.lower_vector_ternary_operation(
            factor_node,
            factor_expr,
            y_node,
            y_expr,
            x_node,
            x_expr,
        )

    def vector_ternary_result_info(
        self,
        condition_info,
        true_type,
        true_info,
        false_type,
        false_info,
    ):
        component_count = len(condition_info["components"])

        if true_info is not None:
            if len(true_info["components"]) != component_count:
                return None
            result_info = true_info
        elif false_info is not None:
            if len(false_info["components"]) != component_count:
                return None
            result_info = false_info
        else:
            component_type = self.scalar_component_type(true_type)
            false_component_type = self.scalar_component_type(false_type)
            if component_type is None:
                component_type = false_component_type
            elif (
                false_component_type is not None
                and false_component_type != component_type
            ):
                return None
            vector_type = self.vector_type_for_components(
                component_type, component_count
            )
            return self.vector_type_info(vector_type)

        if false_info is not None:
            if (
                len(false_info["components"]) != component_count
                or false_info["component_type"] != result_info["component_type"]
            ):
                return None
        elif self.scalar_component_type(false_type) not in {
            None,
            result_info["component_type"],
        }:
            return None

        if true_info is None and self.scalar_component_type(true_type) not in {
            None,
            result_info["component_type"],
        }:
            return None

        return result_info

    def lower_vector_comparison_operation(
        self,
        left_node,
        left_expr,
        right_node,
        right_expr,
        operator,
    ):
        """Lower vector comparisons into bool-vector constructor expressions."""
        if operator not in {"<", "<=", ">", ">=", "==", "!="}:
            return None

        left_type = self.expression_result_type(left_node)
        right_type = self.expression_result_type(right_node)
        left_info = self.vector_type_info(left_type)
        right_info = self.vector_type_info(right_type)
        if not left_info and not right_info:
            return None

        vector_info = left_info or right_info
        component_count = len(vector_info["components"])
        left_piece = self.vector_operation_piece(
            left_node,
            left_expr,
            left_info,
            component_count,
            vector_info["component_type"],
        )
        right_piece = self.vector_operation_piece(
            right_node,
            right_expr,
            right_info,
            component_count,
            vector_info["component_type"],
        )

        if left_info and right_info:
            if len(left_info["components"]) != len(right_info["components"]):
                return None
        elif left_info and right_type is not None:
            pass
        elif right_info and left_type is not None:
            pass
        else:
            return None

        result_type = self.vector_type_for_components(
            "bool",
            len(vector_info["components"]),
        )
        result_info = self.vector_type_info(result_type)
        if result_info is None:
            return None
        operator_names = {
            "<": "lt",
            "<=": "le",
            ">": "gt",
            ">=": "ge",
            "==": "eq",
            "!=": "ne",
        }
        helper_call = self.generate_vector_binary_single_eval_call(
            result_info,
            f"compare_{operator_names[operator]}",
            operator,
            (left_piece, right_piece),
        )
        if helper_call is not None:
            return helper_call

        component_args = []
        for component in vector_info["components"]:
            left_value = self.vector_operation_piece_expr(left_piece, component)
            right_value = self.vector_operation_piece_expr(right_piece, component)
            component_args.append(f"({left_value} {operator} {right_value})")

        return f"{result_info['constructor']}({', '.join(component_args)})"

    def require_vector_min_max_helper(self, vector_info, func_name, operand_shape):
        component_type = vector_info["component_type"]
        if component_type == "bool":
            return None

        vector_type = vector_info["type"]
        scalar_type = self.vector_scalar_parameter_type(vector_info)
        helper_name = f"cgl_{vector_type}_{func_name}"
        if operand_shape == "scalar_right":
            helper_name += "_scalar"
        elif operand_shape == "scalar_left":
            helper_name = f"cgl_scalar_{func_name}_{vector_type}"

        if helper_name in self.helper_functions:
            return helper_name

        components = vector_info["components"]
        constructor = vector_info["constructor"]
        if operand_shape == "vector":
            params = f"{vector_type} lhs, {vector_type} rhs"
            args = [
                self.format_min_max_component(
                    func_name,
                    component_type,
                    f"lhs.{component}",
                    f"rhs.{component}",
                )
                for component in components
            ]
        elif operand_shape == "scalar_right":
            params = f"{vector_type} lhs, {scalar_type} rhs"
            args = [
                self.format_min_max_component(
                    func_name,
                    component_type,
                    f"lhs.{component}",
                    "rhs",
                )
                for component in components
            ]
        else:
            params = f"{scalar_type} lhs, {vector_type} rhs"
            args = [
                self.format_min_max_component(
                    func_name,
                    component_type,
                    "lhs",
                    f"rhs.{component}",
                )
                for component in components
            ]

        helper = (
            f"__device__ inline {vector_type} {helper_name}({params})\n"
            "{\n"
            f"    return {constructor}({', '.join(args)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def format_min_max_component(self, func_name, component_type, left, right):
        if component_type == "float":
            intrinsic = "fminf" if func_name == "min" else "fmaxf"
            return f"{intrinsic}({left}, {right})"
        if component_type == "double":
            intrinsic = "fmin" if func_name == "min" else "fmax"
            return f"{intrinsic}({left}, {right})"

        operator = "<" if func_name == "min" else ">"
        return f"(({left}) {operator} ({right}) ? ({left}) : ({right}))"

    def require_vector_binary_helper(self, vector_info, operator, operand_shape):
        """Register and return a helper function for vector binary arithmetic."""
        if vector_info["component_type"] == "bool":
            return None

        operator_names = {
            "+": "add",
            "-": "sub",
            "*": "mul",
            "/": "div",
            "%": "mod",
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
                self.format_vector_binary_component(
                    vector_info["component_type"],
                    operator,
                    f"lhs.{component}",
                    f"rhs.{component}",
                )
                for component in components
            ]
        elif operand_shape == "scalar_right":
            params = f"{vector_type} lhs, {scalar_type} rhs"
            args = [
                self.format_vector_binary_component(
                    vector_info["component_type"],
                    operator,
                    f"lhs.{component}",
                    "rhs",
                )
                for component in components
            ]
        else:
            params = f"{scalar_type} lhs, {vector_type} rhs"
            args = [
                self.format_vector_binary_component(
                    vector_info["component_type"],
                    operator,
                    "lhs",
                    f"rhs.{component}",
                )
                for component in components
            ]

        helper = (
            f"__device__ inline {vector_type} {helper_name}({params})\n"
            "{\n"
            f"    return {constructor}({', '.join(args)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def format_vector_binary_component(self, component_type, operator, left, right):
        if operator == "%" and component_type in {"float", "double"}:
            func_name = "fmod" if component_type == "double" else "fmodf"
            return f"{func_name}({left}, {right})"
        return f"({left} {operator} {right})"

    def generate_mod_call(self, raw_args, args):
        """Lower vector mod builtins through the same path as binary modulo."""
        if len(raw_args) != 2 or len(args) != 2:
            return None
        return self.lower_vector_binary_operation(
            raw_args[0],
            args[0],
            raw_args[1],
            args[1],
            "%",
        )

    def require_vector_negation_helper(self, vector_info):
        """Register and return a helper function for vector unary negation."""
        if vector_info["component_type"] in {"bool", "uint"}:
            return None

        vector_type = vector_info["type"]
        helper_name = f"cgl_{vector_type}_neg"
        if helper_name in self.helper_functions:
            return helper_name

        components = [
            f"(-value.{component})" for component in vector_info["components"]
        ]
        helper = (
            f"__device__ inline {vector_type} {helper_name}({vector_type} value)\n"
            "{\n"
            f"    return {vector_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def require_vector_logical_not_helper(self, vector_info):
        """Register and return a helper function for bool-vector logical not."""
        if vector_info["component_type"] != "bool":
            return None

        vector_type = vector_info["type"]
        helper_name = f"cgl_{vector_type}_not"
        if helper_name in self.helper_functions:
            return helper_name

        components = [
            f"(!value.{component})" for component in vector_info["components"]
        ]
        helper = (
            f"__device__ inline {vector_type} {helper_name}({vector_type} value)\n"
            "{\n"
            f"    return {vector_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def generate_vector_geometric_call(self, func_name, raw_args, args):
        """Lower vector geometric builtins to generated component-wise helpers."""
        if func_name in {"length", "normalize"}:
            if len(raw_args) != 1:
                return None
            vector_info = self.vector_type_info(
                self.expression_result_type(raw_args[0])
            )
            if vector_info is None:
                return None
            if func_name == "length":
                helper_name = self.require_vector_length_helper(vector_info)
            else:
                helper_name = self.require_vector_normalize_helper(vector_info)
            if helper_name is None:
                return None
            return f"{helper_name}({args[0]})"

        if func_name in {"dot", "cross"}:
            if len(raw_args) != 2:
                return None
            left_info = self.vector_type_info(self.expression_result_type(raw_args[0]))
            right_info = self.vector_type_info(self.expression_result_type(raw_args[1]))
            if (
                left_info is None
                or right_info is None
                or len(left_info["components"]) != len(right_info["components"])
                or left_info["component_type"] != right_info["component_type"]
            ):
                return None
            if func_name == "dot":
                helper_name = self.require_vector_dot_helper(left_info)
            else:
                helper_name = self.require_vector_cross_helper(left_info)
            if helper_name is None:
                return None
            return f"{helper_name}({args[0]}, {args[1]})"

        return None

    def generate_abs_call(self, raw_args, args):
        """Lower abs with scalar type awareness and vector helper support."""
        if len(raw_args) != 1 or len(args) != 1:
            return None

        arg_type = self.expression_result_type(raw_args[0])
        vector_info = self.vector_type_info(arg_type)
        if vector_info is not None:
            helper_name = self.require_vector_abs_helper(vector_info)
            if helper_name is not None:
                return f"{helper_name}({args[0]})"
            return None

        return self.format_abs_component(self.abs_component_type(arg_type), args[0])

    def require_vector_abs_helper(self, vector_info):
        component_type = vector_info["component_type"]
        if component_type == "bool":
            return None

        vector_type = vector_info["type"]
        helper_name = f"cgl_{vector_type}_abs"
        if helper_name in self.helper_functions:
            return helper_name

        components = [
            self.format_abs_component(component_type, f"value.{component}")
            for component in vector_info["components"]
        ]
        helper = (
            f"__device__ inline {vector_type} {helper_name}({vector_type} value)\n"
            "{\n"
            f"    return {vector_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def format_abs_component(self, component_type, value):
        if component_type == "double":
            return f"fabs({value})"
        if component_type == "float" or component_type is None:
            return f"fabsf({value})"
        if component_type == "uint":
            return value
        return f"abs({value})"

    def generate_sign_call(self, raw_args, args):
        """Lower sign with scalar type awareness and vector helper support."""
        if len(raw_args) != 1 or len(args) != 1:
            return None

        arg_type = self.expression_result_type(raw_args[0])
        vector_info = self.vector_type_info(arg_type)
        if vector_info is not None:
            helper_name = self.require_vector_sign_helper(vector_info)
            if helper_name is not None:
                return f"{helper_name}({args[0]})"
            return None

        component_type = self.scalar_component_type(arg_type) or "float"
        if component_type == "bool":
            return None
        return self.format_sign_component(component_type, args[0])

    def require_vector_sign_helper(self, vector_info):
        component_type = vector_info["component_type"]
        if component_type == "bool":
            return None

        vector_type = vector_info["type"]
        helper_name = f"cgl_{vector_type}_sign"
        if helper_name in self.helper_functions:
            return helper_name

        components = [
            self.format_sign_component(component_type, f"value.{component}")
            for component in vector_info["components"]
        ]
        helper = (
            f"__device__ inline {vector_type} {helper_name}({vector_type} value)\n"
            "{\n"
            f"    return {vector_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def format_sign_component(self, component_type, value):
        if component_type == "uint":
            return f"(({value}) > 0u ? 1u : 0u)"

        if component_type == "double":
            zero = "0.0"
            one = "1.0"
            minus_one = "-1.0"
        elif component_type == "float" or component_type is None:
            zero = "0.0f"
            one = "1.0f"
            minus_one = "-1.0f"
        else:
            zero = "0"
            one = "1"
            minus_one = "-1"

        return (
            f"(({value}) > {zero} ? {one} : "
            f"(({value}) < {zero} ? {minus_one} : {zero}))"
        )

    def generate_scalar_math_call(self, func_name, raw_args, args):
        """Lower scalar float/double math builtins with precision-aware names."""
        function_map = self.scalar_float_math_functions()
        signatures = function_map.get(func_name)
        if signatures is None or len(raw_args) != len(args):
            return None

        expected_arity = 2 if func_name == "pow" else 1
        if len(raw_args) != expected_arity:
            return None

        component_types = []
        for raw_arg in raw_args:
            arg_type = self.expression_result_type(raw_arg)
            if self.vector_type_info(arg_type) is not None:
                return None
            component_type = self.scalar_component_type(arg_type)
            if component_type not in {"float", "double", None}:
                return None
            component_types.append(component_type)

        scalar_type = "double" if "double" in component_types else "float"
        if func_name == "inversesqrt" and scalar_type == "double":
            return f"(1.0 / sqrt({args[0]}))"

        target = signatures[scalar_type]
        return f"{target}({', '.join(args)})"

    def generate_vector_scalar_math_call(self, func_name, raw_args, args):
        """Lower unary scalar math builtins applied to vectors component-wise."""
        function_map = self.scalar_float_math_functions()
        signatures = function_map.get(func_name)
        if func_name == "pow":
            return self.generate_vector_pow_call(raw_args, args)
        if signatures is None or len(raw_args) != len(args) or len(raw_args) != 1:
            return None

        vector_info = self.vector_type_info(self.expression_result_type(raw_args[0]))
        if vector_info is None:
            return None

        component_type = vector_info["component_type"]
        if component_type not in {"float", "double"}:
            return None

        helper_name = self.require_vector_scalar_math_helper(
            func_name,
            component_type,
            vector_info,
        )
        if helper_name is None:
            return None
        return f"{helper_name}({args[0]})"

    def require_vector_scalar_math_helper(
        self,
        func_name,
        component_type,
        vector_info,
    ):
        """Register and return a helper for component-wise vector math calls."""
        vector_type = vector_info["type"]
        helper_name = f"cgl_{vector_type}_{func_name}"
        if helper_name in self.helper_functions:
            return helper_name

        components = [
            self.format_scalar_math_component(
                func_name,
                component_type,
                f"value.{component}",
            )
            for component in vector_info["components"]
        ]
        helper = (
            f"__device__ inline {vector_type} {helper_name}({vector_type} value)\n"
            "{\n"
            f"    return {vector_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def generate_vector_pow_call(self, raw_args, args):
        """Lower vector pow(base, exponent) component-wise."""
        if len(raw_args) != 2 or len(args) != 2:
            return None

        base_info = self.vector_type_info(self.expression_result_type(raw_args[0]))
        if base_info is None or base_info["component_type"] not in {"float", "double"}:
            return None

        exponent_info = self.vector_type_info(self.expression_result_type(raw_args[1]))
        if exponent_info is not None:
            if exponent_info["component_type"] != base_info["component_type"] or len(
                exponent_info["components"]
            ) != len(base_info["components"]):
                return None
            helper_name = self.require_vector_pow_helper(base_info, exponent_info)
            return f"{helper_name}({args[0]}, {args[1]})"

        exponent_type = self.scalar_component_type(
            self.expression_result_type(raw_args[1])
        )
        if exponent_type not in {"float", "double", None}:
            return None
        helper_name = self.require_vector_pow_helper(base_info, None)
        return f"{helper_name}({args[0]}, {args[1]})"

    def require_vector_pow_helper(self, base_info, exponent_info):
        vector_type = base_info["type"]
        component_type = base_info["component_type"]
        exponent_shape = "vector" if exponent_info is not None else "scalar"
        helper_name = f"cgl_{vector_type}_pow_{exponent_shape}"
        if helper_name in self.helper_functions:
            return helper_name

        exponent_type = (
            exponent_info["type"]
            if exponent_info is not None
            else self.vector_scalar_parameter_type(base_info)
        )
        components = []
        for component in base_info["components"]:
            exponent_component = (
                f"exponent.{component}" if exponent_info is not None else "exponent"
            )
            components.append(
                self.format_scalar_math_component(
                    "pow", component_type, f"base.{component}, {exponent_component}"
                )
            )
        helper = (
            f"__device__ inline {vector_type} {helper_name}"
            f"({vector_type} base, {exponent_type} exponent)\n"
            "{\n"
            f"    return {base_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def generate_smoothstep_call(self, raw_args, args):
        """Lower scalar smoothstep through a CUDA/HIP-compatible helper."""
        if len(raw_args) != 3 or len(args) != 3:
            return None
        if self.vector_type_info(self.expression_result_type(raw_args[2])) is not None:
            return None

        component_types = [
            self.scalar_component_type(self.expression_result_type(raw_arg))
            for raw_arg in raw_args
        ]
        if any(
            component_type not in {"float", "double", None}
            for component_type in component_types
        ):
            return None
        scalar_type = "double" if "double" in component_types else "float"
        helper_name = self.require_smoothstep_helper(scalar_type)
        return f"{helper_name}({', '.join(args)})"

    def require_smoothstep_helper(self, scalar_type):
        helper_name = f"cgl_smoothstep_{scalar_type}"
        if helper_name in self.helper_functions:
            return helper_name

        min_func = "fmin" if scalar_type == "double" else "fminf"
        max_func = "fmax" if scalar_type == "double" else "fmaxf"
        zero = "0.0" if scalar_type == "double" else "0.0f"
        one = "1.0" if scalar_type == "double" else "1.0f"
        three = "3.0" if scalar_type == "double" else "3.0f"
        two = "2.0" if scalar_type == "double" else "2.0f"
        helper = (
            f"__device__ inline {scalar_type} {helper_name}"
            f"({scalar_type} edge0, {scalar_type} edge1, {scalar_type} value)\n"
            "{\n"
            f"    {scalar_type} t = {max_func}({zero}, "
            f"{min_func}({one}, (value - edge0) / (edge1 - edge0)));\n"
            f"    return t * t * ({three} - {two} * t);\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def format_scalar_math_component(self, func_name, component_type, value):
        """Format one scalar component for a precision-aware math builtin."""
        signatures = self.scalar_float_math_functions()[func_name]
        if func_name == "inversesqrt" and component_type == "double":
            return f"(1.0 / sqrt({value}))"
        target = signatures[component_type]
        return f"{target}({value})"

    def scalar_float_math_functions(self):
        return {
            "sqrt": {"float": "sqrtf", "double": "sqrt"},
            "pow": {"float": "powf", "double": "pow"},
            "sin": {"float": "sinf", "double": "sin"},
            "cos": {"float": "cosf", "double": "cos"},
            "tan": {"float": "tanf", "double": "tan"},
            "asin": {"float": "asinf", "double": "asin"},
            "acos": {"float": "acosf", "double": "acos"},
            "atan": {"float": "atanf", "double": "atan"},
            "sinh": {"float": "sinhf", "double": "sinh"},
            "cosh": {"float": "coshf", "double": "cosh"},
            "tanh": {"float": "tanhf", "double": "tanh"},
            "exp": {"float": "expf", "double": "exp"},
            "exp2": {"float": "exp2f", "double": "exp2"},
            "log": {"float": "logf", "double": "log"},
            "log2": {"float": "log2f", "double": "log2"},
            "floor": {"float": "floorf", "double": "floor"},
            "ceil": {"float": "ceilf", "double": "ceil"},
            "round": {"float": "roundf", "double": "round"},
            "trunc": {"float": "truncf", "double": "trunc"},
            "inversesqrt": {"float": "rsqrtf", "double": "sqrt"},
        }

    def abs_component_type(self, type_name):
        if type_name is None:
            return None
        if not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        mapped_type = self.map_vector_arithmetic_type(type_name)
        if mapped_type == "double":
            return "double"
        if mapped_type == "float":
            return "float"
        if mapped_type in {
            "unsigned char",
            "unsigned short",
            "unsigned int",
            "unsigned long long",
            "uint",
            "uint8_t",
            "uint16_t",
            "uint32_t",
            "uint64_t",
        }:
            return "uint"
        if mapped_type in {
            "char",
            "short",
            "int",
            "long",
            "long long",
            "int8_t",
            "int16_t",
            "int32_t",
            "int64_t",
        }:
            return "int"
        return None

    def scalar_component_type(self, type_name):
        """Return the vector component type corresponding to a scalar type."""
        if type_name is None:
            return None
        if not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        mapped_type = self.map_vector_arithmetic_type(type_name)
        if mapped_type == "bool":
            return "bool"
        if mapped_type in {"float", "double"}:
            return mapped_type
        return self.abs_component_type(type_name)

    def require_vector_dot_helper(self, vector_info):
        if vector_info["component_type"] == "bool":
            return None

        vector_type = vector_info["type"]
        helper_name = f"cgl_{vector_type}_dot"
        if helper_name in self.helper_functions:
            return helper_name

        scalar_type = self.vector_scalar_parameter_type(vector_info)
        terms = [
            f"(lhs.{component} * rhs.{component})"
            for component in vector_info["components"]
        ]
        helper = (
            f"__device__ inline {scalar_type} {helper_name}"
            f"({vector_type} lhs, {vector_type} rhs)\n"
            "{\n"
            f"    return {' + '.join(terms)};\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def require_vector_cross_helper(self, vector_info):
        if (
            vector_info["component_type"] not in {"float", "double"}
            or len(vector_info["components"]) != 3
        ):
            return None

        vector_type = vector_info["type"]
        helper_name = f"cgl_{vector_type}_cross"
        if helper_name in self.helper_functions:
            return helper_name

        constructor = vector_info["constructor"]
        helper = (
            f"__device__ inline {vector_type} {helper_name}"
            f"({vector_type} lhs, {vector_type} rhs)\n"
            "{\n"
            f"    return {constructor}("
            "(lhs.y * rhs.z - lhs.z * rhs.y), "
            "(lhs.z * rhs.x - lhs.x * rhs.z), "
            "(lhs.x * rhs.y - lhs.y * rhs.x));\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def require_vector_length_helper(self, vector_info):
        component_type = vector_info["component_type"]
        if component_type not in {"float", "double"}:
            return None

        vector_type = vector_info["type"]
        helper_name = f"cgl_{vector_type}_length"
        if helper_name in self.helper_functions:
            return helper_name

        sqrt_name = "sqrt" if component_type == "double" else "sqrtf"
        squares = [
            f"(value.{component} * value.{component})"
            for component in vector_info["components"]
        ]
        helper = (
            f"__device__ inline {component_type} {helper_name}({vector_type} value)\n"
            "{\n"
            f"    return {sqrt_name}({' + '.join(squares)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def require_vector_normalize_helper(self, vector_info):
        component_type = vector_info["component_type"]
        if component_type not in {"float", "double"}:
            return None

        length_helper = self.require_vector_length_helper(vector_info)
        if length_helper is None:
            return None

        vector_type = vector_info["type"]
        helper_name = f"cgl_{vector_type}_normalize"
        if helper_name in self.helper_functions:
            return helper_name

        one_literal = "1.0" if component_type == "double" else "1.0f"
        components = [
            f"(value.{component} * inv_length)"
            for component in vector_info["components"]
        ]
        helper = (
            f"__device__ inline {vector_type} {helper_name}({vector_type} value)\n"
            "{\n"
            f"    {component_type} inv_length = {one_literal} / {length_helper}(value);\n"
            f"    return {vector_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def vector_scalar_parameter_type(self, vector_info):
        """Return the scalar parameter type used by generated vector helpers."""
        if vector_info["component_type"] == "uint":
            return "unsigned int"
        return vector_info["component_type"]
