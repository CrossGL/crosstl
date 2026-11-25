"""Lambdas and math built-ins of the HIP generator: mod, fract, clamp, saturate, step, mix, fma and atan2, with their vector helpers."""

from ..ast import (
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    PrimitiveType,
)


class HipMathCallMixin:
    """Math call members of ``HipCodeGen``."""

    def scalar_float_math_functions(self):
        """The shared precision-aware spellings, and ``exp10``, which the HIP math API declares as ``exp10f`` and ``exp10``."""
        return {
            **super().scalar_float_math_functions(),
            "exp10": {"float": "exp10f", "double": "exp10"},
        }

    def generate_lambda_expression(self, args):
        """Render CrossGL's pseudo-lambda as a HIP device lambda."""
        if not args:
            return "[&] __device__ () {}"

        params = ", ".join(self.generate_lambda_parameter(arg) for arg in args[:-1])
        body = self.generate_lambda_body(args[-1])
        return f"[&] __device__ ({params}) {body}"

    def generate_lambda_parameter(self, arg):
        raw = self.lambda_raw_argument_text(arg).strip()
        typed_param = self.split_lambda_typed_parameter(raw)
        if typed_param is None:
            param_name = self.lambda_fallback_parameter_name(raw)
            return f"auto {param_name}" if param_name else "auto"

        type_name, param_name = typed_param
        mapped_type = self.lambda_parameter_type(type_name)
        return f"{mapped_type} {param_name}"

    def generate_lambda_body(self, arg):
        raw = self.lambda_raw_argument_text(arg).strip()
        if raw.startswith("{") and raw.endswith("}"):
            return raw
        if raw:
            return f"{{ return {raw}; }}"
        return "{}"

    def lambda_raw_argument_text(self, arg):
        if isinstance(arg, IdentifierNode):
            return arg.name
        if isinstance(arg, str):
            return arg
        return self.visit(arg)

    def split_lambda_typed_parameter(self, raw):
        if not raw:
            return None
        if any(char in raw for char in "{}()"):
            return None
        parts = raw.rsplit(None, 1)
        if len(parts) != 2:
            return None
        type_name, param_name = parts
        if not param_name.isidentifier():
            return None
        return type_name, param_name

    def lambda_parameter_type(self, type_name):
        if "<" in type_name or ">" in type_name:
            return "auto"
        return self.map_type(type_name)

    def lambda_fallback_parameter_name(self, raw):
        if not raw:
            return ""
        candidate = raw.rsplit(None, 1)[-1]
        if candidate.isidentifier():
            return candidate
        return raw

    def generate_vector_constructor_args(self, vector_info, raw_args, args):
        """Flatten vector arguments passed to HIP make_* constructors."""
        if len(args) == 1:
            arg_type = self.expression_result_type(raw_args[0])
            if arg_type is not None and not self.swizzle_result_vector_info(arg_type):
                return args * len(vector_info["components"])

        generated_args = []
        for raw_arg, arg_expr in zip(raw_args, args):
            swizzle_components = self.member_swizzle_components(raw_arg)
            if swizzle_components is not None:
                generated_args.extend(
                    self.vector_argument_lane_expressions(
                        raw_arg,
                        arg_expr,
                        vector_info,
                    )
                )
                if len(generated_args) >= len(vector_info["components"]):
                    return generated_args[: len(vector_info["components"])]
                continue

            arg_info = self.swizzle_result_vector_info(
                self.expression_result_type(raw_arg)
            )
            if arg_info is None:
                generated_args.append(arg_expr)
            else:
                generated_args.extend(
                    self.vector_argument_lane_expressions(
                        raw_arg,
                        arg_expr,
                        arg_info,
                    )
                )

            if len(generated_args) >= len(vector_info["components"]):
                return generated_args[: len(vector_info["components"])]

        return generated_args

    def vector_argument_lane_expressions(self, raw_arg, arg_expr, arg_info):
        swizzle_components = self.member_swizzle_components(raw_arg)
        if swizzle_components is not None:
            object_node = getattr(
                raw_arg,
                "object_expr",
                getattr(raw_arg, "object", None),
            )
            role = self.hip_compute_builtin_role_for_name(
                getattr(object_node, "name", None)
            )
            if role is not None:
                lanes = [
                    self.hip_compute_builtin_expression(role, component)
                    for component in swizzle_components
                ]
                if all(lane is not None for lane in lanes):
                    return lanes
            object_expr = self.visit(object_node)
            return [f"{object_expr}.{component}" for component in swizzle_components]

        return [f"{arg_expr}.{component}" for component in arg_info["components"]]

    def member_swizzle_components(self, node):
        if not isinstance(node, MemberAccessNode):
            return None

        object_node = getattr(node, "object_expr", getattr(node, "object", None))
        vector_info = self.swizzle_result_vector_info(
            self.expression_result_type(object_node)
        )
        if vector_info is None:
            return None

        component_aliases = {
            "x": "x",
            "y": "y",
            "z": "z",
            "w": "w",
            "r": "x",
            "g": "y",
            "b": "z",
            "a": "w",
        }
        member = getattr(node, "member", "")
        components = [component_aliases.get(component) for component in member]
        if not components or any(component is None for component in components):
            return None

        available_components = vector_info["components"]
        if any(component not in available_components for component in components):
            return None

        return components

    def generate_mod_call(self, raw_args, args):
        """Lower CrossGL/GLSL mod with GLSL floor semantics.

        The ``%`` operator is the truncated remainder (``fmodf``) instead; see
        VectorArithmeticMixin.lower_scalar_modulo_operation.
        """
        if len(raw_args) != 2 or len(args) != 2:
            return None

        vector_call = self.lower_vector_binary_operation(
            raw_args[0],
            args[0],
            raw_args[1],
            args[1],
            "mod",
        )
        if vector_call is not None:
            return vector_call

        scalar_type = self.hip_mod_scalar_type(raw_args[0], raw_args[1])
        if scalar_type is None:
            return None
        helper_name = self.require_hip_scalar_mod_helper(scalar_type)
        return f"{helper_name}({args[0]}, {args[1]})"

    def hip_mod_scalar_type(self, left_node, right_node):
        component_types = []
        for node in (left_node, right_node):
            arg_type = self.expression_result_type(node)
            if self.vector_type_info(arg_type) is not None:
                return None
            component_type = self.scalar_component_type(arg_type)
            if component_type is not None:
                component_types.append(component_type)

        if "double" in component_types:
            return "double"
        if "float" in component_types:
            return "float"
        return None

    def require_hip_scalar_mod_helper(self, scalar_type):
        helper_name = f"cgl_mod_{scalar_type}"
        if helper_name in self.helper_functions:
            return helper_name

        floor_name = "floor" if scalar_type == "double" else "floorf"
        helper = (
            f"__device__ inline {scalar_type} {helper_name}"
            f"({scalar_type} lhs, {scalar_type} rhs)\n"
            "{\n"
            f"    return lhs - rhs * {floor_name}(lhs / rhs);\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def generate_fract_call(self, raw_args, args):
        arg_type = self.expression_result_type(raw_args[0])
        vector_info = self.vector_type_info(arg_type)
        if vector_info is not None:
            helper_name = self.require_vector_fract_helper(vector_info)
            if helper_name is not None:
                return f"{helper_name}({args[0]})"

        scalar_type = self.fract_scalar_type(arg_type)
        helper_name = self.require_scalar_fract_helper(scalar_type)
        return f"{helper_name}({args[0]})"

    def fract_scalar_type(self, type_name):
        if type_name is not None and not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        mapped_type = self.map_type(type_name) if type_name is not None else None
        if mapped_type == "double" or type_name in {"double", "f64"}:
            return "double"
        return "float"

    def require_scalar_fract_helper(self, scalar_type):
        helper_name = f"cgl_fract_{scalar_type}"
        if helper_name in self.helper_functions:
            return helper_name

        floor_name = "floor" if scalar_type == "double" else "floorf"
        helper = (
            f"__device__ inline {scalar_type} {helper_name}({scalar_type} value)\n"
            "{\n"
            f"    return value - {floor_name}(value);\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def require_vector_fract_helper(self, vector_info):
        component_type = vector_info["component_type"]
        if component_type not in {"float", "double", "half"}:
            return None

        vector_type = vector_info["type"]
        helper_name = f"cgl_{vector_type}_fract"
        if helper_name in self.helper_functions:
            return helper_name

        # __half has no floor overload, so widen through the float helper.
        scalar_helper_name = self.require_scalar_fract_helper(
            "float" if component_type == "half" else component_type
        )
        if component_type == "half":
            components = [
                f"__float2half({scalar_helper_name}(__half2float(value.{component})))"
                for component in vector_info["components"]
            ]
        else:
            components = [
                f"{scalar_helper_name}(value.{component})"
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

    def generate_clamp_call(self, raw_args, args):
        value_type = self.expression_result_type(raw_args[0])
        value_info = self.vector_type_info(value_type)
        if not value_info:
            scalar_type = self.clamp_scalar_type(value_type)
            scalar_call = self.generate_scalar_clamp_single_eval_call(
                scalar_type,
                raw_args,
                args,
            )
            if scalar_call is not None:
                return scalar_call
            return self.format_clamp_component(
                scalar_type,
                args[0],
                args[1],
                args[2],
            )

        min_info = self.vector_type_info(self.expression_result_type(raw_args[1]))
        max_info = self.vector_type_info(self.expression_result_type(raw_args[2]))
        if min_info and len(min_info["components"]) != len(value_info["components"]):
            return f"fmaxf({args[1]}, fminf({args[2]}, {args[0]}))"
        if max_info and len(max_info["components"]) != len(value_info["components"]):
            return f"fmaxf({args[1]}, fminf({args[2]}, {args[0]}))"

        helper_name = self.require_vector_clamp_helper(
            value_info,
            min_is_vector=min_info is not None,
            max_is_vector=max_info is not None,
        )
        if helper_name is None:
            return f"fmaxf({args[1]}, fminf({args[2]}, {args[0]}))"
        return f"{helper_name}({args[0]}, {args[1]}, {args[2]})"

    def generate_saturate_call(self, raw_args, args):
        value_type = self.expression_result_type(raw_args[0])
        value_info = self.vector_type_info(value_type)
        component_type = (
            value_info["component_type"]
            if value_info is not None
            else self.scalar_component_type(value_type)
        )
        if component_type in {"int", "uint", "unsigned int"}:
            # saturate keeps its argument's type (semantics.hub_types), so an integer clamps to 0 and 1 as the
            # GLSL output's clamp(n, 0, 1) does.
            integer_type = PrimitiveType(component_type)
            return self.generate_clamp_call(
                [
                    raw_args[0],
                    LiteralNode(0, integer_type),
                    LiteralNode(1, integer_type),
                ],
                [args[0], "0", "1"],
            )
        if component_type not in {"float", "double"}:
            return None

        float_type = PrimitiveType("float")
        return self.generate_clamp_call(
            [
                raw_args[0],
                LiteralNode(0.0, float_type),
                LiteralNode(1.0, float_type),
            ],
            [args[0], "0.0", "1.0"],
        )

    def generate_step_call(self, raw_args, args):
        edge_type = self.expression_result_type(raw_args[0])
        value_type = self.expression_result_type(raw_args[1])
        edge_info = self.vector_type_info(edge_type)
        value_info = self.vector_type_info(value_type)

        if edge_info is None and value_info is None:
            scalar_type = self.step_scalar_type(raw_args)
            if scalar_type is None:
                return None
            return self.format_step_component(scalar_type, args[0], args[1])

        result_info = value_info or edge_info
        if result_info["component_type"] not in {"float", "double", "half"}:
            return None
        if not self.compatible_step_operand(edge_info, result_info):
            return None
        if not self.compatible_step_operand(value_info, result_info):
            return None
        if edge_info is None and not self.compatible_step_scalar(
            edge_type, result_info["component_type"]
        ):
            return None
        if value_info is None and not self.compatible_step_scalar(
            value_type, result_info["component_type"]
        ):
            return None

        helper_name = self.require_vector_step_helper(
            result_info,
            edge_is_vector=edge_info is not None,
            value_is_vector=value_info is not None,
        )
        return f"{helper_name}({args[0]}, {args[1]})"

    def step_scalar_type(self, raw_args):
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

    def compatible_step_operand(self, operand_info, result_info):
        if operand_info is None:
            return True
        return operand_info["component_type"] == result_info["component_type"] and len(
            operand_info["components"]
        ) == len(result_info["components"])

    def compatible_step_scalar(self, type_name, result_component_type=None):
        component_type = self.scalar_component_type(type_name)
        if result_component_type == "half":
            return component_type in {"half", None}
        return component_type in {"float", "double", None}

    def require_vector_step_helper(self, result_info, edge_is_vector, value_is_vector):
        edge_shape = "vector" if edge_is_vector else "scalar"
        value_shape = "vector" if value_is_vector else "scalar"
        helper_name = (
            f"cgl_{result_info['type']}_step_{edge_shape}_edge_{value_shape}_value"
        )
        if helper_name in self.helper_functions:
            return helper_name

        scalar_type = self.vector_scalar_parameter_type(result_info)
        edge_type = result_info["type"] if edge_is_vector else scalar_type
        value_type = result_info["type"] if value_is_vector else scalar_type
        components = []
        for component in result_info["components"]:
            edge = f"edge.{component}" if edge_is_vector else "edge"
            value = f"value.{component}" if value_is_vector else "value"
            if result_info["component_type"] == "half":
                components.append(
                    "__float2half("
                    + self.format_step_component(
                        "float", f"__half2float({edge})", f"__half2float({value})"
                    )
                    + ")"
                )
            else:
                components.append(
                    self.format_step_component(
                        result_info["component_type"], edge, value
                    )
                )

        helper = (
            f"__device__ inline {result_info['type']} {helper_name}"
            f"({edge_type} edge, {value_type} value)\n"
            "{\n"
            f"    return {result_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def format_step_component(self, scalar_type, edge, value):
        if scalar_type == "double":
            zero = "0.0"
            one = "1.0"
        else:
            zero = "0.0f"
            one = "1.0f"
        return f"(({value}) < ({edge}) ? {zero} : {one})"

    def generate_mix_call(self, raw_args, args):
        left_type = self.expression_result_type(raw_args[0])
        right_type = self.expression_result_type(raw_args[1])
        factor_type = self.expression_result_type(raw_args[2])
        left_info = self.vector_type_info(left_type)
        right_info = self.vector_type_info(right_type)
        factor_info = self.vector_type_info(factor_type)

        if not left_info and not right_info and not factor_info:
            scalar_bool_mix = self.lower_bool_scalar_mix_operation(
                raw_args[0],
                args[0],
                raw_args[1],
                args[1],
                raw_args[2],
                args[2],
            )
            if scalar_bool_mix is not None:
                return scalar_bool_mix
            scalar_mix_call = self.generate_scalar_mix_single_eval_call(
                raw_args,
                args,
            )
            if scalar_mix_call is not None:
                return scalar_mix_call
            return self.format_mix_component(args[0], args[1], args[2])

        if left_info is None or right_info is None:
            return None
        if (
            len(left_info["components"]) != len(right_info["components"])
            or left_info["component_type"] != right_info["component_type"]
        ):
            return None

        bool_mix = self.lower_bool_vector_mix_operation(
            raw_args[0],
            args[0],
            raw_args[1],
            args[1],
            raw_args[2],
            args[2],
            left_info,
            right_info,
            factor_info,
        )
        if bool_mix is not None:
            return bool_mix

        if factor_info is not None and (
            len(factor_info["components"]) != len(left_info["components"])
            or factor_info["component_type"] != left_info["component_type"]
        ):
            return None

        helper_name = self.require_vector_mix_helper(
            left_info,
            factor_is_vector=factor_info is not None,
        )
        if helper_name is None:
            return None
        return f"{helper_name}({args[0]}, {args[1]}, {args[2]})"

    def require_vector_mix_helper(self, vector_info, factor_is_vector):
        component_type = vector_info["component_type"]
        if component_type not in {"float", "double", "half"}:
            return None

        vector_type = vector_info["type"]
        factor_shape = "vector" if factor_is_vector else "scalar"
        helper_name = f"cgl_{vector_type}_mix_{factor_shape}"
        if helper_name in self.helper_functions:
            return helper_name

        factor_type = (
            vector_type
            if factor_is_vector
            else self.vector_scalar_parameter_type(vector_info)
        )
        components = []
        for component in vector_info["components"]:
            factor_component = f"a.{component}" if factor_is_vector else "a"
            if component_type == "half":
                components.append(
                    "__float2half("
                    + self.format_mix_component(
                        f"__half2float(x.{component})",
                        f"__half2float(y.{component})",
                        f"__half2float({factor_component})",
                    )
                    + ")"
                )
            else:
                components.append(
                    self.format_mix_component(
                        f"x.{component}",
                        f"y.{component}",
                        factor_component,
                    )
                )

        helper = (
            f"__device__ inline {vector_type} {helper_name}"
            f"({vector_type} x, {vector_type} y, {factor_type} a)\n"
            "{\n"
            f"    return {vector_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def format_mix_component(self, left, right, factor):
        return f"({left} + (({right} - {left}) * {factor}))"

    def generate_fused_multiply_add_call(self, raw_args, args):
        vector_infos = [
            self.vector_type_info(self.expression_result_type(raw_arg))
            for raw_arg in raw_args
        ]
        vector_info = next((info for info in vector_infos if info is not None), None)
        if vector_info is None:
            scalar_type = self.fused_multiply_add_scalar_type(raw_args)
            if scalar_type is None:
                return None
            return self.format_fused_multiply_add_component(
                scalar_type,
                args[0],
                args[1],
                args[2],
            )

        if vector_info["component_type"] not in {"float", "double", "half"}:
            return None
        for info in vector_infos:
            if info is None:
                continue
            if (
                len(info["components"]) != len(vector_info["components"])
                or info["component_type"] != vector_info["component_type"]
            ):
                return None

        component_count = len(vector_info["components"])
        pieces = []
        for raw_arg, arg_expr, info in zip(raw_args, args, vector_infos):
            if info is None and not self.compatible_fma_scalar(
                raw_arg, vector_info["component_type"]
            ):
                return None
            pieces.append(
                self.vector_operation_piece(
                    raw_arg,
                    arg_expr,
                    info,
                    component_count,
                    vector_info["component_type"],
                )
            )

        helper_name = self.require_vector_fused_multiply_add_helper(
            vector_info,
            pieces,
        )
        return f"{helper_name}({', '.join(piece['arg_expr'] for piece in pieces)})"

    def fused_multiply_add_scalar_type(self, raw_args):
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

    def compatible_fma_scalar(self, raw_arg, vector_component_type):
        component_type = self.scalar_component_type(
            self.expression_result_type(raw_arg)
        )
        if component_type is None:
            return True
        if component_type in {"int", "uint"}:
            return True
        return component_type == vector_component_type

    def require_vector_fused_multiply_add_helper(self, vector_info, pieces):
        signature = "_".join(
            self.vector_constructor_piece_signature(piece) for piece in pieces
        )
        helper_name = self.sanitize_helper_name(
            f"cgl_{vector_info['type']}_fma_{signature}"
        )
        if helper_name in self.helper_functions:
            return helper_name

        params = [
            f"{piece['param_type']} arg{index}" for index, piece in enumerate(pieces)
        ]

        def widen(piece, expr):
            # Half lanes widen to float; int scalars convert implicitly in fmaf.
            if piece["kind"] == "vector" or piece["param_type"] == "half":
                return f"__half2float({expr})"
            return expr

        component_args = []
        for component in vector_info["components"]:
            multiply_left = self.vector_operation_piece_param_expr(
                pieces[0], 0, component
            )
            multiply_right = self.vector_operation_piece_param_expr(
                pieces[1], 1, component
            )
            addend = self.vector_operation_piece_param_expr(pieces[2], 2, component)
            if vector_info["component_type"] == "half":
                component_args.append(
                    "__float2half("
                    + self.format_fused_multiply_add_component(
                        "float",
                        widen(pieces[0], multiply_left),
                        widen(pieces[1], multiply_right),
                        widen(pieces[2], addend),
                    )
                    + ")"
                )
            else:
                component_args.append(
                    self.format_fused_multiply_add_component(
                        vector_info["component_type"],
                        multiply_left,
                        multiply_right,
                        addend,
                    )
                )

        helper = (
            f"__device__ inline {vector_info['type']} {helper_name}"
            f"({', '.join(params)})\n"
            "{\n"
            f"    return {vector_info['constructor']}({', '.join(component_args)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def format_fused_multiply_add_component(self, scalar_type, left, right, addend):
        target = "fma" if scalar_type == "double" else "fmaf"
        return f"{target}({left}, {right}, {addend})"

    def generate_atan2_call(self, raw_args, args):
        y_type = self.expression_result_type(raw_args[0])
        x_type = self.expression_result_type(raw_args[1])
        y_info = self.vector_type_info(y_type)
        x_info = self.vector_type_info(x_type)

        if y_info is None and x_info is None:
            return None
        if y_info is None or x_info is None:
            return None
        if (
            len(y_info["components"]) != len(x_info["components"])
            or y_info["component_type"] != x_info["component_type"]
        ):
            return None

        helper_name = self.require_vector_atan2_helper(y_info)
        if helper_name is None:
            return None
        return f"{helper_name}({args[0]}, {args[1]})"

    def generate_scalar_atan2_call(self, raw_args, args):
        component_types = []
        for raw_arg in raw_args:
            arg_type = self.expression_result_type(raw_arg)
            if self.vector_type_info(arg_type) is not None:
                return None
            component_type = self.scalar_component_type(arg_type)
            if component_type not in {"float", "double", None}:
                return None
            component_types.append(component_type)

        target = "atan2" if "double" in component_types else "atan2f"
        return f"{target}({', '.join(args)})"

    def require_vector_atan2_helper(self, vector_info):
        component_type = vector_info["component_type"]
        if component_type not in {"float", "double", "half"}:
            return None

        vector_type = vector_info["type"]
        helper_name = f"cgl_{vector_type}_atan2"
        if helper_name in self.helper_functions:
            return helper_name

        if component_type == "half":
            components = [
                f"__float2half(atan2f(__half2float(y.{component}), __half2float(x.{component})))"
                for component in vector_info["components"]
            ]
        else:
            scalar_func = "atan2" if component_type == "double" else "atan2f"
            components = [
                f"{scalar_func}(y.{component}, x.{component})"
                for component in vector_info["components"]
            ]
        helper = (
            f"__device__ inline {vector_type} {helper_name}"
            f"({vector_type} y, {vector_type} x)\n"
            "{\n"
            f"    return {vector_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def clamp_scalar_type(self, type_name):
        if type_name is None:
            return "float"
        if not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        mapped_type = self.map_type(type_name)
        if mapped_type in {"float", "double"}:
            return mapped_type
        if mapped_type in {
            "bool",
            "char",
            "unsigned char",
            "short",
            "unsigned short",
            "int",
            "unsigned int",
            "long long",
            "unsigned long long",
        }:
            return mapped_type
        return "float"

    def require_vector_clamp_helper(self, vector_info, min_is_vector, max_is_vector):
        if vector_info["component_type"] == "bool":
            return None

        vector_type = vector_info["type"]
        scalar_type = self.vector_scalar_parameter_type(vector_info)
        min_shape = "vector" if min_is_vector else "scalar"
        max_shape = "vector" if max_is_vector else "scalar"
        helper_name = f"cgl_{vector_type}_clamp"
        if min_shape != "vector" or max_shape != "vector":
            helper_name += f"_{min_shape}_min_{max_shape}_max"
        if helper_name in self.helper_functions:
            return helper_name

        min_type = vector_type if min_is_vector else scalar_type
        max_type = vector_type if max_is_vector else scalar_type
        components = vector_info["components"]
        constructor = vector_info["constructor"]
        args = []
        for component in components:
            value_component = f"value.{component}"
            min_component = f"min_value.{component}" if min_is_vector else "min_value"
            max_component = f"max_value.{component}" if max_is_vector else "max_value"
            args.append(
                self.format_clamp_component(
                    vector_info["component_type"],
                    value_component,
                    min_component,
                    max_component,
                )
            )

        helper = (
            f"__device__ inline {vector_type} {helper_name}"
            f"({vector_type} value, {min_type} min_value, {max_type} max_value)\n"
            "{\n"
            f"    return {constructor}({', '.join(args)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def format_clamp_component(self, component_type, value, min_value, max_value):
        if component_type == "float":
            return f"fmaxf({min_value}, fminf({max_value}, {value}))"
        if component_type == "double":
            return f"fmax({min_value}, fmin({max_value}, {value}))"
        return (
            f"(({value}) < ({min_value}) ? ({min_value}) : "
            f"(({value}) > ({max_value}) ? ({max_value}) : ({value})))"
        )
