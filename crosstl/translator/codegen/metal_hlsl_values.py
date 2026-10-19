"""Metal emission of HLSL value conversions.

HLSL applies the usual arithmetic conversions, implicit vector truncation and
scalar splats, and overloads its intrinsics for every numeric component type.
Metal does not, so these members make the conversions explicit: mixed-type
binary operands and builtin arguments, integer ``dot``, ``radians``/``degrees``,
emulated integer and boolean matrices, and HLSL logical intrinsics.
"""

import re

from ..ast import FunctionCallNode, LiteralNode
from .metal_hlsl_matrix import hlsl_emulated_matrix_shape


class MetalHlslValueMixin:
    """HLSL value-conversion members of ``MetalCodeGen``."""

    # Builtins Metal overloads for both floating-point and integer elements.
    # Metal has no usual arithmetic conversions between those overload
    # families, so mixed-family operands must be converted explicitly.
    METAL_COMPONENT_UNIFIED_BUILTINS = {"clamp", "dot", "max", "min"}

    # Builtins Metal provides only for floating-point elements; integer
    # operands are ambiguous between the float and half overloads.
    METAL_FLOATING_POINT_BUILTINS = {
        "acos",
        "asin",
        "atan",
        "atan2",
        "ceil",
        "cos",
        "cross",
        "distance",
        "exp",
        "exp2",
        "faceforward",
        "floor",
        "fract",
        "length",
        "log",
        "log2",
        "mix",
        "normalize",
        "pow",
        "reflect",
        "refract",
        "round",
        "rsqrt",
        "saturate",
        "sin",
        "smoothstep",
        "sqrt",
        "step",
        "tan",
        "trunc",
    }

    METAL_SCALAR_RESULT_BUILTINS = {"distance", "dot", "length"}

    METAL_BUILTIN_COMPONENT_RANKS = {"int": 1, "uint": 2, "half": 3, "float": 4}

    # The Metal standard library has no radians()/degrees().
    METAL_ANGLE_CONVERSION_FACTORS = {
        "radians": "0.017453292519943295",
        "degrees": "57.29577951308232",
    }

    METAL_IMPLICIT_VECTOR_COMPONENTS = frozenset(
        {"float", "half", "int", "uint", "bool"}
    )

    METAL_SIMPLE_OPERAND = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*|\[[^\[\]]*\])*")

    def metal_implicit_vector_conversion(self, expr, rendered, expected_type_name):
        """Spell out source conversions Metal does not apply between vectors.

        Source languages convert vectors with another component type or more
        components implicitly (truncating to the leading components); Metal
        only converts scalars implicitly, so the conversion is made explicit
        where the destination type is known.
        """
        if not expected_type_name or not isinstance(rendered, str):
            return rendered
        expected = self.map_type(expected_type_name)
        if not isinstance(expected, str) or self.is_matrix_value_type(expected):
            return rendered
        expected_component = self.vector_component_type(expected)
        if expected_component not in self.METAL_IMPLICIT_VECTOR_COMPONENTS:
            return rendered
        if not (
            self.is_vector_value_type(expected) or self.is_scalar_value_type(expected)
        ):
            return rendered
        actual_type = self.expression_result_type(expr)
        if actual_type is None:
            return rendered
        actual = self.map_type(actual_type)
        if not isinstance(actual, str) or not self.is_vector_value_type(actual):
            return rendered
        actual_component = self.vector_component_type(actual)
        if actual_component not in self.METAL_IMPLICIT_VECTOR_COMPONENTS:
            return rendered
        expected_width = self.vector_value_width(expected) or 1
        actual_width = self.vector_value_width(actual)
        if actual_width is None or actual_width < expected_width:
            return rendered
        converted = rendered
        if actual_width > expected_width:
            if isinstance(expr, FunctionCallNode):
                # Builtin result widths are not modelled precisely enough to
                # truncate call results.
                return rendered
            if not self.METAL_SIMPLE_OPERAND.fullmatch(converted):
                converted = f"({converted})"
            converted = f"{converted}.{'xyzw'[:expected_width]}"
        if actual_component != expected_component:
            converted = f"{expected}({converted})"
        return converted

    METAL_ARITHMETIC_COMPONENT_RANKS = {
        "bool": 0,
        "int": 1,
        "uint": 2,
        "half": 3,
        "float": 4,
    }

    METAL_COMMON_TYPE_OPERATORS = frozenset(
        {"+", "-", "*", "/", "%", "<", ">", "<=", ">=", "==", "!=", "&", "|", "^"}
    )

    def metal_binary_operand_vector(self, operand):
        """``(mapped type, component, width)`` of a known vector operand."""
        operand_type = self.expression_result_type(operand)
        if operand_type is None:
            return None
        mapped = self.map_type(operand_type)
        if not isinstance(mapped, str) or not self.is_vector_value_type(mapped):
            return None
        component = self.vector_component_type(mapped)
        if component not in self.METAL_ARITHMETIC_COMPONENT_RANKS:
            return None
        return mapped, component, self.vector_value_width(mapped)

    def metal_binary_common_component(self, operator, left_component, right_component):
        if left_component == right_component:
            return left_component
        common = max(
            (left_component, right_component),
            key=self.METAL_ARITHMETIC_COMPONENT_RANKS.get,
        )
        if common == "bool":
            common = "int"
        if operator in {"&", "|", "^"} and common in {"half", "float"}:
            return None
        return common

    def metal_mixed_vector_binary_operands(self, expr, operator, left, right):
        """Convert vector operands with different component types.

        Source languages apply the usual arithmetic conversions to vectors of
        one width; Metal rejects mixed vector operands. Shifts keep the left
        operand type. Returns ``(left, right, common component or None)``.
        """
        left_info = self.metal_binary_operand_vector(expr.left)
        right_info = self.metal_binary_operand_vector(expr.right)
        if left_info is None and right_info is None:
            if operator != "%":
                return left, right, None
            scalar_components = {
                self.vector_component_type(
                    self.map_type(self.expression_result_type(operand))
                )
                for operand in (expr.left, expr.right)
            }
            if "float" in scalar_components:
                return left, right, "float"
            if "half" in scalar_components:
                return left, right, "half"
            return left, right, None
        if left_info is None or right_info is None:
            info = left_info or right_info
            other = expr.right if left_info is not None else expr.left
            other_type = self.map_type(self.expression_result_type(other))
            if operator == "%" and info[1] in {"float", "half"}:
                # fmod has no vector-scalar overloads.
                if left_info is None and self.is_scalar_value_type(other_type):
                    left = f"{info[0]}({left})"
                elif right_info is None and self.is_scalar_value_type(other_type):
                    right = f"{info[0]}({right})"
            return left, right, info[1]
        left_type, left_component, left_width = left_info
        right_type, right_component, right_width = right_info
        if left_width != right_width:
            return left, right, None
        if operator in {"<<", ">>"}:
            if left_component != right_component and left_component in {"int", "uint"}:
                right = f"{left_type}({right})"
            return left, right, left_component
        if operator not in self.METAL_COMMON_TYPE_OPERATORS:
            return left, right, None
        common = self.metal_binary_common_component(
            operator, left_component, right_component
        )
        if common is None:
            return left, right, None
        common_type = f"{common}{left_width}"
        if left_component != common:
            left = f"{common_type}({left})"
        if right_component != common:
            right = f"{common_type}({right})"
        return left, right, common

    def metal_builtin_argument_components(self, arguments):
        """Describe builtin arguments as ``(mapped type, component type)`` pairs.

        Returns ``None`` when any argument type is unknown, a matrix, or outside
        the scalar families whose conversions are unified.
        """
        described = []
        for argument in arguments:
            argument_type = self.expression_result_type(argument)
            if argument_type is None:
                return None
            mapped_type = self.map_type(argument_type)
            if not isinstance(mapped_type, str) or self.is_matrix_value_type(
                mapped_type
            ):
                return None
            component_type = self.vector_component_type(mapped_type)
            if component_type not in self.METAL_BUILTIN_COMPONENT_RANKS:
                return None
            described.append((mapped_type, component_type))
        return described

    def metal_builtin_common_component(self, func_name, arguments):
        """Return the common component type of a mixed builtin call.

        Returns ``(component, described arguments)`` when the call mixes
        component types (or passes integers to a floating-point-only builtin)
        and ``None`` when the arguments need no conversion.
        """
        if func_name in self.user_function_names:
            return None
        floating_only = func_name in self.METAL_FLOATING_POINT_BUILTINS
        if not floating_only and func_name not in self.METAL_COMPONENT_UNIFIED_BUILTINS:
            return None
        arguments = list(arguments or [])
        if not arguments:
            return None
        described = self.metal_builtin_argument_components(arguments)
        if not described:
            return None
        components = {component for _, component in described}
        common = max(components, key=self.METAL_BUILTIN_COMPONENT_RANKS.get)
        if floating_only and common in {"int", "uint"}:
            common = "float"
        if components == {common}:
            return None
        return common, described

    def metal_converted_numeric_literal(self, argument, component_type):
        """Spell an integer literal directly in another component type."""
        value = getattr(argument, "value", argument)
        if isinstance(argument, bool) or isinstance(value, bool):
            return None
        if not isinstance(value, int):
            return None
        if not (isinstance(argument, int) or "Literal" in type(argument).__name__):
            return None
        if component_type == "float":
            return f"{value}.0"
        if component_type == "half":
            return f"{value}.0h"
        if component_type == "uint" and value >= 0:
            return f"{value}u"
        return None

    def generate_metal_unified_builtin_arguments(self, func_name, arguments):
        """Render builtin arguments converted to one component type.

        Metal overloads ``min``/``max``/``clamp`` for floating-point and integer
        elements but applies no usual arithmetic conversions between those
        families, so ``max(3.14, 2)`` or ``clamp(uint, int, int)`` is ambiguous,
        and floating-point-only builtins such as ``sin(2)`` are ambiguous
        between their ``float`` and ``half`` overloads. CrossGL follows the
        GLSL and HLSL implicit conversions (integer to floating point, ``int``
        to ``uint``): integer literals are spelled in the common type and other
        operands are converted with a constructor of the same width.
        """
        unified = self.metal_builtin_common_component(func_name, arguments)
        if unified is None:
            return None
        common, described = unified
        rendered = []
        for argument, (mapped_type, component_type) in zip(arguments, described):
            code = self.generate_expression(argument)
            if component_type == common:
                rendered.append(code)
                continue
            literal = self.metal_converted_numeric_literal(argument, common)
            if literal is not None:
                rendered.append(literal)
                continue
            width = self.vector_value_width(mapped_type)
            target_type = common if width is None else f"{common}{width}"
            rendered.append(f"{target_type}({code})")
        return rendered

    def metal_unified_builtin_result_type(self, func_name, arguments):
        """Result type of a builtin whose arguments were unified, if any."""
        unified = self.metal_builtin_common_component(func_name, arguments)
        if unified is None:
            return None
        common, described = unified
        widths = [self.vector_value_width(mapped) for mapped, _ in described]
        widths = [width for width in widths if width is not None]
        if func_name in self.METAL_SCALAR_RESULT_BUILTINS or not widths:
            return common
        return f"{common}{max(widths)}"

    def generate_metal_integer_dot_call(self, func_name, arguments):
        """Lower ``dot`` on integer vectors, which Metal provides only for floats.

        CrossGL and HLSL define ``dot`` for integer vectors with an integer
        result; Metal's ``dot`` accepts floating-point vectors only. Mixed
        ``int``/``uint`` operands follow the usual conversion to ``uint``.
        """
        arguments = list(arguments or [])
        if func_name != "dot" or len(arguments) != 2:
            return None
        described = self.metal_builtin_argument_components(arguments)
        if not described:
            return None
        components = {component for _, component in described}
        if not components <= {"int", "uint"}:
            return None
        widths = {self.vector_value_width(mapped) for mapped, _ in described}
        if len(widths) != 1 or None in widths:
            return None
        width = widths.pop()
        common = "uint" if "uint" in components else "int"
        rendered = []
        for argument, (_mapped, component) in zip(arguments, described):
            code = self.generate_expression(argument)
            if component != common:
                code = f"{common}{width}({code})"
            rendered.append(code)
        self.required_metal_integer_dot_helper = True
        return f"__crossgl_integer_dot({rendered[0]}, {rendered[1]})"

    def generate_metal_integer_dot_helper(self):
        if not getattr(self, "required_metal_integer_dot_helper", False):
            return ""
        return (
            "template <typename T, int N>\n"
            "static inline T __crossgl_integer_dot(vec<T, N> a, vec<T, N> b) {\n"
            "    T sum = T(0);\n"
            "    for (int i = 0; i < N; ++i) {\n"
            "        sum += a[i] * b[i];\n"
            "    }\n"
            "    return sum;\n"
            "}\n\n"
        )

    def generate_metal_angle_conversion_call(self, func_name, arguments):
        """Lower ``radians``/``degrees``, which the Metal standard library lacks."""
        factor = self.METAL_ANGLE_CONVERSION_FACTORS.get(func_name)
        arguments = list(arguments or [])
        if factor is None or len(arguments) != 1:
            return None
        if func_name in self.user_function_names:
            return None
        argument_type = self.map_type(self.expression_result_type(arguments[0]))
        component_type = self.vector_component_type(argument_type)
        value = self.generate_expression(arguments[0])
        if component_type in {"int", "uint"}:
            width = self.vector_value_width(argument_type)
            value = f"float{width or ''}({value})"
        elif component_type == "half":
            return f"(({value}) * half({factor}))"
        return f"(({value}) * {factor})"

    def metal_emulated_matrix_binary_result_type(self, operator, left_type, right_type):
        """Result type of an operator with an emulated integer or bool matrix."""
        left_shape = (
            hlsl_emulated_matrix_shape(self.type_name_string(left_type))
            if left_type is not None
            else None
        )
        right_shape = (
            hlsl_emulated_matrix_shape(self.type_name_string(right_type))
            if right_type is not None
            else None
        )
        if left_shape is None and right_shape is None:
            return None
        shape = left_shape or right_shape
        scalar, rows, columns = shape
        if operator in {"<", ">", "<=", ">=", "==", "!="}:
            return f"bool{rows}x{columns}"
        if operator == "*":
            if left_shape is not None and right_shape is not None:
                return f"{scalar}{left_shape[1]}x{right_shape[2]}"
            other_type = right_type if left_shape is not None else left_type
            if other_type is not None and self.vector_value_width(other_type):
                # matrix * vector gives a column; vector * matrix a row.
                width = rows if left_shape is not None else columns
                return f"{scalar}{width}"
        return self.type_name_string(left_type if left_shape else right_type)

    def metal_expression_emulated_matrix_shape(self, expr):
        expr_type = self.expression_result_type(expr)
        if expr_type is None:
            return None
        return hlsl_emulated_matrix_shape(self.type_name_string(expr_type))

    def metal_value_component_count(self, expr):
        """Scalar components of a value: 1 for scalars, N for vectors."""
        expr_type = self.expression_result_type(expr)
        if expr_type is None:
            return None
        width = self.vector_value_width(expr_type)
        if width is not None:
            return width
        if self.vector_component_type(expr_type) is not None:
            return 1
        return None

    def generate_metal_matrix_constructor_override(self, type_name, args):
        """Constructors that involve emulated HLSL integer and bool matrices."""
        type_name = self.type_name_string(type_name)
        args = list(args or [])
        shape = hlsl_emulated_matrix_shape(type_name)
        if shape is None:
            if len(args) != 1 or not self.metal_expression_emulated_matrix_shape(
                args[0]
            ):
                return None
            metal_type = self.map_type(type_name)
            if not self.is_matrix_value_type(metal_type):
                raise ValueError(
                    f"Cannot convert an HLSL integer or bool matrix to '{metal_type}'"
                )
            component = self.vector_component_type(metal_type)
            value = self.generate_expression_with_expected(args[0], None)
            return f"crosstl_hlsl_matrix_native<{component}>({value})"

        # Like a Metal matrix constructor, the arguments fill whole columns.
        scalar, rows, columns = shape
        mapped_type = self.map_type(type_name)
        column_type = f"{scalar}{rows}"
        rendered = [self.generate_expression_with_expected(arg, None) for arg in args]
        if len(args) == 1:
            if self.metal_expression_emulated_matrix_shape(
                args[0]
            ) or self.is_matrix_value_type(self.expression_result_type(args[0])):
                return f"crosstl_hlsl_matrix_cast<{scalar}>({rendered[0]})"
            return (
                f"crosstl_hlsl_matrix_splat<{scalar}, {columns}, {rows}>"
                f"({rendered[0]})"
            )
        counts = [self.metal_value_component_count(arg) for arg in args]
        if len(args) == rows * columns and all(count in (1, None) for count in counts):
            column_values = [
                f"{column_type}({', '.join(rendered[column * rows:(column + 1) * rows])})"
                for column in range(columns)
            ]
        elif len(args) == columns and all(count == rows for count in counts):
            column_values = [f"{column_type}({value})" for value in rendered]
        else:
            raise ValueError(
                f"Unsupported constructor arguments for HLSL matrix '{type_name}'"
            )
        return f"{mapped_type}{{{{{', '.join(column_values)}}}}}"

    def generate_metal_hlsl_logical_intrinsic(self, func_name, args):
        """HLSL ``and``/``or``: element-wise logic without short-circuiting names."""
        if func_name not in {"and", "or"} or len(args or []) != 2:
            return None
        if func_name in self.user_function_names:
            return None
        left, right = (
            self.generate_expression_with_expected(arg, None) for arg in args
        )
        if any(self.metal_expression_emulated_matrix_shape(arg) for arg in args):
            return f"crosstl_hlsl_{func_name}({left}, {right})"
        operator = "&&" if func_name == "and" else "||"
        left_type = self.expression_result_type(args[0])
        width = self.vector_value_width(left_type) if left_type is not None else None
        if width is not None:
            left = f"bool{width}({left})"
            right = f"bool{width}({right})"
        else:
            left = f"bool({left})"
            right = f"bool({right})"
        return f"({left} {operator} {right})"

    def generate_metal_value_constructor_arguments(self, metal_type, args):
        """Render constructor arguments, converting vectors to the target components.

        Source languages compose vectors from vectors of any component type
        (``float3(int2, 1)``); a Metal composite constructor only accepts vector
        arguments with the target component type.
        """
        rendered = [self.generate_expression_with_expected(arg, None) for arg in args]
        if len(args) < 2 or self.is_matrix_value_type(metal_type):
            return rendered
        target_width = self.vector_value_width(metal_type)
        target_component = self.vector_component_type(metal_type)
        if target_width is None or target_component is None:
            return rendered
        for index, arg in enumerate(args):
            arg_type = self.expression_result_type(arg)
            if arg_type is None:
                continue
            width = self.vector_value_width(arg_type)
            if width is None or width < 2:
                continue
            component = self.vector_component_type(arg_type)
            if component is not None and component != target_component:
                rendered[index] = f"{target_component}{width}({rendered[index]})"
        return rendered

    def metal_matrix_binary_result_type(self, operator, left_type, right_type):
        """Result type of an arithmetic operation with a matrix operand.

        Metal ``floatCxR`` has ``C`` columns of ``R`` rows: ``M * v`` takes a
        ``C`` vector to an ``R`` vector and ``v * M`` an ``R`` vector to ``C``.
        """
        left = self.map_type(left_type) if left_type is not None else None
        right = self.map_type(right_type) if right_type is not None else None
        left_matrix = isinstance(left, str) and self.is_matrix_value_type(left)
        right_matrix = isinstance(right, str) and self.is_matrix_value_type(right)
        if not left_matrix and not right_matrix:
            return None
        if operator not in {"*", "+", "-", "/"}:
            return None

        def parts(matrix):
            scalar, columns, rows = re.fullmatch(
                r"(float|half|double)(\d)x(\d)", matrix
            ).groups()
            return scalar, int(columns), int(rows)

        if left_matrix and right_matrix:
            if operator != "*":
                return left
            scalar, _left_columns, rows = parts(left)
            _scalar, right_columns, _right_rows = parts(right)
            return f"{scalar}{right_columns}x{rows}"
        matrix = left if left_matrix else right
        other = right if left_matrix else left
        if other is None:
            return None
        if self.is_vector_value_type(other) and operator == "*":
            scalar, columns, rows = parts(matrix)
            return f"{scalar}{rows if left_matrix else columns}"
        if self.is_scalar_value_type(other):
            return matrix
        return None

    METAL_EXACT_INTEGER_COMPOUND_OPERATORS = frozenset(
        {"+=", "-=", "*=", "&=", "|=", "^=", "<<="}
    )

    def metal_assignment_value_type(self, target, value, op):
        """Type the right-hand side converts to before it is combined.

        A compound assignment combines in the operands' common type; converting
        the right operand to the target type first is only exact for floating
        targets and for wrapping integer operators.
        """
        target_type = self.expression_result_type(target)
        if op == "=" or target_type is None:
            return target_type
        target_component = self.vector_component_type(self.map_type(target_type))
        if target_component in {"float", "half"}:
            return target_type
        value_type = self.expression_result_type(value)
        value_component = (
            self.vector_component_type(self.map_type(value_type))
            if value_type is not None
            else None
        )
        if (
            target_component in {"int", "uint"}
            and value_component in {"int", "uint", "bool"}
            and op in self.METAL_EXACT_INTEGER_COMPOUND_OPERATORS
        ):
            return target_type
        return None

    @staticmethod
    def metal_expression_is_parenthesized(code):
        """Return whether rendered code is one fully enclosing parenthesis group."""
        text = str(code).strip()
        if not (text.startswith("(") and text.endswith(")")):
            return False
        depth = 0
        for index, char in enumerate(text):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and index != len(text) - 1:
                    return False
        return depth == 0

    def metal_negated_wide_integer_literal(self, expr):
        """Spell the negation of an unsuffixed literal beyond int as signed.

        A literal between 2^31 and 2^63 - 1 is emitted unsigned on its own, but
        its negation is a signed value. Metal types the unsuffixed decimal as
        long, so the negation keeps its value in every integer width.
        """
        if getattr(expr, "op", getattr(expr, "operator", None)) != "-":
            return None
        operand = getattr(expr, "operand", None)
        if getattr(expr, "is_postfix", False) or not isinstance(operand, LiteralNode):
            return None
        value = operand.value
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        literal_type = getattr(getattr(operand, "literal_type", None), "name", None)
        if literal_type not in (None, "int") or not (
            0x7FFFFFFF < value <= 0x7FFFFFFFFFFFFFFF
        ):
            # Typed 64-bit literals keep their own suffix.
            return None
        return f"-{value}"
