"""Math built-in calls in the WGSL generator: componentwise and columnwise matrix built-ins, matrix-scalar operators, clamp, boolean mix, scalar argument replication and integer argument conversion."""

import re

from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    CastNode,
    ConstructorNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    SwizzleNode,
    TernaryOpNode,
    UnaryOpNode,
)
from ..refusals import WGSLRefusal
from ..semantics.rewrite import is_pure
from .wgsl_integer_operands import (
    is_abstract_integer_literal,
    mixed_integer_call_is_unsigned,
    wgsl_mixed_integer_call_arguments,
)
from .wgsl_parameter_lowering import reference_parameter_qualifier


class WGSLMathCallMixin:
    """Math call members of ``WGSLCodeGen``."""

    # WGSL has `*` between a matrix and a scalar but no matrix-scalar `+`, `-` or `/`
    # and no matrixCompMult or outerProduct built-in (WGSL sections 8.7, 17.5), so
    # these build the result one column at a time.
    COLUMNWISE_MATRIX_BUILTINS = frozenset({"matrixCompMult", "outerProduct"})
    # Built-ins that reduce a vector to its component type (GLSL 4.60 section 8.5).
    COMPONENT_RESULT_BUILTINS = frozenset({"length", "distance", "dot"})
    # Component-wise built-ins whose result has the type of their vector argument,
    # or of their first argument when all are scalars (GLSL 4.60 sections 8.1-8.3, 8.5).
    COMPONENTWISE_MATH_BUILTINS = frozenset(
        {
            "abs",
            "sign",
            "floor",
            "ceil",
            "trunc",
            "round",
            "roundEven",
            "fract",
            "mod",
            "sqrt",
            "inversesqrt",
            "exp",
            "exp2",
            "log",
            "log2",
            "pow",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "sinh",
            "cosh",
            "tanh",
            "asinh",
            "acosh",
            "atanh",
            "radians",
            "degrees",
            "normalize",
            "min",
            "max",
            "clamp",
            "mix",
            "step",
            "smoothstep",
            "fma",
        }
    )
    # GLSL math built-ins declared for floating-point arguments (min, max and clamp
    # also for integers), whose int or uint argument GLSL converts to float when
    # another argument is floating point (GLSL 4.60 sections 4.1.10, 6.1, 8.1-8.5).
    FLOAT_ARGUMENT_BUILTINS = frozenset(
        {
            "min",
            "max",
            "clamp",
            "mix",
            "step",
            "smoothstep",
            "pow",
            "atan",
            "mod",
            "distance",
            "fma",
        }
    )
    # GLSL math built-ins declared only for floating-point arguments, whose int or
    # uint argument GLSL converts to float even when no argument is floating point
    # (GLSL 4.60 sections 8.1-8.5). min, max, clamp, mix, abs and sign also have
    # integer overloads and are not listed.
    FLOAT_ONLY_BUILTINS = frozenset(
        {
            "sqrt",
            "inversesqrt",
            "exp",
            "exp2",
            "log",
            "log2",
            "pow",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "sinh",
            "cosh",
            "tanh",
            "asinh",
            "acosh",
            "atanh",
            "radians",
            "degrees",
            "floor",
            "ceil",
            "fract",
            "round",
            "roundEven",
            "trunc",
            "mod",
            "step",
            "smoothstep",
            "fma",
            "length",
            "distance",
            "dot",
            "normalize",
        }
    )
    # Built-ins whose scalar GLSL overloads are floating-point only. Integer
    # arguments are implicitly converted to f32 for these calls.
    FLOAT_ONLY_COMPONENTWISE_MATH_BUILTINS = frozenset(
        COMPONENTWISE_MATH_BUILTINS - {"abs", "sign", "min", "max", "clamp", "mix"}
    )
    VECTOR_COMPONENT_NAMES = "xyzw"

    def generate_matrix_scalar_binary(self, expr):
        if expr.operator not in {"+", "-", "/"}:
            return None
        left_type = self.expression_type(expr.left)
        right_type = self.expression_type(expr.right)
        left_matrix = self.matrix_shape(left_type)
        right_matrix = self.matrix_shape(right_type)
        if left_matrix is not None and right_matrix is None:
            matrix_shape, scalar = left_matrix, expr.right
        elif right_matrix is not None and left_matrix is None:
            matrix_shape, scalar = right_matrix, expr.left
        else:
            if expr.operator == "/" and left_matrix is not None:
                self.refuse_matrix_scalar_binary(expr)
            return None
        if not self.is_matrix_scalar_operand(scalar, matrix_shape[0]):
            # WGSL has no `/` on matrices at all, and a scalar of another type
            # does not convert to the matrix element type.
            if expr.operator == "/" or self.scalar_type_name(
                self.expression_type(scalar)
            ):
                self.refuse_matrix_scalar_binary(expr)
            return None
        self.require_repeatable_matrix_operands(
            f"matrix-scalar {expr.operator}", (expr.left, expr.right)
        )
        left = self.generate_expression(expr.left)
        right = self.generate_expression(expr.right)
        element, columns, rows = matrix_shape
        parts = []
        for index in range(columns):
            column_left = (
                self.matrix_column_expression(left, index)
                if left_matrix is not None
                else left
            )
            column_right = (
                self.matrix_column_expression(right, index)
                if right_matrix is not None
                else right
            )
            parts.append(f"{column_left} {expr.operator} {column_right}")
        return f"mat{columns}x{rows}<{element}>({', '.join(parts)})"

    def refuse_matrix_scalar_binary(self, expr):
        raise WGSLRefusal(
            f"WGSL target lowers matrix {expr.operator} only with a scalar operand "
            "of the matrix element type or an integer literal; convert the operand "
            "first"
        )

    def is_matrix_scalar_operand(self, expr, element_type):
        scalar_type = self.scalar_type_name(self.expression_type(expr))
        if scalar_type is None or scalar_type == "bool":
            return False
        if scalar_type == element_type:
            return True
        # An integer literal is an abstract integer in WGSL and converts to the
        # matrix element type.
        return isinstance(expr, LiteralNode) and scalar_type == "i32"

    def generate_columnwise_matrix_builtin_call(self, node, function_name):
        if len(node.arguments) != 2:
            raise WGSLRefusal(
                f"WGSL target supports {function_name}() calls with exactly 2 "
                f"arguments; got {len(node.arguments)}"
            )
        first, second = node.arguments
        result_type = self.columnwise_matrix_builtin_type(function_name, node.arguments)
        if result_type is None:
            if function_name == "matrixCompMult":
                requirement = "two float matrices of the same type"
            else:
                requirement = "two float vectors of the same component type"
            raise WGSLRefusal(
                f"WGSL target requires {function_name}() operands to be {requirement}"
            )
        self.require_repeatable_matrix_operands(function_name, node.arguments)
        left = self.generate_expression(first)
        right = self.generate_expression(second)
        element, columns, rows = self.matrix_shape(result_type)
        if function_name == "matrixCompMult":
            parts = [
                f"{self.matrix_column_expression(left, index)} * "
                f"{self.matrix_column_expression(right, index)}"
                for index in range(columns)
            ]
        else:
            # outerProduct(c, r)[j] == c * r[j] (GLSL 4.60 section 8.6).
            column = self.matrix_operand_expression(left)
            row = self.matrix_operand_expression(right)
            parts = [
                f"{column} * {row}.{self.VECTOR_COMPONENT_NAMES[index]}"
                for index in range(columns)
            ]
        return f"mat{columns}x{rows}<{element}>({', '.join(parts)})"

    def columnwise_matrix_builtin_type(self, function_name, arguments):
        if len(arguments) != 2:
            return None
        first_type, second_type = (self.expression_type(arg) for arg in arguments)
        if function_name == "matrixCompMult":
            first = self.matrix_shape(first_type)
            if first is None or first != self.matrix_shape(second_type):
                return None
            if first[0] not in {"f32", "f16"}:
                return None
            element, columns, rows = first
            return f"mat{columns}x{rows}<{element}>"
        column = self.vector_shape(first_type)
        row = self.vector_shape(second_type)
        if column is None or row is None or column[0] != row[0]:
            return None
        if column[0] not in {"f32", "f16"}:
            return None
        return f"mat{row[1]}x{column[1]}<{column[0]}>"

    def require_repeatable_matrix_operands(self, operation, operands):
        # Each operand appears once per column in the lowered expression.
        for operand in operands:
            if not self.is_repeatable_matrix_operand(operand):
                raise WGSLRefusal(
                    f"WGSL target lowers {operation} one column at a time and "
                    "requires operands without side effects; assign the operand "
                    "to a local variable first"
                )

    def is_repeatable_matrix_operand(self, expr):
        if isinstance(expr, FunctionCallNode):
            function_name = self.expression_name(expr.function)
            if not isinstance(expr.function, IdentifierNode) or (
                self.is_user_function_name(function_name)
            ):
                return False
            if not (
                self.is_type_constructor_name(function_name)
                or function_name in self.COLUMNWISE_MATRIX_BUILTINS
                or function_name in {"transpose", "inverse"}
                or is_pure(expr)
            ):
                return False
            return all(
                self.is_repeatable_matrix_operand(argument)
                for argument in expr.arguments or []
            )
        if isinstance(expr, BinaryOpNode):
            return self.is_repeatable_matrix_operand(
                expr.left
            ) and self.is_repeatable_matrix_operand(expr.right)
        if isinstance(expr, UnaryOpNode):
            return expr.operator not in {
                "++",
                "--",
            } and self.is_repeatable_matrix_operand(expr.operand)
        if isinstance(expr, TernaryOpNode):
            return all(
                self.is_repeatable_matrix_operand(part)
                for part in (expr.condition, expr.true_expr, expr.false_expr)
            )
        if isinstance(expr, CastNode):
            return self.is_repeatable_matrix_operand(expr.expression)
        if isinstance(expr, MemberAccessNode):
            return self.is_repeatable_matrix_operand(expr.object_expr)
        if isinstance(expr, SwizzleNode):
            return self.is_repeatable_matrix_operand(expr.vector_expr)
        if isinstance(expr, ArrayAccessNode):
            return self.is_repeatable_matrix_operand(
                expr.array_expr
            ) and self.is_repeatable_matrix_operand(expr.index_expr)
        if isinstance(expr, ConstructorNode):
            return all(
                self.is_repeatable_matrix_operand(argument)
                for argument in getattr(expr, "arguments", None) or []
            )
        return is_pure(expr)

    def matrix_operand_expression(self, rendered):
        if re.fullmatch(r"[A-Za-z_]\w*(?:\.\w+|\[[^\[\]]*\])*", rendered):
            return rendered
        # A parenthesized expression or a call such as `mat2x2<f32>(...)` is already
        # a primary expression when its first `(` closes at the end.
        callee = re.match(r"(?:[A-Za-z_]\w*(?:<[\w<>, ]*>)?)?(?=\()", rendered)
        if callee is not None and rendered.endswith(")"):
            depth = 0
            for index in range(callee.end(), len(rendered)):
                depth += {"(": 1, ")": -1}.get(rendered[index], 0)
                if depth == 0 and index < len(rendered) - 1:
                    break
            else:
                return rendered
        return f"({rendered})"

    def matrix_column_expression(self, rendered, index):
        return f"{self.matrix_operand_expression(rendered)}[{index}]"

    def generate_clamp_call(self, node):
        if len(node.arguments) != 3:
            raise WGSLRefusal(
                "WGSL target supports clamp() calls with exactly 3 arguments; got "
                f"{len(node.arguments)}"
            )
        value_expr = self.generate_expression(node.arguments[0])
        value_type = self.expression_type(node.arguments[0])
        vector_type_name = self.vector_type_name(value_type)
        float_component = self.float_call_component("clamp", node.arguments)
        if float_component is None and (
            vector_type_name is None
            or mixed_integer_call_is_unsigned(self, node.arguments)
        ):
            args = wgsl_mixed_integer_call_arguments(self, node.arguments)
            return f"clamp({', '.join(args)})"
        if vector_type_name is None:
            # A float argument picks the float overload, so each integer argument converts to
            # float from its own value; through the mixed-integer u32 form a negative int
            # would wrap first.
            args = self.convert_integer_call_arguments(
                "clamp",
                node.arguments,
                [self.generate_expression(argument) for argument in node.arguments],
            )
            return f"clamp({', '.join(args)})"
        if float_component is not None:
            vector_type_name = (
                f"vec{self.vector_component_count(value_type)}<{float_component}>"
            )
        # generate_clamp_bound replicates a scalar bound, converted or not.
        value_expr, lower, upper = self.convert_integer_call_arguments(
            "clamp",
            node.arguments,
            [value_expr]
            + [self.generate_expression(bound) for bound in node.arguments[1:]],
        )
        lower = self.generate_clamp_bound(node.arguments[1], vector_type_name, lower)
        upper = self.generate_clamp_bound(node.arguments[2], vector_type_name, upper)
        return f"clamp({value_expr}, {lower}, {upper})"

    def generate_boolean_mix_call(self, node):
        """``mix(x, y, a)`` with a ``bool`` or ``bvecN`` selector as ``select(x, y, a)``.

        With a boolean selector, CrossGL's ``mix`` selects (GLSL 4.60 section 8.3): each
        component is ``y`` where ``a`` is true and ``x`` where it is false. WGSL's ``mix``
        takes only a floating-point selector; ``select(f, t, cond)`` (WGSL section 17.3)
        is the selecting builtin, with the same argument order and a scalar ``cond``
        allowed for vector values. Returns ``None`` for a non-boolean selector.
        """
        selector_width = self.boolean_expression_width(node.arguments[2])
        if selector_width is None:
            return None
        # Only values whose type is known can disagree with the selector; a value with no
        # recorded type, such as a built-in call, is not taken to be a scalar.
        value_width = max(
            (
                self.vector_component_count(value_type) or 1
                for value_type in map(self.expression_type, node.arguments[:2])
                if value_type is not None
            ),
            default=selector_width,
        )
        if selector_width not in (1, value_width):
            raise WGSLRefusal(
                "WGSL target does not support mix with a "
                f"{selector_width}-component selector and "
                f"{value_width}-component values"
            )
        x, y, a = (self.generate_expression(arg) for arg in node.arguments)
        return f"select({x}, {y}, {a})"

    BOOLEAN_COMPARISON_OPERATORS = {"<", ">", "<=", ">=", "==", "!="}
    BOOLEAN_VECTOR_RELATIONAL_FUNCTIONS = {
        "lessThan",
        "lessThanEqual",
        "greaterThan",
        "greaterThanEqual",
        "equal",
        "notEqual",
        "not",
    }
    BOOLEAN_CLASSIFICATION_FUNCTIONS = {"isnan", "isinf"}
    BOOLEAN_REDUCTION_FUNCTIONS = {"any", "all"}

    def boolean_expression_width(self, expr):
        """Component count of a ``bool`` or ``vecN<bool>`` expression, else ``None``."""
        expr_type = self.expression_type(expr)
        if self.is_bool_type(expr_type):
            return 1
        shape = self.vector_shape(expr_type)
        if shape is not None:
            return shape[1] if shape[0] == "bool" else None
        if isinstance(expr, BinaryOpNode):
            if expr.operator in {"&&", "||"}:
                return 1
            if expr.operator in self.BOOLEAN_COMPARISON_OPERATORS:
                operand_type = self.expression_type(expr.left) or (
                    self.expression_type(expr.right)
                )
                matrix_shape = self.matrix_shape(operand_type)
                if matrix_shape is not None:
                    return matrix_shape[1] * matrix_shape[2]
                return self.vector_component_count(operand_type) or 1
            return None
        if isinstance(expr, UnaryOpNode) and expr.operator == "!":
            return 1
        if (
            isinstance(expr, FunctionCallNode)
            and isinstance(expr.function, IdentifierNode)
            and expr.function.name in self.BOOLEAN_VECTOR_RELATIONAL_FUNCTIONS
            and expr.arguments
            and not self.is_user_function_name(expr.function.name)
        ):
            return self.vector_component_count(self.expression_type(expr.arguments[0]))
        if (
            isinstance(expr, FunctionCallNode)
            and isinstance(expr.function, IdentifierNode)
            and len(expr.arguments) == 1
            and not self.is_user_function_name(expr.function.name)
        ):
            if expr.function.name in self.BOOLEAN_REDUCTION_FUNCTIONS:
                return 1
            if expr.function.name in self.BOOLEAN_CLASSIFICATION_FUNCTIONS:
                argument_type = self.expression_type(expr.arguments[0])
                if argument_type is None:
                    return None
                return self.vector_component_count(argument_type) or 1
        return None

    def replicate_scalar_call_arguments(
        self, arguments, rendered, component_override=None
    ):
        """Replicate the scalar arguments of a min, max, step or smoothstep call beside a vector.

        GLSL has min(genType, float), max(genType, float), step(float, genType)
        and smoothstep(float, float, genType) (GLSL 4.60 section 8.3), but WGSL
        declares them with one type T for every argument, so a scalar is written
        as the vector type of the other arguments, as clamp's bounds are. Only a
        scalar of the vector's component type or a literal is replicated; a
        scalar of another type is left as written. ``component_override`` is the
        float type of a call whose integer arguments convert_integer_call_arguments
        converted; those scalars are replicated too.
        """
        argument_types = [self.expression_type(arg) for arg in arguments]
        vector_shape = next(
            (
                shape
                for shape in map(self.vector_shape, argument_types)
                if shape is not None
            ),
            None,
        )
        if vector_shape is None:
            return list(rendered)
        component, size = vector_shape
        if component_override is not None:
            component = component_override
        replicated = []
        for arg, arg_type, text in zip(arguments, argument_types, rendered):
            literal = arg.operand if isinstance(arg, UnaryOpNode) else arg
            if self.vector_shape(arg_type) is None and (
                isinstance(literal, LiteralNode)
                or self.scalar_type_name(arg_type) == component
                or (
                    component_override is not None
                    and self.scalar_type_name(arg_type) in {"i32", "u32"}
                )
            ):
                text = f"vec{size}<{component}>({text})"
            replicated.append(text)
        return replicated

    def float_call_component(self, function_name, arguments):
        """The float type an int or uint argument of a float built-in call is converted to.

        Returns ``f32`` or ``f16`` when an argument other than an abstract integer
        literal is an integer and either ``function_name`` is one of
        FLOAT_ARGUMENT_BUILTINS and another argument is floating point, or
        ``function_name`` is one of FLOAT_ONLY_BUILTINS (``f32`` when no argument is
        floating point). Else None.
        """
        float_only = function_name in self.FLOAT_ONLY_BUILTINS
        if function_name not in self.FLOAT_ARGUMENT_BUILTINS and not float_only:
            return None
        components = [
            self.call_argument_component(arg, arg_type)
            for arg, arg_type in zip(
                arguments, (self.expression_type(arg) for arg in arguments)
            )
        ]
        if not {"i32", "u32"} & set(components):
            return None
        float_component = next((c for c in components if c in {"f32", "f16"}), None)
        if float_component is None and float_only:
            return "f32"
        return float_component

    def call_argument_component(self, arg, arg_type):
        """Scalar or vector component type of a call argument; None for an integer literal."""
        literal = arg
        while isinstance(literal, UnaryOpNode) and literal.operator in {"-", "+"}:
            literal = literal.operand
        if is_abstract_integer_literal(literal):
            return None
        shape = self.vector_shape(arg_type)
        return shape[0] if shape is not None else self.scalar_type_name(arg_type)

    def convert_integer_call_arguments(self, function_name, arguments, rendered):
        """Convert the int and uint arguments of a float built-in call to the float type.

        GLSL picks the float overload of max(float, int) and converts the int
        argument (GLSL 4.60 sections 4.1.10 and 6.1). WGSL has no implicit
        conversions and its built-ins take arguments of one type, so the argument
        is written as f32(k), or vecN<f32>(iv) for an ivecN. A converted scalar
        beside a vector is replicated afterwards, by replicate_scalar_call_arguments
        or generate_clamp_bound. Integer literals are abstract in WGSL and are
        left as written.
        """
        rendered = list(rendered)
        float_component = self.float_call_component(function_name, arguments)
        if float_component is None:
            return rendered
        for index, arg in enumerate(arguments):
            arg_type = self.expression_type(arg)
            if self.call_argument_component(arg, arg_type) not in {"i32", "u32"}:
                continue
            shape = self.vector_shape(arg_type)
            target = (
                float_component
                if shape is None
                else f"vec{shape[1]}<{float_component}>"
            )
            rendered[index] = f"{target}({rendered[index]})"
        return rendered

    def builtin_math_call_type(self, function_name, arguments):
        """Return the result type of a GLSL math built-in call, or None if unknown."""
        if not arguments or self.is_user_function_name(function_name):
            return None
        float_component = self.float_call_component(function_name, arguments)
        if float_component is not None:
            # The int and uint arguments are converted (convert_integer_call_arguments).
            if function_name in self.COMPONENT_RESULT_BUILTINS:
                return float_component
            size = next(
                (
                    shape[1]
                    for shape in (
                        self.vector_shape(self.expression_type(arg))
                        for arg in arguments
                    )
                    if shape is not None
                ),
                None,
            )
            return float_component if size is None else f"vec{size}<{float_component}>"
        if function_name in self.COMPONENT_RESULT_BUILTINS:
            argument_type = self.expression_type(arguments[0])
            return self.vector_component_type(argument_type) or self.scalar_type_name(
                argument_type
            )
        if function_name in self.COMPONENTWISE_MATH_BUILTINS:
            argument_types = [self.expression_type(arg) for arg in arguments]
            vector_type = next(
                (t for t in argument_types if self.vector_shape(t) is not None), None
            )
            if vector_type is not None:
                return vector_type
            scalar_arguments = argument_types
            boolean_selector = False
            if function_name == "mix" and len(arguments) == 3:
                # A boolean selector may be a comparison or logical expression;
                # its expression_type is intentionally not a scalar type. Infer
                # the result from x and y, then let generate_boolean_mix_call
                # validate the selector shape.
                boolean_selector = (
                    self.boolean_expression_width(arguments[2]) is not None
                )
                if boolean_selector:
                    scalar_arguments = argument_types[:2]
            scalar_names = [self.scalar_type_name(t) for t in scalar_arguments]
            if any(s is None for s in scalar_names):
                return None
            if function_name in self.FLOAT_ONLY_COMPONENTWISE_MATH_BUILTINS:
                return "f32"
            if (
                function_name == "mix"
                and not boolean_selector
                and "bool" not in scalar_names
            ):
                return "f32"
            for target in ("f32", "u32", "i32"):
                for t, s in zip(argument_types, scalar_names):
                    if s == target:
                        return t
            if scalar_names and scalar_names[0] is not None:
                return argument_types[0]
        return None

    def generate_clamp_bound(self, bound, vector_type_name, rendered=None):
        if rendered is None:
            rendered = self.generate_expression(bound)
        if self.vector_component_count(self.expression_type(bound)) is not None:
            return rendered
        return f"{vector_type_name}({rendered})"

    def generate_call_arguments(self, function_name, arguments, function=None):
        metadata_key = (
            self.function_signature_key(function)
            if function is not None
            else function_name
        )
        texture_parameter_indices = self._function_texture_parameters.get(
            metadata_key, self._function_texture_parameters.get(function_name, ())
        )
        pointer_parameter_indices = self._function_pointer_parameters.get(
            metadata_key, self._function_pointer_parameters.get(function_name, ())
        )
        resource_member_parameters = self._function_resource_member_parameters.get(
            metadata_key,
            self._function_resource_member_parameters.get(function_name, {}),
        )
        parameters = list(getattr(function, "parameters", []) or [])
        reference_parameters = {
            index: parameter
            for index, parameter in enumerate(parameters)
            if reference_parameter_qualifier(parameter) is not None
        }
        if reference_parameters:
            self.validate_reference_argument_aliasing(function, arguments)
        if (
            not texture_parameter_indices
            and not pointer_parameter_indices
            and not resource_member_parameters
            and not reference_parameters
        ):
            parameter_types = [
                getattr(parameter, "param_type", None)
                for parameter in getattr(function, "parameters", []) or []
            ]
            return ", ".join(
                (
                    self.generate_expression_for_target(arg, parameter_types[index])
                    if index < len(parameter_types)
                    else self.generate_expression(arg)
                )
                for index, arg in enumerate(arguments)
            )

        texture_parameter_indices = set(texture_parameter_indices)
        pointer_parameter_indices = set(pointer_parameter_indices)
        parameter_types = [
            getattr(parameter, "param_type", None)
            for parameter in getattr(function, "parameters", []) or []
        ]
        rendered = []
        for index, arg in enumerate(arguments):
            if index in reference_parameters:
                rendered.append(
                    self.reference_argument_expression(
                        arg, reference_parameters[index], function
                    )
                )
            elif index in pointer_parameter_indices:
                rendered.append(self.pointer_argument_expression(arg))
            else:
                rendered.append(
                    self.generate_expression_for_target(arg, parameter_types[index])
                    if index < len(parameter_types)
                    else self.generate_expression(arg)
                )
            if index in texture_parameter_indices:
                rendered.append(self.texture_sampler_expression(arg))
            if index in resource_member_parameters:
                rendered.extend(
                    self.resource_member_call_arguments(
                        arg, resource_member_parameters[index]
                    )
                )
        return ", ".join(rendered)
