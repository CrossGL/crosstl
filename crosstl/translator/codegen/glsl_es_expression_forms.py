"""GLSL ES 3.00 spellings of expressions that desktop GLSL accepts but ESSL 3.00 does not.

ESSL 3.00 has no scalar swizzles (GLSL 4.20 added them), allows ``length()`` on arrays only (section 5.9 of
the ESSL 3.00 specification) and requires ``const`` locals to have constant initializers (section 4.3.3). The
WebGL emitter spells each of them in the ESSL form with the same value instead.
"""

import re

from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    ConstructorNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    PrimitiveType,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)

_FIRST_COMPONENTS = frozenset("xrs")
_SWIZZLE = re.compile(r"[xyzw]{1,4}|[rgba]{1,4}|[stpq]{1,4}")
_MATRIX = re.compile(r"(?:d|f16|f32)?mat([234])(?:x[234])?")
_CONSTRUCTOR = re.compile(
    r"(?:[biud]?vec[234]|d?mat[234](?:x[234])?|float|int|uint|bool)"
)

# ESSL 3.00 section 4.3.3: built-in calls with constant arguments are constant expressions, except texture lookups.
_CONSTANT_BUILTINS = frozenset(
    {
        "radians", "degrees", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
        "asinh", "acosh", "atanh", "pow", "exp", "log", "exp2", "log2", "sqrt", "inversesqrt",
        "abs", "sign", "floor", "trunc", "round", "roundEven", "ceil", "fract", "mod", "min", "max",
        "clamp", "mix", "step", "smoothstep", "length", "distance", "dot", "cross", "normalize",
        "faceforward", "reflect", "refract", "matrixCompMult", "outerProduct", "transpose",
        "determinant", "inverse", "lessThan", "lessThanEqual", "greaterThan", "greaterThanEqual",
        "equal", "notEqual", "any", "all", "not",
    }
)  # fmt: skip
_MUTATING_OPERATORS = frozenset({"++", "--", "INCREMENT", "DECREMENT", "PRE_INCREMENT"})

# ESSL 3.00 section 8: built-ins defined for float genTypes only, whose result follows their widest argument.
_COMPONENTWISE_FLOAT_BUILTINS = frozenset(
    {
        "radians", "degrees", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
        "asinh", "acosh", "atanh", "pow", "exp", "log", "exp2", "log2", "sqrt", "inversesqrt",
        "floor", "trunc", "round", "roundEven", "ceil", "fract", "mod", "mix", "step", "smoothstep",
        "normalize", "faceforward", "reflect", "refract",
    }
)  # fmt: skip
_FLOAT_BUILTINS = _COMPONENTWISE_FLOAT_BUILTINS | {"dot", "length", "distance", "cross"}
# Defined for float, int and uint genTypes, so only a mix of float and integer arguments needs a conversion.
_MIXED_NUMERIC_BUILTINS = frozenset({"min", "max", "clamp"})
_COMPONENTWISE_RESULT_BUILTINS = (
    _COMPONENTWISE_FLOAT_BUILTINS | _MIXED_NUMERIC_BUILTINS | {"abs", "sign"}
)


class GLSLESExpressionFormsMixin:
    """Mixin for :class:`WebGLCodeGen`; relies on the GLSL emitter's type queries."""

    def glsl_es_expression_form(self, expr):
        """Return the ESSL spelling of ``expr`` when desktop GLSL's would be rejected, else ``None``."""
        if isinstance(expr, MemberAccessNode):
            return self.glsl_es_scalar_swizzle(expr)
        if isinstance(expr, FunctionCallNode):
            folded = self.glsl_es_folded_length(expr)
            if folded is not None:
                return folded
            return self.glsl_es_converted_builtin_call(expr)
        return None

    def glsl_es_builtin_name(self, expr):
        """The GLSL built-in a call resolves to, for classifying its overloads; ``None`` for a user function.

        The name is only compared with the built-in tables below and never emitted. A built-in without a
        ``function_map`` entry has the same name in CrossGL and GLSL.
        """
        name = self.function_call_name(expr)
        if not isinstance(name, str) or name in getattr(
            self, "function_return_types", {}
        ):
            return None
        if name in self.function_map:
            return self.function_map[name]
        return name

    def glsl_es_converted_builtin_call(self, expr):
        """A float built-in called with integer arguments, with each one converted explicitly.

        Desktop GLSL converts ``int`` and ``uint`` arguments to ``float`` while it resolves a built-in overload
        (GLSL 4.60 sections 4.1.10 and 6.1); ESSL 3.00 has no implicit conversions (section 4.1.10), so
        ``mod(x, 2)`` or ``round(u)`` matches no overload there. The conversion keeps the desktop meaning.
        """
        name = self.glsl_es_builtin_name(expr)
        if name not in _FLOAT_BUILTINS and name not in _MIXED_NUMERIC_BUILTINS:
            return None
        if id(expr) in self.glsl_es_converting_calls:
            return None
        arguments = list(getattr(expr, "arguments", None) or [])
        infos = [
            self.glsl_value_type_info(self.glsl_source_expression_type(argument))
            for argument in arguments
        ]
        if not arguments or any(info is None for info in infos):
            return None
        families = {info["family"] for info in infos}
        if "bool" in families:
            return None
        if name in _MIXED_NUMERIC_BUILTINS and (
            "float" not in families or families == {"float"}
        ):
            return None
        converted = [
            (
                argument
                if info["family"] == "float"
                else self.glsl_es_float_conversion(argument, info)
            )
            for argument, info in zip(arguments, infos)
        ]
        if all(new is old for new, old in zip(converted, arguments)):
            return None
        self.glsl_es_converting_calls.add(id(expr))
        expr.arguments = expr.args = converted
        try:
            return self.generate_expression(expr)
        finally:
            expr.arguments = expr.args = arguments
            self.glsl_es_converting_calls.discard(id(expr))

    @staticmethod
    def glsl_es_float_conversion(argument, info):
        value = getattr(argument, "value", None)
        if (
            isinstance(argument, LiteralNode)
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            return LiteralNode(float(value), PrimitiveType("float"))
        width = info["width"]
        target = "float" if width == 1 else f"vec{width}"
        return FunctionCallNode(IdentifierNode(target), [argument])

    @property
    def glsl_es_converting_calls(self):
        calls = getattr(self, "_glsl_es_converting_calls", None)
        if calls is None:
            calls = self._glsl_es_converting_calls = set()
        return calls

    def glsl_source_expression_type(self, expression):
        found = super().glsl_source_expression_type(expression)
        if found is None and isinstance(expression, FunctionCallNode):
            return self.glsl_es_builtin_result_type(expression)
        return found

    def glsl_es_builtin_result_type(self, expr):
        """The genType result of a float or mixed-numeric built-in: its widest argument, in float when any is."""
        name = self.glsl_es_builtin_name(expr)
        if name not in _COMPONENTWISE_RESULT_BUILTINS:
            return None
        infos = [
            self.glsl_value_type_info(self.glsl_source_expression_type(argument))
            for argument in getattr(expr, "arguments", None) or []
        ]
        if not infos or any(info is None for info in infos):
            return None
        widest = max(infos, key=lambda info: info["width"])
        float_result = name in _FLOAT_BUILTINS or any(
            info["family"] == "float" for info in infos
        )
        if not float_result:
            return widest["source"]
        return "float" if widest["width"] == 1 else f"vec{widest['width']}"

    def glsl_es_scalar_swizzle(self, expr):
        member = expr.member
        if not isinstance(member, str) or not _SWIZZLE.fullmatch(member):
            return None
        if not set(member) <= _FIRST_COMPONENTS:
            return None
        info = self.glsl_value_type_info(
            self.glsl_source_expression_type(expr.object_expr)
        )
        if info is None and isinstance(expr.object_expr, FunctionCallNode):
            info = self.glsl_value_type_info(
                self.glsl_es_builtin_result_type(expr.object_expr)
            )
        if info is None or info["width"] != 1:
            return None
        value = self.generate_expression(expr.object_expr)
        if len(member) == 1:
            return value
        vector = self.glsl_vector_type_for_component(info["mapped"], len(member))
        return f"{vector}({value})" if vector else None

    def glsl_es_folded_length(self, expr):
        function = getattr(expr, "function", None)
        if not isinstance(function, MemberAccessNode) or function.member != "length":
            return None
        if getattr(expr, "arguments", None):
            return None
        source_type = self.glsl_source_expression_type(function.object_expr)
        type_name = self.type_name_string(source_type) or ""
        if "[" in type_name:
            return None  # arrays keep length()
        count = None
        info = self.glsl_value_type_info(source_type)
        if info is not None and info["width"] > 1:
            count = info["width"]
        else:
            matrix = _MATRIX.fullmatch(self.map_type(type_name)) if type_name else None
            count = int(matrix.group(1)) if matrix else None
        if count is None:
            return None
        if not self.glsl_side_effect_free_expression(function.object_expr):
            return None
        return str(count)

    def glsl_es_local_const_allowed(self, node):
        """ESSL keeps ``const`` only on locals whose initializer is a constant expression."""
        constant = self.glsl_es_constant_expression(
            getattr(node, "initial_value", None)
        )
        if constant and getattr(node, "name", None):
            self.glsl_es_constant_locals.add(node.name)
        return constant

    @property
    def glsl_es_constant_locals(self):
        names = getattr(self, "_glsl_es_constant_locals", None)
        if names is None:
            names = self._glsl_es_constant_locals = set()
        return names

    def glsl_es_constant_name(self, name):
        if (
            name in self.local_variable_types
            and name not in self.glsl_es_constant_locals
        ):
            return False
        return (
            name in self.glsl_es_constant_locals
            or name in (getattr(self, "glsl_constant_types", None) or {})
            or name in (getattr(self, "current_compile_time_int_constants", None) or {})
        )

    def glsl_es_constant_expression(self, expr):
        if expr is None:
            return False
        if isinstance(expr, (int, float, bool)) or isinstance(expr, LiteralNode):
            return True
        if isinstance(expr, str):
            return self.glsl_es_constant_name(expr)
        if isinstance(expr, (IdentifierNode, VariableNode)):
            return self.glsl_es_constant_name(getattr(expr, "name", None))
        if isinstance(expr, BinaryOpNode):
            return self.glsl_es_constant_expression(
                expr.left
            ) and self.glsl_es_constant_expression(expr.right)
        if isinstance(expr, UnaryOpNode):
            return str(
                expr.op
            ) not in _MUTATING_OPERATORS and self.glsl_es_constant_expression(
                expr.operand
            )
        if isinstance(expr, TernaryOpNode):
            return all(
                self.glsl_es_constant_expression(part)
                for part in (expr.condition, expr.true_expr, expr.false_expr)
            )
        if isinstance(expr, MemberAccessNode):
            return self.glsl_es_constant_expression(expr.object_expr)
        if isinstance(expr, ArrayAccessNode):
            return self.glsl_es_constant_expression(
                expr.array_expr
            ) and self.glsl_es_constant_expression(expr.index_expr)
        if isinstance(expr, ConstructorNode):
            return all(
                self.glsl_es_constant_expression(argument)
                for argument in expr.arguments or []
            ) and not getattr(expr, "named_arguments", None)
        if isinstance(expr, FunctionCallNode):
            name = self.function_call_name(expr)
            callee = isinstance(name, str) and (
                _CONSTRUCTOR.fullmatch(name)
                or name in getattr(self, "structs_by_name", {})
                or (
                    name in _CONSTANT_BUILTINS
                    and name not in getattr(self, "function_return_types", {})
                )
            )
            return bool(callee) and all(
                self.glsl_es_constant_expression(argument)
                for argument in expr.arguments or []
            )
        return False
