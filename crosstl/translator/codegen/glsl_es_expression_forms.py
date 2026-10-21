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


class GLSLESExpressionFormsMixin:
    """Mixin for :class:`WebGLCodeGen`; relies on the GLSL emitter's type queries."""

    def glsl_es_expression_form(self, expr):
        """Return the ESSL spelling of ``expr`` when desktop GLSL's would be rejected, else ``None``."""
        if isinstance(expr, MemberAccessNode):
            return self.glsl_es_scalar_swizzle(expr)
        if isinstance(expr, FunctionCallNode):
            return self.glsl_es_folded_length(expr)
        return None

    def glsl_es_scalar_swizzle(self, expr):
        member = expr.member
        if not isinstance(member, str) or not _SWIZZLE.fullmatch(member):
            return None
        if not set(member) <= _FIRST_COMPONENTS:
            return None
        info = self.glsl_value_type_info(
            self.glsl_source_expression_type(expr.object_expr)
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
