"""GLSL's aggregate ``==`` and ``!=`` spelled in CrossGL, whose equality operators are componentwise.

GLSL 4.60 section 5.9 defines ``==`` and ``!=`` on vectors and matrices as aggregate comparisons that yield one
``bool``; vector relational results come only from the built-ins ``equal`` and ``notEqual`` (section 8.7).  CrossGL's
equality operators are componentwise on vectors, as in HLSL, MSL, WGSL and SPIR-V (ADR 0002), so the importer writes

- ``all(a == b)`` and ``any(a != b)`` when the operands are vectors;
- ``(all(a[0] == b[0]) && ...)`` and ``(any(a[0] != b[0]) || ...)`` over the columns when they are matrices; and
- the operator unchanged when they are scalars, structures or arrays, whose CrossGL equality is already aggregate.

The operand types come from :class:`GlslExpressionTypes`, a small typer over the GLSL AST.  An equality whose operand
types it cannot determine is refused with :class:`GlslAggregateEqualityError`: guessing scalar would silently turn an
aggregate comparison into a comparison of the first components on every target that truncates.
"""

from __future__ import annotations

import re

from crosstl.translator.codegen.builtin_result_types import (
    GLSL_FAMILY,
    builtin_call_result_type,
    format_type,
    type_shape,
)
from crosstl.translator.codegen.builtin_variable_types import (
    crossgl_builtin_variable_type,
)
from crosstl.translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

from .OpenglAst import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    FunctionCallNode,
    MemberAccessNode,
    NumberNode,
    PostfixOpNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    VectorConstructorNode,
)

EQUALITY_OPERATORS = {"==": ("all", "&&"), "!=": ("any", "||")}
_EQUALITY_SPELLINGS = {"EQUAL": "==", "NOT_EQUAL": "!=", "==": "==", "!=": "!="}
_BOOLEAN_OPERATORS = frozenset(
    {
        "==",
        "!=",
        "<",
        ">",
        "<=",
        ">=",
        "&&",
        "||",
        "^^",
        "EQUAL",
        "NOT_EQUAL",
        "LESS_THAN",
        "GREATER_THAN",
        "LESS_EQUAL",
        "GREATER_EQUAL",
        "LOGICAL_AND",
        "LOGICAL_OR",
        "LOGICAL_XOR",
    }
)
_ARRAY_SUFFIX = re.compile(r"^(?P<element>[^\[]+)\[[^\]]*\]$")
_SWIZZLE = re.compile(r"^(?:[xyzw]{1,4}|[rgba]{1,4}|[stpq]{1,4})$")
_SCALAR_TYPES = frozenset({"bool", "int", "uint", "float", "double"})
# GL_EXT_shader_explicit_arithmetic_types spellings, which builtin_result_types does not model.
_EXPLICIT_VECTOR = re.compile(r"(?:i|u)(?:8|16|32|64)vec[234]|f(?:16|32|64)vec[234]")
_EXPLICIT_MATRIX = re.compile(r"f(?:16|32|64)mat(?P<columns>[234])(?:x[234])?")
_EXPLICIT_SCALAR = re.compile(r"u?int(?:8|16|32|64)_t|float(?:16|32|64)_t")
_FLOAT_LITERAL = re.compile(
    r"^(?:\d+\.\d*|\.\d+|\d+[eE][+-]?\d+|\d+\.\d*[eE][+-]?\d+|\.\d+[eE][+-]?\d+)(?P<suffix>[fF]|[lL][fF])?$"
)
_INTEGER_LITERAL = re.compile(
    r"^(?:0[xX][0-9a-fA-F]+|\d+)(?P<suffix>[uU])?$"
)
# Sampling and fetch built-ins return a four-component vector of the sampler's component type (GLSL 4.60 section
# 8.9); comparison (shadow) lookups other than gathers return one float.
_TEXEL_FUNCTION = re.compile(r"^(?:texture|texelFetch|imageLoad|subpassLoad)")
_TEXTURE_QUERY_TYPES = {
    "textureQueryLod": "vec2",
    "textureQueryLevels": "int",
    "textureSamples": "int",
    "imageSamples": "int",
}
# Built-in variables and constants that GLSL extensions declare, with the types the extension specifications give
# them.  CrossGL's own built-in variables come from ``crossgl_builtin_variable_type``.
_GLSL_EXTENSION_VARIABLES = {
    # GL_EXT_fragment_shading_rate.
    "gl_ShadingRateEXT": "int",
    "gl_PrimitiveShadingRateEXT": "int",
    "gl_ShadingRateFlag2VerticalPixelsEXT": "int",
    "gl_ShadingRateFlag4VerticalPixelsEXT": "int",
    "gl_ShadingRateFlag2HorizontalPixelsEXT": "int",
    "gl_ShadingRateFlag4HorizontalPixelsEXT": "int",
    # GL_EXT_ray_query.
    "gl_RayQueryCommittedIntersectionNoneEXT": "uint",
    "gl_RayQueryCommittedIntersectionTriangleEXT": "uint",
    "gl_RayQueryCommittedIntersectionGeneratedEXT": "uint",
    "gl_RayQueryCandidateIntersectionTriangleEXT": "uint",
    "gl_RayQueryCandidateIntersectionAABBEXT": "uint",
    # GL_EXT_ray_tracing (the ray flags are shared with GL_EXT_ray_query).
    "gl_RayFlagsNoneEXT": "uint",
    "gl_RayFlagsOpaqueEXT": "uint",
    "gl_RayFlagsNoOpaqueEXT": "uint",
    "gl_RayFlagsTerminateOnFirstHitEXT": "uint",
    "gl_RayFlagsSkipClosestHitShaderEXT": "uint",
    "gl_RayFlagsCullBackFacingTrianglesEXT": "uint",
    "gl_RayFlagsCullFrontFacingTrianglesEXT": "uint",
    "gl_RayFlagsCullOpaqueEXT": "uint",
    "gl_RayFlagsCullNoOpaqueEXT": "uint",
    "gl_HitKindFrontFacingTriangleEXT": "uint",
    "gl_HitKindBackFacingTriangleEXT": "uint",
    "gl_LaunchIDEXT": "uvec3",
    "gl_LaunchSizeEXT": "uvec3",
    "gl_InstanceCustomIndexEXT": "int",
    "gl_GeometryIndexEXT": "int",
    "gl_WorldRayOriginEXT": "vec3",
    "gl_WorldRayDirectionEXT": "vec3",
    "gl_ObjectRayOriginEXT": "vec3",
    "gl_ObjectRayDirectionEXT": "vec3",
    "gl_RayTminEXT": "float",
    "gl_RayTmaxEXT": "float",
    "gl_HitTEXT": "float",
    "gl_HitKindEXT": "uint",
    "gl_IncomingRayFlagsEXT": "uint",
    "gl_ObjectToWorldEXT": "mat4x3",
    "gl_WorldToObjectEXT": "mat4x3",
    # GL_EXT_multiview and GL_EXT_device_group.
    "gl_ViewIndex": "int",
    "gl_DeviceIndex": "int",
    # GL_KHR_shader_subgroup ballot masks.
    "gl_SubgroupEqMask": "uvec4",
    "gl_SubgroupGeMask": "uvec4",
    "gl_SubgroupGtMask": "uvec4",
    "gl_SubgroupLeMask": "uvec4",
    "gl_SubgroupLtMask": "uvec4",
}
# GL_EXT_ray_query's query functions; ``rayQueryProceedEXT`` and the remaining getters are listed for completeness.
_GLSL_EXTENSION_FUNCTIONS = {
    "rayQueryProceedEXT": "bool",
    "rayQueryGetIntersectionTypeEXT": "uint",
    "rayQueryGetRayTMinEXT": "float",
    "rayQueryGetRayFlagsEXT": "uint",
    "rayQueryGetWorldRayOriginEXT": "vec3",
    "rayQueryGetWorldRayDirectionEXT": "vec3",
    "rayQueryGetIntersectionTEXT": "float",
    "rayQueryGetIntersectionInstanceCustomIndexEXT": "int",
    "rayQueryGetIntersectionInstanceIdEXT": "int",
    "rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT": "uint",
    "rayQueryGetIntersectionGeometryIndexEXT": "int",
    "rayQueryGetIntersectionPrimitiveIndexEXT": "int",
    "rayQueryGetIntersectionBarycentricsEXT": "vec2",
    "rayQueryGetIntersectionFrontFaceEXT": "bool",
    "rayQueryGetIntersectionCandidateAABBOpaqueEXT": "bool",
    "rayQueryGetIntersectionObjectRayDirectionEXT": "vec3",
    "rayQueryGetIntersectionObjectRayOriginEXT": "vec3",
    "rayQueryGetIntersectionObjectToWorldEXT": "mat4x3",
    "rayQueryGetIntersectionWorldToObjectEXT": "mat4x3",
}


class GlslAggregateEqualityError(TranslationDiagnostic):
    """A GLSL ``==`` or ``!=`` whose operand types the importer cannot determine."""

    code = CODE_PREFIX + "glsl-aggregate-equality-untyped"


class GlslMatrixEqualityOperandError(TranslationDiagnostic):
    """A GLSL matrix ``==`` or ``!=`` whose operands would be evaluated once per column."""

    code = CODE_PREFIX + "glsl-matrix-equality-impure-operand"


class GlslExpressionTypes:
    """GLSL types of importer expressions, spelled as GLSL spells them; ``None`` where they cannot be decided.

    ``importer`` supplies ``lookup_variable_type(name)``, ``lookup_variable_is_array(name)``, ``structs_by_name``
    and ``glsl_function_return_types`` (user function name to its return types).
    """

    def __init__(self, importer):
        self.importer = importer

    def type_of(self, expr):
        if isinstance(expr, bool) or expr in ("true", "false"):
            return "bool"
        if isinstance(expr, NumberNode):
            return _literal_type(expr.value)
        if isinstance(expr, str):
            return self.variable_type(expr)
        if isinstance(expr, VariableNode):
            return self.variable_type(expr.name)
        if isinstance(expr, MemberAccessNode):
            return self.member_type(expr)
        if isinstance(expr, ArrayAccessNode):
            return self.element_type(self.type_of(expr.array))
        if isinstance(expr, VectorConstructorNode):
            return expr.type_name
        if isinstance(expr, FunctionCallNode):
            return self.call_type(expr)
        if isinstance(expr, BinaryOpNode):
            return self.binary_type(expr)
        if isinstance(expr, UnaryOpNode):
            if expr.op in ("!", "LOGICAL_NOT", "NOT"):
                return "bool"
            return self.type_of(expr.operand)
        if isinstance(expr, PostfixOpNode):
            return self.type_of(expr.operand)
        if isinstance(expr, TernaryOpNode):
            return self.type_of(expr.true_expr) or self.type_of(expr.false_expr)
        if isinstance(expr, AssignmentNode):
            return self.type_of(getattr(expr, "left", None))
        return None

    def variable_type(self, name):
        if not isinstance(name, str):
            return None
        declared = _normalized(self.importer.lookup_variable_type(name))
        if declared is not None:
            if self.importer.lookup_variable_is_array(name):
                return f"{declared}[]"
            return declared
        return crossgl_builtin_variable_type(name) or _GLSL_EXTENSION_VARIABLES.get(
            name
        )

    def member_type(self, expr):
        base = self.type_of(expr.object)
        if base is None:
            return None
        member = expr.member
        shape = type_shape(base)
        if shape is not None and shape.matrix is None:
            if isinstance(member, str) and _SWIZZLE.match(member):
                return format_type(shape.scalar, len(member), GLSL_FAMILY)
            return None
        struct = self.importer.structs_by_name.get(base)
        if struct is None:
            return None
        for field in getattr(struct, "members", None) or []:
            if getattr(field, "name", None) == member:
                field_type = _normalized(getattr(field, "vtype", None))
                if field_type and getattr(field, "is_array", False):
                    return f"{field_type}[]"
                return field_type
        return None

    def element_type(self, base):
        if base is None:
            return None
        array = _ARRAY_SUFFIX.match(base)
        if array:
            return array.group("element").strip()
        shape = type_shape(base)
        if shape is None or shape.is_scalar:
            return None
        if shape.matrix is not None:
            return format_type(shape.scalar, shape.matrix[1], GLSL_FAMILY)
        return format_type(shape.scalar, 1, GLSL_FAMILY)

    def call_type(self, expr):
        name = expr.name
        if isinstance(name, VariableNode):
            name = name.name
        if isinstance(name, MemberAccessNode):
            return "int" if name.member == "length" else None
        if not isinstance(name, str):
            return None
        if _is_type_name(name) or name in self.importer.structs_by_name:
            return name
        returns = self.importer.glsl_function_return_types.get(name)
        if returns is not None:
            return _normalized(next(iter(returns))) if len(returns) == 1 else None
        arguments = list(getattr(expr, "args", None) or [])
        if name in _TEXTURE_QUERY_TYPES:
            return _TEXTURE_QUERY_TYPES[name]
        if name in _GLSL_EXTENSION_FUNCTIONS:
            return _GLSL_EXTENSION_FUNCTIONS[name]
        argument_types = [self.type_of(argument) for argument in arguments]
        result = builtin_call_result_type(name, argument_types, GLSL_FAMILY)
        if result is not None:
            return result
        if _TEXEL_FUNCTION.match(name) and argument_types:
            return _texel_type(name, argument_types[0])
        return None

    def binary_type(self, expr):
        op = expr.op
        if op in _BOOLEAN_OPERATORS:
            return "bool"
        if op == ",":
            return self.type_of(expr.right)
        left, right = self.type_of(expr.left), self.type_of(expr.right)
        return _arithmetic_type(op, left, right)


def lower_aggregate_equality(importer, node, left, right):
    """CrossGL text for the GLSL equality ``node`` whose operands rendered as ``left`` and ``right``.

    Returns ``None`` when ``node`` is not ``==`` or ``!=``.  Raises :class:`GlslAggregateEqualityError` when neither
    operand's type is known.
    """
    operator = _EQUALITY_SPELLINGS.get(node.op)
    if operator is None:
        return None
    typer = GlslExpressionTypes(importer)
    shapes = [
        _operand_kind(typer.type_of(side), importer.structs_by_name)
        for side in (node.left, node.right)
    ]
    kinds = {kind for kind, _ in shapes if kind is not None}
    reduction, joiner = EQUALITY_OPERATORS[operator]
    if "vector" in kinds:
        return f"{reduction}({left} {operator} {right})"
    if "matrix" in kinds:
        columns = next(count for kind, count in shapes if kind == "matrix")
        if not (_is_pure(node.left) and _is_pure(node.right)):
            raise GlslMatrixEqualityOperandError(
                f"GLSL matrix '{operator}' compares every column, so its operands are repeated; they must be free of "
                "side effects",
                target="crossgl",
            )
        terms = [
            f"{reduction}({left}[{column}] {operator} {right}[{column}])"
            for column in range(columns)
        ]
        return "(" + f" {joiner} ".join(terms) + ")"
    if kinds:
        return f"({left} {operator} {right})"
    raise GlslAggregateEqualityError(
        f"cannot determine the operand types of GLSL '{left} {operator} {right}'; GLSL compares vectors and matrices "
        "as a whole while CrossGL compares them componentwise",
        target="crossgl",
    )


def _normalized(type_name):
    """``type_name`` without precision qualifiers (``highp vec3`` is ``vec3``), or ``None`` when empty."""
    if type_name is None:
        return None
    words = str(type_name).split()
    return words[-1] if words else None


def _operand_kind(type_name, struct_names=()):
    """``("vector"|"matrix"|"scalar"|"aggregate", columns)`` for a GLSL type, or ``(None, 0)`` when unknown.

    Only arrays and declared structures are aggregates; a type name that is neither, nor a scalar, vector or matrix
    (including GL_EXT_shader_explicit_arithmetic_types' ``i16vec2``, ``f64mat3`` and ``int64_t``), is unknown.
    """
    if type_name is None:
        return None, 0
    if _ARRAY_SUFFIX.match(type_name) or type_name in struct_names:
        return "aggregate", 0
    explicit = _EXPLICIT_VECTOR.fullmatch(type_name)
    if explicit:
        return "vector", 0
    explicit = _EXPLICIT_MATRIX.fullmatch(type_name)
    if explicit:
        return "matrix", int(explicit.group("columns"))
    if _EXPLICIT_SCALAR.fullmatch(type_name):
        return "scalar", 0
    shape = type_shape(type_name)
    if shape is None:
        return None, 0
    if shape.matrix is not None:
        return "matrix", shape.matrix[0]
    return ("scalar", 0) if shape.width == 1 else ("vector", 0)


def _literal_type(value):
    text = str(value)
    match = _FLOAT_LITERAL.match(text)
    if match:
        suffix = (match.group("suffix") or "").lower()
        return "double" if suffix == "lf" else "float"
    match = _INTEGER_LITERAL.match(text)
    if match:
        return "uint" if match.group("suffix") else "int"
    return None


def _is_type_name(name):
    if name in _SCALAR_TYPES:
        return True
    shape = type_shape(name)
    return shape is not None and shape.family == GLSL_FAMILY


def _texel_type(name, sampler_type):
    sampler = str(sampler_type or "")
    if "Shadow" in sampler and "Gather" not in name:
        return "float"
    prefix = sampler[:1] if sampler[:1] in ("i", "u") else ""
    if not re.match(r"^[iu]?(?:sampler|image|texture|subpassInput)", sampler):
        return None
    return f"{prefix}vec4"


def _arithmetic_type(op, left, right):
    """The type of GLSL arithmetic or bitwise ``op`` on ``left`` and ``right`` (sections 5.9 and 5.10)."""
    if op in ("<<", ">>", "BITWISE_SHIFT_LEFT", "BITWISE_SHIFT_RIGHT"):
        return left
    lhs, rhs = type_shape(left), type_shape(right)
    if lhs is None or rhs is None:
        return None
    multiply = op in ("*", "MULTIPLY")
    if lhs.matrix and rhs.matrix:
        if multiply:
            return _matrix_type(lhs.scalar, rhs.matrix[0], lhs.matrix[1])
        return left
    if lhs.matrix:
        return format_type(lhs.scalar, lhs.matrix[1], GLSL_FAMILY) if (multiply and rhs.width > 1) else left
    if rhs.matrix:
        return format_type(rhs.scalar, rhs.matrix[0], GLSL_FAMILY) if (multiply and lhs.width > 1) else right
    scalar = lhs.scalar
    if "double" in (lhs.scalar, rhs.scalar):
        scalar = "double"
    elif "float" in (lhs.scalar, rhs.scalar):
        scalar = "float"
    return format_type(scalar, max(lhs.width, rhs.width), GLSL_FAMILY)


def _matrix_type(scalar, columns, rows):
    prefix = {"double": "d", "half": "f16"}.get(scalar, "")
    return f"{prefix}mat{columns}" if columns == rows else f"{prefix}mat{columns}x{rows}"


def _is_pure(expr):
    """Whether evaluating ``expr`` more than once is equivalent to evaluating it once."""
    if expr is None or isinstance(expr, (bool, int, float, str, NumberNode)):
        return True
    if isinstance(expr, VariableNode):
        return True
    if isinstance(expr, MemberAccessNode):
        return _is_pure(expr.object)
    if isinstance(expr, ArrayAccessNode):
        return _is_pure(expr.array) and _is_pure(expr.index)
    if isinstance(expr, VectorConstructorNode):
        return all(_is_pure(argument) for argument in expr.args or [])
    if isinstance(expr, FunctionCallNode):
        name = expr.name.name if isinstance(expr.name, VariableNode) else expr.name
        return (
            isinstance(name, str)
            and _is_type_name(name)
            and all(_is_pure(argument) for argument in expr.args or [])
        )
    if isinstance(expr, BinaryOpNode):
        return _is_pure(expr.left) and _is_pure(expr.right)
    if isinstance(expr, UnaryOpNode):
        return expr.op not in ("++", "--") and _is_pure(expr.operand)
    if isinstance(expr, TernaryOpNode):
        return all(
            _is_pure(part) for part in (expr.condition, expr.true_expr, expr.false_expr)
        )
    return False
