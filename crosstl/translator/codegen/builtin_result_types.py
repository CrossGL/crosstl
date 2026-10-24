"""Result types of CrossGL built-in calls and of component access, shared by the emitters.

An emitter that declares ``let r = smoothstep(0.0, 1.0, v);`` or ``let c = m[0];`` must write the type CrossGL gives
the initializer.  CrossGL spells its built-in functions as GLSL does, so the result types below are the overloads of
GLSL 4.60 chapter 8 (the HLSL spellings CrossGL also accepts, such as ``lerp`` and ``frac``, follow the GLSL function
they name).  Argument types arrive in the emitter's spelling: the GLSL family (``vec3``, ``ivec2``, ``mat2x3``) or the
C family HLSL and MSL share (``float3``, ``int2``, ``float3x3``).  A result built from an argument keeps that
argument's family; ``None`` means the rule cannot decide, and the emitter must not guess a type in its place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..ast import ArrayAccessNode, FunctionCallNode

GLSL_FAMILY = "glsl"
C_FAMILY = "c"

_GLSL_VECTOR_SCALARS = {
    "": "float",
    "i": "int",
    "u": "uint",
    "b": "bool",
    "d": "double",
    "f16": "half",
}
_GLSL_VECTOR_PREFIXES = {
    scalar: prefix for prefix, scalar in _GLSL_VECTOR_SCALARS.items()
}
_GLSL_SCALAR_SPELLINGS = {"float16_t": "half"}
_SCALARS = {"float", "int", "uint", "bool", "double", "half"}
_GLSL_VECTOR = re.compile(r"(i|u|b|d|f16)?vec([234])")
_GLSL_MATRIX = re.compile(r"(d|f16)?mat([234])(?:x([234]))?")
_C_VECTOR = re.compile(r"(float|int|uint|bool|half|double)([234])")
_C_MATRIX = re.compile(r"(float|half|double)([234])x([234])")


@dataclass(frozen=True)
class TypeShape:
    """A scalar, vector or matrix type: its scalar, vector width and, for a matrix, both spelled extents."""

    scalar: str
    width: int = 1
    matrix: tuple[int, int] | None = None
    family: str | None = None

    @property
    def is_scalar(self):
        return self.matrix is None and self.width == 1


def type_shape(type_name):
    """Parse ``type_name`` into a :class:`TypeShape`, or ``None`` if it is not a scalar, vector or matrix."""
    if not isinstance(type_name, str):
        return None
    name = type_name.strip()
    if name in _SCALARS:
        return TypeShape(name)
    if name in _GLSL_SCALAR_SPELLINGS:
        return TypeShape(_GLSL_SCALAR_SPELLINGS[name], family=GLSL_FAMILY)
    match = _GLSL_VECTOR.fullmatch(name)
    if match:
        return TypeShape(
            _GLSL_VECTOR_SCALARS[match.group(1) or ""],
            int(match.group(2)),
            family=GLSL_FAMILY,
        )
    match = _GLSL_MATRIX.fullmatch(name)
    if match:
        columns = int(match.group(2))
        rows = int(match.group(3) or columns)
        scalar = {"d": "double", "f16": "half"}.get(match.group(1) or "", "float")
        return TypeShape(scalar, matrix=(columns, rows), family=GLSL_FAMILY)
    match = _C_VECTOR.fullmatch(name)
    if match:
        return TypeShape(match.group(1), int(match.group(2)), family=C_FAMILY)
    match = _C_MATRIX.fullmatch(name)
    if match:
        return TypeShape(
            match.group(1),
            matrix=(int(match.group(2)), int(match.group(3))),
            family=C_FAMILY,
        )
    return None


def format_type(scalar, width, family):
    """Spell a scalar or vector type in ``family``; ``None`` when the family has no spelling for it."""
    if scalar not in _SCALARS or width not in {1, 2, 3, 4}:
        return None
    if width == 1:
        return "float16_t" if (scalar == "half" and family == GLSL_FAMILY) else scalar
    if family == GLSL_FAMILY:
        return f"{_GLSL_VECTOR_PREFIXES[scalar]}vec{width}"
    if family == C_FAMILY:
        return f"{scalar}{width}"
    return None


def component_access_type(type_name, family):
    """The type of ``value[i]`` for a vector or matrix ``value``, or ``None`` for any other type.

    Indexing a vector selects a component; indexing a CrossGL ``matCxR`` selects a column, a ``vecR`` (GLSL 4.60
    section 5.6).  A non-square matrix in the C family is left undecided: HLSL and MSL index rows and columns
    respectively, so its column type depends on how the emitter laid the matrix out.
    """
    shape = type_shape(type_name)
    if shape is None or shape.is_scalar:
        return None
    if shape.matrix is None:
        return format_type(shape.scalar, 1, shape.family or family)
    first, second = shape.matrix
    if shape.family == GLSL_FAMILY:
        return format_type(shape.scalar, second, GLSL_FAMILY)
    if first != second:
        return None
    return format_type(shape.scalar, first, C_FAMILY)


# Componentwise functions whose genType parameters share one type and whose scalar parameters broadcast, so the
# result has the first argument's scalar and the widest argument's width (GLSL 4.60 sections 8.1-8.5, 8.8 and 8.14).
_COMPONENTWISE = frozenset(
    {
        "abs", "sign", "floor", "trunc", "round", "roundEven", "ceil", "fract", "frac", "mod", "fmod",
        "min", "max", "clamp", "mix", "lerp", "step", "smoothstep", "saturate", "fma", "ldexp", "frexp", "modf",
        "pow", "exp", "exp2", "log", "log2", "sqrt", "inversesqrt", "rsqrt",
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
        "radians", "degrees", "normalize", "faceforward", "reflect", "refract", "cross",
        "dFdx", "dFdy", "dFdxFine", "dFdyFine", "dFdxCoarse", "dFdyCoarse", "fwidth", "fwidthFine", "fwidthCoarse",
        "ddx", "ddy", "ddx_fine", "ddy_fine", "ddx_coarse", "ddy_coarse",
        "bitfieldExtract", "bitfieldInsert", "bitfieldReverse",
    }
)  # fmt: skip
# Functions that keep a matrix argument's type (sections 8.6).
_MATRIX_PRESERVING = frozenset({"inverse", "matrixCompMult"})
# Functions whose result is the scalar of their first argument (sections 8.5 and 8.6).
_SCALAR_OF_FIRST = frozenset({"length", "distance", "dot", "determinant"})
# Componentwise relational functions return a bool vector of their arguments' width (section 8.7).
_BOOL_VECTOR = frozenset(
    {"lessThan", "lessThanEqual", "greaterThan", "greaterThanEqual", "equal", "notEqual", "not", "isnan", "isinf"}
)  # fmt: skip
_BOOL_SCALAR = frozenset({"any", "all"})
# Functions with a fixed result scalar and their argument's width (sections 8.3 and 8.8).
_FIXED_SCALAR = {
    "floatBitsToInt": "int",
    "floatBitsToUint": "uint",
    "intBitsToFloat": "float",
    "uintBitsToFloat": "float",
    "bitCount": "int",
    "findLSB": "int",
    "findMSB": "int",
}
# Packing functions (section 8.4).
_PACKING = {
    "packUnorm2x16": ("uint", 1),
    "packSnorm2x16": ("uint", 1),
    "packUnorm4x8": ("uint", 1),
    "packSnorm4x8": ("uint", 1),
    "packHalf2x16": ("uint", 1),
    "unpackUnorm2x16": ("float", 2),
    "unpackSnorm2x16": ("float", 2),
    "unpackHalf2x16": ("float", 2),
    "unpackUnorm4x8": ("float", 4),
    "unpackSnorm4x8": ("float", 4),
}
# Texture residency queries (texture_residency.py): an opaque int code, and whether it reports residency.
_RESIDENCY = {"textureResidency": ("int", 1), "textureResident": ("bool", 1)}
_TEXTURE_SIZE_FUNCTIONS = frozenset({"textureSize", "imageSize"})
_TEXTURE_SIZE_WIDTHS = (
    ("Buffer", 1),
    ("1DArray", 2),
    ("1D", 1),
    ("2DMSArray", 3),
    ("2DArray", 3),
    ("CubeArray", 3),
    ("Cube", 2),
    ("2DMS", 2),
    ("2DRect", 2),
    ("2D", 2),
    ("3D", 3),
)


def builtin_call_result_type(name, argument_types, family):
    """The CrossGL result type of built-in ``name`` applied to ``argument_types``, spelled for the emitter.

    ``argument_types`` holds the emitter's types of the arguments (``None`` where unknown) and ``family`` the
    emitter's spelling family, used only when no argument fixes one.  Returns ``None`` when ``name`` is not a
    built-in listed here or the argument types do not determine the result.
    """
    if not isinstance(name, str):
        return None
    shapes = [type_shape(argument) for argument in argument_types]
    if name in _PACKING or name in _RESIDENCY:
        scalar, width = _PACKING.get(name) or _RESIDENCY[name]
        return format_type(scalar, width, family)
    if name in _TEXTURE_SIZE_FUNCTIONS:
        return _texture_size_type(argument_types, family)
    if not shapes or shapes[0] is None:
        return None
    first = shapes[0]
    result_family = next(
        (shape.family for shape in shapes if shape and shape.family), family
    )
    if name == "transpose":
        return _transposed_matrix(argument_types[0])
    if name == "outerProduct":
        return _outer_product_type(shapes)
    if name in _MATRIX_PRESERVING:
        return argument_types[0] if first.matrix is not None else None
    if first.matrix is not None:
        return (
            format_type(first.scalar, 1, result_family)
            if name == "determinant"
            else None
        )
    if any(shape is not None and shape.matrix is not None for shape in shapes):
        return None
    if name in _SCALAR_OF_FIRST:
        return format_type(first.scalar, 1, result_family)
    if name in _BOOL_SCALAR:
        return "bool"
    if name in _BOOL_VECTOR:
        return format_type("bool", first.width, result_family)
    if name in _FIXED_SCALAR:
        return format_type(_FIXED_SCALAR[name], first.width, result_family)
    if name in _COMPONENTWISE:
        if any(shape is None for shape in shapes):
            return None
        width = max(shape.width for shape in shapes)
        if name in {"step", "smoothstep"}:
            # step(edge, x) and smoothstep(edge0, edge1, x) take their genType from x (section 8.3).
            return format_type(shapes[-1].scalar, width, result_family)
        return format_type(first.scalar, width, result_family)
    return None


def inferred_value_type(emitter, expression, family):
    """The CrossGL type of ``expression`` in ``emitter``'s spelling, or ``None`` when it cannot be determined.

    Built-in calls and vector or matrix component access are typed by the rules above, with their operands typed
    the same way; anything else, and a call to a user function that shadows a built-in name, keeps the emitter's own
    ``expression_result_type``.  ``emitter`` provides ``function_call_name``, ``function_return_types``,
    ``expression_result_type`` and ``type_name_string``.
    """
    if expression is None:
        return None
    if isinstance(expression, FunctionCallNode):
        name = emitter.function_call_name(expression)
        if name and name not in getattr(emitter, "function_return_types", {}):
            arguments = getattr(expression, "arguments", None) or []
            argument_types = [
                inferred_value_type(emitter, argument, family) for argument in arguments
            ]
            result = builtin_call_result_type(name, argument_types, family)
            if result is not None:
                return result
    elif isinstance(expression, ArrayAccessNode):
        base_type = inferred_value_type(emitter, expression.array, family)
        component = component_access_type(base_type, family)
        if component is not None:
            return component
    return emitter.type_name_string(emitter.expression_result_type(expression)) or None


def _transposed_matrix(type_name):
    shape = type_shape(type_name)
    if shape is None or shape.matrix is None:
        return None
    first, second = shape.matrix
    if shape.family == GLSL_FAMILY:
        prefix = {"double": "d", "half": "f16"}.get(shape.scalar, "")
        return f"{prefix}mat{second}x{first}" if first != second else type_name
    return f"{shape.scalar}{second}x{first}"


def _outer_product_type(shapes):
    if len(shapes) != 2 or any(
        shape is None or shape.matrix or shape.is_scalar for shape in shapes
    ):
        return None
    column, row = shapes
    if column.scalar != row.scalar or column.scalar not in {"float", "double", "half"}:
        return None
    if column.family == GLSL_FAMILY:
        # outerProduct(c, r) has r's width as its column count and c's as its row count (section 8.6).
        prefix = {"double": "d", "half": "f16"}.get(column.scalar, "")
        return f"{prefix}mat{row.width}x{column.width}"
    if column.width != row.width:
        return None
    return f"{column.scalar}{column.width}x{column.width}"


def _texture_size_type(argument_types, family):
    if not argument_types or not isinstance(argument_types[0], str):
        return None
    texture_type = argument_types[0]
    for marker, width in _TEXTURE_SIZE_WIDTHS:
        if marker in texture_type:
            return format_type("int", width, family)
    return None
