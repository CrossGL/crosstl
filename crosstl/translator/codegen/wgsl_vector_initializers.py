"""WGSL value constructors for CrossGL vector and matrix brace initializers.

CrossGL accepts a brace initializer for a vector or a matrix, as GLSL does: ``vec3 v = {x, 1.0, 2.0};`` and
``mat2 m = {{1.0, 0.0}, {0.0, x}};`` (GLSL 4.60 section 4.1.11, "Initializers"). WGSL has no initializer lists, so
WGSL output writes the type's value constructor instead, ``vec3<f32>(x, 1.0, 2.0)`` or
``mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, x))`` (WGSL, "Value Constructor Built-in Functions").

A vector initializer gives one scalar for each component. As in GLSL output, components after the final supplied
value are zero-initialized, written as the component type's zero value, ``T()`` (WGSL, "Zero Value Expressions").

A matrix initializer gives either one value for each column, a vector or a nested brace list, or one scalar for each
component in column-major order. Both are matrix constructor forms in WGSL. ``{}`` gives the zero value of either
type. Nested lists in an array of vectors are written the same way, as elements of ``array<vecN<T>, M>(...)``.
WGSL has no implicit conversion between concrete types, so each value is converted to the component or column
type as a structure member is (``wgsl_struct_initializers.wgsl_converted_value``).

Any other initializer, such as one with more values than the vector has components or a short matrix list, raises
:class:`WGSLVectorInitializerError`.
"""

from __future__ import annotations

from ..ast import ArrayLiteralNode, ArrayType
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .wgsl_array_constructors import wgsl_array_element
from .wgsl_struct_initializers import wgsl_converted_value

__all__ = [
    "WGSLVectorInitializerError",
    "is_vector_braced_initializer",
    "wgsl_vector_array_initializer",
    "wgsl_vector_initializer",
]


class WGSLVectorInitializerError(TranslationDiagnostic):
    """A vector or matrix brace initializer that WGSL cannot write as the type's value constructor."""

    def __init__(self, type_name, reason):
        self.type_name = type_name
        self.missing_capabilities = ("wgsl.vector-initializer",)
        super().__init__(
            f"WGSL target cannot write the brace initializer of {type_name}: {reason}",
            target="wgsl",
            code=f"{CODE_PREFIX}wgsl-vector-initializer-unsupported",
        )


def wgsl_vector_array_initializer(codegen, expr, target_type):
    """Return WGSL ``array<vecN<T>, M>(...)`` for an array of vectors whose initializer nests brace lists, else ``None``.

    Each nested list, as in ``vec3 vs[2] = {{x, 1.0, 2.0}, {3.0, 4.0, 5.0}};``, is written as a vector constructor,
    and the element type and count are spelled out so the array has its declared type. An unsized array takes its
    count from the list. A short list is left to the padded array path, and one longer than its array to the array
    literal path.
    """
    if not isinstance(expr, ArrayLiteralNode) or not isinstance(target_type, ArrayType):
        return None
    element_type = target_type.element_type
    if codegen.vector_shape(element_type) is None:
        return None
    values = list(expr.elements or [])
    if not any(isinstance(value, ArrayLiteralNode) for value in values):
        return None
    size = (
        len(values)
        if target_type.size is None
        else codegen.resolve_integer_expression(target_type.size)
    )
    if size != len(values):
        return None
    element = codegen.type_name_string(element_type)
    rendered = [wgsl_array_element(codegen, value, element_type) for value in values]
    return f"array<{element}, {size}>({', '.join(rendered)})"


def is_vector_braced_initializer(codegen, expr, target_type):
    """Whether ``expr`` is a brace initializer for a vector or a matrix."""
    return isinstance(expr, ArrayLiteralNode) and (
        codegen.vector_shape(target_type) is not None
        or codegen.matrix_shape(target_type) is not None
    )


def wgsl_vector_initializer(codegen, expr, target_type):
    """Return WGSL ``vecN<T>(...)`` or ``matCxR<T>(...)`` for a vector or matrix brace initializer, else ``None``."""
    if not isinstance(expr, ArrayLiteralNode):
        return None
    vector = codegen.vector_shape(target_type)
    if vector is not None:
        return _vector_value(codegen, list(expr.elements or []), *vector)
    matrix = codegen.matrix_shape(target_type)
    if matrix is not None:
        return _matrix_value(codegen, list(expr.elements or []), *matrix)
    return None


def _count(number, noun):
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _is_aggregate(codegen, value):
    """Whether ``value`` is a brace list, a vector or a matrix rather than a scalar."""
    if isinstance(value, ArrayLiteralNode):
        return True
    value_type = codegen.expression_type(value)
    return (
        codegen.vector_shape(value_type) is not None
        or codegen.matrix_shape(value_type) is not None
    )


def _is_column(codegen, value, rows):
    """Whether ``value`` can be a column of ``rows`` components: a brace list or a vector of that size."""
    if isinstance(value, ArrayLiteralNode):
        return True
    shape = codegen.vector_shape(codegen.expression_type(value))
    return shape is not None and shape[1] == rows


def _vector_value(codegen, values, scalar, lanes):
    vector = f"vec{lanes}<{scalar}>"
    if not values:
        return f"{vector}()"
    if len(values) > lanes:
        raise WGSLVectorInitializerError(
            vector,
            f"it has {lanes} components and the initializer gives {_count(len(values), 'value')}",
        )
    if any(_is_aggregate(codegen, value) for value in values):
        raise WGSLVectorInitializerError(
            vector, "each value of a vector initializer must be a scalar"
        )
    rendered = [wgsl_converted_value(codegen, value, scalar) for value in values]
    rendered += [f"{scalar}()"] * (lanes - len(values))
    return f"{vector}({', '.join(rendered)})"


def _matrix_value(codegen, values, scalar, columns, rows):
    matrix = f"mat{columns}x{rows}<{scalar}>"
    if not values:
        return f"{matrix}()"
    if len(values) == columns and all(
        _is_column(codegen, value, rows) for value in values
    ):
        column = f"vec{rows}<{scalar}>"
        rendered = [wgsl_converted_value(codegen, value, column) for value in values]
    elif len(values) == columns * rows and not any(
        _is_aggregate(codegen, value) for value in values
    ):
        rendered = [wgsl_converted_value(codegen, value, scalar) for value in values]
    else:
        raise WGSLVectorInitializerError(
            matrix,
            f"the initializer gives {_count(len(values), 'value')}; give one vector for each of its {columns} columns "
            f"or one scalar for each of its {columns * rows} components",
        )
    return f"{matrix}({', '.join(rendered)})"
