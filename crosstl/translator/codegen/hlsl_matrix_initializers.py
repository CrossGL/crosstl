"""HLSL text for a CrossGL brace list ``{...}`` that initializes a matrix.

GLSL 4.60 section 4.1.11 makes each element of a matrix initializer list one column: a nested
list or a vector, as in ``mat2 m = {vec2(1.0, 0.0), vec2(0.0, x)}``. CrossGL also accepts a flat
list of scalars, which fills the columns in order. Missing trailing elements are zero.

HLSL constructors fill rows, so the columns are written as the rows of the transposed matrix,
``transpose(float2x2(...))``, which keeps the elements in evaluation order. A vector column is
passed whole: an HLSL matrix constructor takes vectors and spreads their components.
"""

from __future__ import annotations

from ..ast import ArrayLiteralNode

__all__ = ("hlsl_matrix_aggregate_initializer",)


def _is_column(codegen, element):
    if isinstance(element, ArrayLiteralNode):
        return True
    width = codegen.expression_component_count(element)
    return width is not None and width > 1


def _zeros(codegen, node, component_type, count):
    return [
        codegen.hlsl_value_initialized_expression(node, component_type)
        for _index in range(count)
    ]


def _column(codegen, expr, column, mapped_type, component_type, rows):
    """The HLSL constructor arguments of one column: its components, padded to ``rows``."""
    if isinstance(column, ArrayLiteralNode):
        elements = list(getattr(column, "elements", []) or [])
        if len(elements) > rows:
            codegen.hlsl_aggregate_error(
                column,
                expected_type=mapped_type,
                reason="element-count-mismatch",
                expected_element_count=rows,
            )
        rendered = [
            codegen.generate_expression_with_expected(element, component_type)
            for element in elements
        ]
        return rendered + _zeros(codegen, column, component_type, rows - len(elements))
    width = codegen.expression_component_count(column)
    if width != rows:
        codegen.hlsl_aggregate_error(
            expr,
            expected_type=mapped_type,
            reason="matrix-shape-mismatch",
            detail=(
                f"a matrix column must be a list or a vector of {rows} components, "
                f"got {width if width is not None else 'an unknown number of'} components"
            ),
        )
    column_type = f"{component_type}{rows}"
    return [codegen.generate_expression_with_expected(column, column_type)]


def hlsl_matrix_aggregate_initializer(codegen, expr, mapped_type):
    """Lower ``{...}`` for the HLSL matrix ``mapped_type``, or return ``None`` for another type."""
    dimensions = codegen.hlsl_matrix_dimensions(mapped_type)
    if dimensions is None:
        return None
    if codegen.hlsl_is_zero_aggregate_initializer(expr):
        return codegen.hlsl_value_initialized_expression(expr, mapped_type)

    component_type, rows, columns = dimensions
    elements = list(getattr(expr, "elements", []) or [])
    column_wise = [_is_column(codegen, element) for element in elements]
    rendered = []
    if any(column_wise):
        if not all(column_wise):
            codegen.hlsl_aggregate_error(
                expr,
                expected_type=mapped_type,
                reason="matrix-shape-mismatch",
                detail="matrix columns cannot mix nested and flat initializer elements",
            )
        if len(elements) > columns:
            codegen.hlsl_aggregate_error(
                expr,
                expected_type=mapped_type,
                reason="element-count-mismatch",
                expected_element_count=columns,
            )
        for column in elements:
            rendered.extend(
                _column(codegen, expr, column, mapped_type, component_type, rows)
            )
        missing = (columns - len(elements)) * rows
    else:
        expected_count = rows * columns
        if len(elements) > expected_count:
            codegen.hlsl_aggregate_error(
                expr,
                expected_type=mapped_type,
                reason="element-count-mismatch",
                expected_element_count=expected_count,
            )
        rendered.extend(
            codegen.generate_expression_with_expected(element, component_type)
            for element in elements
        )
        missing = expected_count - len(elements)
    rendered.extend(_zeros(codegen, expr, component_type, missing))
    transposed_type = f"{component_type}{columns}x{rows}"
    return f"transpose({transposed_type}({', '.join(rendered)}))"
