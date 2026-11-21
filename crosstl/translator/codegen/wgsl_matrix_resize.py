"""Matrix constructors that resize another matrix, for WGSL.

CrossGL, like GLSL (4.60 section 5.4.2, "Matrix Constructors"), constructs a matrix from a matrix of another shape:
element ``[c][r]`` of the result is element ``[c][r]`` of the argument where the argument has one, and element
``[c][r]`` of the identity matrix elsewhere. ``mat2(m3)`` keeps the upper-left 2x2 block of ``m3`` and ``mat3(m2)``
pads ``m2`` with the identity. WGSL converts a matrix only to a matrix of the same shape (WGSL section 17.1.2,
"Conversion Built-in Functions"), so ``mat2x2<f32>(m)`` of a ``mat3x3<f32>`` is rejected.

The constructor becomes a call of a helper per pair of shapes and element types, so the argument is evaluated once::

    fn crosstl_resize_mat3x3_f32_to_mat2x2_f32(m: mat3x3<f32>) -> mat2x2<f32> {
        return mat2x2<f32>(m[0].xy, m[1].xy);
    }
"""

from __future__ import annotations

__all__ = ["wgsl_matrix_resize", "wgsl_matrix_resize_helpers"]

_COMPONENTS = "xyzw"


def wgsl_matrix_resize(codegen, target_type, argument):
    """The helper call that converts matrix ``argument`` to ``target_type`` of another shape, or ``None``."""
    target = codegen.matrix_shape(target_type)
    if target is None:
        return None
    source_type = codegen.expression_type(argument)
    source = codegen.matrix_shape(source_type)
    if source is None or source[1:] == target[1:]:
        return None
    helper = codegen.safe_wgsl_identifier(
        f"crosstl_resize_mat{source[1]}x{source[2]}_{source[0]}"
        f"_to_mat{target[1]}x{target[2]}_{target[0]}"
    )
    codegen._matrix_resize_helpers.setdefault(helper, (source, target))
    value = codegen.generate_expression_for_target(argument, source_type)
    return f"{helper}({value})"


def wgsl_matrix_resize_helpers(codegen):
    """The helper functions the generated calls use, in the order of their first call."""
    helpers = codegen._matrix_resize_helpers
    return "\n\n".join(_helper(name, *helpers[name]) for name in helpers)


def _helper(name, source, target):
    source_element, source_columns, source_rows = source
    element, columns, rows = target
    source_matrix = f"mat{source_columns}x{source_rows}<{source_element}>"
    matrix = f"mat{columns}x{rows}<{element}>"
    column_values = [_column(column, source, target) for column in range(columns)]
    return (
        f"fn {name}(m: {source_matrix}) -> {matrix} {{\n"
        f"    return {matrix}({', '.join(column_values)});\n"
        "}"
    )


def _identity(column, row):
    return "1.0" if row == column else "0.0"


def _column(column, source, target):
    source_element, source_columns, source_rows = source
    element, _columns, rows = target
    vector = f"vec{rows}<{element}>"
    if column >= source_columns:
        return f"{vector}({', '.join(_identity(column, row) for row in range(rows))})"
    value = f"m[{column}]"
    if source_element != element:
        value = f"vec{source_rows}<{element}>({value})"
    if rows < source_rows:
        return f"{value}.{_COMPONENTS[:rows]}"
    if rows == source_rows:
        return value
    padding = ", ".join(_identity(column, row) for row in range(source_rows, rows))
    return f"{vector}({value}, {padding})"
