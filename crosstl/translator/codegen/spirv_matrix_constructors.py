"""SPIR-V columns for the one-argument matrix constructors of GLSL 4.60 section 5.4.2.

``matN(s)`` puts the scalar ``s`` on the diagonal and zero elsewhere. ``matCxR(m)`` copies the overlapping
elements of the matrix ``m`` and fills the rest from the identity matrix. Flattening either argument into a
component list would fill the matrix with ``s`` followed by zeros, or with ``m``'s columns run together, so these
forms are built column by column here.
"""

from __future__ import annotations

__all__ = ["one_argument_matrix_columns"]


def one_argument_matrix_columns(
    gen, function_name, arg, component_type, vector_type, cols, rows
):
    """The column vectors for ``function_name(arg)``, or None when ``arg`` is neither a scalar nor a matrix."""
    zero = gen.register_constant(0.0, component_type)
    one = gen.register_constant(1.0, component_type)
    if gen.is_scalar_numeric_value(arg):
        diagonal = gen.convert_vector_constructor_scalar(
            arg, component_type, function_name
        )
        return [
            gen.composite_construct(
                vector_type, [diagonal if row == col else zero for row in range(rows)]
            )
            for col in range(cols)
        ]
    matrix_info = gen.matrix_type_info_from_type(
        gen.find_registered_type_by_base(arg.type.base_type)
    )
    if matrix_info is None:
        return None
    source_column_type, source_cols = matrix_info
    column_info = gen.vector_type_info_from_type(source_column_type)
    if column_info is None:
        return None
    source_component_type, source_rows = column_info
    columns = []
    for col in range(cols):
        source_column = (
            gen.composite_extract(arg, source_column_type, col)
            if col < source_cols
            else None
        )
        components = []
        for row in range(rows):
            if source_column is not None and row < source_rows:
                element = gen.composite_extract(
                    source_column, source_component_type, row
                )
                components.append(
                    gen.convert_vector_constructor_scalar(
                        element, component_type, function_name
                    )
                )
            else:
                components.append(one if row == col else zero)
        columns.append(gen.composite_construct(vector_type, components))
    return columns
