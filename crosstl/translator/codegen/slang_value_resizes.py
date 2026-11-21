"""Slang for the one-argument CrossGL constructors that resize a matrix or truncate a vector.

GLSL 4.60 section 5.4.2 lets a matrix constructor take a matrix of another shape: the result takes the source's
element ``[c][r]`` wherever the source has one and the identity matrix's element everywhere else. It also lets a
vector constructor take a wider vector, whose leading components it keeps. CrossGL follows GLSL. Slang has neither
form: slangc rejects ``float3x3(float2x2)`` and ``float2(float4)`` with "type mismatch in expression". Both are
rendered here in forms Slang accepts:

- A matrix resize calls a helper function, one per pair of shapes, that builds the result row by row. The Slang
  emitter keeps a CrossGL ``matCxR`` as a ``floatRxC`` whose row ``r`` is the matrix's row ``r``, so the upper-left
  rule is the same in both: a row the source has keeps its leading components and is padded with the identity's
  elements, and a row it lacks is the identity's row. The helper evaluates its argument once.
- A vector truncation swizzles the argument to the target's width first: ``float2(v)`` becomes ``float2(v.xy)``.
"""

from __future__ import annotations

from .c_family_operands import postfix_operand

__all__ = ["slang_resize_constructor"]

_SWIZZLES = {1: "x", 2: "xy", 3: "xyz", 4: "xyzw"}
_IDENTITY_VALUES = {
    "bool": ("false", "true"),
    "int": ("0", "1"),
    "uint": ("0u", "1u"),
}


def slang_resize_constructor(codegen, constructor_type, args):
    """Slang for ``constructor_type(arg)`` when it resizes a matrix or truncates a vector, or ``None``."""
    if len(args or []) != 1:
        return None
    arg = args[0]
    arg_type = codegen.expression_result_type(arg)
    target_matrix = codegen.hlsl_matrix_dimensions(constructor_type)
    if target_matrix is not None:
        source_matrix = codegen.hlsl_matrix_dimensions(arg_type)
        if source_matrix is None or source_matrix[1:] == target_matrix[1:]:
            return None
        helper = _matrix_resize_helper(codegen, source_matrix, target_matrix)
        return f"{helper}({codegen.generate_expression_with_expected(arg, None)})"
    if not codegen.is_vector_value_type(constructor_type):
        return None
    if not codegen.is_vector_value_type(arg_type):
        return None
    width = codegen.value_component_count(constructor_type)
    source_width = codegen.value_component_count(arg_type)
    if not width or not source_width or source_width <= width:
        return None
    rendered = postfix_operand(codegen.generate_expression(arg))
    return f"{codegen.convert_type(constructor_type)}({rendered}.{_SWIZZLES[width]})"


def _matrix_resize_helper(codegen, source, target):
    source_component, source_rows, source_columns = source
    component, rows, columns = target
    source_type = f"{source_component}{source_rows}x{source_columns}"
    target_type = f"{component}{rows}x{columns}"
    name = codegen.helper_function_name(
        f"cgl_matrix_resize_{source_type}_to_{target_type}"
    )
    zero, one = _IDENTITY_VALUES.get(component, ("0.0", "1.0"))
    row_values = []
    for row in range(rows):
        if row >= source_rows:
            values = ", ".join(
                one if column == row else zero for column in range(columns)
            )
            row_values.append(f"{component}{columns}({values})")
            continue
        source_row = f"m[{row}]"
        if columns < source_columns:
            source_row = f"{source_row}.{_SWIZZLES[columns]}"
        padding = [
            one if column == row else zero for column in range(source_columns, columns)
        ]
        if padding or component != source_component:
            source_row = f"{component}{columns}({', '.join([source_row, *padding])})"
        row_values.append(source_row)
    codegen.register_helper_function(
        name,
        f"{target_type} {name}({source_type} m)\n"
        "{\n"
        f"    return {target_type}({', '.join(row_values)});\n"
        "}",
    )
    return name
