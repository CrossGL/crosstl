"""WGSL lowering of the matrix ``inverse`` built-in.

CrossGL spells GLSL's ``inverse(m)`` (GLSL 4.60 section 8.6), which returns the inverse of a square float matrix.
WGSL declares ``determinant`` and ``transpose`` but no inverse (WGSL section 17.5, "Numeric Built-in Functions"), so
the call is written to a helper per matrix type::

    fn crosstl_inverse_mat3x3_f32(m: mat3x3<f32>) -> mat3x3<f32> { ... }

The helper runs Gauss-Jordan elimination with partial pivoting, the same algorithm the Slang generator's helper uses.
WGSL indexes a matrix by column, so the elimination treats the columns of ``m`` as the rows of its transpose; the
columns of the result are then the rows of ``inverse(transpose(m))``, which is ``inverse(m)``. GLSL leaves the result
undefined for a singular or poorly-conditioned matrix, and so does the helper. WGSL output computes ``half`` values as
``f32``, so a matrix of another element type, a non-square matrix or an operand of unknown type is refused with
:class:`WGSLMatrixInverseError`.
"""

from __future__ import annotations

from ..ast import IdentifierNode

__all__ = [
    "WGSLMatrixInverseError",
    "is_matrix_inverse_call",
    "wgsl_matrix_inverse_call",
    "wgsl_matrix_inverse_helpers",
]


class WGSLMatrixInverseError(ValueError):
    """Raised when an ``inverse`` call has no WGSL helper: its operand is not a square ``f32`` matrix."""

    project_diagnostic_code = "project.translate.wgsl-matrix-inverse-unsupported"
    missing_capabilities = ("wgsl.matrix-inverse-lowering",)


def is_matrix_inverse_call(codegen, call):
    """Whether ``call`` is the built-in ``inverse`` with one operand."""
    function = call.function
    return (
        isinstance(function, IdentifierNode)
        and function.name == "inverse"
        and len(call.arguments) == 1
        and not codegen.is_user_function_name(function.name)
    )


def wgsl_matrix_inverse_call(codegen, call):
    """The helper call for ``inverse(m)``, registering the helper on ``codegen``."""
    (operand,) = call.arguments
    operand_type = codegen.expression_type(operand)
    shape = codegen.matrix_shape(operand_type)
    if shape is None:
        raise WGSLMatrixInverseError(
            "WGSL target cannot lower inverse(): its operand is not a matrix of a "
            "known type"
        )
    element, columns, rows = shape
    if columns != rows or element != "f32":
        raise WGSLMatrixInverseError(
            f"WGSL target cannot lower inverse() on mat{columns}x{rows}<{element}>; "
            "only square f32 matrices have an inverse"
        )
    helper = codegen.safe_wgsl_identifier(f"crosstl_inverse_mat{rows}x{rows}_f32")
    codegen._matrix_inverse_helpers.setdefault(helper, rows)
    argument = codegen.generate_expression_for_target(operand, operand_type)
    return f"{helper}({argument})"


def wgsl_matrix_inverse_helpers(codegen):
    """The helper functions that the generated calls use, in the order of their first call."""
    helpers = codegen._matrix_inverse_helpers
    return "\n\n".join(_helper(helper, helpers[helper]) for helper in helpers)


def _helper(helper, size):
    matrix = f"mat{size}x{size}<f32>"
    identity_columns = ", ".join(
        "vec{}<f32>({})".format(
            size, ", ".join("1.0" if row == column else "0.0" for row in range(size))
        )
        for column in range(size)
    )
    return (
        f"fn {helper}(m: {matrix}) -> {matrix} {{\n"
        "    var a = m;\n"
        f"    var inv = {matrix}({identity_columns});\n"
        f"    for (var i = 0; i < {size}; i++) {{\n"
        "        var pivot = i;\n"
        "        var pivotAbs = abs(a[i][i]);\n"
        f"        for (var c = i + 1; c < {size}; c++) {{\n"
        "            let candidate = abs(a[c][i]);\n"
        "            if (candidate > pivotAbs) {\n"
        "                pivotAbs = candidate;\n"
        "                pivot = c;\n"
        "            }\n"
        "        }\n"
        "        if (pivot != i) {\n"
        "            let column = a[i];\n"
        "            a[i] = a[pivot];\n"
        "            a[pivot] = column;\n"
        "            let invColumn = inv[i];\n"
        "            inv[i] = inv[pivot];\n"
        "            inv[pivot] = invColumn;\n"
        "        }\n"
        "        let scale = 1.0 / a[i][i];\n"
        "        a[i] = a[i] * scale;\n"
        "        inv[i] = inv[i] * scale;\n"
        f"        for (var c = 0; c < {size}; c++) {{\n"
        "            if (c != i) {\n"
        "                let factor = a[c][i];\n"
        "                a[c] = a[c] - factor * a[i];\n"
        "                inv[c] = inv[c] - factor * inv[i];\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "    return inv;\n"
        "}"
    )
