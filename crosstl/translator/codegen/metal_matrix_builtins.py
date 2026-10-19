"""GLSL's matrix built-ins ``matrixCompMult`` and ``outerProduct`` for MSL.

CrossGL spells its built-ins as GLSL does (GLSL 4.60 section 8.6). MSL has neither function: ``*`` on two MSL
matrices is the linear-algebra product. :class:`MetalMatrixBuiltins` lowers both to one helper per matrix shape;
the helpers take their operands by value, so each argument is evaluated once.

* ``matrixCompMult(x, y)`` multiplies column ``i`` of ``x`` by column ``i`` of ``y``.
* ``outerProduct(c, r)`` with ``c`` a ``vecR`` and ``r`` a ``vecC`` is the ``matCxR`` whose column ``j`` is
  ``c * r[j]``.
"""

from __future__ import annotations

__all__ = ["MetalMatrixBuiltins"]

MATRIX_BUILTINS = frozenset({"matrixCompMult", "outerProduct"})


class MetalMatrixBuiltins:
    def __init__(self, generator):
        self.generator = generator
        self.helpers = {}

    def call(self, func_name, args):
        """Render a call of a GLSL matrix built-in, or ``None`` for any other call."""
        generator = self.generator
        if (
            func_name not in MATRIX_BUILTINS
            or func_name in generator.user_function_names
        ):
            return None
        if len(args) != 2:
            return None
        if func_name == "matrixCompMult":
            shape = generator.metal_matrix_dimensions(
                generator.expression_result_type(args[0]) or ""
            )
            if shape is None:
                return None
            name = self.component_product(*shape)
        else:
            name = self.outer_product(args)
            if name is None:
                return None
        rendered = ", ".join(generator.generate_expression(arg) for arg in args)
        return f"{name}({rendered})"

    def component_product(self, scalar, columns, rows):
        matrix = f"{scalar}{columns}x{rows}"
        name = f"__crossgl_matrixCompMult_{matrix}"
        if name not in self.helpers:
            columns_text = ", ".join(
                f"x[{column}] * y[{column}]" for column in range(columns)
            )
            self.helpers[name] = (
                f"inline {matrix} {name}({matrix} x, {matrix} y) {{\n"
                f"    return {matrix}({columns_text});\n"
                "}\n"
            )
        return name

    def outer_product(self, args):
        generator = self.generator
        column_type = generator.map_type(
            generator.expression_result_type(args[0]) or ""
        )
        row_type = generator.map_type(generator.expression_result_type(args[1]) or "")
        column = _vector_shape(column_type)
        row = _vector_shape(row_type)
        if column is None or row is None or column[0] != row[0]:
            return None
        scalar, rows = column
        columns = row[1]
        matrix = f"{scalar}{columns}x{rows}"
        name = f"__crossgl_outerProduct_{matrix}"
        if name not in self.helpers:
            columns_text = ", ".join(f"c * r[{index}]" for index in range(columns))
            self.helpers[name] = (
                f"inline {matrix} {name}({column_type} c, {row_type} r) {{\n"
                f"    return {matrix}({columns_text});\n"
                "}\n"
            )
        return name

    def code(self):
        return "".join(self.helpers[name] for name in sorted(self.helpers))


def _vector_shape(type_name):
    for scalar in ("float", "half"):
        suffix = type_name[len(scalar) :] if type_name.startswith(scalar) else ""
        if suffix in {"2", "3", "4"}:
            return scalar, int(suffix)
    return None
