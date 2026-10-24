"""GLSL's matrix built-ins ``matrixCompMult`` and ``outerProduct``, and matrix shape conversions, for MSL.

CrossGL spells its built-ins as GLSL does (GLSL 4.60 section 8.6). MSL has neither function: ``*`` on two MSL
matrices is the linear-algebra product. :class:`MetalMatrixBuiltins` lowers both to one helper per matrix shape;
the helpers take their operands by value, so each argument is evaluated once.

* ``matrixCompMult(x, y)`` multiplies column ``i`` of ``x`` by column ``i`` of ``y``.
* ``outerProduct(c, r)`` with ``c`` a ``vecR`` and ``r`` a ``vecC`` is the ``matCxR`` whose column ``j`` is
  ``c * r[j]``.
"""

from __future__ import annotations

import re

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

    def component_constructor(self, metal_type, args):
        """A matrix built from scalars and vectors whose components do not line up with its columns.

        GLSL (section 5.4.2) fills the columns with the components of the arguments in order; MSL only takes one
        column per argument or one scalar per element. Each argument is passed to a helper once, by value.
        """
        generator = self.generator
        shape = generator.metal_matrix_dimensions(metal_type)
        if shape is None or not args:
            return None
        scalar, columns, rows = shape
        parameter_types = []
        widths = []
        for arg in args:
            arg_type = generator.map_type(generator.expression_result_type(arg) or "")
            width = _component_width(arg_type)
            if width is None:
                return None
            parameter_types.append(arg_type)
            widths.append(width)
        if len(args) == 1 and widths[0] == 1:
            return None
        if all(width == 1 for width in widths) or all(
            width == rows for width in widths
        ):
            return None
        if sum(widths) != columns * rows:
            return None
        matrix = f"{scalar}{columns}x{rows}"
        name = f"__crossgl_{matrix}_from_" + "_".join(parameter_types)
        if name not in self.helpers:
            components = []
            for index, width in enumerate(widths):
                components += (
                    [f"p{index}"]
                    if width == 1
                    else [f"p{index}.{'xyzw'[lane]}" for lane in range(width)]
                )
            columns_text = ", ".join(
                f"{scalar}{rows}({', '.join(components[column * rows : (column + 1) * rows])})"
                for column in range(columns)
            )
            parameters = ", ".join(
                f"{parameter_type} p{index}"
                for index, parameter_type in enumerate(parameter_types)
            )
            self.helpers[name] = (
                f"inline {matrix} {name}({parameters}) {{\n"
                f"    return {matrix}({columns_text});\n"
                "}\n"
            )
        rendered = ", ".join(
            generator.generate_expression_with_expected(arg, None) for arg in args
        )
        return f"{name}({rendered})"

    def resize(self, target, source):
        """The helper converting a matrix of shape ``source`` to shape ``target``, each a ``(scalar, columns, rows)``.

        GLSL and HLSL matrix conversions to another shape keep the overlapping columns and rows, and fill the rest
        from the identity matrix (GLSL 4.60 section 5.4.2). MSL has no such conversion.
        """
        scalar, columns, rows = target
        source_scalar, source_columns, source_rows = source
        matrix = f"{scalar}{columns}x{rows}"
        source_matrix = f"{source_scalar}{source_columns}x{source_rows}"
        name = f"__crossgl_resize_{source_matrix}_to_{matrix}"
        if name not in self.helpers:
            swizzle = "xyzw"[:rows]
            column_texts = []
            for column in range(columns):
                if column >= source_columns:
                    values = ["1.0" if row == column else "0.0" for row in range(rows)]
                    column_texts.append(f"{scalar}{rows}({', '.join(values)})")
                elif rows == source_rows:
                    column_texts.append(f"m[{column}]")
                elif rows < source_rows:
                    column_texts.append(f"m[{column}].{swizzle}")
                else:
                    fill = [
                        "1.0" if row == column else "0.0"
                        for row in range(source_rows, rows)
                    ]
                    column_texts.append(
                        f"{scalar}{rows}(m[{column}], {', '.join(fill)})"
                    )
            self.helpers[name] = (
                f"inline {matrix} {name}({source_matrix} m) {{\n"
                f"    return {matrix}({', '.join(column_texts)});\n"
                "}\n"
            )
        return name

    def code(self):
        return "".join(self.helpers[name] for name in sorted(self.helpers))


def _component_width(type_name):
    match = re.fullmatch(r"(float|half|int|uint|short|ushort|bool)([234])?", type_name)
    if match is None:
        return None
    return int(match.group(2) or 1)


def _vector_shape(type_name):
    for scalar in ("float", "half"):
        suffix = type_name[len(scalar) :] if type_name.startswith(scalar) else ""
        if suffix in {"2", "3", "4"}:
            return scalar, int(suffix)
    return None
