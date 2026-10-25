"""CrossGL matrix orientation in HLSL.

CrossGL ``matCxR`` (ADR 0004) has C columns and R rows. The HLSL emitter declares it as ``floatRxC``, the same
R rows and C columns, so ``m * v`` stays ``mul(m, v)`` and HLSL's default ``column_major`` packing stores the
columns contiguously as std140 and std430 do. HLSL disagrees with CrossGL in three places, which this module
lowers:

* ``m[i]`` is row ``i`` in HLSL and column ``i`` in CrossGL (GLSL 4.60 section 5.6). A constant column becomes
  the matrix swizzle ``m._m0i_m1i...``, which is also an lvalue; a dynamic column read becomes
  ``transpose(m)[i]``; a dynamic column write calls a helper; ``m[i][j]`` becomes ``m[j][i]``.
* ``floatRxC(a, b, ...)`` fills rows, while CrossGL constructors fill columns (GLSL 4.60 section 5.4.2), so a
  constructor with several arguments becomes ``transpose(floatCxR(a, b, ...))``. ``floatRxC(s)`` splats ``s``
  while ``matCxR(s)`` is ``s`` on the diagonal and zero elsewhere.
* ``m.length()`` does not exist in HLSL; it is the column count.
"""

from __future__ import annotations

import re

from ..ast import ArrayAccessNode, MemberAccessNode
from ..matrix_shape import crossgl_matrix_shape

__all__ = ["HLSLMatrixOrientation"]

_SIMPLE_OPERAND = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\s*(?:\.[A-Za-z_][A-Za-z0-9_]*|\[[^\[\]]*\]))*"
    r"|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[fFhHlLuU]*"
)


def _array(node):
    return getattr(node, "array_expr", getattr(node, "array", None))


def _index(node):
    return getattr(node, "index_expr", getattr(node, "index", None))


def _operand(rendered):
    """``rendered`` as an operand of a postfix swizzle or subscript."""
    text = rendered.strip()
    return text if _SIMPLE_OPERAND.fullmatch(text) else f"({text})"


class HLSLMatrixOrientation:
    """Matrix subscripts, constructors and ``length()`` for one HLSL generation."""

    def __init__(self, generator):
        self.generator = generator
        self.column_setters = {}
        self.diagonal_builders = {}

    # Shapes -----------------------------------------------------------------------------------------

    def dimensions(self, type_name):
        """``(scalar, rows, columns)`` of an HLSL or CrossGL matrix type, or ``None``."""
        if not type_name:
            return None
        return self.generator.hlsl_matrix_dimensions(
            self.generator.type_name_string(type_name)
        )

    def expression_dimensions(self, expr):
        if expr is None or isinstance(expr, (str, int, float)):
            return None
        if isinstance(expr, MemberAccessNode):
            # A buffer block member declared as an array of matrices reports its element type.
            member = self.generator.glsl_buffer_block_member_access(expr)
            if member is not None and member.get("is_array"):
                return None
        return self.dimensions(self.generator.expression_result_type(expr))

    def column_type(self, matrix_type):
        """The type of ``m[i]``: a column of ``rows`` components."""
        dimensions = self.dimensions(matrix_type)
        if dimensions is None:
            return None
        scalar, rows, _columns = dimensions
        return f"{scalar}{rows}"

    def product_type(self, left_type, right_type):
        """The type of CrossGL ``left * right`` when either operand is a matrix (lowered to ``mul``)."""
        left = self.dimensions(left_type)
        right = self.dimensions(right_type)
        if left is None and right is None:
            return None
        generator = self.generator
        if left is not None and right is not None:
            scalar, rows, inner = left
            return f"{scalar}{rows}x{right[2]}" if right[1] == inner else None
        if left is not None:
            scalar, rows, columns = left
            other = right_type
        else:
            scalar, rows, columns = right
            other = left_type
        if not generator.is_vector_value_type(other):
            return generator.map_type(left_type if left is not None else right_type)
        width = generator.value_component_count(other)
        if left is not None:
            return f"{scalar}{rows}" if width == columns else None
        return f"{scalar}{columns}" if width == rows else None

    def arithmetic_type(self, operator, left_type, right_type):
        """The type of a componentwise CrossGL ``+``, ``-`` or ``/`` with a matrix operand.

        A matrix and a scalar, or two matrices of one shape, give that matrix type (GLSL 4.60 section 5.9).
        ``None`` for every other operation, which keeps the caller's typing.
        """
        if operator not in {"+", "-", "/"}:
            return None
        left = self.dimensions(left_type)
        right = self.dimensions(right_type)
        if left is None and right is None:
            return None
        generator = self.generator
        if left is not None and right is not None:
            return generator.map_type(left_type) if left == right else None
        matrix_type, other = (
            (left_type, right_type) if left is not None else (right_type, left_type)
        )
        if other is None or not generator.is_scalar_value_type(other):
            return None
        return generator.map_type(matrix_type)

    def call_type(self, func_name, arg_types):
        """The type of ``matrixCompMult(a, b)`` or ``outerProduct(c, r)``, or ``None``.

        ``matrixCompMult`` keeps its operands' type, and ``outerProduct(c, r)`` has as many rows as ``c`` has
        components and as many columns as ``r`` (GLSL 4.60 section 8.6).
        """
        if len(arg_types) != 2:
            return None
        generator = self.generator
        if func_name == "matrixCompMult":
            if self.dimensions(arg_types[0]) is None:
                return None
            return generator.map_type(arg_types[0])
        if func_name == "outerProduct":
            column, row = arg_types
            if not (
                generator.is_vector_value_type(column)
                and generator.is_vector_value_type(row)
            ):
                return None
            rows = generator.value_component_count(column)
            columns = generator.value_component_count(row)
            return f"{generator.vector_component_type(column)}{rows}x{columns}"
        return None

    def constant_index(self, index_expr):
        return self.generator.literal_int_value(
            index_expr, getattr(self.generator, "literal_int_constants", {})
        )

    # Subscripts -------------------------------------------------------------------------------------

    def subscript(self, array_expr, index_expr):
        """Render ``array_expr[index_expr]`` when ``array_expr`` is a matrix or one of its columns."""
        if isinstance(array_expr, ArrayAccessNode):
            matrix_expr = _array(array_expr)
            if self.expression_dimensions(matrix_expr) is not None:
                return self.element(matrix_expr, _index(array_expr), index_expr)
        dimensions = self.expression_dimensions(array_expr)
        if dimensions is None:
            return None
        return self.column(array_expr, index_expr, dimensions)

    def element(self, matrix_expr, column_expr, row_expr):
        generator = self.generator
        if not (
            generator.hlsl_expression_is_repeatable(column_expr)
            or generator.hlsl_expression_is_repeatable(row_expr)
        ):
            return generator.refuse_lowering(
                "DirectX cannot reorder the column and row subscripts of a matrix element "
                "when both have side effects"
            )
        base = _operand(generator.generate_expression(matrix_expr))
        column = generator.generate_expression(column_expr)
        row = generator.generate_expression(row_expr)
        return f"{base}[{row}][{column}]"

    def column(self, matrix_expr, index_expr, dimensions):
        generator = self.generator
        _scalar, rows, columns = dimensions
        base = _operand(generator.generate_expression(matrix_expr))
        constant = self.constant_index(index_expr)
        if constant is not None:
            if not 0 <= constant < columns:
                return generator.refuse_lowering(
                    f"matrix column {constant} is out of range for a matrix of {columns} columns"
                )
            return base + "." + "".join(f"_m{row}{constant}" for row in range(rows))
        return f"transpose({base})[{generator.generate_expression(index_expr)}]"

    def column_assignment(self, target, value, op, statement_context):
        """``m[i] op= v`` for a dynamic column ``i``; ``None`` for every other assignment."""
        if not isinstance(target, ArrayAccessNode):
            return None
        matrix_expr = _array(target)
        index_expr = _index(target)
        dimensions = self.expression_dimensions(matrix_expr)
        if dimensions is None or self.constant_index(index_expr) is not None:
            return None
        generator = self.generator
        if not statement_context:
            return generator.refuse_lowering(
                "DirectX cannot use a dynamic matrix column assignment as a value"
            )
        repeatable = generator.hlsl_expression_is_repeatable
        if not repeatable(matrix_expr) or (op != "=" and not repeatable(index_expr)):
            return generator.refuse_lowering(
                "DirectX writes a dynamic matrix column through a copy of the matrix, which "
                "needs a matrix and column expression without side effects"
            )
        scalar, rows, columns = dimensions
        base = generator.generate_expression(matrix_expr)
        index = generator.generate_expression(index_expr)
        rendered_value = generator.generate_expression(value)
        if op != "=":
            operator = op[:-1]
            if operator == "*" and self.expression_dimensions(value) is not None:
                return generator.refuse_lowering(
                    "DirectX has no lowering for a column-times-matrix compound assignment "
                    "with a dynamic column"
                )
            rendered_value = (
                f"(transpose({_operand(base)})[{index}] {operator} ({rendered_value}))"
            )
        helper = self.column_setter(scalar, rows, columns)
        return f"{base} = {helper}({base}, {index}, {rendered_value})"

    def column_setter(self, scalar, rows, columns):
        name = f"__crossgl_set_column_{scalar}{rows}x{columns}"
        if name not in self.column_setters:
            matrix = f"{scalar}{rows}x{columns}"
            stores = " ".join(f"m[{row}][c] = v[{row}];" for row in range(rows))
            self.column_setters[name] = (
                f"{matrix} {name}({matrix} m, int c, {scalar}{rows} v) {{\n"
                f"    {stores}\n"
                "    return m;\n"
                "}\n"
            )
        return name

    # Constructors -----------------------------------------------------------------------------------

    def constructor(self, constructor_type, args, render):
        """Render a CrossGL matrix constructor, or ``None`` when ``constructor_type`` is not a CrossGL matrix.

        ``render(arg)`` renders one argument. A one-matrix argument is handled by the resize constructor, or is a
        copy when the shapes match.
        """
        shape = crossgl_matrix_shape(constructor_type)
        if shape is None or not args:
            return None
        scalar_kind, columns, rows = shape
        scalar = "double" if scalar_kind == "double" else "float"
        generator = self.generator
        if len(args) == 1:
            if self.expression_dimensions(args[0]) is not None:
                return None
            count = generator.expression_component_count(args[0])
            if count == 1:
                return self.diagonal(scalar, rows, columns, args[0], render(args[0]))
            if count != rows * columns:
                return None
        rendered = ", ".join(render(arg) for arg in args)
        return f"transpose({scalar}{columns}x{rows}({rendered}))"

    def diagonal(self, scalar, rows, columns, arg, rendered):
        if not self.generator.hlsl_expression_is_repeatable(arg):
            name = f"__crossgl_diagonal_{scalar}{rows}x{columns}"
            if name not in self.diagonal_builders:
                self.diagonal_builders[name] = (
                    f"{scalar}{rows}x{columns} {name}({scalar} s) {{\n"
                    f"    return {self.diagonal_value(scalar, rows, columns, 's')};\n"
                    "}\n"
                )
            return f"{name}({rendered})"
        return self.diagonal_value(scalar, rows, columns, _operand(rendered))

    @staticmethod
    def diagonal_value(scalar, rows, columns, value):
        elements = [
            value if row == column else "0.0"
            for row in range(rows)
            for column in range(columns)
        ]
        return f"{scalar}{rows}x{columns}({', '.join(elements)})"

    # Built-ins --------------------------------------------------------------------------------------

    def builtin_call(self, func_name, args):
        """``matrixCompMult`` and ``outerProduct`` (GLSL 4.60 section 8.6); ``None`` for any other call."""
        generator = self.generator
        if len(args) != 2 or func_name in getattr(generator, "user_function_names", ()):
            return None
        if func_name == "matrixCompMult":
            if self.expression_dimensions(args[0]) is None:
                return None
            # HLSL's * on two matrices is componentwise.
            left, right = (_operand(generator.generate_expression(arg)) for arg in args)
            return f"({left} * {right})"
        if func_name != "outerProduct":
            return None
        widths = []
        for arg in args:
            arg_type = generator.expression_result_type(arg)
            if not generator.is_vector_value_type(arg_type):
                return None
            widths.append(generator.value_component_count(arg_type))
        scalar = generator.vector_component_type(
            generator.map_type(generator.expression_result_type(args[0]))
        )
        column, row = (generator.generate_expression(arg) for arg in args)
        return f"mul({scalar}{widths[0]}x1({column}), {scalar}1x{widths[1]}({row}))"

    # length() ---------------------------------------------------------------------------------------

    def length_call(self, func_expr, args):
        """``m.length()`` on a matrix: its column count, an ``int`` constant."""
        if args or not isinstance(func_expr, MemberAccessNode):
            return None
        if getattr(func_expr, "member", None) != "length":
            return None
        dimensions = self.expression_dimensions(
            getattr(func_expr, "object_expr", getattr(func_expr, "object", None))
        )
        if dimensions is None:
            return None
        return f"int({dimensions[2]})"

    def helper_code(self):
        helpers = {**self.column_setters, **self.diagonal_builders}
        return "".join(helpers[name] for name in sorted(helpers))
