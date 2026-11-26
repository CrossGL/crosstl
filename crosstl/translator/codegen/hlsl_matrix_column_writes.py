"""Writes through a dynamic CrossGL matrix column in HLSL output.

CrossGL ``m[k]`` is column ``k`` while HLSL's ``m[k]`` is row ``k`` (see :mod:`.hlsl_matrix_orientation`), so a
dynamic column read is ``transpose(m)[k]``. That is not an lvalue, but DXC accepts an assignment, an increment and
an ``out`` or ``inout`` argument through it: checked with DXC 1.9.2602, ``transpose(m)[k].y = 7.0``,
``transpose(m)[k].y++`` and ``bump(transpose(m)[k])`` all compile to DXIL that writes a temporary nothing reads,
so the source's write is lost without a diagnostic. This module routes each such write through the matrix:

* ``m[k] op= v`` and ``m[k].yx op= v`` whose value is discarded store each element they write in place, since
  HLSL ``m[r][k]`` is CrossGL ``m[k][r]`` and is an lvalue. A value that more than one element store reads is
  evaluated once, first, into a function-scope temporary::

      (__crossgl_column = (v), m[1][k] op= __crossgl_column[0], m[0][k] op= __crossgl_column[1])

  The whole column assigned in a statement of its own keeps the column setter,
  ``m = __crossgl_set_column_float2x2(m, k, v)``. No operand of a comma expression has a matrix type, which
  DXC's SPIR-V back end cannot compile.
* ``m[k]++`` and ``m[k].y--`` whose value is discarded are the compound assignments ``+= 1`` and ``-= 1``.
* An ``out`` or ``inout`` argument ``m[k]`` or ``m[k].y``, and the result argument of a built-in such as
  ``modf`` or ``frexp``, is passed through a temporary whose elements are stored back after the call
  (:mod:`.hlsl_call_argument_order`).

A matrix in a GLSL buffer block is held as bytes, and the generator's assignment lowering stores its columns; its
increments and reference arguments take the same routes, through that lowering. The column index is evaluated once: an index with a side effect goes into a temporary first. Any other write
through a dynamic column, such as an assignment or increment whose value is used, is refused.
"""

from __future__ import annotations

import copy
import re

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    PrimitiveType,
    SwizzleNode,
)
from .glsl_buffer_block_arrays import glsl_buffer_matrix_column_access

__all__ = [
    "column_copy_back",
    "dynamic_matrix_column",
    "hlsl_column_increment",
    "hlsl_column_write",
]

_SWIZZLE_SETS = ("xyzw", "rgba", "stpq")
# CrossGL matrices hold floating-point values (GLSL 4.60 section 4.1.6), so these are their compound operators.
_COMPONENTWISE = frozenset({"=", "+=", "-=", "*=", "/="})
_SWIZZLE = re.compile(r"[xyzw]{1,4}|[rgba]{1,4}|[stpq]{1,4}")


def _array(node):
    return getattr(node, "array_expr", getattr(node, "array", None))


def _index(node):
    return getattr(node, "index_expr", getattr(node, "index", None))


def _swizzle_indices(components):
    """The component indices a swizzle selects, or ``None`` when it is not a swizzle."""
    if not isinstance(components, str) or not _SWIZZLE.fullmatch(components):
        return None
    letters = next(group for group in _SWIZZLE_SETS if components[0] in group)
    return tuple(letters.index(letter) for letter in components)


def _swizzle_of(expr):
    """``(vector_expr, indices)`` when ``expr`` is a swizzle, else ``None``."""
    if isinstance(expr, SwizzleNode):
        indices = _swizzle_indices(expr.components)
        return None if indices is None else (expr.vector_expr, indices)
    if isinstance(expr, MemberAccessNode):
        indices = _swizzle_indices(getattr(expr, "member", None))
        if indices is None:
            return None
        return getattr(expr, "object_expr", getattr(expr, "object", None)), indices
    return None


def dynamic_matrix_column(generator, expr, *, buffers=False):
    """``(column, indices)`` when ``expr`` names a dynamic column of a matrix HLSL holds, else ``None``.

    ``column`` is the ``m[k]`` node and ``indices`` the column components a swizzle chain selects, or ``None``
    for the whole column. A matrix in a GLSL buffer block is held as bytes and has its own store path, so its
    columns count only with ``buffers``.
    """
    orientation = generator.hlsl_matrix_orientation
    indices = None
    node = expr
    while True:
        swizzle = _swizzle_of(node)
        if swizzle is None:
            break
        node, inner = swizzle
        if indices is not None and any(index >= len(inner) for index in indices):
            # A struct member spelled like a swizzle (``s.b.zx``): a matrix column is a vector, so every link of
            # a chain that ends at one is a swizzle that selects among the components of the link inside it.
            return None
        # An outer swizzle selects among the components the inner one selected.
        indices = inner if indices is None else tuple(inner[i] for i in indices)
    if not isinstance(node, ArrayAccessNode):
        return None
    matrix_expr = _array(node)
    if orientation.expression_dimensions(matrix_expr) is None:
        return None
    if orientation.constant_index(_index(node)) is not None:
        return None
    if not buffers and glsl_buffer_matrix_column_access(generator, node) is not None:
        return None
    return node, indices


def _temporary(generator, type_name, stem):
    """Declare a function-scope temporary of HLSL type ``type_name``; its name, or ``None``."""
    declarations = generator.hlsl_wide_integer_compound_indices
    if declarations is None or not type_name:
        return None
    used = generator.current_identifier_reserved_names | set(
        generator.local_variable_types
    )
    used.update(generator.hlsl_wide_integer_reserved_names)
    used.update(generator.global_variable_types)
    used.update(name for _type, name in declarations)
    name = stem
    while name in used or name in generator.HLSL_RESERVED_LOCAL_IDENTIFIER_NAMES:
        name += "_"
    generator.current_identifier_reserved_names.add(name)
    generator.local_variable_types[name] = type_name
    declarations.append((type_name, name))
    return name


def _refuse(generator, detail):
    return generator.refuse_lowering(
        f"DirectX writes a dynamic matrix column through a copy of the matrix, and {detail}"
    )


def hlsl_column_write(generator, target, value, op, *, discarded, standalone=True):
    """HLSL for ``target op= value`` when ``target`` writes through a dynamic matrix column, else ``None``.

    The result is an expression whose value is meaningless, so it is only valid where the value is
    ``discarded``; anywhere else the write is refused. A ``standalone`` write is a whole statement, not an
    operand of a comma expression.
    """
    parsed = dynamic_matrix_column(generator, target)
    if parsed is None:
        return None
    if not discarded:
        return _refuse(generator, "the assignment's value is used")
    column, indices = parsed
    orientation = generator.hlsl_matrix_orientation
    matrix_expr = _array(column)
    index_expr = _index(column)
    repeatable = generator.hlsl_expression_is_repeatable
    if not repeatable(matrix_expr):
        return _refuse(generator, "the matrix expression has a side effect")
    if op not in _COMPONENTWISE:
        return _refuse(generator, f"'{op}' has no componentwise column lowering")
    if op == "*=" and orientation.expression_dimensions(value) is not None:
        return _refuse(
            generator,
            "a column-times-matrix compound assignment has no column lowering",
        )
    scalar, rows, columns = orientation.expression_dimensions(matrix_expr)
    if indices is None:
        indices = tuple(range(rows))
    elif any(component >= rows for component in indices) or len(set(indices)) != len(
        indices
    ):
        return _refuse(
            generator, "the swizzle does not name distinct column components"
        )
    steps = []
    rendered_value = generator.generate_expression(value)
    index = generator.generate_expression(index_expr)
    base = generator.generate_expression(matrix_expr)
    whole = indices == tuple(range(rows))
    if standalone and op == "=" and whole and repeatable(index_expr):
        # The whole column in one statement: the setter returns the updated matrix.
        setter = orientation.column_setter(scalar, rows, columns)
        return f"{base} = {setter}({base}, {index}, {rendered_value})"
    if not repeatable(index_expr):
        name = _temporary(generator, "int", "__crossgl_column_index")
        if name is None:
            return _refuse(generator, "the column index has a side effect")
        steps.append(f"{name} = {index}")
        index = name
    if len(indices) == 1:
        components = [f"({rendered_value})"]
    elif isinstance(value, IdentifierNode) and generator.expression_component_count(
        value
    ) == len(indices):
        components = [f"{rendered_value}[{j}]" for j in range(len(indices))]
    else:
        # Every element store reads the value, so it is evaluated once, first.
        name = _temporary(generator, f"{scalar}{len(indices)}", "__crossgl_column")
        if name is None:
            return _refuse(generator, "the column has no function-scope temporary")
        steps.append(f"{name} = {rendered_value}")
        components = [f"{name}[{j}]" for j in range(len(indices))]
    # HLSL m[r][c] is CrossGL m[c][r]; each element store writes the matrix in place, and no comma
    # operand has a matrix type (DXC's SPIR-V back end has no comma operator over matrices).
    for row, component in zip(indices, components):
        steps.append(f"{base}[{row}][{index}] {op} {component}")
    return steps[0] if len(steps) == 1 else "(" + ", ".join(steps) + ")"


def hlsl_column_increment(generator, expr, operand, operator):
    """HLSL for ``operand++`` or ``operand--`` through a dynamic matrix column, else ``None``."""
    if dynamic_matrix_column(generator, operand, buffers=True) is None:
        return None
    discarded = expr is getattr(generator, "hlsl_statement_expression", None)
    if not discarded:
        return _refuse(generator, "the increment's value is used")
    op = "+=" if operator == "++" else "-="
    one = LiteralNode(1, PrimitiveType("int"))
    write = hlsl_column_write(generator, operand, one, op, discarded=True)
    if write is not None:
        return write
    # A buffer block column: the generator's assignment lowering stores its bytes.
    return generator.generate_assignment(
        AssignmentNode(copy.deepcopy(operand), one, op), statement_context=False
    )


def column_copy_back(generator, target, name):
    """The copy-back of temporary ``name`` into the dynamic column ``target``, or ``None``."""
    return hlsl_column_write(
        generator, target, IdentifierNode(name), "=", discarded=True, standalone=False
    )
