"""Dynamic row reads of HLSL matrices that have more rows than columns.

DXC lowers a dynamically indexed row of a ``float3x2``, ``float4x2`` or ``float4x3`` (any matrix with more rows than
columns) that lives in a constant buffer, in an array element or in a structure member through a temporary sized by
the column count, and then writes every row into it. DXIL validation rejects the program ("Access to out-of-bounds
memory is disallowed"); with validation off the row read is out of bounds. DXC 1.9.2602 shows this for
``m[r]``, ``m[r][c]``, ``a[k][r]`` and ``s.m[r]`` alike. A matrix held in a local variable or a parameter is lowered
correctly, so such a read takes its matrix from a parameter copy::

    float3x2 crossgl_matrix_value_float3x2(float3x2 value) { return value; }
    ... crossgl_matrix_value_float3x2(m[k])[r][c] ...

The helper returns its argument unchanged, so the value read is the same. A row written through an assignment, an
increment, or an ``out``/``inout`` argument keeps the direct form, which DXC lowers correctly and which a call result
could not replace: those lvalue paths are marked before their function body or assignment is rendered.
"""

from __future__ import annotations

import re

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    IdentifierNode,
    MemberAccessNode,
    SwizzleNode,
    UnaryOpNode,
)

__all__ = ["HlslTallMatrixRows"]

_COMPOUND_ASSIGNMENTS = frozenset(
    {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="}
)


class HlslTallMatrixRows:
    """Route dynamic row reads of tall matrices through a parameter copy."""

    def __init__(self, codegen):
        self.codegen = codegen
        # Mapped matrix type -> helper name.
        self.helpers = {}
        # id -> node for every node on a written lvalue path; the node is kept so its id stays unique.
        self.lvalues = {}

    # Lvalue paths -----------------------------------------------------------------------------------------------

    def mark_chain(self, node):
        """Mark ``node`` and the array, member and swizzle bases it is written through."""
        while node is not None and id(node) not in self.lvalues:
            self.lvalues[id(node)] = node
            if isinstance(node, ArrayAccessNode):
                node = getattr(node, "array_expr", None)
            elif isinstance(node, MemberAccessNode):
                node = getattr(node, "object_expr", None)
            elif isinstance(node, SwizzleNode):
                node = getattr(node, "vector_expr", None)
            else:
                return

    def mark_writes(self, root):
        """Mark the assignment targets and increment operands under ``root``."""
        for node in self.codegen.walk_ast(root):
            if isinstance(node, AssignmentNode):
                self.mark_chain(getattr(node, "target", None))
            elif isinstance(node, UnaryOpNode) and getattr(node, "operator", None) in {
                "++",
                "--",
            }:
                self.mark_chain(getattr(node, "operand", None))
            elif (
                isinstance(node, BinaryOpNode)
                and getattr(node, "operator", None) in _COMPOUND_ASSIGNMENTS
            ):
                self.mark_chain(getattr(node, "left", None))

    # Reads ------------------------------------------------------------------------------------------------------

    def tall_matrix_type(self, expr):
        codegen = self.codegen
        value_type = codegen.expression_result_type(expr)
        if value_type is None:
            return None
        shape = codegen.hlsl_matrix_shape(value_type)
        if shape is None or int(shape[1]) <= int(shape[2]):
            return None
        return codegen.map_type(value_type)

    def held_in_a_local(self, expr):
        if not isinstance(expr, IdentifierNode):
            return False
        return getattr(expr, "name", None) in (self.codegen.local_variable_types or {})

    def row_read(self, matrix_expr, row_expr, rendered_matrix):
        """The operand to take row ``row_expr`` of ``matrix_expr`` from; ``None`` keeps ``rendered_matrix``.

        A written path is marked from its outermost access down, so ``matrix_expr`` is marked when the row (or an
        element of it) is assigned.
        """
        if id(matrix_expr) in self.lvalues or self.held_in_a_local(matrix_expr):
            return None
        if self.codegen.literal_int_value(row_expr, {}) is not None:
            return None
        matrix_type = self.tall_matrix_type(matrix_expr)
        if matrix_type is None:
            return None
        name = self.helpers.get(matrix_type)
        if name is None:
            name = "crossgl_matrix_value_" + re.sub(r"\W+", "_", matrix_type)
            self.helpers[matrix_type] = name
        return f"{name}({rendered_matrix})"

    def declarations(self):
        code = ""
        for matrix_type, name in sorted(self.helpers.items(), key=lambda item: item[1]):
            code += f"{matrix_type} {name}({matrix_type} value)\n{{\n    return value;\n}}\n\n"
        return code
