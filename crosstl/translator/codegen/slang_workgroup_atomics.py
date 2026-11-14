"""Slang lowering of CrossGL atomics on workgroup-shared variables.

CrossGL, like GLSL, updates a ``shared`` (Slang ``groupshared``) variable with
``atomicAdd(target, value)`` and the other ``atomic*`` built-ins, each of which
returns the value the target held before. Slang has no such functions: it
writes ``InterlockedAdd(target, value, original)``, which stores that value in
an ``original`` output. A statement keeps the call; any other use declares an
``original`` temporary in the enclosing statement's prelude and reads it.

Only ``int`` and ``uint`` scalars, and elements of arrays of them, have Slang
``Interlocked*`` operations; atomics on other workgroup values are refused.
"""

from __future__ import annotations

from ..ast import ArrayAccessNode, ArrayNode, IdentifierNode, VariableNode
from .ast_walk import walk_ast_values

__all__ = ["SlangWorkgroupAtomics", "workgroup_variable_names"]

_WORKGROUP_QUALIFIERS = frozenset({"groupshared", "shared", "threadgroup", "workgroup"})


def workgroup_variable_names(ast):
    """Names of the variables ``ast`` declares in workgroup-shared storage."""
    names = set()
    for node in walk_ast_values(ast):
        if not isinstance(node, (VariableNode, ArrayNode)):
            continue
        qualifiers = getattr(node, "qualifiers", None) or ()
        if any(
            str(qualifier).lower() in _WORKGROUP_QUALIFIERS for qualifier in qualifiers
        ):
            if isinstance(node.name, str):
                names.add(node.name)
    return names


def _workgroup_root(target):
    """The variable an atomic target names, through any array subscripts."""
    while isinstance(target, ArrayAccessNode):
        target = getattr(target, "array", getattr(target, "array_expr", None))
    if isinstance(target, (IdentifierNode, VariableNode)) and isinstance(
        target.name, str
    ):
        return target.name
    return None


class SlangWorkgroupAtomics:
    """Writes ``Interlocked*`` calls for CrossGL atomics on ``groupshared`` values."""

    def __init__(self, codegen, workgroup_names):
        self.codegen = codegen
        self.workgroup_names = frozenset(workgroup_names)

    def refuse(self, func_name, reason):
        return self.codegen.refuse_lowering(
            f"Slang groupshared atomic: {func_name} {reason}", ""
        )

    def element_type(self, target):
        """The declared element type of the workgroup variable ``target`` reaches."""
        declared = self.codegen.get_expression_type(target)
        if not isinstance(declared, str):
            return None
        element = declared.split("[", 1)[0].strip()
        depth = 0
        node = target
        while isinstance(node, ArrayAccessNode):
            depth += 1
            node = getattr(node, "array", getattr(node, "array_expr", None))
        dimensions = declared.count("[")
        if dimensions and depth != dimensions:
            return None
        return self.codegen.convert_type(element)

    def expression(self, func_name, args, statement_context):
        """``Interlocked*`` text for ``func_name(args)``, ``None`` when no workgroup target."""
        operation = self.codegen.structured_buffer_atomic_operations().get(func_name)
        if operation is None or not args:
            return None
        root = _workgroup_root(args[0])
        if root is None or root not in self.workgroup_names:
            return None
        intrinsic, value_count = operation
        if len(args) != 1 + value_count:
            return self.refuse(
                func_name,
                f"takes a target and {value_count} value(s), not {len(args) - 1}",
            )
        element_type = self.element_type(args[0])
        if element_type not in {"int", "uint"}:
            return self.refuse(
                func_name,
                f"needs an int or uint target, not '{element_type or 'an aggregate'}'",
            )
        codegen = self.codegen
        target = codegen.generate_expression(args[0])
        values = [
            codegen.generate_expression_with_expected(value, element_type)
            for value in args[1:]
        ]
        if statement_context:
            return f"{intrinsic}({', '.join([target, *values])})"
        if not codegen.expression_prelude_active():
            return self.refuse(
                func_name,
                "returns its original value only where a statement can declare it",
            )
        original = codegen.unique_expression_temp_name(f"cgl_{func_name}_original")
        codegen.add_expression_prelude(
            [
                f"{element_type} {original};",
                f"{intrinsic}({', '.join([target, *values, original])});",
            ],
            result_name=original,
        )
        return original
