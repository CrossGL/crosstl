"""Comma expressions in the source importers.

The C-family sources (HLSL, Metal, GLSL, CUDA, HIP, OpenCL C, Slang) all have the comma operator: it
evaluates its operands from left to right and yields the last one (C11 6.5.17). The importers' parsers
represent ``a, b, c`` as a left-nested chain of binary ``,`` nodes. CrossGL spells the operator
``(a, b, c)``; the parentheses are part of the CrossGL syntax, since an unparenthesized comma would split a
declaration, an argument list or a ternary arm.

Because ``(a, b)`` is the comma operator, an importer of a language with tuples (Mojo, Rust) writes a tuple
literal with a trailing comma, ``(a, b,)``, which CrossGL reads as an aggregate literal of its elements.
"""

from __future__ import annotations

__all__ = [
    "comma_operands",
    "crossgl_comma_expression",
    "crossgl_tuple_literal",
    "is_binary_comma",
]


def crossgl_tuple_literal(rendered_elements) -> str:
    """The CrossGL spelling of a tuple literal: ``(a, b,)``, never the comma operator ``(a, b)``."""
    elements = list(rendered_elements)
    if not elements:
        return "()"
    return f"({', '.join(elements)},)"


def is_binary_comma(expr) -> bool:
    """True when ``expr`` is a binary node for the comma operator."""
    return (
        getattr(expr, "op", getattr(expr, "operator", None)) == ","
        and hasattr(expr, "left")
        and hasattr(expr, "right")
    )


def comma_operands(expr):
    """The operands of a chain of binary ``,`` nodes, in evaluation order."""
    operands = []
    pending = [expr]
    while pending:
        node = pending.pop()
        if is_binary_comma(node):
            pending.append(node.right)
            pending.append(node.left)
        else:
            operands.append(node)
    return operands


def crossgl_comma_expression(expr, generate):
    """The CrossGL ``(a, b, ...)`` for a binary ``,`` chain, or None when ``expr`` is not one."""
    if not is_binary_comma(expr):
        return None
    rendered = [generate(operand) for operand in comma_operands(expr)]
    return f"({', '.join(rendered)})"
