"""Operands of C-family importers that must stay grouped in CrossGL output.

An importer writes an assignment without parentheses, because as a statement it
needs none. As the operand of another operator it does: ``(t -= 3) > 0`` written
as ``t -= 3 > 0`` reads back as ``t -= (3 > 0)``, since assignment binds more
loosely than every other operator except the comma.
"""

from .common_ast import AssignmentNode


def grouped_operand(visitor, node):
    """Render ``node`` as an operand, keeping an assignment as one group."""
    text = visitor.visit(node)
    if isinstance(node, AssignmentNode) and text is not None:
        return f"({text})"
    return text
