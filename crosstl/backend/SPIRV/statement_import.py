"""Expression statements and unknown AST nodes in the Vulkan GLSL importer.

The importer's statement loop used to end in ``// Unhandled statement type: <Node>``. An expression statement whose
root is not a call, assignment or operator, such as ``flag ? bump() : reset();`` or ``data.values[i];``, reached
that branch, so the comment replaced the statement and the calls in it never ran on any target. An expression
statement evaluates its expression for its side effects and discards the value (GLSL 4.60 section 6.1.3, "Expression
Statements"), so each expression root is now emitted as ``<expression>;``. Any node the importer has no CrossGL
spelling for, as a statement or as an operand, raises :class:`SpirvStatementUnsupportedError` instead of being
written as a comment or as the node's Python ``repr``.
"""

from __future__ import annotations

from crosstl.translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

from .VulkanAst import (
    ArrayAccessNode,
    MemberAccessNode,
    TernaryOpNode,
)

__all__ = [
    "EXPRESSION_STATEMENT_TYPES",
    "SpirvStatementUnsupportedError",
    "unsupported_expression",
    "unsupported_statement",
]

# Expression roots the statement loop has no dedicated branch for. Calls, assignments and operators keep theirs.
EXPRESSION_STATEMENT_TYPES = (
    TernaryOpNode,
    MemberAccessNode,
    ArrayAccessNode,
    int,
    float,
)


class SpirvStatementUnsupportedError(TranslationDiagnostic):
    """A Vulkan GLSL statement or expression node with no CrossGL spelling, refused rather than dropped."""

    def __init__(self, kind: str, node):
        self.node_type = type(node).__name__
        self.missing_capabilities = (f"spirv.import.{kind}.{self.node_type}",)
        super().__init__(
            f"The Vulkan importer has no CrossGL translation for the {kind} "
            f"{self.node_type}; refusing to drop it",
            target="cgl",
            code=f"{CODE_PREFIX}spirv-{kind}-unsupported",
        )


def unsupported_statement(node):
    """Raise for a statement node the importer cannot emit."""
    raise SpirvStatementUnsupportedError("statement", node)


def unsupported_expression(node):
    """Raise for an expression node the importer cannot emit."""
    raise SpirvStatementUnsupportedError("expression", node)
