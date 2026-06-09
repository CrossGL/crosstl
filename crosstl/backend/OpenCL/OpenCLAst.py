"""OpenCL AST node definitions."""

from crosstl.backend.HIP.HipAst import *  # noqa: F401,F403
from crosstl.backend.HIP.HipAst import ASTNode


class OpenCLProgramNode(ASTNode):
    """Root node representing a complete OpenCL program."""

    def __init__(self, statements=None):
        self.statements = statements or []

    def __repr__(self):
        return f"OpenCLProgramNode(statements={self.statements})"


class OpenCLStatementExpressionNode(ASTNode):
    """GNU statement-expression block used by some OpenCL C corpora."""

    def __init__(self, statements=None):
        self.statements = statements or []

    def __repr__(self):
        return f"OpenCLStatementExpressionNode(statements={self.statements})"


class OpenCLMacroBlockNode(ASTNode):
    """Unexpanded statement-like macro call with a braced block argument."""

    def __init__(self, name, args=None):
        self.name = name
        self.args = args or []

    def __repr__(self):
        return f"OpenCLMacroBlockNode(name={self.name}, args={self.args})"


class OpenCLBlockLiteralNode(ASTNode):
    """Unsupported OpenCL C block literal captured as an inert expression."""

    def __init__(self, params=None, body=""):
        self.params = params or []
        self.body = body

    def __repr__(self):
        return f"OpenCLBlockLiteralNode(params={self.params}, body={self.body!r})"
