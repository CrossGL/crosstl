"""OpenCL AST node definitions."""

from crosstl.backend.HIP.HipAst import *  # noqa: F401,F403
from crosstl.backend.HIP.HipAst import ASTNode


class OpenCLProgramNode(ASTNode):
    """Root node representing a complete OpenCL program."""

    def __init__(self, statements=None):
        self.statements = statements or []

    def __repr__(self):
        return f"OpenCLProgramNode(statements={self.statements})"

