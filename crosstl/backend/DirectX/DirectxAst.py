"""DirectX/HLSL AST Node definitions"""

from ..common_ast import *


# DirectX-specific nodes

class CbufferNode(StructNode):
    """Node representing a constant buffer (cbuffer)"""

    def __init__(self, name, members):
        super().__init__(name, members)

    def __repr__(self):
        return f"CbufferNode(name={self.name}, members={self.members})"


class PragmaNode(ASTNode):
    """Node representing a pragma directive"""

    def __init__(self, directive, value=None):
        self.directive = directive
        self.value = value

    def __repr__(self):
        return f"PragmaNode(directive={self.directive}, value={self.value})"


class IncludeNode(ASTNode):
    """Node representing an include directive"""

    def __init__(self, path, is_system=False):
        self.path = path
        self.is_system = is_system

    def __repr__(self):
        return f"IncludeNode(path={self.path}, is_system={self.is_system})"


class SwitchStatementNode(ASTNode):
    """Node representing a switch statement"""

    def __init__(self, expression, cases):
        self.expression = expression
        self.cases = cases

    def __repr__(self):
        return f"SwitchStatementNode(expression={self.expression}, cases={self.cases})"


class SwitchCaseNode(ASTNode):
    """Node representing a case in a switch statement"""

    def __init__(self, case_value, statements, is_default=False):
        self.case_value = case_value
        self.statements = statements
        self.is_default = is_default

    def __repr__(self):
        return f"SwitchCaseNode(case_value={self.case_value}, statements={len(self.statements)}, is_default={self.is_default})"
