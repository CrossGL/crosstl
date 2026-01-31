"""OpenGL/GLSL AST Node definitions"""

# pylint: disable=unused-import
from crosstl.backend.common_ast import (
    ASTNode,
    ShaderNode,
    VariableNode,
    AssignmentNode,
    FunctionNode,
    ArrayAccessNode,
    BinaryOpNode,
    UnaryOpNode,
    ReturnNode,
    FunctionCallNode,
    IfNode,
    ForNode,
    WhileNode,
    DoWhileNode,
    VectorConstructorNode,
    MemberAccessNode,
    TernaryOpNode,
    StructNode,
    SwitchNode,
    CaseNode,
    PostfixOpNode,
    BreakNode,
    ContinueNode,
    DiscardNode,
)

# GLSL-specific nodes


class LayoutNode(ASTNode):
    """Node representing layout qualifiers"""

    def __init__(self, qualifiers=None, declaration=None):
        self.qualifiers = qualifiers or {}
        self.declaration = declaration

    def __repr__(self):
        return (
            f"LayoutNode(qualifiers={self.qualifiers}, declaration={self.declaration})"
        )


class UniformNode(ASTNode):
    """Node representing a uniform variable"""

    def __init__(self, vtype, name, value=None):
        self.vtype = vtype
        self.name = name
        self.value = value

    def __repr__(self):
        return f"UniformNode(vtype={self.vtype}, name={self.name}, value={self.value})"


class ConstantNode(ASTNode):
    """Node representing a constant declaration"""

    def __init__(self, vtype, name, value=None):
        self.vtype = vtype
        self.name = name
        self.value = value

    def __repr__(self):
        return f"ConstantNode(vtype={self.vtype}, name={self.name}, value={self.value})"


class BlockNode(ASTNode):
    """Node representing a block of statements"""

    def __init__(self, statements):
        self.statements = statements

    def __repr__(self):
        return f"BlockNode(statements={len(self.statements)})"


class NumberNode(ASTNode):
    """Node representing a numeric literal"""

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"NumberNode(value={self.value})"
