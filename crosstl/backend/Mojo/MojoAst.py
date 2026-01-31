"""Mojo AST Node definitions"""

from ..common_ast import (
    ASTNode,
    ArrayAccessNode,
    AssignmentNode,
    AttributeNode,
    BinaryOpNode,
    BreakNode,
    CastNode,
    ConstantBufferNode,
    ContinueNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    MemberAccessNode,
    ReturnNode,
    ShaderNode,
    StructNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    VectorConstructorNode,
    WhileNode,
)

# Keep common AST imports used for re-exports (autoflake-safe).
_COMMON_NODES = (
    ArrayAccessNode,
    AssignmentNode,
    AttributeNode,
    BinaryOpNode,
    BreakNode,
    CastNode,
    ConstantBufferNode,
    ContinueNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    MemberAccessNode,
    ReturnNode,
    ShaderNode,
    StructNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    VectorConstructorNode,
    WhileNode,
)

# Mojo-specific nodes


class VariableDeclarationNode(VariableNode):
    """Node representing a Mojo variable declaration with type inference"""

    def __init__(self, vtype, name, value=None, is_var=True):
        super().__init__(vtype, name, value)
        self.is_var = is_var  # True for 'var', False for 'let'
        self.var_type = "var" if is_var else "let"
        self.initial_value = value

    def __repr__(self):
        keyword = "var" if self.is_var else "let"
        return f"VariableDeclarationNode({keyword} {self.name}: {self.vtype})"


class ImportNode(ASTNode):
    """Node representing an import statement"""

    def __init__(self, module, items=None, alias=None):
        self.module = module
        self.module_name = module
        self.items = items or []
        self.alias = alias

    def __repr__(self):
        return (
            f"ImportNode(module={self.module}, items={self.items}, alias={self.alias})"
        )


class ClassNode(ASTNode):
    """Node representing a class definition"""

    def __init__(self, name, members, methods, base_classes=None):
        self.name = name
        self.members = members
        self.methods = methods
        self.base_classes = base_classes or []

    def __repr__(self):
        return f"ClassNode(name={self.name}, members={len(self.members)}, methods={len(self.methods)})"


class DecoratorNode(ASTNode):
    """Node representing a decorator"""

    def __init__(self, name, args=None):
        super().__init__(name, args)

    def __repr__(self):
        return f"DecoratorNode(name={self.name}, args={self.args})"


class SwitchCaseNode(ASTNode):
    """Node representing a switch case"""

    def __init__(self, value, statements):
        self.value = value
        self.statements = statements
        self.condition = value
        self.body = statements

    def __repr__(self):
        return f"SwitchCaseNode(value={self.value}, statements={len(self.statements)})"


class PragmaNode(ASTNode):
    """Node representing a pragma directive"""

    def __init__(self, directive, value=None):
        self.directive = directive
        self.value = value

    def __repr__(self):
        return f"PragmaNode(directive={self.directive}, value={self.value})"


class IncludeNode(ASTNode):
    """Node representing an include directive"""

    def __init__(self, path):
        self.path = path

    def __repr__(self):
        return f"IncludeNode(path={self.path})"


class PassNode(ASTNode):
    """Node representing a pass statement"""

    def __repr__(self):
        return "PassNode()"
