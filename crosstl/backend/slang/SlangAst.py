"""Slang AST Node definitions"""

from ..common_ast import (
    ArrayAccessNode,
    AssignmentNode,
    ASTNode,
    BinaryOpNode,
    BreakNode,
    CallNode,
    CaseNode,
    CastNode,
    ContinueNode,
    DiscardNode,
    DoWhileNode,
    EnumNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    InitializerListNode,
    MemberAccessNode,
    MethodCallNode,
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
    BinaryOpNode,
    BreakNode,
    CallNode,
    CastNode,
    CaseNode,
    ContinueNode,
    DiscardNode,
    DoWhileNode,
    EnumNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    InitializerListNode,
    MemberAccessNode,
    MethodCallNode,
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

# Slang-specific nodes


class ImportNode(ASTNode):
    """Node representing an import statement"""

    def __init__(self, module_path, imported_items=None, alias=None):
        self.module_path = module_path
        self.module_name = module_path
        self.imported_items = imported_items or []
        self.alias = alias

    def __repr__(self):
        return f"ImportNode(module_path={self.module_path}, imported_items={self.imported_items}, alias={self.alias})"


class ExportNode(ASTNode):
    """Node representing an export statement"""

    def __init__(self, exported_items):
        self.exported_items = exported_items
        self.item = exported_items

    def __repr__(self):
        return f"ExportNode(exported_items={self.exported_items})"


class TypedefNode(ASTNode):
    """Node representing a type alias"""

    def __init__(self, name, target_type):
        self.name = name
        self.target_type = target_type
        self.original_type = name
        self.new_type = target_type

    def __repr__(self):
        return f"TypedefNode(name={self.name}, target_type={self.target_type})"


class GenericNode(ASTNode):
    """Node representing a generic type parameter"""

    def __init__(self, name, constraints=None):
        self.name = name
        self.constraints = constraints or []

    def __repr__(self):
        return f"GenericNode(name={self.name}, constraints={self.constraints})"


class GenericConstraintNode(ASTNode):
    """Node representing a simple generic conformance constraint."""

    def __init__(self, parameter, constraint_type):
        self.parameter = parameter
        self.constraint_type = constraint_type

    def __repr__(self):
        return (
            "GenericConstraintNode("
            f"parameter={self.parameter}, constraint_type={self.constraint_type})"
        )


class AssociatedTypeNode(ASTNode):
    """Node representing a Slang interface associated type requirement."""

    def __init__(self, name, constraint_type=None, target_type=None, qualifiers=None):
        self.name = name
        self.constraint_type = constraint_type
        self.target_type = target_type
        self.qualifiers = qualifiers or []

    def __repr__(self):
        return (
            "AssociatedTypeNode("
            f"name={self.name}, constraint_type={self.constraint_type}, "
            f"target_type={self.target_type})"
        )


class InterfaceNode(ASTNode):
    """Node representing a Slang interface declaration."""

    def __init__(
        self,
        name,
        methods=None,
        generic_parameters=None,
        associated_types=None,
    ):
        self.name = name
        self.methods = methods or []
        self.generic_parameters = generic_parameters
        self.associated_types = associated_types or []

    def __repr__(self):
        return f"InterfaceNode(name={self.name}, methods={len(self.methods)})"


class ExtensionNode(ASTNode):
    """Node representing a Slang extension"""

    def __init__(self, extended_type, methods, conformances=None):
        self.extended_type = extended_type
        self.methods = methods
        self.conformances = conformances or []

    def __repr__(self):
        return (
            "ExtensionNode("
            f"extended_type={self.extended_type}, conformances={self.conformances}, "
            f"methods={len(self.methods)})"
        )
