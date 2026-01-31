"""Slang AST Node definitions"""

from ..common_ast import ASTNode

# Slang-specific nodes


class ImportNode(ASTNode):
    """Node representing an import statement"""

    def __init__(self, module_path, imported_items=None, alias=None):
        self.module_path = module_path
        self.imported_items = imported_items or []
        self.alias = alias

    def __repr__(self):
        return f"ImportNode(module_path={self.module_path}, imported_items={self.imported_items}, alias={self.alias})"


class ExportNode(ASTNode):
    """Node representing an export statement"""

    def __init__(self, exported_items):
        self.exported_items = exported_items

    def __repr__(self):
        return f"ExportNode(exported_items={self.exported_items})"


class TypedefNode(ASTNode):
    """Node representing a type alias"""

    def __init__(self, name, target_type):
        self.name = name
        self.target_type = target_type

    def __repr__(self):
        return f"TypedefNode(name={self.name}, target_type={self.target_type})"


class GenericNode(ASTNode):
    """Node representing a generic type parameter"""

    def __init__(self, name, constraints=None):
        self.name = name
        self.constraints = constraints or []

    def __repr__(self):
        return f"GenericNode(name={self.name}, constraints={self.constraints})"


class ExtensionNode(ASTNode):
    """Node representing a Slang extension"""

    def __init__(self, extended_type, methods):
        self.extended_type = extended_type
        self.methods = methods

    def __repr__(self):
        return f"ExtensionNode(extended_type={self.extended_type}, methods={len(self.methods)})"
