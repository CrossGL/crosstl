"""Rust AST Node definitions"""

from ..common_ast import *


# Rust-specific nodes

class ImplNode(ASTNode):
    """Node representing an impl block"""

    def __init__(self, struct_name, methods, trait_name=None):
        self.struct_name = struct_name
        self.methods = methods
        self.trait_name = trait_name  # For trait implementations

    def __repr__(self):
        if self.trait_name:
            return f"ImplNode(trait={self.trait_name}, for={self.struct_name}, methods={len(self.methods)})"
        return f"ImplNode(for={self.struct_name}, methods={len(self.methods)})"


class TraitNode(ASTNode):
    """Node representing a trait definition"""

    def __init__(self, name, methods, *args, **kwargs):
        self.name = name
        self.methods = methods
        
        # Handle additional arguments for compatibility
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def __repr__(self):
        return f"TraitNode(name={self.name}, methods={len(self.methods)})"


class LetNode(ASTNode):
    """Node representing a let binding"""

    def __init__(self, name, value, var_type=None, is_mutable=False):
        self.name = name
        self.value = value
        self.var_type = var_type
        self.is_mutable = is_mutable

    def __repr__(self):
        mut = "mut " if self.is_mutable else ""
        return f"LetNode({mut}{self.name}: {self.var_type} = {self.value})"


class LoopNode(ASTNode):
    """Node representing an infinite loop"""

    def __init__(self, body, label=None):
        self.body = body
        self.label = label

    def __repr__(self):
        return f"LoopNode(label={self.label}, body={self.body})"


class MatchNode(ASTNode):
    """Node representing a match expression"""

    def __init__(self, expression, arms):
        self.expression = expression
        self.arms = arms

    def __repr__(self):
        return f"MatchNode(expression={self.expression}, arms={len(self.arms)})"


class MatchArmNode(ASTNode):
    """Node representing a match arm"""

    def __init__(self, pattern, guard, body):
        self.pattern = pattern
        self.guard = guard
        self.body = body

    def __repr__(self):
        return f"MatchArmNode(pattern={self.pattern}, guard={self.guard}, body={self.body})"


class UseNode(ASTNode):
    """Node representing a use statement"""

    def __init__(self, path, alias=None, items=None):
        self.path = path
        self.alias = alias
        self.items = items  # For use path::{item1, item2}

    def __repr__(self):
        return f"UseNode(path={self.path}, alias={self.alias}, items={self.items})"


class GenericParameterNode(ASTNode):
    """Node representing a generic parameter"""

    def __init__(self, name, bounds=None, default=None):
        self.name = name
        self.bounds = bounds or []
        self.default = default

    def __repr__(self):
        return f"GenericParameterNode(name={self.name}, bounds={self.bounds})"


class RangeNode(ASTNode):
    """Node representing a range expression"""

    def __init__(self, start, end, inclusive=False, step=None):
        self.start = start
        self.end = end
        self.inclusive = inclusive
        self.step = step

    def __repr__(self):
        op = "..=" if self.inclusive else ".."
        return f"RangeNode({self.start}{op}{self.end})"


class TupleNode(ASTNode):
    """Node representing a tuple"""

    def __init__(self, elements):
        self.elements = elements

    def __repr__(self):
        return f"TupleNode(elements={self.elements})"


class ArrayNode(ASTNode):
    """Node representing an array literal"""

    def __init__(self, elements, size=None):
        self.elements = elements
        self.size = size

    def __repr__(self):
        return f"ArrayNode(elements={len(self.elements)}, size={self.size})"


class ReferenceNode(ASTNode):
    """Node representing a reference (&)"""

    def __init__(self, expression, is_mutable=False):
        self.expression = expression
        self.is_mutable = is_mutable

    def __repr__(self):
        mut = "mut " if self.is_mutable else ""
        return f"ReferenceNode(&{mut}{self.expression})"


class DereferenceNode(ASTNode):
    """Node representing a dereference (*)"""

    def __init__(self, expression):
        self.expression = expression

    def __repr__(self):
        return f"DereferenceNode(*{self.expression})"


class BlockNode(ASTNode):
    """Node representing a block expression"""

    def __init__(self, statements, returns_value=False):
        self.statements = statements
        self.returns_value = returns_value

    def __repr__(self):
        return f"BlockNode(statements={len(self.statements)}, returns_value={self.returns_value})"


class ConstNode(ASTNode):
    """Node representing a const declaration"""

    def __init__(self, name, value, const_type=None):
        self.name = name
        self.value = value
        self.const_type = const_type

    def __repr__(self):
        return f"ConstNode(name={self.name}, const_type={self.const_type}, value={self.value})"


class StaticNode(ASTNode):
    """Node representing a static variable"""

    def __init__(self, name, value, static_type=None, is_mutable=False):
        self.name = name
        self.value = value
        self.static_type = static_type
        self.is_mutable = is_mutable

    def __repr__(self):
        mut = "mut " if self.is_mutable else ""
        return f"StaticNode({mut}{self.name}: {self.static_type} = {self.value})"


class StructInitializationNode(ASTNode):
    """Node representing struct initialization"""

    def __init__(self, struct_name, fields):
        self.struct_name = struct_name
        self.fields = fields  # Dict of field_name: value

    def __repr__(self):
        return f"StructInitializationNode(struct_name={self.struct_name}, fields={self.fields})"
