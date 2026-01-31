"""Rust AST Node definitions"""

from ..common_ast import *


class ShaderNode(ASTNode):
    """Root node representing a Rust module/program."""

    def __init__(
        self,
        structs=None,
        functions=None,
        global_variables=None,
        impl_blocks=None,
        use_statements=None,
        traits=None,
    ):
        self.structs = structs or []
        self.functions = functions or []
        self.global_variables = global_variables or []
        self.impl_blocks = impl_blocks or []
        self.use_statements = use_statements or []
        self.traits = traits or []

    def __repr__(self):
        return (
            "ShaderNode("
            f"structs={len(self.structs)}, "
            f"functions={len(self.functions)}, "
            f"globals={len(self.global_variables)}, "
            f"impl_blocks={len(self.impl_blocks)}, "
            f"use_statements={len(self.use_statements)})"
        )


class StructNode(ASTNode):
    """Node representing a Rust struct with visibility and attributes."""

    def __init__(self, name, members, attributes=None, visibility=None, generics=None):
        self.name = name
        self.members = members
        self.attributes = attributes or []
        self.visibility = visibility
        self.generics = generics or []

    def __repr__(self):
        return (
            f"StructNode(name={self.name}, members={len(self.members)}, "
            f"visibility={self.visibility})"
        )


class FunctionNode(ASTNode):
    """Node representing a Rust function."""

    def __init__(
        self, return_type, name, params, body, attributes=None, visibility=None, generics=None
    ):
        self.return_type = return_type
        self.name = name
        self.params = params
        self.body = body
        self.attributes = attributes or []
        self.visibility = visibility
        self.generics = generics or []

    def __repr__(self):
        return (
            f"FunctionNode(name={self.name}, return_type={self.return_type}, "
            f"params={len(self.params)}, visibility={self.visibility})"
        )

# Rust-specific nodes


class ImplNode(ASTNode):
    """Node representing an impl block"""

    def __init__(self, struct_name, methods, trait_name=None, generics=None):
        self.struct_name = struct_name
        self.methods = methods
        self.functions = methods
        self.trait_name = trait_name  # For trait implementations
        self.generics = generics or []

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
        self.vtype = var_type
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


class ForNode(ASTNode):
    """Node representing a Rust for-in loop"""

    def __init__(self, pattern, iterable, body):
        self.pattern = pattern
        self.iterable = iterable
        self.body = body

    def __repr__(self):
        return f"ForNode(pattern={self.pattern}, iterable={self.iterable}, body={self.body})"


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

    def __init__(self, name, const_type, value, visibility=None):
        self.name = name
        self.const_type = const_type
        self.vtype = const_type
        self.value = value
        self.visibility = visibility

    def __repr__(self):
        return f"ConstNode(name={self.name}, const_type={self.const_type}, value={self.value})"


class StaticNode(ASTNode):
    """Node representing a static variable"""

    def __init__(self, name, static_type, value, is_mutable=False, visibility=None):
        self.name = name
        self.static_type = static_type
        self.vtype = static_type
        self.value = value
        self.is_mutable = is_mutable
        self.visibility = visibility

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
