"""Rust AST Node definitions"""

from ..common_ast import (
    ASTNode,
    ArrayAccessNode,
    AssignmentNode,
    AttributeNode,
    BinaryOpNode,
    CastNode,
    FunctionCallNode,
    IfNode,
    MemberAccessNode,
    ReturnNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    VectorConstructorNode,
)


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
        enums=None,
        type_aliases=None,
    ):
        self.structs = structs or []
        self.functions = functions or []
        self.global_variables = global_variables or []
        self.impl_blocks = impl_blocks or []
        self.use_statements = use_statements or []
        self.traits = traits or []
        self.enums = enums or []
        self.type_aliases = type_aliases or []

    def __repr__(self):
        return (
            "ShaderNode("
            f"structs={len(self.structs)}, "
            f"functions={len(self.functions)}, "
            f"globals={len(self.global_variables)}, "
            f"impl_blocks={len(self.impl_blocks)}, "
            f"use_statements={len(self.use_statements)}, "
            f"traits={len(self.traits)}, "
            f"enums={len(self.enums)}, "
            f"type_aliases={len(self.type_aliases)})"
        )


class StructNode(ASTNode):
    """Node representing a Rust struct with visibility and attributes."""

    def __init__(
        self,
        name,
        members,
        attributes=None,
        visibility=None,
        generics=None,
        where_clauses=None,
    ):
        self.name = name
        self.members = members
        self.attributes = attributes or []
        self.visibility = visibility
        self.generics = generics or []
        self.where_clauses = where_clauses or []

    def __repr__(self):
        return (
            f"StructNode(name={self.name}, members={len(self.members)}, "
            f"visibility={self.visibility})"
        )


class EnumVariantNode(ASTNode):
    """Node representing one Rust enum variant."""

    def __init__(self, name, kind="unit", fields=None, value=None, attributes=None):
        self.name = name
        self.kind = kind
        self.fields = fields or []
        self.value = value
        self.attributes = attributes or []

    def __repr__(self):
        return (
            f"EnumVariantNode(name={self.name}, kind={self.kind}, "
            f"fields={len(self.fields)}, value={self.value})"
        )


class EnumNode(ASTNode):
    """Node representing a Rust enum declaration."""

    def __init__(
        self,
        name,
        variants,
        attributes=None,
        visibility=None,
        generics=None,
        where_clauses=None,
    ):
        self.name = name
        self.variants = variants
        self.attributes = attributes or []
        self.visibility = visibility
        self.generics = generics or []
        self.where_clauses = where_clauses or []

    def __repr__(self):
        return (
            f"EnumNode(name={self.name}, variants={len(self.variants)}, "
            f"visibility={self.visibility})"
        )


class FunctionNode(ASTNode):
    """Node representing a Rust function."""

    def __init__(
        self,
        return_type,
        name,
        params,
        body,
        attributes=None,
        visibility=None,
        generics=None,
        where_clauses=None,
        is_async=False,
        is_unsafe=False,
        abi=None,
        is_const=False,
    ):
        self.return_type = return_type
        self.name = name
        self.params = params
        self.body = body
        self.attributes = attributes or []
        self.visibility = visibility
        self.generics = generics or []
        self.where_clauses = where_clauses or []
        self.is_async = is_async
        self.is_unsafe = is_unsafe
        self.abi = abi
        self.is_const = is_const

    def __repr__(self):
        return (
            f"FunctionNode(name={self.name}, return_type={self.return_type}, "
            f"params={len(self.params)}, visibility={self.visibility}, "
            f"is_async={self.is_async}, is_unsafe={self.is_unsafe}, "
            f"abi={self.abi}, is_const={self.is_const})"
        )


# Rust-specific nodes


class ImplNode(ASTNode):
    """Node representing an impl block"""

    def __init__(
        self,
        struct_name,
        methods,
        trait_name=None,
        generics=None,
        where_clauses=None,
        type_aliases=None,
    ):
        self.struct_name = struct_name
        self.methods = methods
        self.functions = methods
        self.trait_name = trait_name  # For trait implementations
        self.generics = generics or []
        self.where_clauses = where_clauses or []
        self.type_aliases = type_aliases or []

    def __repr__(self):
        if self.trait_name:
            return f"ImplNode(trait={self.trait_name}, for={self.struct_name}, methods={len(self.methods)})"
        return f"ImplNode(for={self.struct_name}, methods={len(self.methods)})"


class AssociatedTypeNode(ASTNode):
    """Node representing a Rust trait associated type declaration."""

    def __init__(self, name, bounds=None, default_type=None, where_clauses=None):
        self.name = name
        self.bounds = bounds or []
        self.default_type = default_type
        self.where_clauses = where_clauses or []

    def __repr__(self):
        return (
            f"AssociatedTypeNode(name={self.name}, bounds={self.bounds}, "
            f"default_type={self.default_type})"
        )


class TypeAliasNode(ASTNode):
    """Node representing a Rust type alias declaration."""

    def __init__(
        self,
        name,
        alias_type,
        generics=None,
        visibility=None,
        where_clauses=None,
        attributes=None,
    ):
        self.name = name
        self.alias_type = alias_type
        self.generics = generics or []
        self.visibility = visibility
        self.where_clauses = where_clauses or []
        self.attributes = attributes or []

    def __repr__(self):
        return (
            f"TypeAliasNode(name={self.name}, alias_type={self.alias_type}, "
            f"generics={self.generics})"
        )


class TraitNode(ASTNode):
    """Node representing a trait definition"""

    def __init__(
        self,
        name,
        methods,
        generics=None,
        visibility=None,
        where_clauses=None,
        associated_types=None,
        *args,
        **kwargs,
    ):
        self.name = name
        self.methods = methods
        self.generics = generics or []
        self.visibility = visibility
        self.where_clauses = where_clauses or []
        self.associated_types = associated_types or []

        # Handle additional arguments for compatibility
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def __repr__(self):
        return f"TraitNode(name={self.name}, methods={len(self.methods)})"


class LetNode(ASTNode):
    """Node representing a let binding"""

    def __init__(
        self,
        name,
        value,
        var_type=None,
        is_mutable=False,
        else_body=None,
    ):
        self.name = name
        self.value = value
        self.var_type = var_type
        self.vtype = var_type
        self.is_mutable = is_mutable
        self.else_body = else_body

    def __repr__(self):
        mut = "mut " if self.is_mutable else ""
        else_part = f" else {self.else_body}" if self.else_body is not None else ""
        return f"LetNode({mut}{self.name}: {self.var_type} = {self.value}{else_part})"


class LetPatternConditionNode(ASTNode):
    """Node representing a Rust `let PATTERN = expression` condition."""

    def __init__(self, pattern, expression):
        self.pattern = pattern
        self.expression = expression

    def __repr__(self):
        return (
            f"LetPatternConditionNode(pattern={self.pattern}, "
            f"expression={self.expression})"
        )


class MatchesMacroNode(ASTNode):
    """Node representing a Rust `matches!(expression, pattern if guard)` macro."""

    def __init__(self, expression, pattern, guard=None):
        self.expression = expression
        self.pattern = pattern
        self.guard = guard

    def __repr__(self):
        return (
            f"MatchesMacroNode(expression={self.expression}, "
            f"pattern={self.pattern}, guard={self.guard})"
        )


class ConditionChainNode(ASTNode):
    """Node representing a Rust condition chain joined by top-level `&&`."""

    def __init__(self, operands):
        self.operands = operands

    def __repr__(self):
        return f"ConditionChainNode(operands={self.operands})"


class ClosureParameterNode(ASTNode):
    """Node representing a Rust closure parameter."""

    def __init__(self, pattern, param_type=None):
        self.pattern = pattern
        self.param_type = param_type
        self.vtype = param_type

    def __repr__(self):
        return f"ClosureParameterNode(pattern={self.pattern}, param_type={self.param_type})"


class ClosureNode(ASTNode):
    """Node representing a Rust closure expression."""

    def __init__(self, params, body, is_move=False, return_type=None, is_async=False):
        self.params = params
        self.body = body
        self.is_move = is_move
        self.return_type = return_type
        self.is_async = is_async

    def __repr__(self):
        async_prefix = "async " if self.is_async else ""
        move = "move " if self.is_move else ""
        return (
            f"ClosureNode({async_prefix}{move}params={self.params}, "
            f"return_type={self.return_type}, body={self.body})"
        )


class AwaitNode(ASTNode):
    """Node representing a Rust `expression.await` postfix expression."""

    def __init__(self, expression):
        self.expression = expression

    def __repr__(self):
        return f"AwaitNode(expression={self.expression})"


class AsyncBlockNode(ASTNode):
    """Node representing a Rust `async { ... }` block expression."""

    def __init__(self, block, is_move=False):
        self.block = block
        self.is_move = is_move

    def __repr__(self):
        move = "move " if self.is_move else ""
        return f"AsyncBlockNode({move}block={self.block})"


class UnsafeBlockNode(ASTNode):
    """Node representing a Rust `unsafe { ... }` block expression."""

    def __init__(self, block):
        self.block = block

    def __repr__(self):
        return f"UnsafeBlockNode(block={self.block})"


class ConstBlockNode(ASTNode):
    """Node representing a Rust `const { ... }` block expression."""

    def __init__(self, block):
        self.block = block

    def __repr__(self):
        return f"ConstBlockNode(block={self.block})"


class TryNode(ASTNode):
    """Node representing a Rust `expression?` try-propagation expression."""

    def __init__(self, expression):
        self.expression = expression

    def __repr__(self):
        return f"TryNode(expression={self.expression})"


class TryBlockNode(ASTNode):
    """Node representing a Rust `try { ... }` block expression."""

    def __init__(self, block):
        self.block = block

    def __repr__(self):
        return f"TryBlockNode(block={self.block})"


class WhileNode(ASTNode):
    """Node representing a Rust while loop."""

    def __init__(self, condition, body, label=None):
        self.condition = condition
        self.body = body
        self.label = label

    def __repr__(self):
        return f"WhileNode(label={self.label}, condition={self.condition}, body={self.body})"


class LoopNode(ASTNode):
    """Node representing an infinite loop"""

    def __init__(self, body, label=None):
        self.body = body
        self.label = label

    def __repr__(self):
        return f"LoopNode(label={self.label}, body={self.body})"


class ForNode(ASTNode):
    """Node representing a Rust for-in loop"""

    def __init__(self, pattern, iterable, body, label=None):
        self.pattern = pattern
        self.iterable = iterable
        self.body = body
        self.label = label

    def __repr__(self):
        return (
            f"ForNode(label={self.label}, pattern={self.pattern}, "
            f"iterable={self.iterable}, body={self.body})"
        )


class BreakNode(ASTNode):
    """Node representing a Rust break statement."""

    def __init__(self, label=None, value=None):
        self.label = label
        self.value = value

    def __repr__(self):
        return f"BreakNode(label={self.label}, value={self.value})"


class ContinueNode(ASTNode):
    """Node representing a Rust continue statement."""

    def __init__(self, label=None):
        self.label = label

    def __repr__(self):
        return f"ContinueNode(label={self.label})"


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


class MatchOrPatternNode(ASTNode):
    """Node representing a Rust match or-pattern."""

    def __init__(self, patterns):
        self.patterns = patterns

    def __repr__(self):
        return f"MatchOrPatternNode(patterns={self.patterns})"


class MatchBindingPatternNode(ASTNode):
    """Node representing a Rust binding pattern like `name @ pattern`."""

    def __init__(self, name, pattern):
        self.name = name
        self.pattern = pattern

    def __repr__(self):
        return f"MatchBindingPatternNode(name={self.name}, pattern={self.pattern})"


class MatchRestPatternNode(ASTNode):
    """Node representing a Rust slice/array rest pattern (`..`)."""

    def __repr__(self):
        return "MatchRestPatternNode(..)"


class MatchStructPatternNode(ASTNode):
    """Node representing a Rust struct or record variant match pattern."""

    def __init__(self, name, fields=None, has_rest=False):
        self.name = name
        self.fields = fields or []
        self.has_rest = has_rest

    def __repr__(self):
        return (
            f"MatchStructPatternNode(name={self.name}, fields={self.fields}, "
            f"has_rest={self.has_rest})"
        )


class UseNode(ASTNode):
    """Node representing a use statement"""

    def __init__(self, path, alias=None, items=None, visibility=None):
        self.path = path
        self.alias = alias
        self.items = items  # For use path::{item1, item2}
        self.visibility = visibility

    def __repr__(self):
        return (
            f"UseNode(path={self.path}, alias={self.alias}, "
            f"items={self.items}, visibility={self.visibility})"
        )


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

    def __init__(self, statements, returns_value=None):
        self.statements = statements
        self.returns_value = returns_value
        self.expression = returns_value

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
