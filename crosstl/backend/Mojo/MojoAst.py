"""Mojo AST Node definitions"""

from ..common_ast import (
    ArrayAccessNode,
    AssignmentNode,
    ASTNode,
    AttributeNode,
    BinaryOpNode,
    BreakNode,
    CallNode,
    CastNode,
    ConstantBufferNode,
    ContinueNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    MemberAccessNode,
    MethodCallNode,
    RangeForNode,
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
    CallNode,
    CastNode,
    ConstantBufferNode,
    ContinueNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    MemberAccessNode,
    MethodCallNode,
    RangeForNode,
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

    def __init__(self, vtype, name, value=None, is_var=True, attributes=None):
        super().__init__(vtype, name, value, attributes=attributes)
        self.is_var = is_var  # True for 'var', False for 'let'
        self.var_type = "var" if is_var else "let"
        self.initial_value = value

    def __repr__(self):
        keyword = "var" if self.is_var else "let"
        return f"VariableDeclarationNode({keyword} {self.name}: {self.vtype})"


class TupleNode(ASTNode):
    """Node representing a Mojo tuple expression."""

    def __init__(self, elements):
        self.elements = elements

    def __repr__(self):
        return f"TupleNode(elements={self.elements})"


class ListLiteralNode(ASTNode):
    """Node representing a Mojo list literal expression."""

    def __init__(self, elements):
        self.elements = elements

    def __repr__(self):
        return f"ListLiteralNode(elements={self.elements})"


class ListComprehensionNode(ASTNode):
    """Node representing a Mojo list comprehension expression."""

    def __init__(self, expression, clauses):
        self.expression = expression
        self.clauses = clauses

    def __repr__(self):
        return (
            "ListComprehensionNode("
            f"expression={self.expression}, clauses={self.clauses})"
        )


class DictLiteralNode(ASTNode):
    """Node representing a Mojo dictionary display."""

    def __init__(self, entries):
        self.entries = entries

    def __repr__(self):
        return f"DictLiteralNode(entries={self.entries})"


class BracedLiteralNode(ASTNode):
    """Node representing a non-dictionary Mojo braced display or initializer list."""

    def __init__(self, elements):
        self.elements = elements

    def __repr__(self):
        return f"BracedLiteralNode(elements={self.elements})"


class DictComprehensionNode(ASTNode):
    """Node representing a Mojo dictionary comprehension expression."""

    def __init__(self, key, value, clauses):
        self.key = key
        self.value = value
        self.clauses = clauses

    def __repr__(self):
        return (
            "DictComprehensionNode("
            f"key={self.key}, value={self.value}, clauses={self.clauses})"
        )


class SetComprehensionNode(ASTNode):
    """Node representing a Mojo set comprehension expression."""

    def __init__(self, expression, clauses):
        self.expression = expression
        self.clauses = clauses

    def __repr__(self):
        return (
            "SetComprehensionNode("
            f"expression={self.expression}, clauses={self.clauses})"
        )


class SliceNode(ASTNode):
    """Node representing a Mojo slice index."""

    def __init__(self, start=None, stop=None, step=None, has_step=False):
        self.start = start
        self.stop = stop
        self.step = step
        self.has_step = has_step

    def __repr__(self):
        return (
            "SliceNode("
            f"start={self.start}, stop={self.stop}, step={self.step}, "
            f"has_step={self.has_step})"
        )


class SpreadExpressionNode(ASTNode):
    """Node representing Mojo positional or keyword unpacking in expressions."""

    def __init__(self, expression, kind="positional"):
        self.expression = expression
        self.kind = kind

    def __repr__(self):
        return (
            "SpreadExpressionNode(" f"kind={self.kind}, expression={self.expression})"
        )


class WithNode(ASTNode):
    """Node representing a Mojo with/as block."""

    def __init__(self, context_expr, alias, body, contexts=None):
        self.context_expr = context_expr
        self.alias = alias
        self.body = body
        self.contexts = contexts or [(context_expr, alias)]

    def __repr__(self):
        return (
            f"WithNode(contexts={len(self.contexts)}, "
            f"alias={self.alias}, body={len(self.body)})"
        )


class TryExceptNode(ASTNode):
    """Node representing a Mojo try/except/else/finally block."""

    def __init__(
        self,
        try_body,
        except_body=None,
        exception_name=None,
        else_body=None,
        finally_body=None,
    ):
        self.try_body = try_body
        self.except_body = except_body or []
        self.exception_name = exception_name
        self.else_body = else_body or []
        self.finally_body = finally_body or []

    def __repr__(self):
        return (
            "TryExceptNode("
            f"try_body={len(self.try_body)}, "
            f"except_body={len(self.except_body)}, "
            f"else_body={len(self.else_body)}, "
            f"finally_body={len(self.finally_body)})"
        )


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

    def __init__(
        self, name, members=None, methods=None, base_classes=None, attributes=None
    ):
        self.name = name
        self.members = members or []
        self.methods = methods or []
        self.base_classes = base_classes or []
        self.attributes = attributes or []

    def __repr__(self):
        return f"ClassNode(name={self.name}, members={len(self.members)}, methods={len(self.methods)})"


class ExtensionNode(ASTNode):
    """Node representing a Mojo __extension block."""

    def __init__(self, name, members=None, methods=None, attributes=None):
        self.name = name
        self.members = members or []
        self.methods = methods or []
        self.attributes = attributes or []

    def __repr__(self):
        return (
            f"ExtensionNode(name={self.name}, members={len(self.members)}, "
            f"methods={len(self.methods)})"
        )


class TraitNode(ASTNode):
    """Node representing a Mojo trait definition."""

    def __init__(
        self, name, members=None, methods=None, base_classes=None, attributes=None
    ):
        self.name = name
        self.members = members or []
        self.methods = methods or []
        self.base_classes = base_classes or []
        self.attributes = attributes or []

    def __repr__(self):
        return f"TraitNode(name={self.name}, members={len(self.members)}, methods={len(self.methods)})"


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
