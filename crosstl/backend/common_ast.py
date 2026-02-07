"""
Common AST Node Definitions.

This module defines the Abstract Syntax Tree (AST) node classes that are shared
across all backend parsers in CrossTL. These nodes represent the common
elements of shader programs.

The AST hierarchy includes:
    - ShaderNode: Root node representing a complete program
    - FunctionNode: Function declarations
    - StructNode: Structure definitions
    - Statement nodes: If, For, While, Switch, Return, etc.
    - Expression nodes: Binary operations, function calls, member access, etc.
    - Type nodes: Variables, constants, arrays

Example:
    >>> from crosstl.backend.common_ast import ShaderNode, FunctionNode
    >>> shader = ShaderNode(functions=[FunctionNode("void", "main", [], [])])
"""


class ASTNode:
    """Base class for all AST nodes"""


class ShaderNode(ASTNode):
    """Root node representing a complete program"""

    def __init__(
        self,
        includes=None,
        functions=None,
        structs=None,
        global_variables=None,
        kernels=None,
        *args,  # Accept extra positional args
        **kwargs,  # Accept extra keyword args for compatibility
    ):
        self.includes = includes or []
        self.functions = functions or []
        self.structs = structs or []
        self.global_variables = global_variables or []
        self.kernels = kernels or []

        # Initialize common backend-specific attributes with defaults
        self.uniforms = kwargs.get("uniforms", [])
        self.in_out = kwargs.get("in_out", [])
        self.constant = kwargs.get("constant", [])
        self.io_variables = kwargs.get("io_variables", [])
        self.processors = kwargs.get("processors", [])
        self.cbuffers = kwargs.get("cbuffers", [])
        self.imports = kwargs.get("imports", [])
        self.exports = kwargs.get("exports", [])
        self.typedefs = kwargs.get("typedefs", [])
        self.extensions = kwargs.get("extensions", [])
        self.global_vars = kwargs.get("global_vars", self.global_variables)  # Alias

        # Support different backend signatures
        if args:
            # Handle positional arguments from different backends
            if len(args) >= 1:
                self.uniforms = args[0]
            if len(args) >= 2:
                self.in_out = args[1]
            if len(args) >= 3:
                self.constant = args[2]

        # Store any extra kwargs as attributes
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def __repr__(self):
        return f"ShaderNode(includes={self.includes}, functions={self.functions}, structs={self.structs}, global_variables={self.global_variables}, kernels={self.kernels})"


class FunctionNode(ASTNode):
    """Node representing a function declaration"""

    def __init__(
        self,
        return_type,
        name,
        params,
        body,
        qualifiers=None,
        attributes=None,
        *args,
        **kwargs,
    ):
        self.return_type = return_type
        self.name = name
        self.params = params
        self.body = body
        self.qualifiers = qualifiers or []
        self.attributes = attributes or []
        self.generics = kwargs.get("generics", [])  # For generic/template functions

        # Support additional arguments from different backends
        if args:
            if len(args) >= 1:
                self.qualifier = args[0]  # Some backends use singular

        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def __repr__(self):
        return f"FunctionNode(return_type={self.return_type}, name={self.name}, params={self.params}, body={self.body}, qualifiers={self.qualifiers})"


class StructNode(ASTNode):
    """Node representing a struct declaration"""

    def __init__(self, name, members, attributes=None):
        self.name = name
        self.members = members
        self.attributes = attributes or []

    def __repr__(self):
        return f"StructNode(name={self.name}, members={self.members})"


class EnumNode(ASTNode):
    """Node representing an enum declaration"""

    def __init__(self, name, members):
        self.name = name
        self.members = members  # list of (name, value_or_None)

    def __repr__(self):
        return f"EnumNode(name={self.name}, members={self.members})"


class TypeAliasNode(ASTNode):
    """Node representing a typedef/alias"""

    def __init__(self, alias_type, name):
        self.alias_type = alias_type
        self.name = name

    def __repr__(self):
        return f"TypeAliasNode(alias_type={self.alias_type}, name={self.name})"


class StaticAssertNode(ASTNode):
    """Node representing a static_assert"""

    def __init__(self, condition, message=None):
        self.condition = condition
        self.message = message

    def __repr__(self):
        return f"StaticAssertNode(condition={self.condition}, message={self.message})"


class VariableNode(ASTNode):
    """Node representing a variable declaration"""

    def __init__(
        self,
        vtype,
        name,
        value=None,
        qualifiers=None,
        attributes=None,
        is_const=False,
        **kwargs,
    ):
        self.vtype = vtype
        self.name = name
        self.value = value
        self.qualifiers = qualifiers or []
        self.attributes = attributes or []
        self.is_const = is_const
        self.semantic = kwargs.get("semantic", None)  # Common in shader languages

        # Support additional parameters from different backends
        for key, val in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, val)

    def __repr__(self):
        return f"VariableNode(vtype={self.vtype}, name={self.name}, value={self.value}, qualifiers={self.qualifiers})"


class AssignmentNode(ASTNode):
    """Node representing an assignment operation"""

    def __init__(self, left, right, operator="="):
        self.left = left
        self.right = right
        self.operator = operator

    def __repr__(self):
        return f"AssignmentNode(left={self.left}, operator={self.operator}, right={self.right})"


class BinaryOpNode(ASTNode):
    """Node representing a binary operation"""

    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self):
        return f"BinaryOpNode(left={self.left}, op={self.op}, right={self.right})"


class UnaryOpNode(ASTNode):
    """Node representing a unary operation"""

    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

    def __repr__(self):
        return f"UnaryOpNode(op={self.op}, operand={self.operand})"


class PostfixOpNode(ASTNode):
    """Node representing a postfix operation (e.g., i++, i--)"""

    def __init__(self, operand, op):
        self.operand = operand
        self.op = op

    def __repr__(self):
        return f"PostfixOpNode(operand={self.operand}, op={self.op})"


class FunctionCallNode(ASTNode):
    """Node representing a function call"""

    def __init__(self, name, args):
        self.name = name
        self.args = args

    def __repr__(self):
        return f"FunctionCallNode(name={self.name}, args={self.args})"


class MethodCallNode(ASTNode):
    """Node representing a method call on an object"""

    def __init__(self, object, method, args):
        self.object = object
        self.method = method
        self.args = args

    def __repr__(self):
        return f"MethodCallNode(object={self.object}, method={self.method}, args={self.args})"


class CallNode(ASTNode):
    """Node representing a call on a callee expression"""

    def __init__(self, callee, args):
        self.callee = callee
        self.args = args

    def __repr__(self):
        return f"CallNode(callee={self.callee}, args={self.args})"


class MemberAccessNode(ASTNode):
    """Node representing member access (dot or arrow operator)"""

    def __init__(self, object, member, is_pointer=False):
        self.object = object
        self.member = member
        self.is_pointer = is_pointer

    def __repr__(self):
        op = "->" if self.is_pointer else "."
        return f"MemberAccessNode(object={self.object}, member={self.member}, operator={op})"


class ArrayAccessNode(ASTNode):
    """Node representing array access"""

    def __init__(self, array, index):
        self.array = array
        self.index = index

    def __repr__(self):
        return f"ArrayAccessNode(array={self.array}, index={self.index})"


class IfNode(ASTNode):
    """Node representing an if statement"""

    def __init__(
        self,
        condition=None,
        if_body=None,
        else_body=None,
        if_chain=None,
        else_if_chain=None,
    ):
        # Support both old and new style
        if if_chain is not None or else_if_chain is not None:
            self.if_chain = if_chain or []
            self.else_if_chain = else_if_chain or []
            self.else_body = else_body
            # Extract condition and if_body from if_chain if available
            if self.if_chain:
                self.condition = condition or (
                    self.if_chain[0][0] if self.if_chain else None
                )
                self.if_body = if_body or (
                    self.if_chain[0][1] if self.if_chain else None
                )
            else:
                self.condition = condition
                self.if_body = if_body
        else:
            self.condition = condition
            self.if_body = if_body
            self.else_body = else_body
            self.if_chain = []
            self.else_if_chain = []

    def __repr__(self):
        return f"IfNode(condition={self.condition}, if_body={self.if_body}, else_body={self.else_body})"


class ForNode(ASTNode):
    """Node representing a for loop"""

    def __init__(self, init, condition, update, body):
        self.init = init
        self.condition = condition
        self.update = update
        self.body = body

    def __repr__(self):
        return f"ForNode(init={self.init}, condition={self.condition}, update={self.update}, body={self.body})"


class WhileNode(ASTNode):
    """Node representing a while loop"""

    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

    def __repr__(self):
        return f"WhileNode(condition={self.condition}, body={self.body})"


class DoWhileNode(ASTNode):
    """Node representing a do-while loop"""

    def __init__(self, body, condition):
        self.body = body
        self.condition = condition

    def __repr__(self):
        return f"DoWhileNode(body={self.body}, condition={self.condition})"


class SwitchNode(ASTNode):
    """Node representing a switch statement"""

    def __init__(self, expression, cases, default_case=None, default=None):
        self.expression = expression
        self.cases = cases
        # Support both parameter names for compatibility
        self.default_case = default_case or default
        self.default = self.default_case

    def __repr__(self):
        return f"SwitchNode(expression={self.expression}, cases={self.cases}, default_case={self.default_case})"


class CaseNode(ASTNode):
    """Node representing a case in a switch statement"""

    def __init__(self, value, body=None, statements=None):
        self.value = value
        # Support both parameter names for compatibility
        self.body = body or statements or []
        self.statements = self.body

    def __repr__(self):
        return f"CaseNode(value={self.value}, body={self.body})"


class ReturnNode(ASTNode):
    """Node representing a return statement"""

    def __init__(self, value=None):
        self.value = value

    def __repr__(self):
        return f"ReturnNode(value={self.value})"


class ContinueNode(ASTNode):
    """Node representing a continue statement"""

    def __init__(self, *args, **kwargs):
        # Accept any arguments for compatibility
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return "ContinueNode()"


class BreakNode(ASTNode):
    """Node representing a break statement"""

    def __init__(self, *args, **kwargs):
        # Accept any arguments for compatibility
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return "BreakNode()"


class DiscardNode(ASTNode):
    """Node representing a discard statement"""

    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return "DiscardNode()"


class VectorConstructorNode(ASTNode):
    """Node representing vector constructor"""

    def __init__(self, vector_type, args, type_name=None):
        # Support both parameter names for compatibility
        self.vector_type = vector_type or type_name
        self.type_name = self.vector_type
        self.args = args

    def __repr__(self):
        return (
            f"VectorConstructorNode(vector_type={self.vector_type}, args={self.args})"
        )


class TernaryOpNode(ASTNode):
    """Node representing a ternary conditional operator"""

    def __init__(self, condition, true_expr, false_expr):
        self.condition = condition
        self.true_expr = true_expr
        self.false_expr = false_expr

    def __repr__(self):
        return f"TernaryOpNode(condition={self.condition}, true_expr={self.true_expr}, false_expr={self.false_expr})"


class CastNode(ASTNode):
    """Node representing a type cast"""

    def __init__(self, target_type, expression):
        self.target_type = target_type
        self.expression = expression

    def __repr__(self):
        return f"CastNode(target_type={self.target_type}, expression={self.expression})"


class PreprocessorNode(ASTNode):
    """Node representing preprocessor directives"""

    def __init__(self, directive, content):
        self.directive = directive
        self.content = content

    def __repr__(self):
        return f"PreprocessorNode(directive={self.directive}, content={self.content})"


class AttributeNode(ASTNode):
    """Attributes/annotations"""

    def __init__(self, name, args=None, arguments=None):
        self.name = name
        # Support both parameter names for compatibility
        self.args = args or arguments or []
        self.arguments = self.args

    def __repr__(self):
        return f"AttributeNode(name='{self.name}', args={self.args})"


class TextureSampleNode(ASTNode):
    """Node representing texture sampling"""

    def __init__(self, texture, sampler, coordinates, lod=None):
        self.texture = texture
        self.sampler = sampler
        self.coordinates = coordinates
        self.lod = lod

    def __repr__(self):
        if self.lod is not None:
            return f"TextureSampleNode(texture={self.texture}, sampler={self.sampler}, coordinates={self.coordinates}, lod={self.lod})"
        return f"TextureSampleNode(texture={self.texture}, sampler={self.sampler}, coordinates={self.coordinates})"


class SyncNode(ASTNode):
    """Node representing synchronization operations"""

    def __init__(self, sync_type, args=None, arguments=None):
        self.sync_type = sync_type
        # Support both parameter names
        self.args = args or arguments or []
        self.arguments = self.args

    def __repr__(self):
        return f"SyncNode(sync_type={self.sync_type}, args={self.args})"


class ThreadgroupSyncNode(ASTNode):
    """Node representing threadgroup synchronization"""

    def __init__(self):
        pass

    def __repr__(self):
        return "ThreadgroupSyncNode()"


class ConstantBufferNode(ASTNode):
    """Node representing constant buffer"""

    def __init__(self, name, members):
        self.name = name
        self.members = members

    def __repr__(self):
        return f"ConstantBufferNode(name={self.name}, members={self.members})"
