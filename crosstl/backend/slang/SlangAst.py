"""
Slang Abstract Syntax Tree (AST) implementation
"""


class ASTNode:
    """Base class for all AST nodes."""


class ShaderNode(ASTNode):
    """Root node for a shader program."""

    def __init__(self, functions=None, variables=None, structs=None):
        self.functions = functions or []
        self.variables = variables or []
        self.structs = structs or []


class VariableNode(ASTNode):
    """Node representing a variable declaration."""

    def __init__(self, type_name, name, value=None, qualifiers=None, array_size=None):
        self.type_name = type_name
        self.name = name
        self.value = value
        self.qualifiers = qualifiers or []
        self.array_size = array_size


class AssignmentNode(ASTNode):
    """Node representing a variable assignment."""

    def __init__(self, left, right):
        self.left = left
        self.right = right


class FunctionNode(ASTNode):
    """Node representing a function declaration/definition."""

    def __init__(self, return_type, name, parameters, body, semantic=None):
        self.return_type = return_type
        self.name = name
        self.parameters = parameters or []
        self.body = body or []
        self.semantic = semantic


class ParameterNode(ASTNode):
    """Node representing a function parameter."""

    def __init__(self, type_name, name, semantic=None):
        self.type_name = type_name
        self.name = name
        self.semantic = semantic


class ArrayAccessNode(ASTNode):
    """Node representing array access (array[index])."""

    def __init__(self, array, index):
        self.array = array
        self.index = index


class BinaryOpNode(ASTNode):
    """Node representing a binary operation (a + b)."""

    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right


class UnaryOpNode(ASTNode):
    """Node representing a unary operation (!a, -b)."""

    def __init__(self, operator, operand):
        self.operator = operator
        self.operand = operand


class ReturnNode(ASTNode):
    """Node representing a return statement."""

    def __init__(self, value=None):
        self.value = value


class FunctionCallNode(ASTNode):
    """Node representing a function call."""

    def __init__(self, function, arguments):
        self.function = function
        self.arguments = arguments or []


class IfNode(ASTNode):
    """Node representing an if statement."""

    def __init__(self, condition, true_branch, false_branch=None):
        self.condition = condition
        self.true_branch = true_branch or []
        self.false_branch = false_branch or []


class ForNode(ASTNode):
    """Node representing a for loop."""

    def __init__(self, init, condition, update, body):
        self.init = init
        self.condition = condition
        self.update = update
        self.body = body or []


class VectorConstructorNode(ASTNode):
    """Node representing a vector constructor (float4(x, y, z, w))."""

    def __init__(self, type_name, arguments):
        self.type_name = type_name
        self.arguments = arguments or []


class ConstantNode(ASTNode):
    """Node representing a constant value (number, boolean, etc.)."""

    def __init__(self, value, type_name=None):
        self.value = value
        self.type_name = type_name


class MemberAccessNode(ASTNode):
    """Node representing member access (obj.member)."""

    def __init__(self, object_expr, member):
        self.object_expr = object_expr
        self.member = member


class TernaryOpNode(ASTNode):
    """Node representing a ternary operation (a ? b : c)."""

    def __init__(self, condition, true_value, false_value):
        self.condition = condition
        self.true_value = true_value
        self.false_value = false_value


class StructNode(ASTNode):
    """Node representing a struct definition."""

    def __init__(self, name, members):
        self.name = name
        self.members = members or []


class SemanticNode(ASTNode):
    """Node representing a semantic (: SV_TARGET, : TEXCOORD, etc.)."""

    def __init__(self, semantic_name):
        self.semantic_name = semantic_name
