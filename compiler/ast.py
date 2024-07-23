class ASTNode:
    pass


class UniformNode(ASTNode):
    def __init__(self, vtype, name):
        self.vtype = vtype  
        self.name = name    
    def __repr__(self):
        return f"UniformNode(vtype={self.vtype}, name={self.name})"
    def __str__(self):
        return f"uniform {self.vtype} {self.name};"

class ShaderNode(ASTNode):
    def __init__(self, name, inputs, outputs, functions):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.functions = functions


class FunctionNode(ASTNode):
    def __init__(self, return_type, name, params, body):
        self.return_type = return_type
        self.name = name
        self.params = params
        self.body = body


class VariableNode(ASTNode):
    def __init__(self, vtype, name):
        self.vtype = vtype
        self.name = name


class AssignmentNode(ASTNode):
    def __init__(self, name, value):
        self.name = name
        self.value = value


class IfNode(ASTNode):
    def __init__(self, condition, if_body, else_body=None):
        self.condition = condition
        self.if_body = if_body
        self.else_body = else_body


class ForNode(ASTNode):
    def __init__(self, init, condition, update, body):
        self.init = init
        self.condition = condition
        self.update = update
        self.body = body


class ReturnNode(ASTNode):
    def __init__(self, value):
        self.value = value


class FunctionCallNode(ASTNode):
    def __init__(self, name, args):
        self.name = name
        self.args = args


class BinaryOpNode(ASTNode):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right


class MemberAccessNode(ASTNode):
    def __init__(self, object, member):
        self.object = object
        self.member = member


class UnaryOpNode(ASTNode):
    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

    def __repr__(self):
        return f"UnaryOpNode(operator={self.op}, operand={self.operand})"

    def __str__(self):
        return f"({self.op}{self.operand})"