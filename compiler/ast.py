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


class TernaryOpNode:
    def __init__(self, condition, true_expr, false_expr):
        self.condition = condition
        self.true_expr = true_expr
        self.false_expr = false_expr

    def __repr__(self):
        return f"TernaryOpNode(condition={self.condition}, true_expr={self.true_expr}, false_expr={self.false_expr})"


class ShaderNode:
    def __init__(
        self,
        name,
        global_inputs,
        global_outputs,
        global_functions,
        vertex_section,
        fragment_section,
    ):
        self.name = name
        self.global_inputs = global_inputs
        self.global_outputs = global_outputs
        self.global_functions = global_functions
        self.vertex_section = vertex_section
        self.fragment_section = fragment_section

    def __repr__(self):
        return f"ShaderNode({self.name!r}) {self.global_inputs!r} {self.global_outputs!r} {self.global_functions!r} {self.vertex_section!r} {self.fragment_section!r}"


class VERTEXShaderNode:
    def __init__(self, inputs, outputs, functions):
        self.inputs = inputs
        self.outputs = outputs
        self.functions = functions

    def __repr__(self):
        return f"VERTEXShaderNode({self.inputs!r}) {self.outputs!r} {self.functions!r}"


class FRAGMENTShaderNode:
    def __init__(self, inputs, outputs, functions):
        self.inputs = inputs
        self.outputs = outputs
        self.functions = functions

    def __repr__(self):
        return (
            f"FRAGMENTShaderNode({self.inputs!r}) {self.outputs!r} {self.functions!r}"
        )


class FunctionNode(ASTNode):
    def __init__(self, return_type, name, params, body):
        self.return_type = return_type
        self.name = name
        self.params = params
        self.body = body

    def __repr__(self):
        return f"FunctionNode(return_type={self.return_type}, name={self.name}, params={self.params}, body={self.body})"


class VariableNode(ASTNode):
    def __init__(self, vtype, name):
        self.vtype = vtype
        self.name = name

    def __repr__(self):
        return f"VariableNode(vtype={self.vtype}, name={self.name})"


class AssignmentNode(ASTNode):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"AssignmentNode(name={self.name}, value={self.value})"


class IfNode(ASTNode):
    def __init__(self, condition, if_body, else_body=None):
        self.condition = condition
        self.if_body = if_body
        self.else_body = else_body

    def __repr__(self):
        return f"IfNode(condition={self.condition}, if_body={self.if_body}, else_body={self.else_body})"


class ForNode(ASTNode):
    def __init__(self, init, condition, update, body):
        self.init = init
        self.condition = condition
        self.update = update
        self.body = body

    def __repr__(self):
        return f"ForNode(init={self.init}, condition={self.condition}, update={self.update}, body={self.body})"


class ReturnNode(ASTNode):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"ReturnNode(value={self.value})"


class FunctionCallNode(ASTNode):
    def __init__(self, name, args):
        self.name = name
        self.args = args

    def __repr__(self):
        return f"FunctionCallNode(name={self.name}, args={self.args})"


class BinaryOpNode(ASTNode):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self):
        return f"BinaryOpNode(left={self.left}, operator={self.op}, right={self.right})"


class MemberAccessNode(ASTNode):
    def __init__(self, object, member):
        self.object = object
        self.member = member

    def __repr__(self):
        return f"MemberAccessNode(object={self.object}, member={self.member})"


class UnaryOpNode(ASTNode):
    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

    def __repr__(self):
        return f"UnaryOpNode(operator={self.op}, operand={self.operand})"

    def __str__(self):
        return f"({self.op}{self.operand})"
