class ASTNode:
    pass


class TernaryOpNode:
    def __init__(self, condition, true_expr, false_expr):
        self.condition = condition
        self.true_expr = true_expr
        self.false_expr = false_expr

    def __repr__(self):
        return f"TernaryOpNode(condition={self.condition}, true_expr={self.true_expr}, false_expr={self.false_expr})"


class ShaderNode:
    def __init__(self, structs, functions, global_variables, cbuffers):
        self.structs = structs
        self.functions = functions
        self.global_variables = global_variables
        self.cbuffers = cbuffers

    def __repr__(self):
        return f"ShaderNode(structs={self.structs}, functions={self.functions}, global_variables={self.global_variables}, cbuffers={self.cbuffers})"


class StructNode:
    def __init__(self, name, members):
        self.name = name
        self.members = members

    def __repr__(self):
        return f"StructNode(name={self.name}, members={self.members})"


class FunctionNode(ASTNode):
    def __init__(self, return_type, name, params, body, qualifier=None, semantic=None):
        self.return_type = return_type
        self.name = name
        self.params = params
        self.body = body
        self.qualifier = qualifier
        self.semantic = semantic

    def __repr__(self):
        return f"FunctionNode(return_type={self.return_type}, name={self.name}, params={self.params}, body={self.body}, qualifier={self.qualifier}, semantic={self.semantic})"


class VariableNode(ASTNode):
    def __init__(self, vtype, name, semantic=None):
        self.vtype = vtype
        self.name = name
        self.semantic = semantic

    def __repr__(self):
        return f"VariableNode(vtype='{self.vtype}', name='{self.name}', semantic={self.semantic})"


class AssignmentNode(ASTNode):
    def __init__(self, left, right, operator="="):
        self.left = left
        self.right = right
        self.operator = operator

    def __repr__(self):
        return f"AssignmentNode(left={self.left}, operator='{self.operator}', right={self.right})"


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


class WhileNode(ASTNode):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

    def __repr__(self):
        return f"WhileNode(condition={self.condition}, body={self.body})"


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
        return f"BinaryOpNode(left={self.left}, op={self.op}, right={self.right})"


class MemberAccessNode(ASTNode):
    def __init__(self, object, member):
        self.object = object
        self.member = member

    def __repr__(self):
        return f"MemberAccessNode(object={self.object}, member={self.member})"


class VectorConstructorNode:
    def __init__(self, type_name, args):
        self.type_name = type_name
        self.args = args

    def __repr__(self):
        return f"VectorConstructorNode(type_name={self.type_name}, args={self.args})"


class UnaryOpNode(ASTNode):
    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

    def __repr__(self):
        return f"UnaryOpNode(operator={self.op}, operand={self.operand})"

    def __str__(self):
        return f"({self.op}{self.operand})"
