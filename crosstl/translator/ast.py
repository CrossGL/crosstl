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


class CbufferNode:
    def __init__(self, name, members):
        self.name = name
        self.members = members

    def __repr__(self):
        return f"CbufferNode(name={self.name}, members={self.members})"


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
    def __init__(
        self,
        if_condition,
        if_body,
        else_if_conditions=None,
        else_if_bodies=None,
        else_body=None,
    ):
        self.if_condition = if_condition
        self.if_body = if_body
        self.else_if_conditions = (
            [] if else_if_conditions is None else else_if_conditions
        )
        self.else_if_bodies = [] if else_if_bodies is None else else_if_bodies
        self.else_body = else_body

    def __repr__(self):
        return f"IfNode(if_condition={self.if_condition}, if_body={self.if_body}, else_if_conditions={self.else_if_conditions}, else_if_bodies={self.else_if_bodies}, else_body={self.else_body})"


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
        return f"BinaryOpNode(left={self.left}, op={self.op}, right={self.right})"


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
        return f"UnaryOpNode(op={self.op}, operand={self.operand})"
