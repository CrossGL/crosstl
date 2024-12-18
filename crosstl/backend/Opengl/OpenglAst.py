class ASTNode:
    pass


class UniformNode(ASTNode):
    def __init__(self, vtype, name):
        self.vtype = vtype
        self.name = name

    def __repr__(self):
        return f"UniformNode(vtype={self.vtype}, name={self.name})"


class ConstantNode(ASTNode):
    def __init__(self, vtype, name, value):
        self.vtype = vtype
        self.name = name
        self.value = value

    def __repr__(self):
        return (
            f"ConstantNode(vtype={self.vtype}, name ={self.name} ,value={self.value})"
        )


class TernaryOpNode:
    def __init__(self, condition, true_expr, false_expr):
        self.condition = condition
        self.true_expr = true_expr
        self.false_expr = false_expr

    def __repr__(self):
        return f"TernaryOpNode(condition={self.condition}, true_expr={self.true_expr}, false_expr={self.false_expr})"


class LayoutNode:
    def __init__(self, location_number, dtype, name, io_type, semantic=None):
        self.location_number = location_number
        self.dtype = dtype
        self.name = name
        self.io_type = io_type
        self.semantic = semantic

    def __repr__(self):
        return f"LayoutNode(location_number={self.location_number}, dtype={self.dtype}, name={self.name}, io_type={self.io_type}, semantic={self.semantic})"


class ShaderNode(ASTNode):
    def __init__(
        self, io_variables, constant, uniforms, global_variables, functions, shader_type
    ):
        self.io_variables = io_variables
        self.constant = constant
        self.uniforms = uniforms
        self.global_variables = global_variables
        self.functions = functions
        self.shader_type = shader_type

    def __repr__(self):
        return f"ShaderNode(io_variables={self.io_variables}, constant={self.constant},uniforms={self.uniforms} , global_variables={self.global_variables}, functions={self.functions}, shader_type={self.shader_type})"


class FunctionNode(ASTNode):
    def __init__(self, return_type, name, params, body, qualifier=None):
        self.return_type = return_type
        self.name = name
        self.params = params
        self.body = body
        self.qualifier = qualifier

    def __repr__(self):
        return f"FunctionNode(return_type={self.return_type}, name={self.name}, params={self.params}, body={self.body}, qualifier={self.qualifier})"


class ArrayAccessNode(ASTNode):
    def __init__(self, array, index):
        self.array = array
        self.index = index

    def __repr__(self):
        return f"ArrayAccessNode(array={self.array}, index={self.index})"


class VariableNode(ASTNode):
    def __init__(self, vtype, name, io_type=None, semantic=None):
        self.vtype = vtype
        self.name = name
        self.io_type = io_type
        self.semantic = semantic

    def __repr__(self):
        return f"VariableNode(vtype='{self.vtype}', name='{self.name}', io_type={self.io_type}, semantic={self.semantic})"


class AssignmentNode(ASTNode):
    def __init__(self, name, value, operator="="):
        self.name = name
        self.value = value
        self.operator = operator

    def __repr__(self):
        return f"AssignmentNode(name={self.name}, value={self.value}, operator='{self.operator}')"


class VectorConstructorNode:
    def __init__(self, type_name, args):
        self.type_name = type_name
        self.args = args

    def __repr__(self):
        return f"VectorConstructorNode(type_name={self.type_name}, args={self.args})"


class IfNode(ASTNode):
    def __init__(
        self,
        if_condition,
        if_body,
        else_if_conditions=[],
        else_if_bodies=[],
        else_body=None,
    ):
        self.if_condition = if_condition
        self.if_body = if_body
        self.else_if_conditions = else_if_conditions
        self.else_if_bodies = else_if_bodies
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
