class ASTNode:
    pass


class TernaryOpNode:
    def __init__(self, condition, true_expr, false_expr):
        self.condition = condition
        self.true_expr = true_expr
        self.false_expr = false_expr

    def __repr__(self):
        return f"TernaryOpNode(condition={self.condition}, true_expr={self.true_expr}, false_expr={self.false_expr})"


class ShaderNode(ASTNode):
    def __init__(self, preprocessors, struct, constant, functions):
        self.processors = preprocessors
        self.struct = struct
        self.constant = constant
        self.functions = functions

    def __repr__(self):
        return f"ShaderNode(processors={self.processors}, struct={self.struct}, constant={self.constant}, functions={self.functions})"


class StructNode(ASTNode):
    def __init__(self, name, members):
        self.name = name
        self.members = members

    def __repr__(self):
        return f"StructNode(name={self.name}, members={self.members})"


class FunctionNode(ASTNode):
    def __init__(self, qualifier, return_type, name, params, body, attributes=None):
        self.qualifier = qualifier
        self.return_type = return_type
        self.name = name
        self.params = params
        self.body = body
        self.attributes = attributes or []

    def __repr__(self):
        return f"FunctionNode(qualifier={self.qualifier}, return_type={self.return_type}, name={self.name}, params={self.params}, body={self.body}, attributes={self.attributes})"


class ArrayAccessNode(ASTNode):
    def __init__(self, array, index):
        self.array = array
        self.index = index

    def __repr__(self):
        return f"ArrayAccessNode(array={self.array}, index={self.index})"


class VariableNode(ASTNode):
    def __init__(self, vtype, name, attributes=None):
        self.vtype = vtype
        self.name = name
        self.attributes = attributes or []
        self.is_const = False  # Default is not constant

    def __repr__(self):
        const_str = "const " if self.is_const else ""
        return f"VariableNode(vtype='{const_str}{self.vtype}', name='{self.name}', attributes={self.attributes})"


class AttributeNode(ASTNode):
    def __init__(self, name, args=None):
        self.name = name
        self.args = args or []

    def __repr__(self):
        return f"AttributeNode(name='{self.name}', args={self.args})"


class AssignmentNode(ASTNode):
    def __init__(self, left, right, operator="="):
        self.left = left
        self.right = right
        self.operator = operator

    def __repr__(self):
        return f"AssignmentNode(left={self.left}, operator='{self.operator}', right={self.right})"


class IfNode(ASTNode):
    def __init__(self, if_chain=None, else_if_chain=None, else_body=None):
        self.if_chain = if_chain or []
        self.else_if_chain = else_if_chain or []
        self.else_body = else_body

    def __repr__(self):
        return f"IfNode(if_chain={self.if_chain}, else_if_chain={self.else_if_chain}, else_body={self.else_body})"


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


class VectorConstructorNode(ASTNode):
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


class TextureSampleNode(ASTNode):
    def __init__(self, texture, sampler, coordinates, lod=None):
        self.texture = texture
        self.sampler = sampler
        self.coordinates = coordinates
        self.lod = lod

    def __repr__(self):
        if self.lod is not None:
            return f"TextureSampleNode(texture={self.texture}, sampler={self.sampler}, coordinates={self.coordinates}, lod={self.lod})"
        return f"TextureSampleNode(texture={self.texture}, sampler={self.sampler}, coordinates={self.coordinates})"


class ThreadgroupSyncNode(ASTNode):
    def __init__(self):
        pass

    def __repr__(self):
        return "ThreadgroupSyncNode()"


class ConstantBufferNode(ASTNode):
    def __init__(self, name, members):
        self.name = name
        self.members = members

    def __repr__(self):
        return f"ConstantBufferNode(name={self.name}, members={self.members})"


class SwitchNode(ASTNode):
    def __init__(self, expression, cases, default=None):
        self.expression = expression
        self.cases = cases  # List of CaseNode
        self.default = default  # List of statements or None

    def __repr__(self):
        return f"SwitchNode(expression={self.expression}, cases={self.cases}, default={self.default})"


class CaseNode(ASTNode):
    def __init__(self, value, statements):
        self.value = value
        self.statements = statements

    def __repr__(self):
        return f"CaseNode(value={self.value}, statements={self.statements})"
