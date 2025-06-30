class MojoASTNode:
    pass


class TernaryOpNode(MojoASTNode):
    def __init__(self, condition, true_expr, false_expr):
        self.condition = condition
        self.true_expr = true_expr
        self.false_expr = false_expr

    def __repr__(self):
        return f"TernaryOpNode(condition={self.condition}, true_expr={self.true_expr}, false_expr={self.false_expr})"


class ShaderNode(MojoASTNode):
    def __init__(self, functions):
        self.functions = functions

    def __repr__(self):
        return f"ShaderNode(functions={self.functions})"


class StructNode(MojoASTNode):
    def __init__(self, name, members):
        self.name = name
        self.members = members

    def __repr__(self):
        return f"StructNode(name={self.name}, members={self.members})"


class FunctionNode(MojoASTNode):
    def __init__(self, qualifier, return_type, name, params, body, attributes=None):
        self.qualifier = qualifier
        self.return_type = return_type
        self.name = name
        self.params = params
        self.body = body
        self.attributes = attributes or []

    def __repr__(self):
        return f"FunctionNode(qualifier={self.qualifier}, return_type={self.return_type}, name={self.name}, params={self.params}, body={self.body}, attributes={self.attributes})"


class VariableDeclarationNode(MojoASTNode):
    def __init__(self, var_type, name, initial_value=None, vtype=None):
        self.var_type = var_type
        self.name = name
        self.initial_value = initial_value
        self.vtype = vtype

    def __repr__(self):
        return f"VariableDeclarationNode(var_type={self.var_type}, name={self.name}, initial_value={self.initial_value}, vtype={self.vtype})"


class ArrayAccessNode(MojoASTNode):
    def __init__(self, array, index):
        self.array = array
        self.index = index

    def __repr__(self):
        return f"ArrayAccessNode(array={self.array}, index={self.index})"


class VariableNode(MojoASTNode):
    def __init__(self, vtype, name, attributes=None):
        self.vtype = vtype
        self.name = name
        self.attributes = attributes or []

    def __repr__(self):
        return f"VariableNode(vtype='{self.vtype}', name='{self.name}', attributes={self.attributes})"


class AttributeNode(MojoASTNode):
    def __init__(self, name, args=None):
        self.name = name
        self.args = args or []

    def __repr__(self):
        return f"AttributeNode(name='{self.name}', args={self.args})"


class AssignmentNode(MojoASTNode):
    def __init__(self, left, right, operator="="):
        self.left = left
        self.right = right
        self.operator = operator

    def __repr__(self):
        return f"AssignmentNode(left={self.left}, operator='{self.operator}', right={self.right})"


class IfNode(MojoASTNode):
    def __init__(self, condition, if_body, else_body=None):
        self.condition = condition
        self.if_body = if_body
        self.else_body = else_body

    def __repr__(self):
        return f"IfNode(condition={self.condition}, if_body={self.if_body}, else_body={self.else_body})"


class ForNode(MojoASTNode):
    def __init__(self, init, condition, update, body):
        self.init = init
        self.condition = condition
        self.update = update
        self.body = body

    def __repr__(self):
        return f"ForNode(init={self.init}, condition={self.condition}, update={self.update}, body={self.body})"


class WhileNode(MojoASTNode):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

    def __repr__(self):
        return f"WhileNode(condition={self.condition}, body={self.body})"


class DoWhileNode(MojoASTNode):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

    def __repr__(self):
        return f"DoWhileNode(condition={self.condition}, body={self.body})"


class ReturnNode(MojoASTNode):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"ReturnNode(value={self.value})"


class FunctionCallNode(MojoASTNode):
    def __init__(self, name, args):
        self.name = name
        self.args = args

    def __repr__(self):
        return f"FunctionCallNode(name={self.name}, args={self.args})"


class BinaryOpNode(MojoASTNode):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self):
        return f"BinaryOpNode(left={self.left}, operator={self.op}, right={self.right})"


class UnaryOpNode(MojoASTNode):
    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

    def __repr__(self):
        return f"UnaryOpNode(operator={self.op}, operand={self.operand})"


class MemberAccessNode(MojoASTNode):
    def __init__(self, object, member):
        self.object = object
        self.member = member

    def __repr__(self):
        return f"MemberAccessNode(object={self.object}, member={self.member})"


class VectorConstructorNode(MojoASTNode):
    def __init__(self, type_name, args):
        self.type_name = type_name
        self.args = args

    def __repr__(self):
        return f"VectorConstructorNode(type_name={self.type_name}, args={self.args})"


class TextureSampleNode(MojoASTNode):
    def __init__(self, texture, sampler, coordinates):
        self.texture = texture
        self.sampler = sampler
        self.coordinates = coordinates

    def __repr__(self):
        return f"TextureSampleNode(texture={self.texture}, sampler={self.sampler}, coordinates={self.coordinates})"


class ThreadgroupSyncNode(MojoASTNode):
    def __init__(self):
        pass

    def __repr__(self):
        return "ThreadgroupSyncNode()"


class ConstantBufferNode(MojoASTNode):
    def __init__(self, name, members):
        self.name = name
        self.members = members

    def __repr__(self):
        return f"ConstantBufferNode(name={self.name}, members={self.members})"


class ImportNode(MojoASTNode):
    def __init__(self, module_name, alias=None):
        self.module_name = module_name
        self.alias = alias

    def __repr__(self):
        if self.alias:
            return f"ImportNode(module_name='{self.module_name}', alias='{self.alias}')"
        return f"ImportNode(module_name='{self.module_name}')"


class ClassNode(MojoASTNode):
    def __init__(self, name, base_classes, members):
        self.name = name
        self.base_classes = base_classes
        self.members = members

    def __repr__(self):
        return f"ClassNode(name={self.name}, base_classes={self.base_classes}, members={self.members})"


class DecoratorNode(MojoASTNode):
    def __init__(self, name, args=None):
        self.name = name
        self.args = args or []

    def __repr__(self):
        return f"DecoratorNode(name={self.name}, args={self.args})"


class SwitchCaseNode(MojoASTNode):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

    def __repr__(self):
        return f"SwitchCaseNode(condition={self.condition}, body={self.body})"


class SwitchNode(MojoASTNode):
    def __init__(self, expression, cases):
        self.expression = expression
        self.cases = cases

    def __repr__(self):
        return f"SwitchNode(expression={self.expression}, cases={self.cases})"


class CaseNode(MojoASTNode):
    def __init__(self, value, body):
        self.value = value
        self.body = body

    def __repr__(self):
        return f"CaseNode(value={self.value}, body={self.body})"


class PragmaNode(MojoASTNode):
    def __init__(self, directive, value):
        self.directive = directive
        self.value = value

    def __repr__(self):
        return f"PragmaNode(directive={self.directive}, value={self.value})"


class IncludeNode(MojoASTNode):
    def __init__(self, path):
        self.path = path

    def __repr__(self):
        return f"IncludeNode(path={self.path})"


class BreakNode(MojoASTNode):
    def __init__(self):
        pass

    def __repr__(self):
        return "BreakNode()"


class ContinueNode(MojoASTNode):
    def __init__(self):
        pass

    def __repr__(self):
        return "ContinueNode()"


class PassNode(MojoASTNode):
    def __init__(self):
        pass

    def __repr__(self):
        return "PassNode()"
