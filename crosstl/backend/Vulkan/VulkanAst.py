from typing import Any


class ASTNode:
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return self.__repr__()


class TernaryOpNode(ASTNode):
    def __init__(self, condition: ASTNode, true_expr: ASTNode, false_expr: ASTNode):
        self.condition = condition
        self.true_expr = true_expr
        self.false_expr = false_expr

    def __repr__(self) -> str:
        return (
            f"TernaryOpNode(condition={self.condition}, "
            f"true_expr={self.true_expr}, false_expr={self.false_expr})"
        )


class ShaderNode(ASTNode):
    def __init__(
        self,
        spirv_version: str,
        descriptor_sets: list,
        shader_stages: list,
        functions: list,
    ):
        self.spirv_version = spirv_version
        self.descriptor_sets = descriptor_sets
        self.shader_stages = shader_stages
        self.functions = functions

    def __repr__(self) -> str:
        return (
            f"ShaderNode(spirv_version={self.spirv_version}, "
            f"descriptor_sets={self.descriptor_sets}, "
            f"shader_stages={self.shader_stages}, functions={self.functions})"
        )


class IfNode(ASTNode):
    def __init__(
        self,
        if_condition: ASTNode,
        if_body: list,
        else_if_conditions: list = None,
        else_if_bodies: list = None,
        else_body: list = None,
    ):
        self.if_condition = if_condition
        self.if_body = if_body
        self.else_if_conditions = (
            else_if_conditions if else_if_conditions is not None else []
        )
        self.else_if_bodies = else_if_bodies if else_if_bodies is not None else []
        self.else_body = else_body

    def __repr__(self) -> str:
        return (
            f"IfNode(if_condition={self.if_condition}, if_body={self.if_body}, "
            f"else_if_conditions={self.else_if_conditions}, "
            f"else_if_bodies={self.else_if_bodies}, else_body={self.else_body})"
        )


class ForNode(ASTNode):
    def __init__(self, init: ASTNode, condition: ASTNode, update: ASTNode, body: list):
        self.init = init
        self.condition = condition
        self.update = update
        self.body = body

    def __repr__(self) -> str:
        return (
            f"ForNode(init={self.init}, condition={self.condition}, "
            f"update={self.update}, body={self.body})"
        )


class ReturnNode(ASTNode):
    def __init__(self, value: Any):
        self.value = value

    def __repr__(self) -> str:
        return f"ReturnNode(value={self.value})"


class FunctionCallNode(ASTNode):
    def __init__(self, name: str, args: list):
        self.name = name
        self.args = args

    def __repr__(self) -> str:
        return f"FunctionCallNode(name={self.name}, args={self.args})"


class BinaryOpNode(ASTNode):
    def __init__(self, left: ASTNode, op: str, right: ASTNode):
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self) -> str:
        return f"BinaryOpNode(left={self.left}, op={self.op}, right={self.right})"


class UnaryOpNode(ASTNode):
    def __init__(self, op: str, operand: ASTNode):
        self.op = op
        self.operand = operand

    def __repr__(self) -> str:
        return f"UnaryOpNode(operator={self.op}, operand={self.operand})"


class DescriptorSetNode(ASTNode):
    def __init__(self, set_number: int, bindings: list):
        self.set_number = set_number
        self.bindings = bindings

    def __repr__(self) -> str:
        return (
            f"DescriptorSetNode(set_number={self.set_number}, bindings={self.bindings})"
        )


class LayoutNode(ASTNode):
    def __init__(
        self,
        bindings: list,
        push_constant: bool,
        layout_type: str,
        data_type: str,
        variable_name: str,
        struct_fields: list,
    ):
        self.bindings = bindings
        self.push_constant = push_constant
        self.layout_type = layout_type
        self.data_type = data_type
        self.variable_name = variable_name
        self.struct_fields = struct_fields

    def __repr__(self) -> str:
        return (
            f"LayoutNode(bindings={self.bindings}, push_constant={self.push_constant}, "
            f"layout_type={self.layout_type}, data_type={self.data_type}, "
            f"variable_name={self.variable_name}, struct_fields={self.struct_fields})"
        )


class UniformNode(ASTNode):
    def __init__(self, name: str, var_type: str, value: Any = None):
        self.name = name
        self.var_type = var_type
        self.value = value

    def __repr__(self) -> str:
        return f"UniformNode(name={self.name}, var_type={self.var_type}, value={self.value})"


class ShaderStageNode(ASTNode):
    def __init__(self, stage: str, entry_point: str):
        self.stage = stage
        self.entry_point = entry_point

    def __repr__(self) -> str:
        return f"ShaderStageNode(stage={self.stage}, entry_point={self.entry_point})"


class PushConstantNode(ASTNode):
    def __init__(self, size: int, values: list):
        self.size = size
        self.values = values

    def __repr__(self) -> str:
        return f"PushConstantNode(size={self.size}, values={self.values})"


class StructNode(ASTNode):
    def __init__(self, name: str, members: list):
        self.name = name
        self.members = members

    def __repr__(self) -> str:
        return f"StructNode(name={self.name}, members={self.members})"


class FunctionNode(ASTNode):
    def __init__(self, name: str, return_type: str, parameters: list, body: list):
        self.name = name
        self.return_type = return_type
        self.parameters = parameters
        self.body = body

    def __repr__(self) -> str:
        return (
            f"FunctionNode(name={self.name}, return_type={self.return_type}, "
            f"parameters={self.parameters}, body={self.body})"
        )


class MemberAccessNode(ASTNode):
    def __init__(self, object: ASTNode, member: str):
        self.object = object
        self.member = member

    def __repr__(self) -> str:
        return f"MemberAccessNode(object={self.object}, member={self.member})"


class VariableNode(ASTNode):
    def __init__(self, name: str, var_type: str):
        self.name = name
        self.var_type = var_type

    def __repr__(self) -> str:
        return f"VariableNode(name={self.name}, var_type={self.var_type})"


class SwitchNode(ASTNode):
    def __init__(self, expression: ASTNode, cases: list):
        self.expression = expression
        self.cases = cases

    def __repr__(self) -> str:
        return f"SwitchNode(expression={self.expression}, cases={self.cases})"


class CaseNode(ASTNode):
    def __init__(self, value: Any, body: list):
        self.value = value
        self.body = body

    def __repr__(self) -> str:
        return f"CaseNode(value={self.value}, body={self.body})"


class DefaultNode(ASTNode):
    def __init__(self, statements: list):
        self.statements = statements

    def __repr__(self) -> str:
        return f"DefaultNode(statements={self.statements})"


class WhileNode(ASTNode):
    def __init__(self, condition: ASTNode, body: list):
        self.condition = condition
        self.body = body

    def __repr__(self) -> str:
        return f"WhileNode(condition={self.condition}, body={self.body})"


class DoWhileNode(ASTNode):
    def __init__(self, body: list, condition: ASTNode):
        self.body = body
        self.condition = condition

    def __repr__(self) -> str:
        return f"DoWhileNode(body={self.body}, condition={self.condition})"


class AssignmentNode(ASTNode):
    def __init__(self, name: str, value: Any):
        self.name = name
        self.value = value

    def __repr__(self) -> str:
        return f"AssignmentNode(name={self.name}, value={self.value})"


class BreakNode(ASTNode):
    def __init__(self):
        pass

    def __repr__(self) -> str:
        return f"BreakNode()"
