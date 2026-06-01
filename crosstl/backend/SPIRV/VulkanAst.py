"""SPIR-V/Vulkan AST Node definitions"""

from ..common_ast import (
    ArrayAccessNode,
    AssignmentNode,
    ASTNode,
    BinaryOpNode,
    BreakNode,
    CaseNode,
    ContinueNode,
    DiscardNode,
    DoWhileNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    MemberAccessNode,
    MethodCallNode,
    ReturnNode,
    ShaderNode,
    StructNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)

# Keep common AST imports used for re-exports (autoflake-safe).
_COMMON_NODES = (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    BreakNode,
    CaseNode,
    ContinueNode,
    DiscardNode,
    DoWhileNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    MemberAccessNode,
    MethodCallNode,
    ReturnNode,
    ShaderNode,
    StructNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)

# SPIR-V/Vulkan-specific nodes


class DescriptorSetNode(ASTNode):
    """Node representing a descriptor set binding"""

    def __init__(self, set_number, binding, descriptor_type, name):
        self.set_number = set_number
        self.binding = binding
        self.descriptor_type = descriptor_type
        self.name = name

    def __repr__(self):
        return f"DescriptorSetNode(set={self.set_number}, binding={self.binding}, type={self.descriptor_type}, name={self.name})"


class LayoutNode(ASTNode):
    """Node representing layout qualifiers"""

    def __init__(
        self,
        qualifiers,
        declaration=None,
        *,
        push_constant=False,
        layout_type=None,
        data_type=None,
        variable_name=None,
        struct_fields=None,
        block_name=None,
        declaration_qualifiers=None,
        spirv_id=None,
        spirv_decorations=None,
        spirv_storage_class=None,
    ):
        self.qualifiers = qualifiers
        self.declaration = declaration
        self.push_constant = push_constant
        self.layout_type = layout_type
        self.data_type = data_type
        self.variable_name = variable_name
        self.struct_fields = struct_fields or []
        self.block_name = block_name
        self.declaration_qualifiers = declaration_qualifiers or []
        self.spirv_id = spirv_id
        self.spirv_decorations = spirv_decorations or []
        self.spirv_storage_class = spirv_storage_class

    def __repr__(self):
        return (
            "LayoutNode("
            f"qualifiers={self.qualifiers}, layout_type={self.layout_type}, "
            f"data_type={self.data_type}, variable_name={self.variable_name}, "
            f"block_name={self.block_name}, "
            f"declaration_qualifiers={self.declaration_qualifiers}, "
            f"spirv_id={self.spirv_id})"
        )


class UniformNode(ASTNode):
    """Node representing uniform variable"""

    def __init__(self, vtype, name):
        self.vtype = vtype
        self.name = name

    def __repr__(self):
        return f"UniformNode(vtype={self.vtype}, name={self.name})"


class ShaderStageNode(ASTNode):
    """Node representing shader stage annotation"""

    def __init__(self, stage):
        self.stage = stage

    def __repr__(self):
        return f"ShaderStageNode(stage={self.stage})"


class PushConstantNode(ASTNode):
    """Node representing push constant block"""

    def __init__(self, members):
        self.members = members

    def __repr__(self):
        return f"PushConstantNode(members={self.members})"


class DefaultNode(ASTNode):
    """Node representing default case in switch"""

    def __init__(self, statements):
        self.statements = statements

    def __repr__(self):
        return f"DefaultNode(statements={len(self.statements)})"
