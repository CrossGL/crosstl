"""SPIR-V/Vulkan AST Node definitions"""

from ..common_ast import ASTNode

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

    def __init__(self, qualifiers, declaration=None):
        self.qualifiers = qualifiers
        self.declaration = declaration

    def __repr__(self):
        return (
            f"LayoutNode(qualifiers={self.qualifiers}, declaration={self.declaration})"
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
