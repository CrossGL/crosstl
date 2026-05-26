from crosstl.translator.ast import (
    AttributeNode,
    BlockNode,
    ExecutionModel,
    FunctionNode,
    LiteralNode,
    ParameterNode,
    PrimitiveType,
    ReturnNode,
    ShaderNode,
    ShaderStage,
    StageNode,
    StructMemberNode,
    StructNode,
    VectorType,
)


def build_shader_ir():
    float_type = PrimitiveType("float")
    position_type = VectorType(float_type, 4)
    location_attr = AttributeNode("location", [LiteralNode(0, PrimitiveType("int"))])
    member = StructMemberNode("position", position_type, attributes=[location_attr])
    output_struct = StructNode("VertexOut", [member])

    parameter = ParameterNode("scale", PrimitiveType("float"))
    body = BlockNode([ReturnNode(LiteralNode(1.0, PrimitiveType("float")))])
    entry = FunctionNode("main", PrimitiveType("void"), [parameter], body=body)
    stage = StageNode(ShaderStage.VERTEX, entry)

    shader = ShaderNode(
        "IrContract",
        ExecutionModel.GRAPHICS_PIPELINE,
        stages={ShaderStage.VERTEX: stage},
        structs=[output_struct],
    )
    shader.annotations["ignored_node"] = StructNode("Ignored", [])
    return shader, stage, output_struct, member, location_attr, entry, parameter, body


def test_ast_walk_visits_structural_children_once_and_skips_metadata():
    shader, stage, output_struct, member, location_attr, entry, parameter, body = (
        build_shader_ir()
    )

    walked = list(shader.walk())

    assert walked[0] is shader
    assert stage in walked
    assert output_struct in walked
    assert member in walked
    assert location_attr in walked
    assert entry in walked
    assert parameter in walked
    assert body in walked
    assert shader.annotations["ignored_node"] not in walked
    assert len({id(node) for node in walked}) == len(walked)


def test_bind_parent_links_populates_structural_parent_chain():
    shader, stage, output_struct, member, location_attr, entry, parameter, body = (
        build_shader_ir()
    )

    assert shader.bind_parent_links() is shader

    assert shader.parent is None
    assert stage.parent is shader
    assert entry.parent is stage
    assert parameter.parent is entry
    assert body.parent is entry
    assert output_struct.parent is shader
    assert member.parent is output_struct
    assert member.member_type.parent is member
    assert location_attr.parent is member
    assert location_attr.arguments[0].parent is location_attr
    assert shader.annotations["ignored_node"].parent is None


def test_ast_walk_tolerates_existing_parent_links_and_cycles():
    shader, _, output_struct, member, _, _, _, _ = build_shader_ir()
    shader.bind_parent_links()
    member.default_value = output_struct

    walked = list(shader.walk())

    assert output_struct in walked
    assert member in walked
    assert len({id(node) for node in walked}) == len(walked)
