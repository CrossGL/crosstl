from crosstl.translator.ast import (
    AttributeNode,
    BlockNode,
    ExecutionModel,
    FunctionNode,
    LayoutQualifierNode,
    LiteralNode,
    ParameterNode,
    PrimitiveType,
    ReturnNode,
    ShaderNode,
    ShaderStage,
    StageNode,
    StructMemberNode,
    StructNode,
    VariableNode,
    VectorType,
    create_legacy_shader_node,
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


def test_shader_node_exposes_independent_empty_cbuffer_collection():
    first = ShaderNode("First", ExecutionModel.GRAPHICS_PIPELINE)
    second = ShaderNode("Second", ExecutionModel.GRAPHICS_PIPELINE)
    globals_block = StructNode("Globals", [])

    first.cbuffers.append(globals_block)

    assert first.cbuffers == [globals_block]
    assert second.cbuffers == []


def test_legacy_shader_node_preserves_cbuffers_on_canonical_field():
    globals_block = StructNode("Globals", [])

    shader = create_legacy_shader_node(
        structs=[],
        functions=[],
        global_variables=[],
        cbuffers=[globals_block],
    )

    assert shader.cbuffers == [globals_block]
    assert shader.constants == []
    assert globals_block in list(shader.walk())


def test_stage_node_walk_and_parent_links_include_stage_local_ir_surfaces():
    local_variable = VariableNode("sharedTile", PrimitiveType("float"))
    local_function = FunctionNode(
        "helper",
        PrimitiveType("float"),
        [],
        body=BlockNode([ReturnNode(LiteralNode(1.0, PrimitiveType("float")))]),
    )
    local_struct = StructNode(
        "Payload",
        [StructMemberNode("value", PrimitiveType("float"))],
    )
    local_cbuffer = StructNode(
        "StageConstants",
        [StructMemberNode("scale", PrimitiveType("float"))],
    )
    layout = LayoutQualifierNode(
        entries=[AttributeNode("local_size_x", [LiteralNode(8, PrimitiveType("int"))])],
        direction="in",
    )
    entry = FunctionNode(
        "main",
        PrimitiveType("void"),
        [],
        body=BlockNode([]),
    )
    stage = StageNode(
        ShaderStage.COMPUTE,
        entry,
        local_variables=[local_variable],
        local_functions=[local_function],
        local_structs=[local_struct],
        local_cbuffers=[local_cbuffer],
        layout_qualifiers=[layout],
        execution_config={"local_size_x": "8"},
    )
    shader = ShaderNode(
        "StageLocalContract",
        ExecutionModel.COMPUTE_KERNEL,
        stages={ShaderStage.COMPUTE: stage},
    )

    walked = list(shader.walk())

    assert local_variable in walked
    assert local_function in walked
    assert local_function.body in walked
    assert local_struct in walked
    assert local_struct.members[0] in walked
    assert local_cbuffer in walked
    assert local_cbuffer.members[0] in walked
    assert layout in walked
    assert layout.entries[0] in walked
    assert layout.entries[0].arguments[0] in walked

    shader.bind_parent_links()

    assert stage.parent is shader
    assert local_variable.parent is stage
    assert local_function.parent is stage
    assert local_function.body.parent is local_function
    assert local_struct.parent is stage
    assert local_struct.members[0].parent is local_struct
    assert local_cbuffer.parent is stage
    assert local_cbuffer.members[0].parent is local_cbuffer
    assert layout.parent is stage
    assert layout.entries[0].parent is layout
    assert layout.entries[0].arguments[0].parent is layout.entries[0]
