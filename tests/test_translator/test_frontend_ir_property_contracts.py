from hypothesis import given, settings, strategies as st

from crosstl.translator.ast import (
    AttributeNode,
    BlockNode,
    ExecutionModel,
    FunctionNode,
    LayoutQualifierNode,
    LiteralNode,
    PrimitiveType,
    ReturnNode,
    ShaderNode,
    ShaderStage,
    StageNode,
    StructMemberNode,
    StructNode,
    VariableNode,
)


def _float_type():
    return PrimitiveType("float")


def _int_type():
    return PrimitiveType("int")


def _literal_int(value):
    return LiteralNode(value, _int_type())


def _literal_float(value):
    return LiteralNode(float(value), _float_type())


def _make_stage(stage_kind, index, local_count, layout_count):
    entry = FunctionNode(
        f"main_{index}",
        PrimitiveType("void"),
        [],
        body=BlockNode([]),
    )
    local_variables = [
        VariableNode(f"local_{index}_{i}", _float_type()) for i in range(local_count)
    ]
    local_functions = [
        FunctionNode(
            f"helper_{index}_{i}",
            _float_type(),
            [],
            body=BlockNode([ReturnNode(_literal_float(i))]),
        )
        for i in range(local_count)
    ]
    local_structs = [
        StructNode(
            f"Payload_{index}_{i}",
            [StructMemberNode(f"value_{i}", _float_type())],
        )
        for i in range(local_count)
    ]
    local_cbuffers = [
        StructNode(
            f"Constants_{index}_{i}",
            [StructMemberNode(f"scale_{i}", _float_type())],
        )
        for i in range(local_count)
    ]
    layout_qualifiers = [
        LayoutQualifierNode(
            entries=[AttributeNode(f"local_size_{axis}", [_literal_int(i + 1)])],
            direction="in",
        )
        for i, axis in enumerate(("x", "y", "z")[:layout_count])
    ]

    stage = StageNode(
        stage_kind,
        entry,
        local_variables=local_variables,
        local_functions=local_functions,
        local_structs=local_structs,
        local_cbuffers=local_cbuffers,
        layout_qualifiers=layout_qualifiers,
    )
    expected_nodes = [
        stage,
        entry,
        entry.body,
        *local_variables,
        *local_functions,
        *local_structs,
        *local_cbuffers,
        *layout_qualifiers,
    ]
    for helper in local_functions:
        expected_nodes.append(helper.body)
        expected_nodes.extend(helper.body.statements)
        expected_nodes.append(helper.body.statements[0].value)
    for local_struct in [*local_structs, *local_cbuffers]:
        expected_nodes.extend(local_struct.members)
    for layout in layout_qualifiers:
        expected_nodes.extend(layout.entries)
        expected_nodes.extend(layout.entries[0].arguments)
    return stage, expected_nodes


@settings(max_examples=25, deadline=None)
@given(
    stage_kinds=st.lists(
        st.sampled_from(tuple(ShaderStage)),
        min_size=1,
        max_size=4,
        unique=True,
    ),
    local_count=st.integers(min_value=0, max_value=3),
    layout_count=st.integers(min_value=0, max_value=3),
)
def test_generated_stage_local_ir_shapes_walk_once_and_bind_parent_links(
    stage_kinds,
    local_count,
    layout_count,
):
    stages = {}
    expected_nodes = []
    for index, stage_kind in enumerate(stage_kinds):
        stage, stage_expected_nodes = _make_stage(
            stage_kind,
            index,
            local_count,
            layout_count,
        )
        stages[stage_kind] = stage
        expected_nodes.extend(stage_expected_nodes)

    shader = ShaderNode(
        "GeneratedStageLocalContract",
        ExecutionModel.GRAPHICS_PIPELINE,
        stages=stages,
    )

    walked = list(shader.walk())

    assert len({id(node) for node in walked}) == len(walked)
    for node in expected_nodes:
        assert node in walked

    shader.bind_parent_links()

    for stage in stages.values():
        assert stage.parent is shader
        assert stage.entry_point.parent is stage
        assert stage.entry_point.body.parent is stage.entry_point
        for variable in stage.local_variables:
            assert variable.parent is stage
            assert variable.var_type.parent is variable
        for helper in stage.local_functions:
            assert helper.parent is stage
            assert helper.body.parent is helper
            if helper.body.statements:
                assert helper.body.statements[0].parent is helper.body
                assert (
                    helper.body.statements[0].value.parent is helper.body.statements[0]
                )
        for local_struct in stage.local_structs:
            assert local_struct.parent is stage
            assert local_struct.members[0].parent is local_struct
            assert local_struct.members[0].member_type.parent is local_struct.members[0]
        for local_cbuffer in stage.local_cbuffers:
            assert local_cbuffer.parent is stage
            assert local_cbuffer.members[0].parent is local_cbuffer
            assert (
                local_cbuffer.members[0].member_type.parent is local_cbuffer.members[0]
            )
        for layout in stage.layout_qualifiers:
            assert layout.parent is stage
            assert layout.entries[0].parent is layout
            assert layout.entries[0].arguments[0].parent is layout.entries[0]
