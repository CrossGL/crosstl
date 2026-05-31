from hypothesis import given, settings
from hypothesis import strategies as st

from crosstl.translator.ast import (
    ASTNode,
    AttributeNode,
    BlockNode,
    ExecutionModel,
    FunctionNode,
    IdentifierNode,
    LayoutQualifierNode,
    LiteralNode,
    PrimitiveType,
    ShaderNode,
    ShaderStage,
    StageNode,
    StructNode,
)

IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)


def _int_type():
    return PrimitiveType("int")


def _int(value):
    return LiteralNode(value, _int_type())


def _id(name):
    return IdentifierNode(name)


def _entry_point(name="main"):
    return FunctionNode(
        name,
        PrimitiveType("void"),
        [],
        body=BlockNode([]),
    )


def _assert_walks_once_with(root, expected_nodes):
    walked = list(root.walk())

    assert len({id(node) for node in walked}) == len(walked)
    for node in expected_nodes:
        assert node in walked
    return walked


class MetadataContainerNode(ASTNode):
    def __init__(
        self,
        list_children=None,
        tuple_children=None,
        dict_children=None,
        set_children=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.list_children = list_children or []
        self.tuple_children = tuple_children or ()
        self.dict_children = dict_children or {}
        self.set_children = set_children or set()


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    binding=st.integers(min_value=0, max_value=63),
    set_index=st.integers(min_value=0, max_value=7),
    local_size=st.integers(min_value=1, max_value=1024),
    direction=st.sampled_from(["in", "out", "inout"]),
)
def test_generated_attribute_and_layout_metadata_walks_structural_arguments(
    suffix,
    binding,
    set_index,
    local_size,
    direction,
):
    ignored_attribute_annotation = AttributeNode("ignored_annotation", [_int(999)])
    ignored_attribute_source = AttributeNode("ignored_source", [_int(998)])
    ignored_layout_annotation = AttributeNode("ignored_layout_annotation", [_int(997)])
    ignored_layout_source = AttributeNode("ignored_layout_source", [_int(996)])
    ignored_shader_annotation = StructNode(f"IgnoredShaderMetadata_{suffix}", [])

    binding_argument = _int(binding)
    set_argument = _int(set_index)
    role_argument = _id(f"textureRole_{suffix}")
    local_size_argument = _int(local_size)
    binding_attribute = AttributeNode(
        "binding",
        [binding_argument],
        annotations={"ignored": ignored_attribute_annotation},
        source_location=ignored_attribute_source,
    )
    set_attribute = AttributeNode("set", [set_argument])
    role_attribute = AttributeNode("texture", [role_argument])
    layout = LayoutQualifierNode(
        [
            binding_attribute,
            set_attribute,
            role_attribute,
            AttributeNode("local_size_x", [local_size_argument]),
        ],
        direction=direction,
        annotations={"ignored": ignored_layout_annotation},
        source_location=ignored_layout_source,
    )
    stage = StageNode(
        ShaderStage.COMPUTE,
        _entry_point(),
        layout_qualifiers=[layout],
        execution_config={"local_size_x": local_size, "backend": f"gpu_{suffix}"},
    )
    shader = ShaderNode(
        f"MetadataContract_{suffix}",
        ExecutionModel.COMPUTE_KERNEL,
        stages={ShaderStage.COMPUTE: stage},
        annotations={"ignored": ignored_shader_annotation},
    )

    walked = _assert_walks_once_with(
        shader,
        [
            stage,
            stage.entry_point,
            stage.entry_point.body,
            layout,
            binding_attribute,
            binding_argument,
            binding_argument.literal_type,
            set_attribute,
            set_argument,
            role_attribute,
            role_argument,
            layout.entries[3],
            local_size_argument,
        ],
    )

    assert ignored_attribute_annotation not in walked
    assert ignored_attribute_source not in walked
    assert ignored_layout_annotation not in walked
    assert ignored_layout_source not in walked
    assert ignored_shader_annotation not in walked

    shader.bind_parent_links()

    assert stage.parent is shader
    assert layout.parent is stage
    assert binding_attribute.parent is layout
    assert binding_argument.parent is binding_attribute
    assert binding_argument.literal_type.parent is binding_argument
    assert set_attribute.parent is layout
    assert set_argument.parent is set_attribute
    assert role_attribute.parent is layout
    assert role_argument.parent is role_attribute
    assert layout.entries[3].parent is layout
    assert local_size_argument.parent is layout.entries[3]
    assert ignored_attribute_annotation.parent is None
    assert ignored_attribute_source.parent is None
    assert ignored_layout_annotation.parent is None
    assert ignored_layout_source.parent is None
    assert ignored_shader_annotation.parent is None


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    x_size=st.integers(min_value=1, max_value=1024),
    z_size=st.integers(min_value=1, max_value=1024),
)
def test_generated_stage_execution_config_ast_values_are_structural_metadata(
    suffix,
    x_size,
    z_size,
):
    local_size_x = _int(x_size)
    local_size_y = _id(f"GROUP_SIZE_{suffix}")
    local_size_z = _int(z_size)
    stage = StageNode(
        ShaderStage.COMPUTE,
        _entry_point(),
        execution_config={
            "local_size": (local_size_x, local_size_y, local_size_z),
            "backend_hint": f"compute_{suffix}",
        },
    )
    shader = ShaderNode(
        f"ExecutionConfigMetadata_{suffix}",
        ExecutionModel.COMPUTE_KERNEL,
        stages={ShaderStage.COMPUTE: stage},
    )

    _assert_walks_once_with(
        shader,
        [
            stage,
            local_size_x,
            local_size_x.literal_type,
            local_size_y,
            local_size_z,
            local_size_z.literal_type,
        ],
    )

    shader.bind_parent_links()

    assert local_size_x.parent is stage
    assert local_size_x.literal_type.parent is local_size_x
    assert local_size_y.parent is stage
    assert local_size_z.parent is stage
    assert local_size_z.literal_type.parent is local_size_z


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    list_value=st.integers(min_value=0, max_value=64),
    set_value=st.integers(min_value=0, max_value=64),
)
def test_generated_ast_child_collections_traverse_only_structural_fields(
    suffix,
    list_value,
    set_value,
):
    ignored_annotation = AttributeNode("ignored_annotation", [_int(1)])
    ignored_source = AttributeNode("ignored_source", [_int(2)])
    list_child = AttributeNode("list_child", [_int(list_value)])
    tuple_entry = AttributeNode("tuple_entry", [_id(f"entry_{suffix}")])
    tuple_child = LayoutQualifierNode([tuple_entry], direction="in")
    dict_child = FunctionNode(
        f"helper_{suffix}",
        _int_type(),
        [],
        body=BlockNode([]),
    )
    set_child = AttributeNode("set_child", [_int(set_value)])
    container = MetadataContainerNode(
        list_children=[list_child],
        tuple_children=(tuple_child,),
        dict_children={"helper": dict_child},
        set_children={set_child},
        annotations={"ignored": ignored_annotation},
        source_location=ignored_source,
    )

    walked = _assert_walks_once_with(
        container,
        [
            list_child,
            list_child.arguments[0],
            tuple_child,
            tuple_entry,
            tuple_entry.arguments[0],
            dict_child,
            dict_child.return_type,
            dict_child.body,
            set_child,
            set_child.arguments[0],
        ],
    )

    assert ignored_annotation not in walked
    assert ignored_source not in walked

    container.bind_parent_links()

    assert list_child.parent is container
    assert list_child.arguments[0].parent is list_child
    assert tuple_child.parent is container
    assert tuple_entry.parent is tuple_child
    assert tuple_entry.arguments[0].parent is tuple_entry
    assert dict_child.parent is container
    assert dict_child.return_type.parent is dict_child
    assert dict_child.body.parent is dict_child
    assert set_child.parent is container
    assert set_child.arguments[0].parent is set_child
    assert ignored_annotation.parent is None
    assert ignored_source.parent is None
