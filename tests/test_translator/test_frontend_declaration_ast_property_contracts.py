from hypothesis import given, settings, strategies as st

from crosstl.translator.ast import (
    ArrayNode,
    AttributeNode,
    BlockNode,
    ConstantNode,
    EnumNode,
    EnumVariantNode,
    ExecutionModel,
    FunctionNode,
    GenericParameterNode,
    IdentifierNode,
    ImportNode,
    LayoutQualifierNode,
    LiteralNode,
    MatrixType,
    NamedType,
    PreprocessorNode,
    PrimitiveType,
    ReturnNode,
    ShaderNode,
    ShaderStage,
    StageNode,
    StructMemberNode,
    StructNode,
    VariableNode,
    VectorType,
)

IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)


def _int_type():
    return PrimitiveType("int")


def _uint_type():
    return PrimitiveType("uint")


def _float_type():
    return PrimitiveType("float")


def _id(name):
    return IdentifierNode(name)


def _int(value):
    return LiteralNode(value, _int_type())


def _float(value):
    return LiteralNode(float(value), _float_type())


def _assert_walks_once_with(root, expected_nodes):
    walked = list(root.walk())

    assert len({id(node) for node in walked}) == len(walked)
    for node in expected_nodes:
        assert node in walked


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    constant_value=st.integers(min_value=0, max_value=1024),
    binding=st.integers(min_value=0, max_value=31),
    set_index=st.integers(min_value=0, max_value=7),
)
def test_generated_shader_root_declarations_walk_all_top_level_surfaces(
    suffix,
    constant_value,
    binding,
    set_index,
):
    import_node = ImportNode(
        f"math_{suffix}",
        alias=f"mx_{suffix}",
        items=[f"sin_{suffix}", f"cos_{suffix}"],
    )
    preprocessor = PreprocessorNode("define", f"FEATURE_{suffix} 1")
    constant = ConstantNode(
        f"COUNT_{suffix}",
        _int_type(),
        _int(constant_value),
        visibility="private",
    )
    global_resource = VariableNode(
        f"colorTex_{suffix}",
        NamedType("Texture2D", [NamedType("float4")]),
        attributes=[
            AttributeNode("set", [_int(set_index)]),
            AttributeNode("binding", [_int(binding)]),
        ],
        qualifiers=["uniform"],
        visibility="public",
    )
    cbuffer_member = StructMemberNode(
        f"exposure_{suffix}",
        _float_type(),
        default_value=_float(1),
        attributes=[AttributeNode("offset", [_int(64)])],
    )
    cbuffer = StructNode(
        f"Frame_{suffix}",
        [
            StructMemberNode(
                f"viewProj_{suffix}",
                MatrixType(_float_type(), 4, 4),
            ),
            cbuffer_member,
        ],
        attributes=[
            AttributeNode("set", [_int(set_index)]),
            AttributeNode("binding", [_int(binding)]),
        ],
    )
    cbuffer.is_cbuffer = True
    helper = FunctionNode(
        f"helper_{suffix}",
        _float_type(),
        [],
        body=BlockNode([ReturnNode(_float(1))]),
    )
    entry = FunctionNode("main", PrimitiveType("void"), [], body=BlockNode([]))
    layout = LayoutQualifierNode(
        [AttributeNode("local_size_x", [_int(8)])],
        direction="in",
    )
    stage = StageNode(
        ShaderStage.COMPUTE,
        entry,
        layout_qualifiers=[layout],
        execution_config={"local_size_x": "8"},
    )
    shader = ShaderNode(
        f"DeclarationRoot_{suffix}",
        ExecutionModel.COMPUTE_KERNEL,
        stages={ShaderStage.COMPUTE: stage},
        functions=[helper],
        global_variables=[global_resource],
        constants=[constant],
        cbuffers=[cbuffer],
        imports=[import_node],
        preprocessors=[preprocessor],
    )

    _assert_walks_once_with(
        shader,
        [
            import_node,
            preprocessor,
            constant,
            constant.const_type,
            constant.value,
            global_resource,
            global_resource.var_type,
            global_resource.var_type.generic_args[0],
            *global_resource.attributes,
            global_resource.attributes[0].arguments[0],
            global_resource.attributes[1].arguments[0],
            cbuffer,
            *cbuffer.members,
            cbuffer.members[0].member_type,
            cbuffer.members[0].member_type.element_type,
            cbuffer_member.default_value,
            *cbuffer_member.attributes,
            cbuffer_member.attributes[0].arguments[0],
            *cbuffer.attributes,
            cbuffer.attributes[0].arguments[0],
            cbuffer.attributes[1].arguments[0],
            helper,
            helper.return_type,
            helper.body,
            helper.body.statements[0],
            helper.body.statements[0].value,
            stage,
            entry,
            entry.body,
            layout,
            layout.entries[0],
            layout.entries[0].arguments[0],
        ],
    )

    shader.bind_parent_links()

    assert import_node.parent is shader
    assert preprocessor.parent is shader
    assert constant.parent is shader
    assert constant.const_type.parent is constant
    assert constant.value.parent is constant
    assert global_resource.parent is shader
    assert global_resource.var_type.parent is global_resource
    assert global_resource.var_type.generic_args[0].parent is global_resource.var_type
    assert all(
        attribute.parent is global_resource for attribute in global_resource.attributes
    )
    assert (
        global_resource.attributes[0].arguments[0].parent
        is global_resource.attributes[0]
    )
    assert cbuffer.parent is shader
    assert cbuffer.members[0].parent is cbuffer
    assert cbuffer.members[0].member_type.parent is cbuffer.members[0]
    assert cbuffer.members[0].member_type.element_type.parent is (
        cbuffer.members[0].member_type
    )
    assert cbuffer_member.parent is cbuffer
    assert cbuffer_member.default_value.parent is cbuffer_member
    assert cbuffer_member.attributes[0].parent is cbuffer_member
    assert all(attribute.parent is cbuffer for attribute in cbuffer.attributes)
    assert helper.parent is shader
    assert helper.body.parent is helper
    assert helper.body.statements[0].parent is helper.body
    assert stage.parent is shader
    assert entry.parent is stage
    assert entry.body.parent is entry
    assert layout.parent is stage
    assert layout.entries[0].parent is layout


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    enum_value=st.integers(min_value=0, max_value=1024),
    alignment=st.sampled_from([4, 8, 16]),
)
def test_generated_struct_and_enum_declarations_bind_metadata_children(
    suffix,
    enum_value,
    alignment,
):
    generic_param = GenericParameterNode(
        f"T_{suffix}",
        constraints=[NamedType(f"Scalar_{suffix}"), NamedType(f"Packable_{suffix}")],
        default_type=NamedType(f"Default_{suffix}"),
    )
    base_type = NamedType(f"Base_{suffix}", [NamedType(f"T_{suffix}")])
    payload_member = StructMemberNode(
        f"payload_{suffix}",
        NamedType(f"T_{suffix}"),
        default_value=_id(f"defaultPayload_{suffix}"),
        attributes=[AttributeNode("location", [_int(0)])],
    )
    color_member = StructMemberNode(
        f"color_{suffix}",
        VectorType(_float_type(), 4),
        attributes=[AttributeNode("semantic", [_id("COLOR0")])],
    )
    struct = StructNode(
        f"Payload_{suffix}",
        [payload_member, color_member],
        generic_params=[generic_param],
        attributes=[AttributeNode("align", [_int(alignment)])],
        inheritance=[base_type],
    )
    read_variant = EnumVariantNode(f"Read_{suffix}", value=_int(enum_value))
    tuple_variant = EnumVariantNode(
        f"Tuple_{suffix}",
        fields=[NamedType(f"Payload_{suffix}"), VectorType(_uint_type(), 2)],
    )
    enum = EnumNode(
        f"Mode_{suffix}",
        [read_variant, tuple_variant],
        underlying_type=_uint_type(),
        attributes=[AttributeNode("repr", [_id("u32")])],
    )
    shader = ShaderNode(
        f"DeclarationMetadata_{suffix}",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[struct, enum],
    )

    _assert_walks_once_with(
        shader,
        [
            struct,
            generic_param,
            *generic_param.constraints,
            generic_param.default_type,
            struct.attributes[0],
            struct.attributes[0].arguments[0],
            base_type,
            base_type.generic_args[0],
            payload_member,
            payload_member.member_type,
            payload_member.default_value,
            payload_member.attributes[0],
            payload_member.attributes[0].arguments[0],
            color_member,
            color_member.member_type,
            color_member.member_type.element_type,
            color_member.attributes[0],
            color_member.attributes[0].arguments[0],
            enum,
            enum.underlying_type,
            enum.attributes[0],
            enum.attributes[0].arguments[0],
            read_variant,
            read_variant.value,
            tuple_variant,
            *tuple_variant.fields,
            tuple_variant.fields[1].element_type,
        ],
    )

    shader.bind_parent_links()

    assert struct.parent is shader
    assert generic_param.parent is struct
    assert all(
        constraint.parent is generic_param for constraint in generic_param.constraints
    )
    assert generic_param.default_type.parent is generic_param
    assert struct.attributes[0].parent is struct
    assert struct.attributes[0].arguments[0].parent is struct.attributes[0]
    assert base_type.parent is struct
    assert base_type.generic_args[0].parent is base_type
    assert payload_member.parent is struct
    assert payload_member.member_type.parent is payload_member
    assert payload_member.default_value.parent is payload_member
    assert payload_member.attributes[0].parent is payload_member
    assert color_member.parent is struct
    assert color_member.member_type.parent is color_member
    assert color_member.member_type.element_type.parent is color_member.member_type
    assert enum.parent is shader
    assert enum.underlying_type.parent is enum
    assert enum.attributes[0].parent is enum
    assert read_variant.parent is enum
    assert read_variant.value.parent is read_variant
    assert tuple_variant.parent is enum
    assert tuple_variant.fields[0].parent is tuple_variant
    assert tuple_variant.fields[1].parent is tuple_variant
    assert tuple_variant.fields[1].element_type.parent is tuple_variant.fields[1]


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    size=st.integers(min_value=1, max_value=16),
)
def test_generated_legacy_array_nodes_preserve_structural_array_aliases(
    suffix,
    size,
):
    element_type = VectorType(_float_type(), 4)
    size_expr = _int(size)
    array = ArrayNode(
        element_type,
        f"weights_{suffix}",
        size_expr,
        semantic="TEXCOORD0",
    )
    struct = StructNode(f"LegacyInput_{suffix}", [array])

    _assert_walks_once_with(
        struct,
        [
            array,
            array.var_type,
            element_type,
            element_type.element_type,
            size_expr,
            array.attributes[0],
        ],
    )

    assert array.element_type is element_type
    assert array.size is size_expr
    assert array.var_type.element_type is element_type
    assert array.var_type.size is size_expr
    assert [attribute.name for attribute in array.attributes] == ["TEXCOORD0"]

    struct.bind_parent_links()

    assert array.parent is struct
    assert array.var_type.parent is array
    assert element_type.parent is array.var_type
    assert element_type.element_type.parent is element_type
    assert size_expr.parent is array.var_type
    assert array.attributes[0].parent is array
