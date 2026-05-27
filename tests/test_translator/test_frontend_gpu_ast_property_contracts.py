from hypothesis import given, settings, strategies as st

from crosstl.translator.ast import (
    ArrayType,
    AtomicOpNode,
    BinaryOpNode,
    BlockNode,
    BufferNode,
    BufferOpNode,
    ConstructorPatternNode,
    ExecutionModel,
    ExpressionStatementNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    IdentifierPatternNode,
    LiteralNode,
    LiteralPatternNode,
    MatchArmNode,
    MatchNode,
    MeshOpNode,
    PrimitiveType,
    RayQueryOpNode,
    RayTracingOpNode,
    ReturnNode,
    SamplerNode,
    ShaderNode,
    ShaderStage,
    StageNode,
    StructPatternNode,
    SyncNode,
    TextureNode,
    TextureOpNode,
    TextureResourceNode,
    VectorType,
    WaveOpNode,
)

IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)


def _int_type():
    return PrimitiveType("int")


def _float_type():
    return PrimitiveType("float")


def _uint_type():
    return PrimitiveType("uint")


def _id(name):
    return IdentifierNode(name)


def _int(value):
    return LiteralNode(value, _int_type())


def _float(value):
    return LiteralNode(float(value), _float_type())


def _bool(value):
    return LiteralNode(bool(value), PrimitiveType("bool"))


def _assert_walks_once_with(root, expected_nodes):
    walked = list(root.walk())

    assert len({id(node) for node in walked}) == len(walked)
    for node in expected_nodes:
        assert node in walked


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    level=st.integers(min_value=0, max_value=8),
    atomic_value=st.integers(min_value=1, max_value=64),
    mesh_x=st.integers(min_value=1, max_value=32),
)
def test_generated_gpu_operation_nodes_walk_and_bind_expression_children(
    suffix,
    level,
    atomic_value,
    mesh_x,
):
    texture = TextureNode(
        _id(f"sampleTex_{suffix}"),
        _id(f"linearSampler_{suffix}"),
        _id(f"uv_{suffix}"),
        level=_int(level),
        offset=_id(f"offset_{suffix}"),
    )
    texture_op = TextureOpNode(
        "SampleGrad",
        _id(f"gradTex_{suffix}"),
        [_id(f"gradUv_{suffix}"), _id(f"ddx_{suffix}"), _id(f"ddy_{suffix}")],
        sampler_expr=_id(f"gradSampler_{suffix}"),
    )
    atomic = AtomicOpNode(
        "atomicAdd",
        _id(f"counter_{suffix}"),
        [_int(atomic_value)],
    )
    buffer_op = BufferOpNode(
        "Load",
        _id(f"byteBuffer_{suffix}"),
        [_id(f"byteOffset_{suffix}")],
    )
    wave = WaveOpNode("WaveActiveSum", [_id(f"laneValue_{suffix}")])
    ray = RayTracingOpNode(
        "TraceRay",
        [_id(f"accel_{suffix}"), _uint_type(), _id(f"ray_{suffix}")],
    )
    ray_query = RayQueryOpNode(
        "CommittedRayT",
        _id(f"query_{suffix}"),
        [_id(f"rayFlags_{suffix}")],
    )
    mesh = MeshOpNode("DispatchMesh", [_int(mesh_x), _int(1), _int(1)])
    sync = SyncNode("workgroupBarrier", [_id(f"memoryScope_{suffix}")])

    statements = [
        ExpressionStatementNode(texture),
        ExpressionStatementNode(texture_op),
        ExpressionStatementNode(atomic),
        ExpressionStatementNode(buffer_op),
        ExpressionStatementNode(wave),
        ExpressionStatementNode(ray),
        ExpressionStatementNode(ray_query),
        ExpressionStatementNode(mesh),
        sync,
        ReturnNode(),
    ]
    body = BlockNode(statements)
    entry = FunctionNode("main", PrimitiveType("void"), [], body=body)
    stage = StageNode(ShaderStage.COMPUTE, entry)
    shader = ShaderNode(
        "GpuOperationContracts",
        ExecutionModel.COMPUTE_KERNEL,
        stages={ShaderStage.COMPUTE: stage},
    )

    _assert_walks_once_with(
        shader,
        [
            texture,
            texture.texture_expr,
            texture.sampler_expr,
            texture.coordinates,
            texture.level,
            texture.offset,
            texture_op,
            texture_op.texture_expr,
            texture_op.sampler_expr,
            *texture_op.arguments,
            atomic,
            atomic.target,
            *atomic.arguments,
            buffer_op,
            buffer_op.buffer_expr,
            *buffer_op.arguments,
            wave,
            *wave.arguments,
            ray,
            *ray.arguments,
            ray_query,
            ray_query.query_expr,
            *ray_query.arguments,
            mesh,
            *mesh.arguments,
            sync,
            *sync.arguments,
        ],
    )

    shader.bind_parent_links()

    assert texture.parent is statements[0]
    assert texture.texture_expr.parent is texture
    assert texture.sampler_expr.parent is texture
    assert texture.coordinates.parent is texture
    assert texture.level.parent is texture
    assert texture.offset.parent is texture

    assert texture_op.parent is statements[1]
    assert texture_op.texture_expr.parent is texture_op
    assert texture_op.sampler_expr.parent is texture_op
    assert all(argument.parent is texture_op for argument in texture_op.arguments)

    assert atomic.parent is statements[2]
    assert atomic.target.parent is atomic
    assert atomic.arguments[0].parent is atomic

    assert buffer_op.parent is statements[3]
    assert buffer_op.buffer_expr.parent is buffer_op
    assert buffer_op.arguments[0].parent is buffer_op

    assert wave.parent is statements[4]
    assert wave.arguments[0].parent is wave
    assert ray.parent is statements[5]
    assert all(argument.parent is ray for argument in ray.arguments)
    assert ray_query.parent is statements[6]
    assert ray_query.query_expr.parent is ray_query
    assert ray_query.arguments[0].parent is ray_query
    assert mesh.parent is statements[7]
    assert all(argument.parent is mesh for argument in mesh.arguments)
    assert sync.parent is body
    assert sync.arguments[0].parent is sync


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    array_size=st.integers(min_value=1, max_value=16),
    binding=st.integers(min_value=0, max_value=31),
    set_index=st.integers(min_value=0, max_value=7),
)
def test_generated_resource_nodes_preserve_structural_type_children(
    suffix,
    array_size,
    binding,
    set_index,
):
    buffer_type = ArrayType(VectorType(_float_type(), 4), _int(array_size))
    buffer = BufferNode(
        f"payloadBuffer_{suffix}",
        buffer_type,
        binding=binding,
        set_=set_index,
        access="read_write",
    )
    texture = TextureResourceNode(
        f"storageImage_{suffix}",
        "image2D",
        format="rgba32f",
        binding=binding,
        set_=set_index,
    )
    sampler = SamplerNode(
        f"linearSampler_{suffix}",
        filter_mode="linear",
        address_mode="repeat",
        binding=binding,
    )

    _assert_walks_once_with(
        buffer,
        [
            buffer,
            buffer_type,
            buffer_type.element_type,
            buffer_type.element_type.element_type,
            buffer_type.size,
        ],
    )
    _assert_walks_once_with(texture, [texture])
    _assert_walks_once_with(sampler, [sampler])

    buffer.bind_parent_links()
    texture.bind_parent_links()
    sampler.bind_parent_links()

    assert buffer.buffer_type.parent is buffer
    assert buffer.buffer_type.element_type.parent is buffer.buffer_type
    assert buffer.buffer_type.element_type.element_type.parent is (
        buffer.buffer_type.element_type
    )
    assert buffer.buffer_type.size.parent is buffer.buffer_type
    assert texture.parent is None
    assert sampler.parent is None


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    enabled=st.booleans(),
    threshold=st.integers(min_value=0, max_value=64),
)
def test_generated_pattern_nodes_walk_and_bind_dict_field_patterns(
    suffix,
    enabled,
    threshold,
):
    payload_pattern = ConstructorPatternNode(
        f"Result_{suffix}::Ok",
        [IdentifierPatternNode(f"value_{suffix}")],
    )
    flag_pattern = LiteralPatternNode(_bool(enabled))
    struct_pattern = StructPatternNode(
        f"Command_{suffix}::Dispatch",
        {
            "payload": payload_pattern,
            "enabled": flag_pattern,
        },
        has_rest=True,
    )
    guard = BinaryOpNode(_id(f"value_{suffix}"), ">", _int(threshold))
    call = FunctionCallNode(
        _id(f"emit_{suffix}"),
        [_id(f"value_{suffix}")],
    )
    arm = MatchArmNode(
        struct_pattern,
        guard,
        ExpressionStatementNode(call, is_tail_expression=True),
    )
    match = MatchNode(_id(f"command_{suffix}"), [arm])

    _assert_walks_once_with(
        match,
        [
            match.expression,
            arm,
            struct_pattern,
            payload_pattern,
            payload_pattern.arguments[0],
            flag_pattern,
            flag_pattern.literal,
            guard,
            guard.left,
            guard.right,
            arm.body,
            call,
            call.function,
            *call.arguments,
        ],
    )

    match.bind_parent_links()

    assert arm.parent is match
    assert struct_pattern.parent is arm
    assert struct_pattern.field_patterns["payload"].parent is struct_pattern
    assert payload_pattern.arguments[0].parent is payload_pattern
    assert struct_pattern.field_patterns["enabled"].parent is struct_pattern
    assert flag_pattern.literal.parent is flag_pattern
    assert guard.parent is arm
    assert guard.left.parent is guard
    assert guard.right.parent is guard
    assert arm.body.parent is arm
    assert call.parent is arm.body
    assert call.function.parent is call
    assert call.arguments[0].parent is call
