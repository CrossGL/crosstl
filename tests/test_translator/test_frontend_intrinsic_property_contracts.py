from dataclasses import dataclass

from hypothesis import given, settings
from hypothesis import strategies as st

from crosstl.translator.ast import (
    IdentifierNode,
    MeshOpNode,
    RayQueryOpNode,
    RayTracingOpNode,
    VariableNode,
    WaveOpNode,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)


@dataclass(frozen=True)
class IntrinsicCase:
    operation: str
    source: str
    node_type: type
    argument_count: int


WAVE_CASES = (
    IntrinsicCase("WaveGetLaneCount", "WaveGetLaneCount()", WaveOpNode, 0),
    IntrinsicCase("WaveGetLaneIndex", "WaveGetLaneIndex()", WaveOpNode, 0),
    IntrinsicCase("WaveActiveSum", "WaveActiveSum(1)", WaveOpNode, 1),
    IntrinsicCase("WaveActiveProduct", "WaveActiveProduct(2)", WaveOpNode, 1),
    IntrinsicCase("WaveActiveBitAnd", "WaveActiveBitAnd(3)", WaveOpNode, 1),
    IntrinsicCase("WaveActiveBitOr", "WaveActiveBitOr(4)", WaveOpNode, 1),
    IntrinsicCase("WaveActiveBitXor", "WaveActiveBitXor(5)", WaveOpNode, 1),
    IntrinsicCase("WaveActiveMin", "WaveActiveMin(6)", WaveOpNode, 1),
    IntrinsicCase("WaveActiveMax", "WaveActiveMax(7)", WaveOpNode, 1),
    IntrinsicCase("WaveActiveAllTrue", "WaveActiveAllTrue(true)", WaveOpNode, 1),
    IntrinsicCase("WaveActiveAnyTrue", "WaveActiveAnyTrue(false)", WaveOpNode, 1),
    IntrinsicCase("WaveReadLaneAt", "WaveReadLaneAt(8, 0)", WaveOpNode, 2),
    IntrinsicCase("WaveReadLaneFirst", "WaveReadLaneFirst(9)", WaveOpNode, 1),
    IntrinsicCase(
        "WaveShuffleAndFillUp",
        "WaveShuffleAndFillUp(10, 0, 1)",
        WaveOpNode,
        3,
    ),
    IntrinsicCase("WavePrefixSum", "WavePrefixSum(11)", WaveOpNode, 1),
    IntrinsicCase("WavePrefixProduct", "WavePrefixProduct(12)", WaveOpNode, 1),
    IntrinsicCase("QuadReadLaneAt", "QuadReadLaneAt(13, 1)", WaveOpNode, 2),
)

RAY_TRACING_CASES = (
    IntrinsicCase("TraceRay", "TraceRay(1, 2, 3, 4, 5, 6, 7, 8)", RayTracingOpNode, 8),
    IntrinsicCase("ReportHit", "ReportHit(1, 0)", RayTracingOpNode, 2),
    IntrinsicCase("CallShader", "CallShader(2, 3)", RayTracingOpNode, 2),
    IntrinsicCase(
        "AcceptHitAndEndSearch", "AcceptHitAndEndSearch()", RayTracingOpNode, 0
    ),
    IntrinsicCase("IgnoreHit", "IgnoreHit()", RayTracingOpNode, 0),
)

MESH_CASES = (
    IntrinsicCase("SetMeshOutputCounts", "SetMeshOutputCounts(32, 64)", MeshOpNode, 2),
    IntrinsicCase("DispatchMesh", "DispatchMesh(1, 2, 3)", MeshOpNode, 3),
)

RAY_QUERY_CASES = (
    IntrinsicCase("Proceed", "rayQuery.Proceed()", RayQueryOpNode, 0),
    IntrinsicCase("Abort", "rayQuery.Abort()", RayQueryOpNode, 0),
    IntrinsicCase("CandidateType", "rayQuery.CandidateType()", RayQueryOpNode, 0),
    IntrinsicCase("CommittedType", "rayQuery.CommittedType()", RayQueryOpNode, 0),
    IntrinsicCase(
        "TraceRayInline",
        "rayQuery.TraceRayInline(accel, 0, 255, ray)",
        RayQueryOpNode,
        4,
    ),
    IntrinsicCase("CandidateRayT", "rayQuery.CandidateRayT()", RayQueryOpNode, 0),
    IntrinsicCase("CommittedRayT", "rayQuery.CommittedRayT()", RayQueryOpNode, 0),
)


def parse_code(code):
    return Parser(Lexer(code).tokens).parse()


def intrinsic_initializer(code, variable_name="value"):
    ast = parse_code(code)
    entry_point = next(iter(ast.stages.values())).entry_point
    variables = [
        statement
        for statement in entry_point.body.statements
        if isinstance(statement, VariableNode)
    ]
    for variable in variables:
        if variable.name == variable_name:
            return variable.initial_value
    raise AssertionError(f"Variable {variable_name!r} was not parsed")


def shader_with_initializer(suffix, initializer):
    return f"""
    shader IntrinsicIR_{suffix} {{
        compute {{
            void main() {{
                int value = {initializer};
            }}
        }}
    }}
    """


@settings(max_examples=30, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    case=st.sampled_from(WAVE_CASES),
)
def test_generated_wave_intrinsics_parse_to_canonical_wave_ir(suffix, case):
    initializer = intrinsic_initializer(shader_with_initializer(suffix, case.source))

    assert isinstance(initializer, case.node_type)
    assert initializer.operation == case.operation
    assert len(initializer.arguments) == case.argument_count


@settings(max_examples=30, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    case=st.sampled_from(RAY_TRACING_CASES),
)
def test_generated_ray_tracing_intrinsics_parse_to_canonical_ray_ir(suffix, case):
    initializer = intrinsic_initializer(shader_with_initializer(suffix, case.source))

    assert isinstance(initializer, case.node_type)
    assert initializer.operation == case.operation
    assert len(initializer.arguments) == case.argument_count


@settings(max_examples=30, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    case=st.sampled_from(MESH_CASES),
)
def test_generated_mesh_intrinsics_parse_to_canonical_mesh_ir(suffix, case):
    initializer = intrinsic_initializer(shader_with_initializer(suffix, case.source))

    assert isinstance(initializer, case.node_type)
    assert initializer.operation == case.operation
    assert len(initializer.arguments) == case.argument_count


@settings(max_examples=30, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    case=st.sampled_from(RAY_QUERY_CASES),
)
def test_generated_ray_query_methods_parse_to_canonical_ray_query_ir(suffix, case):
    initializer = intrinsic_initializer(shader_with_initializer(suffix, case.source))

    assert isinstance(initializer, case.node_type)
    assert initializer.operation == case.operation
    assert isinstance(initializer.query_expr, IdentifierNode)
    assert initializer.query_expr.name == "rayQuery"
    assert len(initializer.arguments) == case.argument_count
