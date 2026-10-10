"""Logical-subgroup matrix products with checked workgroup synchronization."""

import os
import sys
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import (
    DirectXCooperativeMatrixUnsupportedError,
    DirectXSoftwareSubgroupError,
    HLSLCodeGen,
)
from tests.runtime_helpers import _prepare_native_package
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_directx_float_atomics import _compile
from tests.test_translator.test_directx_software_reductions import (
    _execute_words,
    _float,
    _word,
)
from tests.test_translator.test_loop_updates import _execute

REQUIRE_ENV = "CROSTL_REQUIRE_DIRECTX_COOPERATIVE_MATRIX"
MATRIX = (
    "CooperativeMatrix<float, 8, 8, subgroup, unspecified, unspecified, "
    "metal_thread_elements, 32, 2, metal_thread_elements_reference_view, "
    "tile_4x4_row_pair, declared_row_pair_mapping>"
)


def _codegen(**options):
    return HLSLCodeGen(
        cooperative_matrix_software_lowering=True,
        software_subgroup_width=32,
        relative_wave_shuffle_out_of_range="self",
        **options,
    )


def _source(body=None, *, helpers="", width=32, height=4):
    if body is None:
        body = f"""
            {MATRIX} result = cooperative_matrix_multiply_accumulate(left, right, accumulator);
            cooperative_matrix_multiply_accumulate(result, right, left, result);
            outputWords[index * 2u] = asuint(cooperative_matrix_element(result, 0));
            outputWords[index * 2u + 1u] = asuint(cooperative_matrix_element(result, 1));
        """
    return f"""shader SoftwareMatrix {{
        StructuredBuffer<uint> inputWords @register(t0);
        RWStructuredBuffer<uint> outputWords @register(u1);
        {helpers}
        compute {{
            @numthreads({width}, {height}, 1)
            void main(uint invocation @gl_LocalInvocationIndex,
                      uvec3 group @gl_WorkGroupID) {{
                uint index = group.x * {width * height}u + invocation;
                {MATRIX} left;
                {MATRIX} right;
                {MATRIX} accumulator;
                cooperative_matrix_element(left, 0) = asfloat(inputWords[index * 6u]);
                cooperative_matrix_element(left, 1) = asfloat(inputWords[index * 6u + 1u]);
                cooperative_matrix_element(right, 0) = asfloat(inputWords[index * 6u + 2u]);
                cooperative_matrix_element(right, 1) = asfloat(inputWords[index * 6u + 3u]);
                cooperative_matrix_element(accumulator, 0) = asfloat(inputWords[index * 6u + 4u]);
                cooperative_matrix_element(accumulator, 1) = asfloat(inputWords[index * 6u + 5u]);
                {body}
            }}
        }}
    }}"""


def test_software_matrix_compiles_without_native_waves(tmp_path):
    generated = _codegen().generate_stage(parse(_source()), "compute")
    assert generated.count("groupshared float2 ") == 2
    assert generated.count("GroupMemoryBarrierWithGroupSync();") == 2
    assert generated.count("precise float term") == 16
    assert "precise float2 result = accumulator;" in generated
    assert "WaveReadLane" not in generated and "WaveSize" not in generated
    assert "cooperative_matrix_multiply_accumulate(" not in generated
    _compile(generated, tmp_path)


@pytest.mark.parametrize(
    "body",
    [
        "if (invocation < 16u) { CALL; }",
        "if (invocation == 0u) { return; } CALL;",
        "for (uint i = 0u; i < invocation; ++i) { CALL; }",
        f"{MATRIX} result = invocation == 0u ? CALL : accumulator;",
    ],
)
def test_software_matrix_rejects_divergent_barriers(body):
    call = "cooperative_matrix_multiply_accumulate(left, right, accumulator)"
    with pytest.raises(DirectXSoftwareSubgroupError):
        _codegen().generate_stage(parse(_source(body.replace("CALL", call))), "compute")


@pytest.mark.parametrize("width,height", [(16, 1), (33, 1), (16, 3)])
def test_software_matrix_requires_complete_logical_subgroups(width, height):
    with pytest.raises(DirectXSoftwareSubgroupError) as error:
        _codegen().generate_stage(parse(_source(width=width, height=height)), "compute")
    assert error.value.reason == "incomplete-matrix-subgroup"


def test_software_matrix_checks_callee_control_flow(tmp_path):
    helper = f"""{MATRIX} product({MATRIX} a, {MATRIX} b, {MATRIX} c, uint count) {{
        for (uint step = 0u; step < count; ++step) {{
            cooperative_matrix_multiply_accumulate(c, a, b, c);
        }}
        return c;
    }}"""
    body = f"""{MATRIX} result = product(left, right, accumulator, 2u);
        outputWords[index] = asuint(cooperative_matrix_element(result, 0));"""
    generated = _codegen().generate_stage(
        parse(_source(body, helpers=helper)), "compute"
    )
    _compile(generated, tmp_path)
    with pytest.raises(DirectXSoftwareSubgroupError):
        _codegen().generate_stage(
            parse(
                _source(
                    body.replace("accumulator, 2u", "accumulator, invocation"),
                    helpers=helper,
                )
            ),
            "compute",
        )


@pytest.mark.parametrize(
    "replacement,reason",
    [
        ("int", "incompatible-multiply-accumulate-contract"),
        ("uint", "incompatible-multiply-accumulate-contract"),
    ],
)
def test_software_matrix_rejects_other_arithmetic_contracts(replacement, reason):
    source = _source().replace(
        "CooperativeMatrix<float,", f"CooperativeMatrix<{replacement},"
    )
    with pytest.raises(DirectXCooperativeMatrixUnsupportedError) as error:
        _codegen().generate_stage(parse(source), "compute")
    assert error.value.reason == reason


def test_software_matrix_generation_state_is_reset():
    generator = _codegen()
    first = generator.generate_stage(parse(_source()), "compute")
    second = generator.generate_stage(parse(_source()), "compute")
    assert first == second
    generator.set_software_subgroup_width(None)
    plain = generator.generate_stage(
        parse("""shader Plain { compute {
        @numthreads(1, 1, 1) void main() {}
    }}"""),
        "compute",
    )
    assert "__crossgl_software_matrix" not in plain
    assert "groupshared" not in plain


@pytest.mark.parametrize(
    "arguments,reason",
    [
        ("left, right", "invalid-operation-arity"),
        ("left, right, 1.0", "missing-operand-contract"),
        (
            "cooperative_matrix_add(left, right), left, right, accumulator",
            "destination-requires-lvalue",
        ),
    ],
)
def test_software_matrix_rejects_invalid_operands(arguments, reason):
    with pytest.raises(DirectXCooperativeMatrixUnsupportedError) as error:
        _codegen().generate_stage(
            parse(_source(f"cooperative_matrix_multiply_accumulate({arguments});")),
            "compute",
        )
    assert error.value.reason == reason


def test_software_matrix_helpers_avoid_user_identifiers(tmp_path):
    helper = "float __crossgl_software_matrix_mma(float value) { return value; }"
    body = f"""float __crossgl_software_matrix_left = 3.0;
        {MATRIX} result = cooperative_matrix_multiply_accumulate(left, right, accumulator);
        outputWords[index] = asuint(cooperative_matrix_element(result, 0) +
            __crossgl_software_matrix_mma(__crossgl_software_matrix_left));"""
    generated = _codegen().generate_stage(
        parse(_source(body, helpers=helper)), "compute"
    )
    assert "float2 __crossgl_software_matrix_mma_1(" in generated
    assert "groupshared float2 __crossgl_software_matrix_left_1[" in generated
    _compile(generated, tmp_path)


def test_matrix_native_gates_are_required_without_extra_jobs():
    from tests.ci_helpers import assert_paths_covered
    from tools import ci_coverage

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/demo-project-testing.yml").read_text()
    assert (
        workflow.count("tests/test_translator/test_directx_cooperative_matrix.py") == 1
    )
    assert 'CROSTL_REQUIRE_COOPERATIVE_MATRIX_RUNTIME: "1"' in workflow
    assert (
        "CROSTL_REQUIRE_DIRECTX_COOPERATIVE_MATRIX: ${{ runner.os == 'Windows' && '1' || '0' }}"
        in workflow
    )
    for event in ("pull_request", "push"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_directx_cooperative_matrix.py",
        )


def _coordinates(lane):
    # Independently enumerate the four 4x4 tiles in row-major order.
    tile = lane // 8
    row = (tile // 2) * 4 + (lane % 8) // 2
    column = (tile % 2) * 4 + (lane % 2) * 2
    return ((row, column), (row, column + 1))


def _operands(subgroup):
    return [
        [
            [
                ((row * 13 + column * 7 + subgroup * 11 + operand * 5) % 23 - 11) / 4
                for column in range(8)
            ]
            for row in range(8)
        ]
        for operand in range(3)
    ]


def _multiply(left, right, accumulator):
    result = [row[:] for row in accumulator]
    for row in range(8):
        for column in range(8):
            for inner in range(8):
                product = _float(_word(left[row][inner] * right[inner][column]))
                result[row][column] = _float(_word(result[row][column] + product))
    return result


def _case(*, rounding=False):
    words, expected = [], []
    for subgroup in range(12):
        left, right, accumulator = _operands(subgroup)
        if rounding:
            left, right, accumulator = (
                [
                    [
                        _float(_word(value / (3 + column + row)))
                        for column, value in enumerate(values)
                    ]
                    for row, values in enumerate(matrix)
                ]
                for matrix in (left, right, accumulator)
            )
        result = _multiply(right, left, _multiply(left, right, accumulator))
        for lane in range(32):
            coordinates = _coordinates(lane)
            words.extend(
                _word(matrix[row][column])
                for matrix in (left, right, accumulator)
                for row, column in coordinates
            )
            expected.extend(_word(result[row][column]) for row, column in coordinates)
    return words, expected


def test_matrix_reference_exercises_cross_lane_products():
    words, expected = _case()
    assert len(words) == 2304 and len(expected) == 768
    assert len(set(expected)) > 100
    assert (
        len({coordinate for lane in range(32) for coordinate in _coordinates(lane)})
        == 64
    )
    left, right, accumulator = _operands(0)
    actual = _multiply(left, right, accumulator)
    assert actual[0][0] != left[0][0] * right[0][0] + accumulator[0][0]


@pytest.mark.parametrize("shape,rounding", [((32, 4, 1), False), ((16, 8, 1), True)])
def test_software_matrix_executes(tmp_path, monkeypatch, shape, rounding):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for DirectX matrix execution")
    assert sys.platform == "win32", "DirectX matrix execution requires Windows"
    monkeypatch.setenv("CROSTL_REQUIRE_DIRECTX_FLOAT_ATOMICS", "1")
    generated = _codegen().generate_stage(
        parse(_source(width=shape[0], height=shape[1])), "compute"
    )
    words, expected = _case(rounding=rounding)
    actual = _execute_words(tmp_path, generated, words, expected, workgroup_size=shape)
    assert actual == expected


PRECISE_SCALAR = """
StructuredBuffer<uint> inputWords : register(t0);
RWStructuredBuffer<uint> outputWords : register(u1);
[numthreads(32, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    precise float product = asfloat(inputWords[id.x * 3u]) *
                            asfloat(inputWords[id.x * 3u + 1u]);
    precise float result = product + asfloat(inputWords[id.x * 3u + 2u]);
    outputWords[id.x] = asuint(result);
}
"""
PRECISE_OPERANDS = [0x3F800001, 0x3F7FFFFE, 0xBF800000]


def test_precise_scalar_control_distinguishes_contraction():
    left, right, addend = map(_float, PRECISE_OPERANDS)
    assert _word(_float(_word(left * right)) + addend) == 0
    assert _word(left * right + addend) == 0xA8800000


@pytest.mark.parametrize("flags", [(), ("-Gis",)], ids=["precise", "strict"])
def test_precise_scalar_control_compiles(tmp_path, flags):
    _compile(PRECISE_SCALAR, tmp_path, flags=flags)


@pytest.mark.parametrize("flags", [(), ("-Gis",)], ids=["precise", "strict"])
@pytest.mark.parametrize(
    "buffer_byte_offset", [None, 16], ids=["direct", "native-views"]
)
def test_precise_scalar_control_executes(
    tmp_path, monkeypatch, flags, buffer_byte_offset
):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for DirectX precise arithmetic execution")
    assert sys.platform == "win32", "DirectX precise arithmetic requires Windows"
    monkeypatch.setenv("CROSTL_REQUIRE_DIRECTX_FLOAT_ATOMICS", "1")
    # (1 + 2^-23) * (1 - 2^-23) rounds to 1 before subtraction of 1.
    # Contracting the operations instead returns -2^-46, a normal float32 value.
    words = PRECISE_OPERANDS * 32
    expected = [0] * 32
    actual = _execute_words(
        tmp_path,
        PRECISE_SCALAR,
        words,
        expected,
        workgroup_size=(32, 1, 1),
        workgroup_count=(1, 1, 1),
        compile_flags=flags,
        buffer_byte_offset=buffer_byte_offset,
    )
    assert actual == expected


def _metal_source():
    loads = "\n".join(
        f"metal::simdgroup_matrix<float, 8, 8> {name};\n"
        f"reinterpret_cast<thread float2&>({name}.thread_elements()) = "
        f"as_type<float2>(uint2(inputWords[index * 6u + {offset}u], "
        f"inputWords[index * 6u + {offset + 1}u]));"
        for name, offset in (("left", 0), ("right", 2), ("accumulator", 4))
    )
    return f"""#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void matrix_products(const device uint* inputWords [[buffer(0)]],
                            device uint* outputWords [[buffer(1)]],
                            uint invocation [[thread_index_in_threadgroup]],
                            uint3 group [[threadgroup_position_in_grid]]) {{
    uint index = group.x * 128u + invocation;
    {loads}
    simdgroup_multiply_accumulate(accumulator, left, right, accumulator);
    simdgroup_multiply_accumulate(accumulator, right, left, accumulator);
    outputWords[index * 2u] = as_type<uint>(accumulator.thread_elements()[0]);
    outputWords[index * 2u + 1u] = as_type<uint>(accumulator.thread_elements()[1]);
}}
"""


def _project_request(root, target, *, shape=(32, 4, 1), rounding=False):
    source = _metal_source()
    (root / "matrix.metal").write_text(source)
    options = {
        "cooperative_matrix_fragment_mapping": "tile_4x4_row_pair",
        "cooperative_matrix_fragment_mapping_provenance": "declared_row_pair_mapping",
    }
    if target in {"directx", "opengl"}:
        options["target_options"] = {
            target: {
                "cooperative_matrix_software_lowering": True,
                "software_subgroup_width": 32,
                "software_subgroup_applicability": "when-used",
            }
        }
        if target == "directx":
            options["target_options"][target][
                "relative_wave_shuffle_out_of_range"
            ] = "self"
    report = translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("matrix.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=shape,
            source_options={"metal": options},
        ),
        format_output=False,
    )
    report.write_json(root / "report.json")
    assert not report.to_json()["diagnostics"], report.to_json()
    descriptor, package = _prepare_native_package(report, root)
    words, expected = _case(rounding=rounding)
    guard = [0x6A15BEEF] * 32

    def value(values):
        return {"dtype": "uint32", "shape": [len(values)], "values": values}

    outputs = _bound_values(descriptor, {"outputWords": value(expected + guard)})
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(
            descriptor,
            {
                "inputWords": value(words),
                "outputWords": value([0xDEADBEEF] * len(expected) + guard),
            },
        ),
        outputs,
        {"workgroupCount": [3, 1, 1], "workgroupSize": list(shape)},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, outputs


@pytest.mark.parametrize("target", ["directx", "metal"])
def test_project_matrix_packages_preserve_execution_contract(tmp_path, target):
    _, request, _ = _project_request(tmp_path, target)
    assert request.artifact["target"] == target
    if target == "directx":
        generated = request.artifact_path.read_text()
        assert "__crossgl_software_matrix_mma" in generated
        _compile(generated, tmp_path)


def test_project_matrix_native_parity(tmp_path):
    if os.environ.get("CROSTL_REQUIRE_COOPERATIVE_MATRIX_RUNTIME") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_COOPERATIVE_MATRIX_RUNTIME=1 for matrix execution"
        )
    if sys.platform not in {"darwin", "win32"}:
        pytest.skip("This matrix execution contract covers Metal and DirectX")
    target = "metal" if sys.platform == "darwin" else "directx"
    source, request, expected = _project_request(tmp_path, target)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="matrix_products",
    )
