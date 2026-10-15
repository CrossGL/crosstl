"""Checked finite induction preserves every software collective invocation."""

import os
import sys
from pathlib import Path

import pytest

from crosstl.project import build_native_loader_dispatch_request
from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import DirectXSoftwareSubgroupError
from crosstl.translator.codegen.GLSL_codegen import (
    GLSLCodeGen,
    OpenGLPrivatePointerParameterError,
    OpenGLSoftwareSubgroupError,
)
from crosstl.translator.codegen.loop_induction import loop_constant_bindings
from tests.test_translator.test_directx_software_reductions import _codegen, _source
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_COLLECTIVE_STRIDE_RUNTIME"


@pytest.fixture(params=["directx", "opengl"])
def stride_target(request):
    return request.param


def _generator(target):
    return (
        _codegen() if target == "directx" else GLSLCodeGen(software_subgroup_width=32)
    )


STRIDE_ERRORS = (DirectXSoftwareSubgroupError, OpenGLSoftwareSubgroupError)

CASES = {
    "ascending": ("short step = 0; step < 16; step += 8", (0, 8)),
    "descending": ("short step = 16; step >= 0; step -= 8", (16, 8, 0)),
    "negative-stride": ("int step = 7; step > -2; step += -3", (7, 4, 1)),
    "negative-subtract": ("int step = -3; step < 4; step -= -3", (-3, 0, 3)),
    "reversed-comparison": ("short step = 0; 16 > step; step += 8", (0, 8)),
    "zero-trip": ("short step = 16; step < 16; step += 8", ()),
    "narrow-boundary": (
        "short step = 32750; step < 32766; step += 8",
        (32750, 32758),
    ),
    "unit-boundary": ("short step = 32765; step < 32767; ++step", (32765, 32766)),
    "unit-descending": ("ushort step = 3; step > 0; --step", (3, 2, 1)),
    "local-positive": ("short step = 0; step < limit; step += stride", (0, 8)),
    "local-negative": ("int step = 7; step > -2; step += stride", (7, 4, 1)),
    "doubling": ("int step = 1; step <= 16; step *= 2", (1, 2, 4, 8, 16)),
    "tripling": ("uint step = 1u; step < 28u; step *= 3u", (1, 3, 9, 27)),
    "left-shift": ("uint step = 1u; step < 32u; step <<= 1u", (1, 2, 4, 8, 16)),
    "geometric-reversed": ("int step = 1; 16 >= step; step *= 2", (1, 2, 4, 8, 16)),
    "geometric-zero-trip": ("short step = 16; step < 16; step *= 2", ()),
    "geometric-narrow": ("short step = 4096; step < 16384; step *= 2", (4096, 8192)),
    "geometric-narrow-shift": (
        "ushort step = 8192; step < 32768; step <<= 1",
        (8192, 16384),
    ),
    "local-factor": ("int step = 1; step <= limit; step *= factor", (1, 2, 4, 8, 16)),
    "local-shift": ("uint step = 1u; step < limit; step <<= shift", (1, 4, 16)),
}

LOCAL_CONSTANTS = {
    "local-positive": (
        "const short base = 4; const short stride = base * 2; const short limit = stride * 2;"
    ),
    "local-negative": "const int stride = -3;",
    "local-factor": "const int factor = 1 + 1; const int limit = factor * 8;",
    "local-shift": "const uint shift = 2u; const uint limit = 32u;",
}


def _canonical(header, body="", helpers="", declarations=""):
    return _source(
        "uint",
        f"{declarations} uint result = 0u; for ({header}) {{ {body} result += WaveActiveSum(invocation + uint(step)); }} outputWords[index] = result;",
        helpers,
    )


@pytest.mark.parametrize("case", CASES)
def test_fixed_stride_collectives_compile(tmp_path, case, stride_target):
    generator = _generator(stride_target)
    ast = parse(_canonical(CASES[case][0], declarations=LOCAL_CONSTANTS.get(case, "")))
    generated = generator.generate_stage(ast, "compute")
    assert generated == generator.generate_stage(ast, "compute")
    if CASES[case][1]:
        assert (
            "GroupMemoryBarrierWithGroupSync"
            if stride_target == "directx"
            else "barrier();"
        ) in generated
    assert "WaveActiveSum(" not in generated
    _compile(generated, stride_target, tmp_path)


@pytest.mark.parametrize(
    "header",
    [
        "int step = 0; step < 16; step += 0",
        "int step = 0; step < 16; step -= 8",
        "int step = 16; step > 0; step += 8",
        "short step = 32760; step < 32767; step += 8",
        "short step = -32760; step > -32768; step -= 9",
        "short step = 0; step <= 32767; step += 1",
        "short step = 32768; step < 32769; step += 1",
        "uint step = 4294967290u; step < 4294967295u; step += 8u",
        "uint step = 8u; step >= 0u; step -= 8u",
        "int step = 2147483640; step < 2147483647; step += 8",
        "int step = -2; step < 8u; step += 2",
        "int step = 0; step < 16; step += invocation",
        "int step = 0; step < invocation; step += 8",
        "int step = int(invocation); step < 16; step += 8",
        # A uniform start of unknown value must fit every start: the largest
        # terminating update follows the bound by one stride.
        "int step = int(group.x); step < 2147483641; step += 8",
        "int step = int(group.x); step <= 2147483640; step += 8",
        "int step = int(group.x); step > -2147483642; step -= 8",
        "int step = int(group.x); step < 16u; step += 8",
        "uint step = group.x; step > 0u; step -= 8u",
        "int step = 0; step < 16; step += unknown()",
        "int step = 0; step < 16; step += (2147483647 + 8) - 2147483647",
        "int step = 0; step < 16; step += ((-2147483647 - 1) % -1) + 8",
        "int step = 0; step < 16; step *= 2",
        "int step = 1; step < 16; step *= 1",
        "int step = 1; step < 16; step *= 0",
        "int step = 1; step < 16; step *= -2",
        "int step = -1; step < 16; step *= 2",
        "int step = 16; step > 0; step *= 2",
        "int step = 1; step <= 1073741824; step *= 2",
        "uint step = 1u; step <= 2147483648u; step *= 2u",
        "short step = 16384; step <= 16384; step *= 2",
        "ushort step = 32768; step <= 32768; step <<= 1",
        "int step = 1; step < 16; step <<= 0",
        "int step = 1; step < 16; step <<= -1",
        "uint step = 1u; step < 16u; step <<= 32u",
        "short step = 1; step < 16; step <<= 16",
        "int step = 1; step < 16; step *= invocation",
        "int step = 1; step < invocation; step *= 2",
        "int step = int(group.x); step < 16; step <<= 1",
        "int step = 1; step < 16; step *= (2147483647 + 2) - 2147483647",
        "volatile int step = 1; step < 16; step *= 2",
    ],
)
def test_unproved_or_overflowing_induction_is_rejected(header, stride_target):
    with pytest.raises(STRIDE_ERRORS):
        _generator(stride_target).generate_stage(parse(_canonical(header)), "compute")


# Uniform starts whose value the proof does not know, as MLX's two-pass SDPA
# ``for (int i = block_idx; i < N; i += blocks)``: every visited value precedes
# the bound, so the proof holds for any start of the counter's type. Values
# map a workgroup index to the visited counter values.
UNIFORM_START_CASES = {
    "ascending": (
        "int step = int(group.x); step < 16; step += 8",
        lambda group: range(group, 16, 8),
    ),
    "descending": (
        "int step = 7 - int(group.x); step > -2; step -= 3",
        lambda group: range(7 - group, -2, -3),
    ),
    "boundary": (
        "int step = int(group.x) + 2147483630; step < 2147483640; step += 8",
        lambda group: range(2147483630 + group, 2147483640, 8),
    ),
    "inclusive-boundary": (
        "int step = int(group.x) + 2147483630; step <= 2147483639; step += 8",
        lambda group: range(2147483630 + group, 2147483640, 8),
    ),
    "zero-trip": (
        "int step = int(group.x) + 16; step < 16; step += 8",
        lambda group: (),
    ),
    "unsigned": (
        "uint step = group.x; step < 64u; step += 32u",
        lambda group: range(group, 64, 32),
    ),
}


@pytest.mark.parametrize("case", UNIFORM_START_CASES)
def test_uniform_unknown_starts_compile(tmp_path, case, stride_target):
    generator = _generator(stride_target)
    ast = parse(_canonical(UNIFORM_START_CASES[case][0]))
    generated = generator.generate_stage(ast, "compute")
    assert (
        "GroupMemoryBarrierWithGroupSync"
        if stride_target == "directx"
        else "barrier();"
    ) in generated
    assert "WaveActiveSum(" not in generated
    _compile(generated, stride_target, tmp_path)


@pytest.mark.parametrize(
    "body",
    [
        "step = int(invocation);",
        "++step;",
        "int step = int(invocation);",
        "unknown(step);",
        "int& alias = step; alias = int(invocation);",
        "int* alias = &step; *alias = int(invocation);",
        "if (invocation == 0u) { break; }",
        "if (invocation == 0u) { continue; }",
        "if (invocation == 0u) { return; }",
    ],
)
@pytest.mark.parametrize(
    "header",
    (
        "int step = 0; step < 16; step += 8",
        "int step = 1; step <= 16; step *= 2",
        "int step = 1; step <= 16; step <<= 1",
    ),
)
def test_fixed_stride_proof_retains_mutation_and_exit_checks(
    body, header, stride_target
):
    with pytest.raises(STRIDE_ERRORS):
        _generator(stride_target).generate_stage(
            parse(_canonical(header, body)),
            "compute",
        )


@pytest.mark.parametrize(
    "declarations,body,header",
    (
        (
            "const int factor = 2;",
            "factor = int(invocation);",
            "int step = 1; step <= 16; step *= factor",
        ),
        (
            "const int factor = 2;",
            "unknown(factor);",
            "int step = 1; step <= 16; step *= factor",
        ),
        (
            "const int factor = 2;",
            "int& alias = factor; alias = int(invocation);",
            "int step = 1; step <= 16; step *= factor",
        ),
        (
            "const int factor = 2;",
            "int factor = int(invocation);",
            "int step = 1; step <= 16; step *= factor",
        ),
        (
            "volatile const int factor = 2;",
            "",
            "int step = 1; step <= 16; step *= factor",
        ),
        (
            "const uint shift = 1u;",
            "++shift;",
            "uint step = 1u; step <= 16u; step <<= shift",
        ),
        (
            "const int factor = 2147483647 + 2;",
            "",
            "int step = 1; step <= 16; step *= factor",
        ),
    ),
)
def test_geometric_control_values_remain_immutable(
    declarations, body, header, stride_target
):
    with pytest.raises(STRIDE_ERRORS):
        _generator(stride_target).generate_stage(
            parse(_canonical(header, body, declarations=declarations)), "compute"
        )


def test_constant_expression_stride_compiles(tmp_path, stride_target):
    generated = _generator(stride_target).generate_stage(
        parse(_canonical("int step = 0; step < 4 * 4; step += 2 * 4")),
        "compute",
    )
    _compile(generated, stride_target, tmp_path)


@pytest.mark.parametrize("through_helper", [False, True])
def test_lexical_stride_is_available_before_collective_helper_analysis(
    tmp_path, stride_target, through_helper
):
    call = "collect(invocation)" if through_helper else "WaveActiveSum(invocation)"
    source = _source(
        "uint",
        "const int base = 4; const int stride = base * 2; "
        f"for (uint step = 0u; step < 24u; step += stride) {{ uint result = {call}; outputWords[index] = result; }}",
        "uint collect(uint value) { return WaveActiveSum(value); }",
    )
    generator = _generator(stride_target)
    generated = generator.generate_stage(parse(source), "compute")
    assert generated == generator.generate_stage(parse(source), "compute")
    _compile(generated, stride_target, tmp_path)


@pytest.mark.parametrize(
    "declarations,before,header",
    [
        ("const int stride = 0;", "", "int step = 0; step < 16; step += stride"),
        ("const int stride = -8;", "", "int step = 0; step < 16; step += stride"),
        ("const uint stride = 2u;", "", "int step = -2; step < 8; step += stride"),
        (
            "const short stride = 65544;",
            "",
            "short step = 0; step < 16; step += stride",
        ),
        (
            "const int stride = (2147483647 + 8) - 2147483647;",
            "",
            "int step = 0; step < 16; step += stride",
        ),
        (
            "const int stride = 8;",
            "",
            "short step = 32760; step < 32767; step += stride",
        ),
        (
            "const uint stride = 8u;",
            "",
            "uint step = 0u; step < group.x; step += stride",
        ),
        (
            "const uint stride = invocation;",
            "",
            "uint step = 0u; step < 16u; step += stride",
        ),
        (
            "volatile const uint stride = 8u;",
            "",
            "uint step = 0u; step < 16u; step += stride",
        ),
        (
            "uint stride = 8u;",
            "stride = invocation;",
            "uint step = 0u; step < 16u; step += stride",
        ),
        (
            "uint stride = 8u;",
            "unknown(stride);",
            "uint step = 0u; step < 16u; step += stride",
        ),
        (
            "uint stride = 8u;",
            "uint& alias = stride; alias = invocation;",
            "uint step = 0u; step < 16u; step += stride",
        ),
        (
            "uint stride = 8u;",
            "uint* alias = &stride; *alias = invocation;",
            "uint step = 0u; step < 16u; step += stride",
        ),
        (
            "{ const uint stride = 8u; }",
            "uint stride = invocation;",
            "uint step = 0u; step < 16u; step += stride",
        ),
    ],
)
def test_local_constant_proofs_reject_unchecked_induction(
    stride_target, declarations, before, header
):
    with pytest.raises((*STRIDE_ERRORS, OpenGLPrivatePointerParameterError)):
        _generator(stride_target).generate_stage(
            parse(_canonical(header, declarations=f"{declarations} {before}")),
            "compute",
        )


def test_loop_constant_snapshots_preserve_scope_and_initializer_order():
    source = _source(
        "uint",
        "",
        """
        void inspect() {
            const int stride = 8;
            for (int a = 0; a < 16; a += stride) {}
            { const int stride = 4; for (int b = 0; b < 16; b += stride) {} }
            for (int c = 0; c < 16; c += stride) {}
            if (true) { const int bound = 24; for (int d = 0; d < bound; d += stride) {} }
            for (int e = 0; e < bound; e += stride) {}
            const int bound = 32;
            for (int f = 0; f < bound; f += stride) {}
        }
    """,
    )
    function = parse(source).functions[0]
    snapshots = loop_constant_bindings(
        function, {}, set(), _generator("directx").type_name_string
    )
    assert list(snapshots.values()) == [
        {"stride": 8},
        {"stride": 4},
        {"stride": 8},
        {"stride": 8, "bound": 24},
        {"stride": 8},
        {"stride": 8, "bound": 32},
    ]
    invalidated = loop_constant_bindings(
        function, {}, {"stride"}, _generator("directx").type_name_string
    )
    assert all("stride" not in snapshot for snapshot in invalidated.values())


@pytest.mark.parametrize(
    "index", ["index", "index + 4u", "group.x * 128u + invocation"]
)
def test_collective_store_value_compiles(tmp_path, index):
    helper = "uint collect(uint value) { return WaveActiveSum(value); }"
    generated = _generator("opengl").generate_stage(
        parse(
            _source(
                "uint",
                f"buffer_store(outputWords, {index}, collect(invocation));",
                helper,
            )
        ),
        "compute",
    )
    _compile(generated, "opengl", tmp_path)


@pytest.mark.parametrize(
    "body,extra",
    [
        ("buffer_store(outputWords, index++, collect(invocation));", ""),
        ("buffer_store(outputWords, unknown(index), collect(invocation));", ""),
        (
            "buffer_store(outputWords, index, invocation == 0u ? collect(invocation) : 0u);",
            "",
        ),
        (
            "if (invocation == 0u) { buffer_store(outputWords, index, collect(invocation)); }",
            "",
        ),
        (
            "buffer_store(outputWords, index, collect(invocation));",
            "void buffer_store(RWStructuredBuffer<uint> data, uint offset, uint value) { data[offset] = value; }",
        ),
    ],
)
def test_collective_store_proof_rejects_unchecked_evaluation(body, extra):
    helper = "uint collect(uint value) { return WaveActiveSum(value); } " + extra
    with pytest.raises(OpenGLSoftwareSubgroupError):
        _generator("opengl").generate_stage(
            parse(_source("uint", body, helper)), "compute"
        )


@pytest.mark.parametrize(
    "header",
    [
        "short step = 32760; step <= 32767; ++step",
        "ushort step = 65530; step <= 65535; ++step",
        "short step = -32760; step >= -32768; --step",
    ],
)
def test_narrow_unit_loops_require_a_representable_terminating_update(
    header, stride_target
):
    with pytest.raises(STRIDE_ERRORS):
        _generator(stride_target).generate_stage(parse(_canonical(header)), "compute")


def _metal_source(case):
    return f"""#include <metal_stdlib>
using namespace metal;
uint collect(uint lane) {{
    uint result = 0u;
    {LOCAL_CONSTANTS.get(case, "")}
    for ({CASES[case][0]}) {{ result += simd_sum(lane + uint(step)); }}
    return result;
}}
kernel void accumulate(device uint* output [[buffer(0)]],
                       uint lane [[thread_index_in_threadgroup]],
                       uint3 group [[threadgroup_position_in_grid]]) {{
    if (group.x >= 2u) {{ return; }}
    output[4u + group.x * 64u + lane] = collect(lane);
}}
"""


def _request(root, target, case):
    source, descriptor, package = _package(
        root, target, "uint", (32, 2, 1), source=_metal_source(case)
    )
    guard = 0xDEADBEEF
    expected = [guard] * 201
    for lane in range(128):
        first = (lane % 64) // 32 * 32
        expected[4 + lane] = (
            sum(
                neighbor + step
                for step in CASES[case][1]
                for neighbor in range(first, first + 32)
            )
            & 0xFFFFFFFF
        )
    assert len(descriptor["bindings"]) == 1
    name = descriptor["bindings"][0]["name"]
    inputs = {name: {"dtype": "uint32", "shape": [201], "values": [guard] * 201}}
    outputs = {name: {"dtype": "uint32", "shape": [201], "values": expected}}
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [3, 1, 1], "workgroupSize": [32, 2, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, outputs


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("case", CASES)
def test_fixed_stride_helpers_translate_and_compile(tmp_path, target, case):
    _, request, _ = _request(tmp_path, target, case)
    _compile(request.artifact_path.read_text(), target, tmp_path)


@pytest.mark.parametrize("case", CASES)
def test_fixed_stride_helpers_execute(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required fixed-stride execution")
    target = {"darwin": "metal", "win32": "directx"}.get(sys.platform, "opengl")
    source, request, outputs = _request(tmp_path, target, case)
    _execute(
        request,
        outputs,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="accumulate",
    )


def _uniform_start_source(case):
    return f"""#include <metal_stdlib>
using namespace metal;
kernel void accumulate(device uint* output [[buffer(0)]],
                       uint lane [[thread_index_in_threadgroup]],
                       uint3 group [[threadgroup_position_in_grid]]) {{
    if (group.x >= 2u) {{ return; }}
    uint result = 0u;
    for ({UNIFORM_START_CASES[case][0]}) {{ result += simd_sum(lane + uint(step)); }}
    output[4u + group.x * 64u + lane] = result;
}}
"""


def _uniform_start_request(root, target, source, steps, offset=0):
    source, descriptor, package = _package(
        root, target, "uint", (32, 2, 1), source=source
    )
    guard = 0xDEADBEEF
    expected = [guard] * 201
    for lane in range(128):
        first = (lane % 64) // 32 * 32
        expected[4 + lane] = (
            sum(
                neighbor + (step & 0xFFFFFFFF) + offset
                for step in steps(lane // 64)
                for neighbor in range(first, first + 32)
            )
            & 0xFFFFFFFF
        )
    (binding,) = descriptor["bindings"]
    inputs = {
        binding["name"]: {"dtype": "uint32", "shape": [201], "values": [guard] * 201}
    }
    outputs = {binding["name"]: {"dtype": "uint32", "shape": [201], "values": expected}}
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [3, 1, 1], "workgroupSize": [32, 2, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, outputs


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("case", [*UNIFORM_START_CASES, "shadowed"])
def test_uniform_start_loops_translate_and_compile(tmp_path, target, case):
    _, request, _ = _uniform_start_request(tmp_path, target, *_uniform_start(case))
    generated = request.artifact_path.read_text()
    if target != "metal":
        assert "simd_sum" not in generated and "WaveActiveSum" not in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("case", [*UNIFORM_START_CASES, "shadowed"])
def test_uniform_start_loops_execute(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required fixed-stride execution")
    target = {"darwin": "metal", "win32": "directx"}.get(sys.platform, "opengl")
    source, request, outputs = _uniform_start_request(
        tmp_path, target, *_uniform_start(case)
    )
    _execute(
        request,
        outputs,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="accumulate",
    )


# MLX's two-pass SDPA reuses its key-loop counter ``i`` for the nested loops
# over a thread's head dimensions. The nested counter must stay a distinct
# variable: a translation that wrote the outer counter would change every
# later sum and the outer trip count.
SHADOWED_SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void accumulate(device uint* output [[buffer(0)]],
                       uint lane [[thread_index_in_threadgroup]],
                       uint3 group [[threadgroup_position_in_grid]]) {
    if (group.x >= 2u) { return; }
    uint result = 0u;
    for (int step = int(group.x); step < 16; step += 8) {
        uint inner = 0u;
        for (int step = 0; step < 3; step++) { inner += uint(step); }
        result += simd_sum(lane + uint(step) + inner);
    }
    output[4u + group.x * 64u + lane] = result;
}
"""


def _uniform_start(case):
    if case == "shadowed":
        return SHADOWED_SOURCE, lambda group: range(group, 16, 8), 3
    return _uniform_start_source(case), UNIFORM_START_CASES[case][1], 0


@pytest.mark.parametrize(
    "body",
    [
        "for (int step = 0; step < 2; step++) { result += uint(step); }",
        "{ int step = int(invocation); step += 1; result += uint(step); }",
        "if (invocation == 0u) { int step = 3; ++step; result += uint(step); }",
    ],
)
@pytest.mark.parametrize(
    "header",
    (
        "int step = 0; step < 16; step += 8",
        "int step = int(group.x); step < 16; step += 8",
    ),
)
def test_nested_scopes_may_reuse_the_counter_name(
    tmp_path, body, header, stride_target
):
    generated = _generator(stride_target).generate_stage(
        parse(_canonical(header, body)), "compute"
    )
    _compile(generated, stride_target, tmp_path)


@pytest.mark.parametrize(
    "body",
    [
        "{ step += 1; }",
        "{ int step = 5; } step += 1;",
        "{ int step = 5; { step = 1; } } { step = 2; }",
        "for (int step = 0; step < 2; step++) { } ++step;",
        "for (int other = step++; other < 2; other++) { }",
        "for (int other = 0; other < 2; other++) { step = other; }",
        "for (int other = 0; other < 2; other++) { int& alias = step; alias = 1; }",
    ],
)
def test_counter_writes_outside_a_shadowing_scope_are_rejected(body, stride_target):
    with pytest.raises(STRIDE_ERRORS):
        _generator(stride_target).generate_stage(
            parse(_canonical("int step = 0; step < 16; step += 8", body)), "compute"
        )


def test_fixed_stride_execution_is_required_in_existing_ci_job():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_collective_loop_strides.py" in step
    assert "-n auto" in step and "continue-on-error" not in step
