"""Software quad-group sums keep Metal's quads of four linear invocations."""

import hashlib
import itertools
import json
import math
import os
import struct
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    translate_project,
)
from crosstl.translator import parse
from crosstl.translator.codegen.software_quad_reductions import (
    quad_builtin_expression,
)
from tests.ci_helpers import assert_paths_covered
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_native_runtime import _native_request
from tests.test_translator.test_native_loader_dispatch_integration import _executor
from tests.test_translator.test_software_subgroup_votes import _codegen

REQUIRE_ENV = "CROSTL_REQUIRE_SOFTWARE_QUAD_REDUCTIONS"
SHAPES = [(32, 1, 1), (8, 4, 1), (6, 1, 1)]


def test_quad_gate_is_required_on_every_native_target():
    from tests import workflow_text

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/demo-project-testing.yml").read_text()
    step = workflow_text.workflow_step_section(
        workflow, "Validate software subgroup products"
    )
    assert "test_software_quad_reductions.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "continue-on-error" not in step and "if:" not in step
    for event in ("pull_request", "push"):
        assert_paths_covered(
            workflow_text.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_software_quad_reductions.py",
        )


def _canonical(body, shape=(32, 4, 1), helpers=""):
    return f"""shader Quads {{
        RWStructuredBuffer<uint> results @register(u0);
        {helpers}
        compute {{
            @numthreads({shape[0]}, {shape[1]}, {shape[2]})
            void main(uint invocation @gl_LocalInvocationIndex,
                      uint quad @gl_QuadGroupID,
                      uint quadLane @gl_QuadGroupInvocationID) {{
                {body}
            }}
        }}
    }}"""


def _word(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _float(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]


def _add(left, right):
    """The binary32 sum of two binary32 words; NaNs become the quiet NaN."""
    total = _float(left) + _float(right)
    return 0x7FC00000 if math.isnan(total) else _word(total)


def _quad_sum(words, kind):
    """``((q0 + q1) + q2) + q3`` with binary32 rounding at every step."""
    if kind != "float":
        return sum(words) & 0xFFFFFFFF
    total = words[0]
    for word in words[1:]:
        total = _add(total, word)
    return total


def _association_dependent(words):
    """Whether another order or pairing of ``words`` rounds differently."""

    def canonical(word):
        return 0x7FC00000 if word & 0x7FFFFFFF > 0x7F800000 else word

    results = set()
    for order in itertools.permutations(words):
        results.add(canonical(_quad_sum(list(order), "float")))
        if len(order) == 4:
            results.add(
                canonical(_add(_add(order[0], order[1]), _add(order[2], order[3])))
            )
    return len(results) > 1


# Quads whose binary32 sums round differently under another association.
ORDERED_QUADS = (
    (1.0, 2.0**-24, 2.0**-24, 0.0),
    (2.0**24, 1.0, 1.0, 0.0),
    (1.0, -1.0, 2.0**-30, 2.0**-30),
)


def _input_words(kind, count):
    """``count`` lanes of quads; floats lead with order-sensitive quads."""
    if kind != "float":
        return [
            (
                0x7FFFFFFF
                if lane % 7 == 3
                else (-1 - lane if kind == "int" and lane % 2 else lane * 3 + 1)
            )
            & 0xFFFFFFFF
            for lane in range(count)
        ]
    exact = [
        (1.0, 2.0, 3.0, 4.0),
        (-0.0, -0.0, -0.0, -0.0),
        (-0.0, 0.0, -0.0, -0.0),
        (math.inf, 1.0, 2.0, 3.0),
        (math.inf, -math.inf, 1.0, 1.0),
        (math.nan, 1.0, 2.0, 3.0),
        (0.5, 0.25, -0.75, 8.0),
        (2.0**100, -(2.0**100), 2.0**-100, 0.0),
    ]
    quads = [*ORDERED_QUADS, *exact]
    return [_word(quads[(lane // 4) % len(quads)][lane % 4]) for lane in range(count)]


def _quad_words(words, index, invocation_count):
    """The words of the quad that holds lane ``index`` of its workgroup."""
    group_base = index - index % invocation_count
    quad_base = index - (index - group_base) % 4
    return words[quad_base : min(quad_base + 4, group_base + invocation_count)]


def _expected(words, kind, invocation_count):
    return [
        _quad_sum(_quad_words(words, index, invocation_count), kind)
        for index in range(len(words))
    ]


def test_reference_sums_each_quad_in_lane_order():
    words = _input_words("float", 12)
    sums = _expected(words, "float", 32)
    assert sums[0] == _word(1.0)
    assert sums[4] == _word(2.0**24)
    assert sums[8] == _word(2.0**-29)
    assert _expected([1, 2, 3, 4, 5, 6], "uint", 6) == [10, 10, 10, 10, 11, 11]
    assert all(_association_dependent(words[lane : lane + 4]) for lane in (0, 4, 8))
    assert not _association_dependent(_input_words("float", 44)[12:16])
    assert quad_builtin_expression("gl_QuadGroupID", "index") == "(index / 4u)"
    assert quad_builtin_expression("gl_SubgroupID", "index") is None


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("kind", ["int", "uint", "float"])
@pytest.mark.parametrize("shape", SHAPES)
def test_quad_sums_compile_without_native_quads(tmp_path, target, kind, shape):
    body = (
        f"uint counter = invocation; {kind} total = QuadActiveSum({kind}(counter++));"
        " results[quad * 4u + quadLane] = uint(total) + counter;"
    )
    generated = _codegen(target).generate_stage(
        parse(_canonical(body, shape)), "compute"
    )
    assert generated.count("counter++") == 1
    for native in ("QuadActiveSum(", "subgroupClustered", "QuadReadAcross", "gl_Quad"):
        assert native not in generated
    assert "#extension GL_KHR_shader_subgroup" not in generated
    assert "for (uint quadOffset = 1u; quadOffset < 4u; ++quadOffset)" in generated
    partial = math.prod(shape) % 4 != 0
    assert (f"quadBase + quadOffset >= {math.prod(shape)}u" in generated) == partial
    if kind == "float":
        assert "precise float result = " in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_quad_builtins_lower_to_linear_invocation_quads(target):
    generated = _codegen(target).generate_stage(
        parse(
            _canonical(
                "results[invocation] = lane(quad, quadLane) + QuadActiveSum(invocation);",
                helpers="uint lane(uint quad @gl_QuadGroupID, uint quadLane @gl_QuadGroupInvocationID) { return quad * 4u + quadLane; }",
            )
        ),
        "compute",
    )
    assert "gl_Quad" not in generated
    if target == "directx":
        assert "uint quad = (invocation / 4u);" in generated
        assert "uint quadLane = (invocation % 4u);" in generated
    else:
        assert "(gl_LocalInvocationIndex / 4u)" in generated
        assert "(gl_LocalInvocationIndex % 4u)" in generated


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    "body",
    [
        "if (invocation < 16u) { uint value = QuadActiveSum(invocation); }",
        "if (quadLane == 0u) { return; } uint value = QuadActiveSum(invocation);",
        "uint value = invocation > 0u ? QuadActiveSum(invocation) : 0u;",
    ],
)
def test_quad_sums_reject_unproven_barrier_participation(target, body):
    with pytest.raises(ValueError):
        _codegen(target).generate_stage(parse(_canonical(body)), "compute")


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    "kind", ["bool", "half", "double", "float2", "int64_t", "uint16_t", "int8_t"]
)
def test_quad_sums_reject_inexact_payloads(target, kind):
    with pytest.raises(ValueError):
        _codegen(target).generate_stage(
            parse(
                _canonical(
                    f"{kind} input = {kind}(1); {kind} total = QuadActiveSum(input);"
                )
            ),
            "compute",
        )


def test_native_directx_refuses_quad_sums():
    with pytest.raises(ValueError, match="requires software subgroup lowering"):
        _codegen("directx", software=False).generate_stage(
            parse(_canonical("uint total = QuadActiveSum(invocation);")), "compute"
        )


def _source(kind, size):
    cast = (
        f"as_type<{kind}>(inputWords[index])" if kind != "uint" else "inputWords[index]"
    )
    return f"""#include <metal_stdlib>
using namespace metal;
{kind} quad_total({kind} value, uint quad [[quadgroup_index_in_threadgroup]]) {{
    {kind} total = quad_sum(value);
    return quad < 0xFFFFFFFFu ? total : {kind}(0);
}}
kernel void quads(device uint* inputWords [[buffer(0)]],
                  device uint* outputWords [[buffer(1)]],
                  uint invocation [[thread_index_in_threadgroup]],
                  uint quad [[quadgroup_index_in_threadgroup]],
                  uint quadLane [[thread_index_in_quadgroup]],
                  uint3 group [[threadgroup_position_in_grid]]) {{
    uint index = group.x * {size}u + invocation;
    {kind} total = quad_total({cast}, quad);
    outputWords[index * 3u] = as_type<uint>(total);
    outputWords[index * 3u + 1u] = quad;
    outputWords[index * 3u + 2u] = quadLane;
}}
"""


def _package(root, target, kind, shape):
    source = _source(kind, math.prod(shape))
    (root / "quads.metal").write_text(source, encoding="utf-8")
    options = {}
    if target != "metal":
        target_options = {"software_subgroup_width": 32}
        if target == "directx":
            target_options["relative_wave_shuffle_out_of_range"] = "self"
        options = {"metal": {"target_options": {target: target_options}}}
    report = translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("quads.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=shape,
            source_options=options,
        ),
        format_output=False,
    )
    report.write_json(root / "report.json")
    assert report.to_json()["summary"]["failedCount"] == 0, report.to_json()
    manifest = build_runtime_artifact_manifest(root / "report.json")
    assert manifest["success"], manifest
    (root / "artifacts.json").write_text(json.dumps(manifest))
    package = root / "package"
    assert build_runtime_package(root / "artifacts.json", package)["success"]
    loader = build_runtime_loader_manifest(package / "runtime-package.json")
    assert loader["success"] and len(loader["loadUnits"]) == 1, loader
    descriptor = build_native_loader_abi_descriptor(
        loader, load_unit_id=loader["loadUnits"][0]["id"]
    )
    return source, descriptor, package


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("kind", ["int", "uint", "float"])
def test_quad_sums_translate_through_project_packages(tmp_path, target, kind):
    _, descriptor, package = _package(tmp_path, target, kind, (8, 4, 1))
    generated = (package / descriptor["artifact"]["packagePath"]).read_text()
    if target == "metal":
        assert "quad_sum(" in generated
    else:
        assert "QuadSum" in generated or "quad_sum_" in generated
    _compile(generated, target, tmp_path)


def _check(outputs, expected, names, ordered_lanes):
    output = outputs[names["outputWords"]]
    wanted = expected[names["outputWords"]]
    assert output["shape"] == wanted["shape"]
    for index, (got, want) in enumerate(zip(output["values"], wanted["values"])):
        lane = index // 3
        if index % 3 == 0 and lane in ordered_lanes:
            continue
        if index % 3 == 0 and want & 0x7FFFFFFF > 0x7F800000:
            assert got & 0x7FFFFFFF > 0x7F800000, (index, got, want)
        else:
            assert got == want, (index, got, want)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("kind", ["int", "uint", "float"])
def test_quad_sums_execute_on_device(tmp_path, shape, kind):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native quad sums")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, descriptor, package = _package(tmp_path, target, kind, shape)
    size = math.prod(shape)
    groups = 4
    words = _input_words(kind, size * groups)
    sums = _expected(words, kind, size)
    expected_words = []
    for index, total in enumerate(sums):
        local = index % size
        expected_words.extend([total, local // 4, local % 4])
    guards = [0xBAD00000 + index for index in range(9)]
    # Metal's hardware may associate a quad sum differently, so lanes of
    # association-dependent quads are checked only on the software lowerings.
    ordered_lanes = (
        {
            index
            for index in range(len(words))
            if _association_dependent(_quad_words(words, index, size))
        }
        if target == "metal" and kind == "float"
        else set()
    )

    def payload(values):
        return {"dtype": "uint32", "shape": [len(values)], "values": values}

    inputs = {
        "inputWords": payload(words),
        "outputWords": payload([0xDEADBEEF] * len(expected_words) + guards),
    }
    expected = _bound_values(
        descriptor,
        {"inputWords": payload(words), "outputWords": payload(expected_words + guards)},
    )
    names = {
        binding["scalarLayout"].get("memberName", binding["name"]): binding["name"]
        for binding in descriptor["bindings"]
    }
    readbacks = {
        name: {key: value for key, value in data.items() if key != "values"}
        for name, data in expected.items()
    }
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        readbacks,
        {"workgroupCount": [groups, 1, 1], "workgroupSize": list(shape)},
        expected_target=target,
    )
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    _, module = _compile(request.artifact_path.read_text(), target, compiled)
    assert module.is_file() and module.stat().st_size
    executor = _executor(target)
    records = {}
    try:
        assert executor.is_available(request).available
        result = executor.run(request)
        assert result.status == "ok", result
        records["generated"] = result.outputs
        if target == "metal":
            original = tmp_path / "original"
            original.mkdir()
            original_source, original_module = _compile(source, target, original)
            state, native = _native_request(request)
            native = replace(
                native,
                artifact_path=original_source,
                module_path=original_module,
                entry_point="quads",
            )
            records["originalMetal"] = executor.runtime_adapter.runtime.dispatch(
                None, state, native
            )
        (tmp_path / "evidence.json").write_text(
            json.dumps(
                {
                    "target": target,
                    "kind": kind,
                    "shape": shape,
                    "inputs": inputs,
                    "expected": expected,
                    "records": records,
                    "sourceSha256": (
                        hashlib.sha256(request.artifact_path.read_bytes()).hexdigest()
                    ),
                },
                indent=2,
            )
        )
        for outputs in records.values():
            _check(outputs, expected, names, ordered_lanes)
    finally:
        close = getattr(executor.runtime_adapter.runtime, "close", None)
        if close:
            close()
