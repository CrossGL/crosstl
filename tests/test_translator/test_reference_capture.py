"""Preserve binding-time storage identity for aggregate accessor references."""

import os
import sys
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.backend.Metal.preprocessor import MetalPreprocessor, MetalStructMethodError
from crosstl.project import build_native_loader_dispatch_request
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_REFERENCE_CAPTURE"
CASES = (
    "stable",
    "changed-index",
    "side-effect",
    "nested",
    "field-index",
    "shared-alias",
    "loop",
    "prefix",
    "conversion",
    "name-collision",
    "index-shadow",
    "readonly",
    "direct",
)
INPUTS = (0, 11, 0xFFFFFFFF, 0x80000000, 0x89ABCDEF)
GUARD = [0x58A5B6C7] * 4


def _source(case):
    index = "uint index = 1u;"
    argument = "index"
    index_type = "uint"
    backing = "fragments[index]"
    setup = ""
    after_binding = ""
    operation = "accum[0] += uint(delta); accum[1] *= 3u;"
    members = "Tile tile;"
    receiver = "tile"
    qualifier = "thread"
    return_type = "void"
    call = "engine.update(3u);"
    if case in ("changed-index", "nested"):
        after_binding = "index = 0u;"
    if case == "side-effect":
        argument = "index++"
    if case == "nested":
        members = "Holder holder;"
        receiver = "holder.tile"
    if case == "field-index":
        index = "uint index = 0u;"
        backing = "fragments[index + bias]"
        setup = "engine.tile.bias = 1u;"
        after_binding = "tile.bias = 0u;"
    if case == "conversion":
        index = "uint index = 257u;"
        index_type = "uchar"
    if case == "prefix":
        operation = "++accum[0]; accum[1]++; accum[0] += uint(delta);"
    if case == "index-shadow":
        operation = (
            "{ uint index = 0u; accum[0] += uint(delta) + index; } accum[1] *= 3u;"
        )
    if case == "name-collision":
        index += " uint crosstl_reference_argument_0 = 3u; uint crosstl_reference_index_0 = 1u;"
        operation = "accum[0] += uint(delta) + crosstl_reference_argument_0 - 3u; accum[1] *= 3u; index += crosstl_reference_index_0 - 1u;"
    binding = f"thread auto& accum = {receiver}.at({argument});"
    body = f"{index} {binding} {after_binding} {operation} seen = index;"
    if case == "shared-alias":
        body = """uint index = 1u;
        thread auto& first = tile.at(index);
        thread auto& second = tile.at(index);
        first[0] += uint(delta);
        second[1] += first[0];
        seen = index;"""
    if case == "loop":
        body = """for (uint index = 0u; index < 2u; index++) {
            thread auto& accum = tile.at(index);
            accum[0] += uint(delta); accum[1] *= 3u; seen++;
        }"""
    if case == "readonly":
        qualifier = "const thread"
        return_type = "uint"
        body = """uint index = 1u; thread auto& accum = tile.at(index);
        index = 0u; return accum[0] + accum[1] + uint(delta) + index;"""
        call = "engine.seen = engine.update(3u);"
    if case == "direct":
        body = "tile.at(1u)[0] += uint(delta); tile.at(1u)[1] *= 3u; seen = 1u;"
    return f"""#include <metal_stdlib>
using namespace metal;
struct Tile {{
    uint2 fragments[2];
    uint bias;
    thread uint2& at({index_type} index) thread {{ return {backing}; }}
    const thread uint2& at({index_type} index) const thread {{ return {backing}; }}
}};
struct Holder {{ Tile tile; }};
struct Engine {{
    {members}
    uint seen;
    template<typename T> {return_type} update(T delta) {qualifier} {{ {body} }}
}};
kernel void reference_capture(const device uint* values [[buffer(0)]],
                              device uint* results [[buffer(1)]],
                              uint tid [[thread_position_in_grid]]) {{
    Engine engine;
    engine.{receiver}.fragments[0] = uint2(99u, 101u);
    engine.{receiver}.fragments[1] = uint2(values[tid], 7u);
    engine.{receiver}.bias = 0u;
    engine.seen = 0u;
    {setup}
    {call}
    results[4 + 5 * tid] = engine.{receiver}.fragments[1][0];
    results[5 + 5 * tid] = engine.{receiver}.fragments[1][1];
    results[6 + 5 * tid] = engine.{receiver}.fragments[0][0];
    results[7 + 5 * tid] = engine.{receiver}.fragments[0][1];
    results[8 + 5 * tid] = engine.seen;
}}
"""


def _expected(case, initial):
    first, second, other_first, other_second, seen = initial + 3, 21, 99, 101, 1
    if case in ("changed-index", "nested", "field-index"):
        seen = 0
    if case == "side-effect":
        seen = 2
    if case == "conversion":
        seen = 257
    if case == "prefix":
        first, second = initial + 4, 8
    if case == "shared-alias":
        second = 7 + first
    if case == "loop":
        other_first, other_second, seen = 102, 303, 2
    if case == "readonly":
        first, second, seen = initial, 7, initial + 10
    return [
        value & 0xFFFFFFFF for value in (first, second, other_first, other_second, seen)
    ]


def _request(root, target, case):
    source, descriptor, package = _package(
        root,
        target,
        "uint",
        (1, 1, 1),
        source=_source(case),
        software_subgroups=False,
    )
    expected = (
        GUARD
        + [word for initial in INPUTS for word in _expected(case, initial)]
        + GUARD
    )
    inputs = _bound_values(
        descriptor,
        {
            "values": {
                "dtype": "uint32",
                "shape": [len(INPUTS)],
                "values": list(INPUTS),
            },
            "results": {
                "dtype": "uint32",
                "shape": [len(expected)],
                "values": GUARD + [0xDEADBEEF] * (len(INPUTS) * 5) + GUARD,
            },
        },
    )
    outputs = _bound_values(
        descriptor,
        {
            "results": {
                "dtype": "uint32",
                "shape": [len(expected)],
                "values": expected,
            },
        },
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [len(INPUTS), 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, outputs


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_reference_capture_project_compiles(tmp_path, target, case):
    _, request, _ = _request(tmp_path, target, case)
    _compile(
        request.artifact_path.read_text(),
        target,
        tmp_path,
        metal_compile_flags=("-Wall", "-Wextra"),
    )


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_reference_capture_saved_intermediate(tmp_path, target, case):
    source = tmp_path / "source.metal"
    source.write_text(_source(case))
    intermediate = tmp_path / "source.cgl"
    intermediate.write_text(
        translate(str(source), backend="crossgl", format_output=False)
    )
    generated = translate(str(intermediate), backend=target, format_output=False)
    _compile(generated, target, tmp_path, metal_compile_flags=("-Wall", "-Wextra"))


@pytest.mark.parametrize("case", CASES)
def test_reference_capture_executes(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native reference capture")
    from tests.test_translator.test_loop_updates import _execute

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    source, request, expected = _request(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="reference_capture",
        metal_compile_flags=("-Wall", "-Wextra"),
    )


@pytest.mark.parametrize(
    "operation",
    (
        "thread uint2* escaped = &accum;",
        "thread uint& escaped = accum[0];",
        "{ uint2 accum = uint2(0); seen = accum[0]; }",
    ),
)
def test_reference_capture_rejects_unproven_alias_uses(operation):
    source = _source("stable").replace(
        "accum[0] += uint(delta); accum[1] *= 3u;", operation
    )
    with pytest.raises(MetalStructMethodError) as error:
        MetalPreprocessor().preprocess(source)
    assert error.value.missing_capabilities == ("struct.reference-return",)


@pytest.mark.parametrize("operation", ("++accum[0];", "accum[0] += 1u;"))
def test_reference_capture_retains_deduced_constness(operation):
    source = _source("readonly").replace(
        "index = 0u; return accum[0]", "index = 0u; " + operation + " return accum[0]"
    )
    with pytest.raises(MetalStructMethodError):
        MetalPreprocessor().preprocess(source)


def test_reference_capture_is_required_on_all_native_platforms():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_reference_capture.py" in step
    assert "pytest -q -n auto" in step
    assert "if:" not in step and "continue-on-error" not in step
    assert "--timeout-seconds 360" in step
