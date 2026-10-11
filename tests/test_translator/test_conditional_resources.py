"""Conditional Metal resources retain their binding and specialization identity."""

import itertools
import json
import os
import sys
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.project import (
    ProjectConfig,
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    translate_project,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile

REQUIRE_ENV = "CROSTL_REQUIRE_CONDITIONAL_RESOURCES"
CONDITIONS = tuple(itertools.product((False, True), repeat=2))
GUARD = 0x5A1B2C3D
WORDS = [0, 1, 11, 0x7FFFFFFF, 0xFFFFFFFF]
SOURCE = """#include <metal_stdlib>
using namespace metal;
constant bool use_left [[function_constant(2)]];
constant bool use_right [[function_constant(6)]];
kernel void conditional_resources(
    const device uint* left [[buffer(3), function_constant(use_left)]],
    const device uint* right [[function_constant(use_right), buffer(9)]],
    constant uint& params [[buffer(5), function_constant(use_right)]],
    const device uint* base [[buffer(1)]],
    device uint* results [[buffer(7)]],
    uint i [[thread_position_in_grid]]) {
    uint value = base[i];
    if (use_left) { value += left[i]; }
    if (use_right) { value += right[i] + params; }
    results[i + 4u] = value;
}
"""


def _project(root, target, values=None, source=SOURCE):
    path = root / "conditional.metal"
    path.write_text(source, encoding="utf-8")
    report = translate_project(
        ProjectConfig(
            root=root,
            include_patterns=(path.name,),
            targets=(target,),
            output_dir="out",
            workgroup_size=(1, 1, 1),
            specialization_constants=(
                dict(zip(("use_left", "use_right"), values))
                if values is not None
                else {}
            ),
        ),
        format_output=False,
    )
    report.write_json(root / "report.json")
    return report.to_json()


def _package(root, target, values=None):
    report = _project(root, target, values)
    assert not report["diagnostics"], report
    assert report["summary"]["translatedCount"] == 1
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
    (root / "descriptor.json").write_text(json.dumps(descriptor, indent=2))
    return descriptor, package


def _request(root, target, values, deferred=False):
    descriptor, package = _package(root, target, None if deferred else values)
    use_left, use_right = values
    expected_words = (
        [GUARD] * 4
        + [
            (word + (19 if use_left else 0) + (36 if use_right else 0)) & 0xFFFFFFFF
            for word in WORDS
        ]
        + [GUARD] * 4
    )
    inputs = _bound_values(
        descriptor,
        {
            "left": {"dtype": "uint32", "shape": [5], "values": [19] * 5},
            "right": {"dtype": "uint32", "shape": [5], "values": [23] * 5},
            "conditional_resources_params" if target == "directx" else "params": {
                "dtype": "uint32",
                "shape": [1],
                "values": [13],
            },
            "base": {"dtype": "uint32", "shape": [5], "values": WORDS},
            "results": {
                "dtype": "uint32",
                "shape": [len(expected_words)],
                "values": [GUARD] * 4 + [0xDEADBEEF] * 5 + [GUARD] * 4,
            },
        },
    )
    expected = _bound_values(
        descriptor,
        {
            "results": {
                "dtype": "uint32",
                "shape": [len(expected_words)],
                "values": expected_words,
            }
        },
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        expected,
        {"workgroupCount": [len(WORDS), 1, 1], "workgroupSize": [1, 1, 1]},
        {2: use_left, 6: use_right} if deferred else None,
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return request, expected, descriptor


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("values", CONDITIONS)
def test_conditional_resources_project_bindings(tmp_path, target, values):
    request, _, descriptor = _request(tmp_path, target, values)
    assert {
        binding["coordinates"]["binding"] for binding in descriptor["bindings"]
    } == {1, 3, 5, 7, 9}
    assert {
        constant["id"]: constant["value"]
        for constant in descriptor["specializationConstants"]
    } == dict(zip((2, 6), values))
    code = request.artifact_path.read_text()
    if target == "metal":
        assert "[[buffer(3)]] [[function_constant(use_left)]]" in code
        assert "[[buffer(9)]] [[function_constant(use_right)]]" in code
        assert (
            "constant uint& params [[buffer(5)]] [[function_constant(use_right)]]"
            in code
        )
    _compile(code, target, tmp_path)


def test_conditional_resources_saved_intermediate_round_trip(tmp_path):
    original = tmp_path / "source.metal"
    original.write_text(SOURCE)
    intermediate = tmp_path / "source.cgl"
    intermediate.write_text(
        translate(str(original), backend="crossgl", format_output=False)
    )
    generated = translate(str(intermediate), backend="metal", format_output=False)
    assert "[[buffer(3)]] [[function_constant(use_left)]]" in generated
    assert "[[buffer(9)]] [[function_constant(use_right)]]" in generated
    assert "[[buffer(5)]] [[function_constant(use_right)]]" in generated
    _compile(generated, "metal", tmp_path)


def test_conditional_resources_directx_requires_concrete_values(tmp_path):
    report = _project(tmp_path, "directx")
    assert report["summary"]["failedCount"] == 1
    assert {item["code"] for item in report["diagnostics"]} == {
        "project.translate.specialization-value-required"
    }


@pytest.mark.parametrize(
    "condition,message",
    (
        ("", "requires one function constant condition"),
        ("use_left, use_right", "accepts at most one value"),
    ),
)
def test_conditional_resources_reject_invalid_attribute_arity(
    tmp_path, condition, message
):
    path = tmp_path / "invalid.cgl"
    path.write_text(f"""shader Invalid {{
        bool use_left @function_constant(2);
        bool use_right @function_constant(6);
        compute {{
            void main(StructuredBuffer<uint> values @buffer(3) @function_constant({condition})) {{}}
        }}
    }}""")
    with pytest.raises(ValueError, match=message):
        translate(str(path), backend="metal", format_output=False)


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("values", CONDITIONS)
def test_conditional_aggregate_constants_compile(tmp_path, target, values):
    source = (
        SOURCE.replace("kernel void", "struct Params { uint bias; };\nkernel void")
        .replace("constant uint& params", "constant Params& params")
        .replace("+ params;", "+ params.bias;")
    )
    report = _project(tmp_path, target, values, source)
    assert not report["diagnostics"], report
    (artifact,) = report["artifacts"]
    assert artifact["status"] == "translated"
    generated = (tmp_path / artifact["path"]).read_text()
    if target == "metal":
        assert (
            "constant Params& params [[buffer(5)]] [[function_constant(use_right)]]"
            in generated
        )
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("values", CONDITIONS)
@pytest.mark.parametrize("deferred", (False, True))
def test_conditional_resources_execute(tmp_path, values, deferred):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required conditional resource execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    if deferred and target == "directx":
        report = _project(tmp_path, target)
        assert report["summary"]["failedCount"] == 1
        assert all(
            item["code"] == "project.translate.specialization-value-required"
            for item in report["diagnostics"]
        )
        return
    request, expected, _ = _request(tmp_path, target, values, deferred)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=SOURCE,
        original_entry="conditional_resources",
    )


def test_conditional_resource_execution_is_required():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_conditional_resources.py" in step
    assert "if:" not in step and "continue-on-error" not in step
    assert "pytest -q -n auto" in step
    assert "--basetemp=" in step and "--junitxml=" in step
    assert "--timeout-seconds 360" in step
