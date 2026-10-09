"""Non-type template arguments retain precedence in project artifacts."""

import json
import os
import sys
from pathlib import Path

import pytest

from crosstl.backend.Metal.preprocessor import MetalPreprocessor
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

REQUIRE_ENV = "CROSTL_REQUIRE_PROJECT_TEMPLATE_GROUPING"
CASES = {
    "conditional": ("Transpose ? 16 : 32", "Transpose ? 32 : 16", 128, (4, 4)),
    "addition": ("8 + 24", "4 + 12", 128, (4, 4)),
    "shift": ("(1 << 5)", "(1 << 4)", 128, (4, 4)),
    "unparenthesized-shift": ("1 << 5", "1 << 4", 128, (4, 4)),
    "nested": (
        "Transpose ? (2 + 6) : (3 + 13)",
        "Transpose ? 12 : 24",
        16,
        (24, 6),
    ),
}
GUARD = 0x5A1B2C3D
WORDS = [0, 1, 11, 0x7FFFFFFF, 0xFFFFFFFF]


def _source(case):
    rows, cols, threads, _ = CASES[case]
    return f"""#include <metal_stdlib>
using namespace metal;
template<int Rows, int Cols, int Threads, int Reads = (Rows * Cols) / Threads>
struct Tile {{
    static int value() {{ return Reads; }}
}};
template<bool Transpose> struct Layout {{
    using tile_t = Tile<{rows}, {cols}, {threads}>;
    static int value() {{ return tile_t::value(); }}
}};
template<typename T>
kernel void template_values(const device T* values [[buffer(0)]],
                            device uint* results [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {{
    results[2u * i + 4u] = uint(Layout<false>::value()) + values[i];
    results[2u * i + 5u] = uint(Layout<true>::value()) + values[i];
}}
template [[host_name("template_values_uint")]]
kernel decltype(template_values<uint>) template_values<uint>;
"""


def _project(root, target, case, scoped):
    source = _source(case)
    (root / "kernel.metal").write_text(source, encoding="utf-8")
    report = translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("kernel.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=(1, 1, 1),
            entry_points={"kernel.metal": ("template_values_uint",)} if scoped else {},
        ),
        format_output=False,
    )
    report.write_json(root / "report.json")
    payload = report.to_json()
    assert not payload["diagnostics"], payload
    (artifact,) = payload["artifacts"]
    assert artifact["status"] == "translated"
    assert artifact["templateMaterialization"]["status"] == "materialized"
    return source, root / artifact["path"]


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
@pytest.mark.parametrize("scoped", (False, True))
@pytest.mark.parametrize("case", CASES)
def test_project_materialization_keeps_value_grouping(
    tmp_path, monkeypatch, target, scoped, case
):
    grouping = []
    original = MetalPreprocessor.__init__

    def record_initialization(self, *args, **kwargs):
        original(self, *args, **kwargs)
        grouping.append(self.group_non_type_template_substitutions)

    monkeypatch.setattr(MetalPreprocessor, "__init__", record_initialization)
    _, artifact = _project(tmp_path, target, case, scoped)
    assert grouping and all(grouping)
    _compile(artifact.read_text(), target, tmp_path)


@pytest.mark.parametrize("scoped", (False, True))
@pytest.mark.parametrize("case", CASES)
def test_project_template_values_execute(tmp_path, scoped, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required template-value execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, _ = _project(tmp_path, target, case, scoped)
    manifest = build_runtime_artifact_manifest(tmp_path / "report.json")
    assert manifest["success"], manifest
    (tmp_path / "artifacts.json").write_text(json.dumps(manifest))
    package = tmp_path / "package"
    assert build_runtime_package(tmp_path / "artifacts.json", package)["success"]
    loader = build_runtime_loader_manifest(package / "runtime-package.json")
    assert loader["success"] and len(loader["loadUnits"]) == 1, loader
    descriptor = build_native_loader_abi_descriptor(
        loader, load_unit_id=loader["loadUnits"][0]["id"]
    )
    expected_words = (
        [GUARD] * 4
        + [(word + value) & 0xFFFFFFFF for word in WORDS for value in CASES[case][3]]
        + [GUARD] * 4
    )
    inputs = _bound_values(
        descriptor,
        {
            "values": {"dtype": "uint32", "shape": [len(WORDS)], "values": WORDS},
            "results": {
                "dtype": "uint32",
                "shape": [len(expected_words)],
                "values": [GUARD] * 4 + [0xDEADBEEF] * (2 * len(WORDS)) + [GUARD] * 4,
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
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="template_values_uint",
    )


def test_template_grouping_execution_is_required_on_all_platforms():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_project_template_grouping.py" in step
    assert "if:" not in step and "continue-on-error" not in step
    assert "pytest -q -n auto" in step
    assert "--basetemp=" in step and "--junitxml=" in step
    assert "--timeout-seconds 360" in step
