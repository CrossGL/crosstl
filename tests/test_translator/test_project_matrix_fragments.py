"""Project-scoped matrix fragment policies retain checked numerical behavior."""

import json
import os
import sys
from pathlib import Path

import pytest

from crosstl.project import (
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    load_project_config,
    translate_project,
)
from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import (
    DirectXCooperativeMatrixUnsupportedError,
    HLSLCodeGen,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile

REQUIRE_ENV = "CROSTL_REQUIRE_PROJECT_MATRIX_FRAGMENTS"
CASES = [(kind, width) for kind in ("float", "int", "uint") for width in (32, 64)]


def _source(kind, width):
    matrix = (
        f"CooperativeMatrix<{kind}, 8, 8, subgroup, unspecified, unspecified, "
        "metal_thread_elements, 32, 2, metal_thread_elements_reference_view, "
        "tile_4x4_row_pair, project_source_contract>"
    )
    operations = [
        ("copied", "left"),
        ("added", "cooperative_matrix_add(left, right)"),
        ("subtracted", "cooperative_matrix_subtract(left, right)"),
        ("multiplied", "product(left, right)"),
        ("negated", "cooperative_matrix_negate(left)"),
    ]
    body = "\n".join(
        f"{matrix} {name} = {expression};\n"
        f"results[4u + tid.x * 10u + {2 * i}u] = cooperative_matrix_element({name}, 0);\n"
        f"results[4u + tid.x * 10u + {2 * i + 1}u] = cooperative_matrix_element({name}, second);"
        for i, (name, expression) in enumerate(operations)
    )
    return f"""
shader MatrixFragments {{
    {matrix} product({matrix} left, {matrix} right) {{
        return cooperative_matrix_elementwise_multiply(left, right);
    }}
    compute {{
        @numthreads({width}, 1, 1)
        void main(StructuredBuffer<{kind}> inputs @register(t0),
                  RWStructuredBuffer<{kind}> results @register(u0),
                  uint3 tid @SV_DispatchThreadID) {{
            {matrix} left;
            {matrix} right;
            cooperative_matrix_element(left, 0) = inputs[tid.x * 4u];
            cooperative_matrix_element(left, 1) = inputs[tid.x * 4u + 1u];
            cooperative_matrix_element(right, 0) = inputs[tid.x * 4u + 2u];
            cooperative_matrix_element(right, 1) = inputs[tid.x * 4u + 3u];
            int second = 1;
            {body}
        }}
    }}
}}
"""


def _report(root, target, case=("float", 32), *, option=True, source=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / "matrix.cgl").write_text(source or _source(*case), encoding="utf-8")
    config = (
        '[project]\nsource_roots = ["."]\ninclude = ["matrix.cgl"]\n'
        f'targets = ["{target}"]\noutput_dir = "out"\n'
        '[project.entry_points]\n"matrix.cgl" = ["main"]\n'
    )
    if option is not None:
        config += (
            f"[project.source_options.cgl.target_options.{target}]\n"
            f"cooperative_matrix_software_lowering = {json.dumps(option)}\n"
        )
    (root / "crosstl.toml").write_text(config, encoding="utf-8")
    report = translate_project(load_project_config(root), format_output=False)
    report.write_json(root / "report.json")
    return report.to_json()


def _request(root, target, case):
    data = _report(root, target, case)
    assert data["summary"]["translatedCount"] == 1, data
    assert data["summary"]["failedCount"] == 0 and not data["diagnostics"], data
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
    kind, width = case
    count = width * 3
    inputs = []
    output = []
    for invocation in range(count):
        values = [
            invocation % 13 + 1,
            invocation % 11 + 3,
            invocation % 7 + 2,
            invocation % 5 + 4,
        ]
        if kind == "float":
            values = [value * 0.5 for value in values]
        elif kind == "int":
            values[0] = -values[0]
            values[3] = -values[3]
        left, right = values[:2], values[2:]
        inputs.extend(values)
        output.extend(left)
        output.extend(a + b for a, b in zip(left, right))
        output.extend(a - b for a, b in zip(left, right))
        output.extend(a * b for a, b in zip(left, right))
        output.extend(-a for a in left)
    if kind == "uint":
        output = [word & 0xFFFFFFFF for word in output]
    guard = 37.0 if kind == "float" else 37
    dtype = {"float": "float32", "int": "int32", "uint": "uint32"}[kind]

    def value(words):
        return {"dtype": dtype, "shape": [len(words)], "values": words}

    bound = _bound_values(
        descriptor,
        {
            "inputs": value(inputs),
            "results": value([guard] * 4 + [0] * len(output) + [guard] * 4),
        },
    )
    expected = _bound_values(
        descriptor, {"results": value([guard] * 4 + output + [guard] * 4)}
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        bound,
        expected,
        {"workgroupCount": [3, 1, 1], "workgroupSize": [width, 1, 1]},
        expected_target=target,
    )
    return request, expected


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("case", CASES)
def test_project_matrix_fragments_compile_and_package(tmp_path, target, case):
    request, _ = _request(tmp_path, target, case)
    _compile(request.artifact_path.read_text(), target, tmp_path)
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["project"]["sourceOptions"]["cgl"]["target_options"][target] == {
        "cooperative_matrix_software_lowering": True
    }
    artifact = data["artifacts"][0]
    assert artifact["entryPoint"] == {
        "source": "main",
        "stage": "compute",
        "target": "CSMain" if target == "directx" else "main",
    }
    assert artifact["provenance"]["pipeline"] == "entry-scoped-translate"


@pytest.mark.parametrize("case", CASES)
def test_project_matrix_fragments_execute(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required matrix fragment execution")
    if sys.platform == "darwin":
        pytest.skip(
            "DirectX and OpenGL project policies execute on their native CI hosts"
        )
    target = {"win32": "directx", "linux": "opengl"}[sys.platform]
    request, expected = _request(tmp_path, target, case)
    _execute(request, expected, tmp_path)


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("option", [None, False, "yes", 0, 1])
def test_project_matrix_policy_is_strict_and_opt_in(tmp_path, target, option):
    data = _report(tmp_path, target, option=option)
    assert data["summary"]["failedCount"] == 1
    assert data["summary"]["translatedCount"] == 0
    if option is None or option is False:
        assert any(
            diagnostic["code"]
            == f"project.translate.{target}-cooperative-matrix-unsupported"
            for diagnostic in data["diagnostics"]
        )
    else:
        assert "must be a boolean" in data["artifacts"][0]["error"]
    assert not (tmp_path / data["artifacts"][0]["path"]).exists()


@pytest.mark.parametrize(
    "original,replacement",
    [
        ("tile_4x4_row_pair", "unregistered_mapping"),
        ("project_source_contract", "unspecified"),
        ("metal_thread_elements, 32, 2", "metal_thread_elements, 16, 4"),
        ("float, 8, 8, subgroup", "float, 8, 8, workgroup"),
    ],
)
def test_project_matrix_policy_preserves_fragment_contract(
    tmp_path, original, replacement
):
    data = _report(
        tmp_path, "directx", source=_source("float", 32).replace(original, replacement)
    )
    assert data["summary"]["failedCount"] == 1
    diagnostic = data["diagnostics"][0]
    if original == "metal_thread_elements, 32, 2":
        assert diagnostic["code"] == "project.translate.failed"
        assert "no registered contract" in diagnostic["message"]
    else:
        assert (
            diagnostic["code"]
            == "project.translate.directx-cooperative-matrix-unsupported"
        )


@pytest.mark.parametrize(
    "operation", ["multiply", "load", "store", "multiply_accumulate"]
)
def test_project_matrix_policy_does_not_admit_cross_lane_operations(
    tmp_path, operation
):
    source = _source("float", 32)
    expression = {
        "multiply": "cooperative_matrix_multiply(left, right)",
        "load": "cooperative_matrix_load(left, inputs, 8)",
        "store": "cooperative_matrix_store(results, left, 8)",
        "multiply_accumulate": (
            "cooperative_matrix_multiply_accumulate(left, left, right, right)"
        ),
    }[operation]
    source = source.replace("int second = 1;", f"{expression}; int second = 1;")
    data = _report(tmp_path, "directx", source=source)
    assert data["summary"]["failedCount"] == 1
    diagnostic = data["diagnostics"][0]
    assert (
        diagnostic["code"] == "project.translate.directx-cooperative-matrix-unsupported"
    )
    assert diagnostic["details"]["cooperativeMatrix"]["operation"] == operation
    assert not (tmp_path / data["artifacts"][0]["path"]).exists()


@pytest.mark.parametrize("enabled", [None, 0, 1, "true"])
def test_directx_matrix_policy_setter_rejects_non_boolean(enabled):
    with pytest.raises(TypeError, match="must be a boolean"):
        HLSLCodeGen().set_cooperative_matrix_software_lowering(enabled)


def test_directx_matrix_policy_can_be_changed_between_generations():
    tree = parse(_source("float", 32))
    generator = HLSLCodeGen()
    assert generator.set_cooperative_matrix_software_lowering(True) is generator
    enabled = generator.generate(tree)
    generator.set_cooperative_matrix_software_lowering(False)
    with pytest.raises(DirectXCooperativeMatrixUnsupportedError):
        generator.generate(tree)
    generator.set_cooperative_matrix_software_lowering(True)
    assert generator.generate(tree) == enabled


def test_project_matrix_fragment_native_gate():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_project_matrix_fragments.py" in step
    assert "if:" not in step and "continue-on-error" not in step
    assert "pytest -q -n auto" in step
    assert "--basetemp=" in step and "--junitxml=" in step
