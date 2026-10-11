"""Private resource references specialize to concrete OpenGL buffer bindings."""

import os
import shutil

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_resource_aggregates import CASES, _source, _workload
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_OPENGL_RESOURCE_AGGREGATES"


def _assertions(inputs):
    maximum = max(len(value["values"]) - 1 for value in inputs.values())
    assert 0 <= maximum < 2**31
    return (
        {
            "source": "products.metal",
            "expression": "reference.offset + index",
            "minimum": 0,
            "maximum": maximum,
        },
    )


@pytest.mark.parametrize("case", CASES)
def test_opengl_resource_aggregate_compiles(tmp_path, case):
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not installed")
    inputs, _ = _workload(case)
    _, descriptor, package = _package(
        tmp_path,
        "opengl",
        "int",
        (1, 1, 1),
        source=_source(case),
        software_subgroups=False,
        index_range_assertions=_assertions(inputs),
    )
    generated = (package / descriptor["artifact"]["packagePath"]).read_text()
    assert "_glsl_" in generated
    assert "int64_t offset;" in generated
    _, module = _compile(generated, "opengl", tmp_path)
    assert module.is_file() and module.stat().st_size


@pytest.mark.parametrize("case", CASES)
def test_opengl_resource_aggregate_executes_natively(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required OpenGL resource execution")
    inputs, outputs = _workload(case)
    _, descriptor, package = _package(
        tmp_path,
        "opengl",
        "int",
        (1, 1, 1),
        source=_source(case),
        software_subgroups=False,
        index_range_assertions=_assertions(inputs),
    )
    expected = _bound_values(descriptor, outputs)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        expected,
        {"workgroupCount": [4, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target="opengl",
    )
    assert not request.execution_plan.diagnostics
    _execute(request, expected, tmp_path)


def test_opengl_resource_aggregate_requires_index_range_proof(tmp_path):
    (tmp_path / "source.metal").write_text(_source("cursor"))
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("source.metal",),
            targets=("opengl",),
            workgroup_size=(1, 1, 1),
            output_dir="out",
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["failedCount"] == 1
    assert any(
        d["code"] == "project.translate.opengl-index-type-unsupported"
        for d in report["diagnostics"]
    )
    assert not list((tmp_path / "out").rglob("*.glsl"))
