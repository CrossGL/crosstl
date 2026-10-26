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


def test_opengl_overwritten_null_and_local_pointer_aggregate_runtime_readback(tmp_path):
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not installed")
    source = """#include <metal_stdlib>
using namespace metal;

template <typename T>
struct LocalSlice {
    const thread T* reader;
    thread T* writer;
};

struct NestedCursor {
    const device int* input;
    device int* output;
    threadgroup int* shared_slot;
    LocalSlice<int> local_slice;
};

    kernel void run_all(
        const device int* left [[buffer(0)]],
        device int* first [[buffer(1)]],
        uint tid [[thread_position_in_grid]])
    {
        threadgroup int shared_tile[4];
        int local_scratch[4];
        local_scratch[0] = left[0] + 10;
        local_scratch[1] = left[1] + 20;
        local_scratch[2] = 0;
        local_scratch[3] = 0;

        NestedCursor cursor = {
            nullptr,
            nullptr,
            &shared_tile[0] + 1,
            { &local_scratch[0] + 1, &local_scratch[2] }
        };
        cursor.input = left + 1;
        cursor.output = first;

        NestedCursor alias = cursor;
        alias.local_slice.writer[0] = alias.local_slice.reader[0] + alias.input[0];
        alias.shared_slot[0] = local_scratch[2] + 5;
        alias.output[0] = shared_tile[1];
    }
    """
    _, descriptor, package = _package(
        tmp_path,
        "opengl",
        "int",
        (1, 1, 1),
        source=source,
        software_subgroups=False,
    )
    generated = (package / descriptor["artifact"]["packagePath"]).read_text()
    _, module = _compile(generated, "opengl", tmp_path)
    assert module.is_file() and module.stat().st_size
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required OpenGL resource execution")

    inputs = {"left": {"dtype": "int32", "shape": [4], "values": [3, 7, 11, 13]}}
    outputs = {
        "first": {"dtype": "int32", "shape": [4], "values": [(7 + 20) + 7 + 5, 0, 0, 0]}
    }
    expected = _bound_values(descriptor, outputs)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        {
            name: {**val, "values": [0] * len(val["values"])}
            for name, val in expected.items()
        },
        {"workgroupCount": [1, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target="opengl",
    )
    assert not request.execution_plan.diagnostics
    _execute(request, expected, tmp_path)
