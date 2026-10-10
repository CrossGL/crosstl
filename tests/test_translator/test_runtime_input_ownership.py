"""Prepared native requests retain values independently of caller payloads."""

import copy
import os
import sys
from pathlib import Path

import pytest

from crosstl.project import (
    RuntimeAllocationView,
    RuntimeValue,
    build_native_loader_dispatch_request,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_RUNTIME_INPUT_OWNERSHIP"
CASES = ("scalar", "record", "read-write", "offset-view")


def _source(case):
    declaration = (
        "struct Metadata { uint index; uint count; };" if case == "record" else ""
    )
    parameter = (
        "constant Metadata* metadata" if case == "record" else "constant ulong* indices"
    )
    index = (
        "uint index = metadata[0].index;"
        if case == "record"
        else "ulong index = indices[0];"
    )
    guard = (
        "index < metadata[0].count && index < 4u" if case == "record" else "index < 4ul"
    )
    operator = "+=" if case == "read-write" else "="
    return f"""#include <metal_stdlib>
using namespace metal;
{declaration}
kernel void owned_inputs(constant uint* values [[buffer(0)]],
                         {parameter} [[buffer(1)]],
                         device uint* results [[buffer(2)]],
                         uint tid [[thread_position_in_grid]]) {{
    {index}
    uint value = 7u;
    if ({guard}) {{ value = values[uint(index)]; }}
    results[4u + (tid & 3u)] {operator} value + tid;
}}
"""


def _request(root, target, case, typed):
    _, descriptor, package = _package(
        root, target, "uint", (1, 1, 1), source=_source(case), software_subgroups=False
    )
    guard = 0x5A1B2C3D

    def value(words, dtype="uint32", shape=None):
        return {"dtype": dtype, "shape": shape or [len(words)], "values": words}

    initial = 10 if case == "read-write" else 0
    inputs = {
        "values": value([3, 5, 11, 17]),
        "results": value([guard] * 4 + [initial] * 4 + [guard] * 4),
    }
    if case == "record":
        inputs["metadata"] = value([[0, 4]], shape=[1, 2])
    else:
        inputs["indices"] = value([0], "uint64")
    outputs = {
        "results": value(
            [guard] * 4 + [initial + 3 + i for i in range(4)] + [guard] * 4
        )
    }
    expected = _bound_values(descriptor, copy.deepcopy(outputs))
    if typed:
        inputs = {
            name: RuntimeValue(name=name, **data) for name, data in inputs.items()
        }
        outputs = {
            name: RuntimeValue(name=name, **data) for name, data in outputs.items()
        }
    if case == "offset-view":
        values = inputs["values"]
        inputs["values"] = RuntimeValue(
            name="values",
            dtype="uint32",
            shape=(4,),
            values=values.values if typed else values["values"],
            allocation=RuntimeAllocationView("source-values", 256, 16, 288),
        )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        _bound_values(descriptor, outputs),
        {"workgroupCount": [4, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    before = copy.deepcopy(request.fixture.to_json())
    for name, data in {
        **inputs,
        **{f"output-{key}": value for key, value in outputs.items()},
    }.items():
        words = data.values if isinstance(data, RuntimeValue) else data["values"]
        if name == "metadata":
            words[0][0] = 4
        else:
            words[:] = [4294967296 if name == "indices" else 99] * len(words)
    assert request.fixture.to_json() == before
    assert not request.execution_plan.diagnostics
    if case == "offset-view":
        resource = next(
            item
            for item in request.execution_plan.resource_bindings
            if item.allocation.allocation_id == "source-values"
        )
        assert resource.allocation == RuntimeAllocationView(
            "source-values", 256, 16, 288
        )
    return request, expected


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("typed", (False, True))
def test_native_request_owns_project_inputs(tmp_path, target, case, typed):
    request, _ = _request(tmp_path, target, case, typed)
    _compile(request.artifact_path.read_text(), target, tmp_path)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("typed", (False, True))
def test_owned_project_inputs_execute(tmp_path, case, typed):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required input ownership execution")
    target = {"linux": "opengl", "darwin": "metal", "win32": "directx"}[sys.platform]
    request, expected = _request(tmp_path, target, case, typed)
    options = (
        {
            "original_source": _source(case),
            "original_entry": "owned_inputs",
            "metal_compile_flags": ("-Wall", "-Wextra", "-Werror"),
        }
        if target == "metal"
        else {}
    )
    _execute(request, expected, tmp_path, **options)


def test_input_ownership_is_required_in_native_ci():
    workflow = Path(".github/workflows/demo-project-testing.yml").read_text()
    step = workflow.split("- name: Validate collective helper arguments", 1)[1].split(
        "- name:", 1
    )[0]
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_runtime_input_ownership.py" in step
