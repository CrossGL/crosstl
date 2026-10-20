"""Restore visible Metal bindings after nested blocks and control-flow scopes."""

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from crosstl import translate
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalIndexedComponentTypeResolutionError,
    MetalToCrossGLConverter,
)
from crosstl.project import build_native_loader_dispatch_request
from tests.ci_helpers import assert_native_step_prerequisites, assert_paths_covered
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import REQUIRE_ENV, _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

GUARD = 0xABCDEF01
CASES = {
    "scalar": ("{ uint values = 9; results[1] = values; }", 9),
    "array": ("{ uint values[2] = {9, 11}; results[1] = values[1]; }", 11),
    "vector": ("{ uint2 values = uint2(9, 11); results[1] = values[1]; }", 11),
    "nested": (
        "{ uint values[2] = {9, 11}; "
        "{ uint values = 5; results[1] = values; } "
        "results[1] += values[1]; }",
        16,
    ),
    "if": (
        "if (values[1] == 3) { uint values = 9; results[1] = values; } "
        "else if (values[0] == 17) { results[1] = values[0]; } "
        "else { results[1] = values[1]; }",
        9,
    ),
    "else": (
        "if (values[1] == 0) { uint values = 9; results[1] = values; } "
        "else if (values[0] == 17) { results[1] = values[0]; } "
        "else { results[1] = values[1]; }",
        17,
    ),
    "for-body": (
        "for (uint i = 0; i < 1; ++i) { uint values = 9; results[1] = values; }",
        9,
    ),
    "for-initializer": (
        "for (uint values = 0; values < 1; ++values) { results[1] = values + 9; }",
        9,
    ),
    "while": (
        "uint i = 0; while (i < values[1]) { "
        "uint values = 9; results[1] = values + i; ++i; }",
        11,
    ),
    "do-while": (
        "uint i = 0; do { uint values = 9; results[1] = values + i; ++i; } "
        "while (i < values[1]);",
        11,
    ),
    "switch": (
        "switch (values[1]) { "
        "case 3: uint values; values = 11; results[1] = values; break; "
        "default: results[1] = 0; break; }",
        11,
    ),
    "helper": ("results[1] = read_value(values);", 17),
}


def _source(case):
    body, _ = CASES[case]
    helper = (
        "uint read_value(const device uint* values) { "
        "{ uint values = 9; if (values != 9) return 0; } return values[0]; }"
        if case == "helper"
        else ""
    )
    return f"""#include <metal_stdlib>
using namespace metal;
{helper}
kernel void scope_values(const device uint* values [[buffer(0)]],
                         device uint* results [[buffer(1)]]) {{
    {body}
    results[0] = values[0];
}}
"""


def _request(root, target, case):
    source, descriptor, package = _package(
        root,
        target,
        "uint",
        (1, 1, 1),
        source=_source(case),
        software_subgroups=False,
        preserve_resource_origins=True,
    )
    expected = [17, CASES[case][1]] + [GUARD] * 6
    inputs = _bound_values(
        descriptor,
        {
            "values": dict(dtype="uint32", shape=[2], values=[17, 3]),
            "results": dict(dtype="uint32", shape=[8], values=[GUARD] * 8),
        },
    )
    outputs = _bound_values(
        descriptor,
        {
            "results": dict(dtype="uint32", shape=[8], values=expected),
        },
    )
    observed = [
        binding["name"]
        for binding in descriptor["bindings"]
        if binding.get("provenance", {}).get("sourceResource", {}).get("parameter")
        == "values"
    ]
    assert len(observed) == 1
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [1, 1, 1], "workgroupSize": [1, 1, 1]},
        observe_inputs=observed,
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, outputs


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_lexical_bindings_translate_and_compile(tmp_path, target, case):
    _, request, _ = _request(tmp_path, target, case)
    _compile(request.artifact_path.read_text(), target, tmp_path)


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_inner_scalar_does_not_borrow_the_outer_buffer_type(tmp_path, target):
    path = tmp_path / "invalid.metal"
    path.write_text(
        _source("scalar").replace("results[1] = values;", "results[1] = values[0];")
    )
    with pytest.raises(MetalIndexedComponentTypeResolutionError, match="scalar value"):
        translate(str(path), backend=target, format_output=False)


@pytest.mark.parametrize("case", ("array", "nested"))
def test_local_arrays_do_not_inherit_buffer_operations(tmp_path, case):
    path = tmp_path / "local-array.metal"
    path.write_text(_source(case))
    crossgl = translate(str(path), backend="crossgl", format_output=False)
    assert "values[1]" in crossgl
    assert "buffer_load(values, 1)" not in crossgl


@pytest.mark.parametrize("scope", ("block", "switch"))
@pytest.mark.parametrize("fail", (False, True))
def test_scope_restores_binding_tables_after_rendering(monkeypatch, scope, fail):
    converter = MetalToCrossGLConverter()
    converter.current_variable_types = {"values": "uint*"}
    converter.current_variable_type_qualifiers = {"values": ("const", "device")}
    converter.current_storage_texture_names = {"image"}
    converter.current_structured_buffer_names = {"values"}
    names = (
        "current_variable_types",
        "current_variable_type_qualifiers",
        "current_storage_texture_names",
        "current_structured_buffer_names",
    )
    original = {name: getattr(converter, name) for name in names}
    expected = copy.deepcopy(original)

    def render(*args):
        converter.current_variable_types["values"] = "uint"
        converter.current_variable_type_qualifiers["values"] = ()
        converter.current_storage_texture_names.clear()
        converter.current_structured_buffer_names.clear()
        assert original == expected
        if fail:
            raise ValueError("invalid inner expression")
        return ""

    monkeypatch.setattr(converter, "generate_function_body", render)
    monkeypatch.setattr(converter, "generate_expression", lambda *args: "0")
    node = SimpleNamespace(expression="0", cases=[], default=["body"])
    method, argument = (
        (converter.generate_scoped_function_body, [])
        if scope == "block"
        else (converter.generate_switch_statement, node)
    )
    if fail:
        with pytest.raises(ValueError, match="invalid inner expression"):
            method(argument, 0, False)
    else:
        method(argument, 0, False)
    assert original == expected
    assert all(getattr(converter, name) is value for name, value in original.items())


@pytest.mark.parametrize("case", CASES)
def test_lexical_bindings_execute_natively(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required scope execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _request(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="scope_values",
    )
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    for record in evidence["records"].values():
        observations = record["details"]["inputObservations"]
        assert len(observations) == 1
        assert all(item["unchanged"] for item in observations.values())


def test_lexical_scopes_native_gate_is_required():
    from tests import workflow_text

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow_text.workflow_step_section(
        workflow, "Validate ordered loop updates"
    )
    assert "test_metal_lexical_scope.py" in step and f'{REQUIRE_ENV}: "1"' in step
    assert "--timeout-seconds" in step and "--junitxml" in step and "-n auto" in step
    assert_native_step_prerequisites(step, job="portable-host")
    assert "continue-on-error" not in step
    for event in ("push", "pull_request"):
        assert_paths_covered(
            workflow_text.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_metal_lexical_scope.py",
        )
