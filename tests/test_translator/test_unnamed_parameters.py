"""Unnamed source parameters retain argument evaluation and strict compilation."""

import os
import shutil
import sys
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.project import build_native_loader_dispatch_request
from crosstl.translator.ast import AttributeNode, ParameterNode, PrimitiveType
from crosstl.translator.codegen import get_codegen
from tests.test_backend.test_metal.test_codegen import convert_without_preprocessing
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_UNNAMED_PARAMETERS"
TARGETS = ("metal", "directx", "opengl")
STRICT_METAL = ("-std=metal3.2", "-Wall", "-Wextra", "-fno-fast-math")
GUARD = 0x5A1B2C3D
INPUTS = [0, 1, 0xFFFFFFFF, 0x80000000, 17]
CASES = {
    "function": (
        "uint select_value(uint value, uint) { return value + 7u; }",
        "uint value = select_value(11u, cursor++);",
        18,
    ),
    "constructor": (
        "struct Config { uint value; Config(uint, uint = 3u) : value(17u) {} };",
        "Config config(cursor++); uint value = config.value;",
        17,
    ),
    "template": (
        "template<typename T> uint select_value(uint value, T) { return value + 7u; }",
        "uint value = select_value(11u, cursor++);",
        18,
    ),
    "tag": (
        "struct Tag { uint field; }; "
        "uint select_value(uint value, Tag) { return value + 7u; }",
        "Tag tag; tag.field = cursor++; uint value = select_value(11u, tag);",
        18,
    ),
    "overload": (
        "uint select_value(uint value, uint) { return value + 7u; } "
        "uint select_value(uint value, float) { return value + 9u; }",
        "uint value = select_value(11u, float(cursor++));",
        20,
    ),
    "collision": (
        "uint select_value(uint _unnamed_param_1, uint) { return _unnamed_param_1 + 7u; }",
        "uint value = select_value(11u, cursor++);",
        18,
    ),
    "named": (
        "uint select_value(uint value, uint extra) { return value + extra; }",
        "uint value = select_value(11u, cursor++);",
        None,
    ),
}


def _source(case):
    declaration, body, _ = CASES[case]
    return f"""#include <metal_stdlib>
using namespace metal;
{declaration}
kernel void parameters(const device uint* inputs [[buffer(0)]],
                       device uint* results [[buffer(1)]],
                       uint tid [[thread_position_in_grid]]) {{
    uint cursor = inputs[tid];
    {body}
    results[4 + 2 * tid] = value;
    results[5 + 2 * tid] = cursor;
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
    )
    output = [GUARD] * 4
    for word in INPUTS:
        value = CASES[case][2]
        output.extend(
            [
                (word + 11 if value is None else value) & 0xFFFFFFFF,
                (word + 1) & 0xFFFFFFFF,
            ]
        )
    output.extend([GUARD] * 4)
    inputs = _bound_values(
        descriptor,
        {
            "inputs": {"dtype": "uint32", "shape": [len(INPUTS)], "values": INPUTS},
            "results": {
                "dtype": "uint32",
                "shape": [len(output)],
                "values": [GUARD] * 4 + [0xDEADBEEF] * (2 * len(INPUTS)) + [GUARD] * 4,
            },
        },
    )
    expected = _bound_values(
        descriptor,
        {
            "results": {"dtype": "uint32", "shape": [len(output)], "values": output},
        },
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        expected,
        {"workgroupCount": [len(INPUTS), 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, expected


def _validate(artifact, work, target):
    return _compile(
        artifact.read_text(), target, work, metal_compile_flags=STRICT_METAL
    )[1]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("target", TARGETS)
def test_unnamed_parameters_project_compile(tmp_path, case, target):
    _, request, _ = _request(tmp_path, target, case)
    generated = request.artifact_path.read_text()
    if target == "metal":
        assert ("__attribute__((unused))" in generated) == (case != "named")
    else:
        assert "maybe_unused" not in generated
    _validate(request.artifact_path, tmp_path, target)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("target", TARGETS)
def test_unnamed_parameters_saved_intermediate(tmp_path, case, target):
    original = tmp_path / "source.metal"
    original.write_text(_source(case))
    intermediate = tmp_path / "source.cgl"
    code = translate(str(original), backend="crossgl", format_output=False)
    assert ("@maybe_unused" in code) == (case != "named")
    intermediate.write_text(code)
    generated = translate(str(intermediate), backend=target, format_output=False)
    _compile(generated, target, tmp_path, metal_compile_flags=STRICT_METAL)


def test_named_parameters_are_not_marked_from_generated_name_prefix():
    code = convert_without_preprocessing(
        "uint probe(uint _unnamed_param_0) { return 17u; }"
    )
    assert "_unnamed_param_0" in code
    assert "@maybe_unused" not in code


def test_unnamed_template_tags_retain_type_and_intent():
    code = convert_without_preprocessing("""
        template<typename T, bool B> uint probe(T value, bool_constant<B>) {
            return value.field;
        }
    """)
    assert "bool_constant<B> _unnamed_param_1 @maybe_unused" in code


@pytest.mark.parametrize("target", (*TARGETS, "cuda", "hip", "slang", "vulkan"))
def test_unused_parameter_annotation_is_not_an_interface_semantic(target):
    parameter = ParameterNode(
        "discarded",
        PrimitiveType("uint"),
        attributes=[AttributeNode("maybe_unused")],
    )
    assert get_codegen(target).semantic_from_node(parameter) is None


def test_named_unused_parameter_warning_is_not_suppressed(tmp_path):
    if sys.platform != "darwin" or not shutil.which("xcrun"):
        pytest.skip("requires the Metal compiler")
    source = tmp_path / "named.metal"
    source.write_text(
        _source("function").replace(
            "uint select_value(uint value, uint)",
            "uint select_value(uint value, uint _unnamed_param_1)",
        )
    )
    generated = translate(str(source), backend="metal", format_output=False)
    assert "__attribute__((unused))" not in generated
    with pytest.raises(AssertionError, match="unused parameter"):
        _compile(generated, "metal", tmp_path, metal_compile_flags=STRICT_METAL)


@pytest.mark.parametrize("case", CASES)
def test_unnamed_parameters_execute(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required unnamed parameter execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _request(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="parameters",
        metal_compile_flags=STRICT_METAL,
        validate=_validate,
    )


def test_unnamed_parameter_execution_is_required():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_unnamed_parameters.py" in step
    assert "if:" not in step and "continue-on-error" not in step
    assert "pytest -q -n auto" in step
    assert "--basetemp=" in step and "--junitxml=" in step
