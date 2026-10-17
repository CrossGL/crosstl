"""Explicit auto pointers preserve source types before overload selection."""

import os
import sys
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalAutoTypeInferenceError,
    MetalSourceOverloadResolutionError,
)
from crosstl.project import build_native_loader_dispatch_request, translate_project
from tests.test_backend.test_metal.test_codegen import (
    convert_without_preprocessing,
    normalize,
    parse_crossgl,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_AUTO_POINTER"
GUARD = 0x5A1B2C3D


@pytest.mark.parametrize("space", ("device", "constant", "threadgroup", "thread"))
@pytest.mark.parametrize("readonly", (False, True))
def test_explicit_auto_pointer_infers_pointee_and_address_space(space, readonly):
    prefix = "const " if readonly else ""
    source = f"""
    float read_value({prefix}{space} float* values, uint offset) {{
        {space} auto* cursor = values + offset;
        {space} auto* nested = 1u + cursor;
        return nested[0];
    }}
    """
    output = convert_without_preprocessing(source)
    expected_prefix = "const " if readonly and space != "constant" else ""
    assert f"{expected_prefix}{space} float* cursor = values + offset;" in normalize(
        output
    )
    assert f"{expected_prefix}{space} float* nested = 1u + cursor;" in normalize(output)
    assert parse_crossgl(output) is not None


def test_explicit_auto_pointer_retains_selected_callable_qualifiers():
    source = """
    const device uint* locate(const device uint* values) { return values + 1u; }
    uint read_value(const device uint* values) {
        device auto* cursor = locate(values);
        return cursor[0];
    }
    """
    assert "const device uint* cursor = locate(values);" in normalize(
        convert_without_preprocessing(source)
    )


def test_explicit_auto_pointer_retains_callable_return_alias_qualifiers():
    source = """
    using Pointer = const device uint*;
    Pointer locate(const device uint* values) { return values + 1u; }
    uint read_value(const device uint* values) {
        device auto* cursor = locate(values);
        return cursor[0];
    }
    """
    assert "const device uint* cursor = locate(values);" in normalize(
        convert_without_preprocessing(source)
    )


def test_explicit_auto_pointer_retains_conditional_pointee_const():
    source = """
    uint read_value(const device uint* first, device uint* second, bool select) {
        device auto* cursor = select ? first : second;
        return cursor[0];
    }
    """
    assert "const device uint* cursor = select ? first : second;" in normalize(
        convert_without_preprocessing(source)
    )


def test_explicit_auto_pointer_does_not_accept_unresolved_return_pointee():
    source = """
    device auto* locate(device uint* values) { return values; }
    uint read_value(device uint* values) {
        device auto* cursor = locate(values);
        return cursor[0];
    }
    """
    with pytest.raises(MetalAutoTypeInferenceError, match="pointee type remains auto"):
        convert_without_preprocessing(source)


@pytest.mark.parametrize("declaration", ("auto*", "auto**"))
def test_explicit_auto_pointer_keeps_indirection_depth(declaration):
    source = f"""
    uint read_value(thread uint** values) {{
        thread {declaration} cursor = values;
        return cursor[0][0];
    }}
    """
    assert "thread uint** cursor = values;" in normalize(
        convert_without_preprocessing(source)
    )


@pytest.mark.parametrize(
    "declaration,initializer,reason",
    (
        ("auto*", " = 1u", "at least 1 pointer"),
        ("auto**", " = values", "at least 2 pointer"),
        ("auto*", "", "at least 1 pointer"),
        ("auto* const", " = values", "distinct pointer-level qualifiers"),
    ),
)
def test_invalid_or_unrepresented_auto_pointer_is_diagnostic(
    tmp_path, declaration, initializer, reason
):
    source = f"""
    kernel void probe(device uint* values [[buffer(0)]]) {{
        device {declaration} cursor{initializer};
        values[0] = 1u;
    }}
    """
    with pytest.raises(MetalAutoTypeInferenceError, match=reason):
        convert_without_preprocessing(source)
    (tmp_path / "probe.metal").write_text(source)
    report = translate_project(tmp_path, targets=["cgl"], output_dir="out").to_json()
    assert report["summary"]["translatedCount"] == 0
    assert report["summary"]["diagnosticsByCode"] == {
        "project.translate.metal-auto-type-unresolved": 1
    }


def test_explicit_auto_pointer_cannot_bind_readonly_storage_to_mutable_helper():
    source = """
    void write_value(device uint* values, uint offset) { values[offset] = 7u; }
    void write_value(device uint* values, uint3 offset) { values[offset.x] = 9u; }
    void probe(const device uint* values) {
        device auto* cursor = values + 1u;
        write_value(cursor, 0u);
    }
    """
    with pytest.raises(MetalSourceOverloadResolutionError):
        convert_without_preprocessing(source)


def test_explicit_auto_pointer_shadowing_and_comma_declarations():
    source = """
    uint probe(const device uint* values, thread float* other) {
        device auto* cursor = values, *next = values + 1u;
        uint result = cursor[0];
        { thread auto* cursor = other; result += uint(cursor[0]); }
        return result + next[0] + cursor[0];
    }
    """
    output = normalize(convert_without_preprocessing(source))
    assert "const device uint* cursor = values;" in output
    assert "const device uint* next = values + 1u;" in output
    assert "thread float* cursor = other;" in output


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_explicit_auto_pointer_selects_integer_overload(tmp_path, target):
    source = """#include <metal_stdlib>
    using namespace metal;
    long read_value(long index, const constant long* pointer) { return pointer[index]; }
    long read_value(uint3 index, const constant long* pointer) { return pointer[index.x] + 7; }
    kernel void pointer_alias(const constant long* values [[buffer(0)]],
                             device uint* results [[buffer(1)]]) {
        const constant auto* cursor = values + 1;
        results[0] = uint(read_value(0L, cursor));
    }
    """
    path = tmp_path / "probe.metal"
    path.write_text(source)
    _compile(
        translate(str(path), backend=target, format_output=False), target, tmp_path
    )


def _source(case):
    body = {
        "device": (
            """
        uint offset = 0u;
        const device auto* cursor = values + offset++;
        device auto* destination = results + 4u;
        destination[0] = read_value(1u, cursor) + offset;
        destination += 1u;
        destination[0] = cursor[2];
        """
        ),
        "private": (
            """
        uint local = values[1];
        thread auto* cursor = &local;
        cursor[0] += 1u;
        results[4] = local;
        results[5] = values[2];
        """
        ),
        "array": (
            """
        uint local[2];
        local[0] = values[1];
        local[1] = values[2];
        thread auto* cursor = local;
        cursor[0] += 1u;
        results[4] = local[0];
        results[5] = cursor[1];
        """
        ),
        "threadgroup": (
            """
        threadgroup uint shared[2];
        shared[0] = values[1];
        shared[1] = values[2];
        threadgroup auto* cursor = &shared[0];
        cursor[0] += 1u;
        results[4] = shared[0];
        results[5] = cursor[1];
        """
        ),
    }[case]
    source = """#include <metal_stdlib>
    using namespace metal;
    uint read_value(uint index, const device uint* pointer) { return pointer[index]; }
    uint read_value(uint3 index, const device uint* pointer) { return pointer[index.x] + 7u; }
    kernel void pointer_alias(device uint* values [[buffer(0)]],
                             device uint* results [[buffer(1)]]) {
    BODY
    }
    """.replace("BODY", body)
    if case == "device":
        source = source.replace(
            "pointer_alias(device uint* values",
            "pointer_alias(const device uint* values",
        )
    return source


def _execute_case(tmp_path, target, case):
    source = _source(case)
    _, descriptor, package = _package(
        tmp_path, target, "uint", (1, 1, 1), source=source, software_subgroups=False
    )
    words = [13, 0xFFFFFFFF, 0x89ABCDEF, 29]
    expected_words = [GUARD] * 4 + [0, words[2]] + [GUARD] * 4
    inputs = _bound_values(
        descriptor,
        {
            "values": {"dtype": "uint32", "shape": [4], "values": words},
            "results": {
                "dtype": "uint32",
                "shape": [10],
                "values": [GUARD] * 4 + [0xDEADBEEF] * 2 + [GUARD] * 4,
            },
        },
    )
    outputs = {"results": {"dtype": "uint32", "shape": [10], "values": expected_words}}
    if case != "device":
        outputs["values"] = {"dtype": "uint32", "shape": [4], "values": words}
    expected = _bound_values(descriptor, outputs)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        expected,
        {"workgroupCount": [1, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="pointer_alias",
    )


@pytest.mark.parametrize("case", ("device", "threadgroup"))
def test_explicit_auto_pointer_executes(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required pointer deduction execution")
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    _execute_case(tmp_path, target, case)


@pytest.mark.parametrize("case", ("private", "array"))
def test_explicit_auto_private_pointer_round_trip_executes(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1" or sys.platform != "darwin":
        pytest.skip(f"requires Metal and {REQUIRE_ENV}=1")
    _execute_case(tmp_path, "metal", case)


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize("case", ("private", "array"))
def test_explicit_auto_private_pointer_keeps_target_diagnostic(tmp_path, target, case):
    (tmp_path / "pointer.metal").write_text(_source(case))
    payload = translate_project(tmp_path, targets=[target], output_dir="out").to_json()
    assert payload["summary"]["failedCount"] == 1, payload
    assert payload["summary"]["diagnosticsByCode"] == {
        f"project.translate.{target}-private-pointer-unsupported": 1
    }, payload


def test_auto_pointer_execution_is_required_on_all_platforms():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_metal_auto_pointer.py" in step
    assert "if:" not in step and "continue-on-error" not in step
    assert "pytest -q -n auto" in step
