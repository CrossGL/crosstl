"""Source byte conversions retain their range across widened arithmetic values."""

import os
import sys
from pathlib import Path

import pytest

from crosstl._crosstl import translate
from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import (
    DirectXContextualConversionError,
    HLSLCodeGen,
)
from crosstl.translator.codegen.metal_codegen import UnsupportedMetalFeatureError
from tests.test_translator.test_boolean_buffer_runtime import _bound_values, _request
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_NARROW_INTEGER_CONVERSIONS"
VALUES = [-257, -256, -129, -128, -1, 0, 1, 127, 128, 255, 256, 257, 65535]
FORMS = (
    "constructor",
    "cast",
    "initializer",
    "alias",
    "assignment",
    "argument",
    "return",
    "array",
    "member",
    "conditional",
    "single_evaluation",
    "vector",
    "vector_braces",
    "array_assignment",
    "buffer_store",
    "buffer_offset",
    "compound",
    "prefix",
    "postfix",
    "member_compound",
    "member_or",
    "member_prefix",
    "member_postfix",
    "nested_member_compound",
    "reference",
    "reference_nested",
    "reference_pair",
    "reference_alias",
    "reference_vector",
    "reference_array",
    "reference_index_effect",
    "reference_conditional",
    "reference_loop",
)


def _narrow(value, signed):
    value %= 256
    return value - 256 if signed and value >= 128 else value


def _case(root, target, signed, form):
    byte = "char" if signed else "uchar"
    declarations = ""
    expression = "int(value)"
    body = f"{byte} value = input_value;"
    if form == "constructor":
        body = f"int value = int({byte}(input_value));"
    elif form == "cast":
        body = f"int value = int(({byte})input_value);"
    elif form == "alias":
        declarations = f"using Byte = {byte};"
        body = "Byte value = Byte(input_value);"
    elif form == "assignment":
        body = f"{byte} value = 0; value = input_value;"
    elif form == "argument":
        declarations = f"int promote({byte} value) {{ return int(value); }}"
        body = "int value = promote(input_value);"
    elif form == "return":
        declarations = f"{byte} narrow(int value) {{ return value; }}"
        body = "int value = int(narrow(input_value));"
    elif form == "array":
        body = f"{byte} values[2] = {{{byte}(0), {byte}(input_value)}};"
        expression = "int(values[1])"
    elif form in {
        "member",
        "member_compound",
        "member_or",
        "member_prefix",
        "member_postfix",
        "nested_member_compound",
    }:
        declarations = f"struct Cell {{ {byte} value; }};"
        member = "cell.value"
        owner_type = "Cell"
        if form == "nested_member_compound":
            declarations += " struct Outer { Cell inner; };"
            member, owner_type = "cell.inner.value", "Outer"
        body = f"{owner_type} cell; {member} = input_value;"
        expression = f"int({member})"
        if form in {"member_compound", "nested_member_compound"}:
            body += f" {member} += 3;"
        elif form == "member_or":
            body += f" {member} |= {byte}(1);"
        elif form == "member_prefix":
            expression = f"int(++{member})"
        elif form == "member_postfix":
            body += f" int old = int({member}++);"
            expression = f"old + int({member}) * 1024"
    elif form == "conditional":
        body = f"{byte} value = input_value < 0 ? input_value : input_value + 256;"
    elif form == "single_evaluation":
        declarations = "int next_value(thread int& value) { int old = value; value += 1; return old; }"
        body = f"int previous = input_value; int value = int({byte}(next_value(previous)));"
        expression = "value + (previous - input_value - 1) * 1000"
    elif form == "vector":
        body = f"{byte}2 value = {byte}2(int2(input_value, input_value + 2));"
        expression = "int(value.x) + int(value.y) * 1024"
    elif form == "vector_braces":
        body = f"{byte}2 value = {{{byte}(input_value), {byte}(input_value + 2)}};"
        expression = "int(value.x) + int(value.y) * 1024"
    elif form == "array_assignment":
        body = f"{byte} values[2]; values[1] = input_value;"
        expression = "int(values[1])"
    elif form in {"buffer_store", "buffer_offset"}:
        body = ""
        expression = "input_value"
        if form == "buffer_offset":
            body = "results += 1;"
    elif form == "compound":
        body += " value += 3;"
    elif form == "prefix":
        expression = "int(++value)"
    elif form == "postfix":
        body += " int old = int(value++);"
        expression = "old + int(value) * 1024"
    elif form.startswith("reference"):
        declarations = (
            f"int increment(thread {byte}& value) {{ value += 3; return int(value); }}"
        )
        body += " increment(value);"
        if form == "reference_nested":
            declarations += f" void twice(thread {byte}& value) {{ increment(value); increment(value); }}"
            body = f"{byte} value = input_value; twice(value);"
        elif form == "reference_pair":
            declarations += f" void pair(thread {byte}& value, thread {byte}& other) {{ increment(value); increment(other); }}"
            body = f"{byte} value = input_value; {byte} other = 13; pair(value, other);"
            expression = "int(value) + (int(other) - 16) * 1024"
        elif form == "reference_alias":
            declarations = f"using Byte = {byte};\n" + declarations.replace(
                f"thread {byte}&", "thread Byte&"
            )
            body = "Byte value = input_value; increment(value);"
        elif form == "reference_member":
            declarations += f" struct Cell {{ {byte} value; }};"
            body = "Cell cell; cell.value = input_value; increment(cell.value);"
            expression = "int(cell.value)"
        elif form == "reference_vector":
            declarations = f"void increment(thread {byte}2& value) {{ value = {byte}2(int2(value) + 3); }}"
            body = f"{byte}2 value = {byte}2(input_value, input_value + 2); increment(value);"
            expression = "int(value.x) + int(value.y) * 1024"
        elif form in {
            "reference_array",
            "reference_index_effect",
            "reference_conditional",
            "reference_loop",
        }:
            body = f"{byte} values[2] = {{{byte}(input_value), {byte}(91)}}; int index = 0;"
            if form == "reference_array":
                body += " increment(values[index]);"
            elif form == "reference_index_effect":
                body += " increment(values[index++]);"
            elif form == "reference_conditional":
                body += (
                    " int result = input_value < 0 ? increment(values[index++]) : 17;"
                )
            else:
                body += " for (int pass = 0; pass < 2; ++pass) { index = 0; increment(values[index++]); }"
            expression = "int(values[0]) + (int(values[1]) - 91) * 1024"
            expected_index = (
                "0"
                if form == "reference_array"
                else (
                    "(input_value < 0 ? 1 : 0)"
                    if form == "reference_conditional"
                    else "1"
                )
            )
            expression += f" + (index - {expected_index}) * 65536"
            if form == "reference_conditional":
                expression += (
                    " + (result - (input_value < 0 ? int(values[0]) : 17)) * 4096"
                )
    output_type = byte if form in {"buffer_store", "buffer_offset"} else "int"
    result_index = "tid" if form == "buffer_offset" else "tid + 1u"
    source = f"""#include <metal_stdlib>
using namespace metal;
{declarations}
kernel void byte_conversions(const device int* inputs [[buffer(0)]],
                             device {output_type}* results [[buffer(1)]],
                             uint tid [[thread_position_in_grid]]) {{
    int input_value = inputs[tid];
    {body}
    results[{result_index}] = {expression};
}}
"""
    _, descriptor, package = _package(
        root, target, "int", (1, 1, 1), source=source, software_subgroups=False
    )
    values = []
    for value in VALUES:
        narrowed = _narrow(value, signed)
        if form in {"vector", "vector_braces"}:
            narrowed += _narrow(value + 2, signed) * 1024
        elif form in {"compound", "member_compound", "nested_member_compound"}:
            narrowed = _narrow(narrowed + 3, signed)
        elif form == "member_or":
            narrowed = _narrow(narrowed | 1, signed)
        elif form in {"prefix", "member_prefix"}:
            narrowed = _narrow(narrowed + 1, signed)
        elif form in {"postfix", "member_postfix"}:
            narrowed += _narrow(narrowed + 1, signed) * 1024
        elif form.startswith("reference"):
            steps = 2 if form in {"reference_nested", "reference_loop"} else 1
            if form == "reference_conditional" and value >= 0:
                steps = 0
            narrowed = _narrow(narrowed + 3 * steps, signed)
            if form == "reference_vector":
                narrowed += _narrow(value + 5, signed) * 1024
        values.append(narrowed)
    count = len(VALUES) + 2
    dtype = "int32"
    guard = 987654
    if form in {"buffer_store", "buffer_offset"}:
        dtype = ("int" if signed else "uint") + ("8" if target == "metal" else "32")
        guard = 91
    inputs = {
        "inputs": {"dtype": "int32", "shape": [len(VALUES)], "values": VALUES},
        "results": {"dtype": dtype, "shape": [count], "values": [guard] * count},
    }
    outputs = {
        "results": {
            "dtype": dtype,
            "shape": [count],
            "values": [guard, *values, guard],
        }
    }
    return (
        source,
        _request(descriptor, package, inputs, outputs, len(VALUES)),
        _bound_values(descriptor, outputs),
    )


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("signed", (True, False), ids=("signed", "unsigned"))
@pytest.mark.parametrize("form", FORMS)
def test_narrow_integer_conversion_packages(tmp_path, target, signed, form):
    _, request, _ = _case(tmp_path, target, signed, form)
    if form.startswith("reference"):
        _compile(request.artifact_path.read_text(), target, tmp_path)
    if target == "directx" and form in {"buffer_store", "buffer_offset"}:
        stores = [
            line
            for line in request.artifact_path.read_text().splitlines()
            if line.strip().startswith("results[") and " = " in line
        ]
        assert len(stores) == 1
        assert (" >> 24)" if signed else " & 255u)") in stores[0]


@pytest.mark.parametrize("signed", (True, False), ids=("signed", "unsigned"))
@pytest.mark.parametrize("form", FORMS)
def test_narrow_integer_conversions_execute_natively(tmp_path, signed, form):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native byte conversions")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _case(tmp_path, target, signed, form)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="byte_conversions",
    )


@pytest.mark.parametrize("target", ("metal", "directx"))
@pytest.mark.parametrize("update", ("values[index++] += 3", "values[index++]++"))
@pytest.mark.parametrize("member", (False, True), ids=("element", "field"))
def test_byte_update_does_not_repeat_an_index_effect(tmp_path, target, update, member):
    source = tmp_path / "update.metal"
    declaration = "uchar values[2] = {uchar(255), uchar(1)};"
    result = "values[0]"
    if member:
        declaration = (
            "Cell values[2]; values[0].value = uchar(255); values[1].value = uchar(1);"
        )
        update = update.replace("values[index++]", "values[index++].value")
        result += ".value"
    source.write_text(
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "struct Cell { uchar value; };\n"
        "kernel void update(device int* results [[buffer(0)]]) {\n"
        f"{declaration} int index = 0;\n"
        f"{update}; results[0] = int({result}) + index;\n}}\n"
    )
    error_type = (
        UnsupportedMetalFeatureError
        if target == "metal"
        else DirectXContextualConversionError
    )
    with pytest.raises(error_type) as error:
        translate(
            str(source), backend=target, source_backend="metal", format_output=False
        )
    assert error.value.reason == "byte-update-lvalue-unsupported"


def test_byte_conversions_and_directx_random_are_required_in_ci():
    workflow = (
        Path(__file__).parents[2] / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    assert workflow.count(f'{REQUIRE_ENV}: "1"') == 3
    assert (
        workflow.count("tests/test_translator/test_narrow_integer_conversions.py") == 3
    )
    step = workflow.split("- name: Validate translated DirectX random kernels\n", 1)[
        1
    ].split("      - name:", 1)[0]
    assert "--target directx --output-dir .mlx-gather-directx/random" in step
    assert "set -euo pipefail" in step and "--timeout-seconds 300" in step
    assert "continue-on-error" not in step


@pytest.mark.parametrize("signed", (True, False), ids=("signed", "unsigned"))
@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_writable_byte_fields_compile(tmp_path, signed, target):
    _, request, expected = _case(tmp_path, target, signed, "reference_member")
    _compile(request.artifact_path.read_text(), target, tmp_path)
    if os.environ.get(REQUIRE_ENV) == "1" and target == {
        "win32": "directx",
        "linux": "opengl",
    }.get(sys.platform):
        _execute(request, expected, tmp_path)


@pytest.mark.parametrize("direction", ("out", "inout"))
@pytest.mark.parametrize("width", (1, 2, 4))
@pytest.mark.parametrize("signed", (True, False), ids=("signed", "unsigned"))
def test_hlsl_writable_overloads_keep_lvalues(tmp_path, direction, width, signed):
    byte = ("char" if signed else "uchar") + (str(width) if width > 1 else "")
    wide = ("int" if signed else "uint") + (str(width) if width > 1 else "")
    source = f"""shader References {{
        RWStructuredBuffer<int> outputs @register(u0);
        void update({direction} {byte} value) {{ value = {byte}(255); }}
        int update({wide} value) {{ return int(value{'.x' if width > 1 else ''}); }}
        int consume({byte} value) {{ return int(value{'.x' if width > 1 else ''}); }}
        compute {{
            @numthreads(1, 1, 1)
            void main() {{
                {byte} value = {byte}(127);
                update(value);
                outputs[0] = update({wide}(257)) + consume({wide}(257)) + int(value{'.x' if width > 1 else ''});
            }}
        }}
    }}"""
    generated = HLSLCodeGen().generate(parse(source))
    assert f"update_{byte}(value);" in generated
    assert f"{direction} {wide} value" in generated
    assert "consume((int" in generated if signed else "consume((uint" in generated
    _compile(generated, "directx", tmp_path)


@pytest.mark.parametrize("kind", ("char", "uchar"))
@pytest.mark.parametrize("readonly", (False, True))
@pytest.mark.parametrize("indexed", (False, True))
def test_hlsl_preserves_exact_byte_aliases_and_rejects_unknown_overlap(
    tmp_path, kind, readonly, indexed
):
    second = "const thread" if readonly else "thread"
    declaration = f"{kind} value = {kind}(1);"
    arguments = "value, value"
    result = "value"
    if indexed:
        declaration = f"{kind} values[2] = {{{kind}(1), {kind}(2)}}; uint index = 0;"
        arguments = "values[index], values[0]"
        result = "values[0]"
    source = tmp_path / "aliases.metal"
    source.write_text(
        "#include <metal_stdlib>\nusing namespace metal;\n"
        f"void change(thread {kind}& first, {second} {kind}& other) {{ first += 1; first += other; }}\n"
        "kernel void aliases(device uint* outputs [[buffer(0)]]) {\n"
        f"{declaration} change({arguments}); outputs[0] = uint({result});\n}}\n"
    )
    if indexed:
        with pytest.raises(DirectXContextualConversionError) as error:
            translate(str(source), backend="directx", format_output=False)
        assert error.value.reason == "byte-reference-alias-unsupported"
    else:
        generated = translate(str(source), backend="directx", format_output=False)
        assert "change_shared_references" in generated
        _compile(generated, "directx", tmp_path)
