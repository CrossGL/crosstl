"""Generated HLSL half buffers separate arithmetic from bit-preserving storage."""

import os
import struct
import sys
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.hlsl_half_storage import DirectXHalfStorageError
from crosstl.translator.resource_storage import (
    BINARY16_STORAGE,
    parse_resource_storage_header,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values, _request
from tests.test_translator.test_codegen.test_directx_codegen import (
    assert_directx_native_16_bit_compute_validates_if_available,
)
from tests.test_translator.test_half_buffer_runtime import (
    _case,
    _directx_buffers,
    _validate_half,
)
from tests.test_translator.test_half_copy_identity import CASES, _source
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_HALF_BUFFER_RUNTIME"


def _translate(root, source):
    path = root / "storage.metal"
    path.write_text(source, encoding="utf-8")
    return translate(str(path), backend="directx", format_output=False)


@pytest.mark.parametrize("case", CASES)
def test_half_copy_lowering_preserves_logical_and_physical_types(tmp_path, case):
    generated = _translate(tmp_path, _source(case))
    assert parse_resource_storage_header(generated) == {
        "values": BINARY16_STORAGE,
        "results": BINARY16_STORAGE,
    }
    assert "StructuredBuffer<float16_t" not in generated
    assert "asfloat16(" in generated and "asuint16(" in generated
    if case in {"structure", "structure-members"}:
        assert "struct Cell" in generated and "float16_t first;" in generated
        assert "uint16_t first;" in generated
    else:
        assert "StructuredBuffer<uint16_t" in generated


@pytest.mark.parametrize(
    "body",
    [
        "results[i] = values[i] + 0.5f;",
        "results[i] += values[i];",
        "results[i] -= values[i];",
        "results[i] *= values[i];",
        "results[i] /= values[i];",
        "results[i] = ++values[i];",
        "results[i] = values[i]++;",
        "results[i] = --values[i];",
        "results[i] = values[i]--;",
        "results[i] = (values[i] = half(2.0f));",
    ],
)
def test_half_updates_encode_writeback(tmp_path, body):
    source = _update_source(body)
    generated = _translate(tmp_path, source)
    assert parse_resource_storage_header(generated)
    assert "asuint16(" in generated and "asfloat16(" in generated
    (tmp_path / "source.hlsl").write_text(generated)
    assert not any(
        line.strip().startswith("asfloat16(") for line in generated.splitlines()
    )
    assert_directx_native_16_bit_compute_validates_if_available(generated, tmp_path)


def _update_source(body):
    return f"""#include <metal_stdlib>
using namespace metal;
kernel void update(device half* values [[buffer(0)]], device half* results [[buffer(1)]], uint i [[thread_position_in_grid]]) {{
    {body}
}}
"""


STRUCTURES = {
    "nested": (
        "struct Inner { half value; }; struct Cell { Inner inner; float marker; };"
    ),
    "array": "struct Cell { half values[2]; uint marker; };",
    "vector": "struct Cell { half2 value; uint marker; };",
    "mixed": "struct Cell { half value; bool enabled; uint marker; };",
    "collision": (
        "struct __crossgl_half_storage_Cell { uint value; }; struct Cell { half first; half second; };"
    ),
}


@pytest.mark.parametrize("case", STRUCTURES)
def test_storage_structures_do_not_change_local_member_types(tmp_path, case):
    generated = _translate(
        tmp_path,
        _source("structure").replace(
            "struct Cell { half first; half second; };", STRUCTURES[case]
        ),
    )
    assert "struct Cell" in generated and "float16_t" in generated
    assert "uint16_t" in generated
    assert (
        "__crossgl_half_load_Cell" in generated
        and "__crossgl_half_store_Cell" in generated
    )
    if case == "collision":
        assert "struct __crossgl_half_storage_Cell_1" in generated
    assert_directx_native_16_bit_compute_validates_if_available(generated, tmp_path)


@pytest.mark.parametrize("scalar", ["float", "ushort", "uint", "int"])
def test_non_half_buffers_do_not_get_storage_codecs(tmp_path, scalar):
    generated = _translate(tmp_path, _source("direct").replace("half", scalar))
    assert parse_resource_storage_header(generated) == {}
    assert "__crossgl_half_storage_" not in generated
    assert "asfloat16(" not in generated


UPDATE_CASES = {
    "add": ("results[i] = values[i] + 0.5f;", lambda value: (value, value + 0.5)),
    "compound": ("results[i] += values[i];", lambda value: (value, value + 2.0)),
    "multiply": ("results[i] *= values[i];", lambda value: (value, value * 2.0)),
    "prefix": ("results[i] = ++values[i];", lambda value: (value + 1, value + 1)),
    "postfix": ("results[i] = values[i]++;", lambda value: (value + 1, value)),
    "assignment": ("results[i] = (values[i] = half(2.0f));", lambda value: (2.0, 2.0)),
}


@pytest.mark.parametrize("case", UPDATE_CASES)
def test_half_storage_updates_execute_with_logical_results(tmp_path, case):
    if sys.platform != "win32" or os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip("required native half storage updates run on Windows")
    body, operation = UPDATE_CASES[case]
    body = body.replace("[i]", "[i + 1u]")
    source = _update_source(body)
    _, descriptor, package = _package(
        tmp_path, "directx", "uint", (1, 1, 1), source=source, software_subgroups=False
    )
    numbers = [i / 16 - 32 for i in range(1024)]

    def payload(values):
        words = [struct.unpack("<H", struct.pack("<e", value))[0] for value in values]
        return {
            "dtype": "float16",
            "encoding": "ieee754-binary16",
            "shape": [len(words)],
            "values": words,
        }

    inputs = {
        "values": payload([0.25, *numbers, 0.25]),
        "results": payload([0.25, *([2.0] * len(numbers)), 0.25]),
    }
    updated = [operation(value) for value in numbers]
    outputs = {
        "values": payload([0.25, *[pair[0] for pair in updated], 0.25]),
        "results": payload([0.25, *[pair[1] for pair in updated], 0.25]),
    }
    request = _request(descriptor, package, inputs, outputs, len(numbers))
    _execute(
        request, _bound_values(descriptor, outputs), tmp_path, validate=_validate_half
    )


def test_half_lowering_controls_are_required_on_windows():
    from tools import ci_coverage

    step = ci_coverage.workflow_step_section(
        Path(".github/workflows/demo-project-testing.yml").read_text(),
        "Validate indexed DirectX gather and resource aggregates",
    )
    assert "tests/test_translator/test_directx_half_storage.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step and "-n auto" in step
    assert "continue-on-error" not in step


def _constant_case(root, word):
    _, descriptor, package, inputs, outputs, _ = _case(root, "directx", "reference")
    input_name = next(name for name in inputs if name != "results")
    inputs[input_name]["values"] = [word]
    outputs["results"]["values"] = [0x3555, *([word] * 16), 0x3555]
    request = _request(descriptor, package, inputs, outputs, 16)
    return request, _bound_values(descriptor, outputs)


@pytest.mark.parametrize("word", [0, 0x8000, 1, 0x8001, 0x7D01, 0xFD55, 0x7E01, 0xFE55])
def test_constant_half_storage_preserves_payload_and_alignment(tmp_path, word):
    request, _ = _constant_case(tmp_path, word)
    buffer = next(
        value for value in _directx_buffers(request) if value.namespace == "cbv"
    )
    assert buffer.dtype == "float16"
    assert buffer.payload == struct.pack("<H", word)
    assert buffer.byte_length == 16 and buffer.allocation_size == 256
    assert buffer.stride == 0


@pytest.mark.parametrize("word", [0, 0x8000, 1, 0x8001, 0x7D01, 0xFD55, 0x7E01, 0xFE55])
def test_constant_half_storage_executes_exact_payloads(tmp_path, word):
    if sys.platform != "win32" or os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip("required native half constant storage runs on Windows")
    request, expected = _constant_case(tmp_path, word)
    _execute(request, expected, tmp_path, validate=_validate_half)


def _program(body, declarations="", scalar="float16"):
    return parse(f"""shader Storage {{
        {declarations}
        RWStructuredBuffer<{scalar}> values;
        RWStructuredBuffer<{scalar}> results;
        compute {{
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() {{ {body} }}
        }}
    }}""")


@pytest.mark.parametrize("intrinsic", ["asfloat16", "asuint16"])
@pytest.mark.parametrize("scope", ["local", "function", "global"])
def test_half_storage_intrinsic_shadowing_is_diagnosed(intrinsic, scope):
    declarations, body = "", "results[0] = values[0];"
    if scope == "local":
        body = f"int {intrinsic} = 0; " + body
    elif scope == "global":
        declarations = f"int {intrinsic};"
    else:
        declarations = f"float16 {intrinsic}(float16 value) {{ return value; }}"
    with pytest.raises(DirectXHalfStorageError) as error:
        HLSLCodeGen().generate(_program(body, declarations))
    assert error.value.reason == "storage-intrinsic-shadowed"


def test_half_storage_state_does_not_leak_between_programs():
    generator = HLSLCodeGen()
    assert parse_resource_storage_header(
        generator.generate(_program("results[0] = ++values[0];"))
    )
    generated = generator.generate(_program("results[0] = values[0];", scalar="float"))
    assert parse_resource_storage_header(generated) == {}
    assert "__crossgl_half_storage_update" not in generated


def test_half_storage_compound_side_effects_are_diagnosed():
    with pytest.raises(DirectXHalfStorageError) as error:
        HLSLCodeGen().generate(_program("uint i = 0; values[i++] += float16(1.0);"))
    assert error.value.reason == "side-effecting-assignment-target"


def test_half_storage_load_retains_resource_validation(monkeypatch):
    generator = HLSLCodeGen()
    original = generator.validate_buffer_call_access
    calls = []

    def validate(name, arguments):
        calls.append(name)
        return original(name, arguments)

    monkeypatch.setattr(generator, "validate_buffer_call_access", validate)
    generator.generate(_program("results[0] = buffer_load(values, 0);"))
    assert "buffer_load" in calls


@pytest.mark.parametrize(
    "body", ["readOnly[0] = values[0];", "readOnly[0]++;", "readOnly[0] += values[0];"]
)
def test_half_storage_does_not_make_read_only_resources_writable(body):
    with pytest.raises(ValueError, match="writ"):
        HLSLCodeGen().generate(_program(body, "StructuredBuffer<float16> readOnly;"))


@pytest.mark.parametrize(
    "body",
    [
        "results[0].first = values[0].second;",
        "results[0].first += values[0].second;",
        "results[0].first = ++values[0].second;",
    ],
)
def test_structure_member_updates_keep_logical_half_type(tmp_path, body):
    generated = HLSLCodeGen().generate(
        _program(body, "struct Cell { float16 first; float16 second; };", scalar="Cell")
    )
    assert ".first = asuint16(" in generated
    assert "asfloat16(" in generated
    assert_directx_native_16_bit_compute_validates_if_available(generated, tmp_path)


@pytest.mark.parametrize(
    "body",
    [
        "results[0].marker |= values[0].marker;",
        "results[0].weight += values[0].weight;",
        "results[0].marker = ++values[0].marker;",
    ],
)
def test_other_structure_members_retain_their_update_semantics(tmp_path, body):
    declarations = "struct Cell { float16 value; uint marker; float weight; };"
    generated = HLSLCodeGen().generate(_program(body, declarations, scalar="Cell"))
    assert ".marker" in generated or ".weight" in generated
    assert_directx_native_16_bit_compute_validates_if_available(generated, tmp_path)


def test_half_storage_array_values_require_element_access():
    declarations = "struct Cell { float16 elements[2]; };"
    with pytest.raises(DirectXHalfStorageError) as error:
        HLSLCodeGen().generate(
            _program(
                "results[0].elements = values[0].elements;", declarations, scalar="Cell"
            )
        )
    assert error.value.reason == "array-storage-value-unsupported"


@pytest.mark.parametrize(
    "body",
    [
        "results[i].x = values[i].y;",
        "results[i].xy = values[i].yx;",
        "results[i][0] = values[i][1];",
        "results[i].x += values[i].y;",
        "results[i].xy += values[i].yx;",
    ],
)
def test_half_vector_component_writes_preserve_storage(tmp_path, body):
    generated = _translate(tmp_path, _update_source(body).replace("half*", "half2*"))
    assert "asuint16(" in generated
    assert_directx_native_16_bit_compute_validates_if_available(generated, tmp_path)
