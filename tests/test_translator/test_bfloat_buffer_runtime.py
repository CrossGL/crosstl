"""Bfloat storage is distinct from binary16 and uses the reflected target ABI."""

import copy
import os
import struct
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from crosstl.project import NativeLoaderDispatchError
from crosstl.project.host_reflection import reflect_target_host_interface
from crosstl.project.native_runtime_drivers import (
    _normalize_dtype,
    _pack_values,
    _unpack_values,
)
from crosstl.project.runtime_value_encoding import (
    BFLOAT16_BITS,
    FLOAT16_BITS,
    FLOAT32_BITS,
)
from crosstl.project.runtime_verification import (
    RuntimeExecutorUnavailable,
    RuntimeTolerance,
    RuntimeValue,
    _compare_runtime_value,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values, _request
from tests.test_translator.test_half_buffer_runtime import (
    _directx_buffers,
    _validate_half,
)
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_BFLOAT_BUFFER_RUNTIME"
WORDS = (
    0,
    0x8000,
    1,
    0x8001,
    0x7F,
    0x80,
    0x3F80,
    0xBF80,
    0x7F7F,
    0xFF7F,
    0x7F80,
    0xFF80,
    0x7FC1,
    0xFFC5,
    0x7F81,
    0xFF85,
)
FORMS = ("scalar", "constant", "reference", "vector2", "vector4", "alias", "struct")
TARGET_FORMS = {
    "metal": FORMS,
    "directx": ("scalar", "constant", "reference", "alias"),
    "opengl": FORMS,
}
NATIVE_TARGET = {"darwin": "metal", "win32": "directx", "linux": "opengl"}.get(
    sys.platform
)


def _storage(target, words):
    if target == "metal":
        return {
            "dtype": "bfloat16",
            "encoding": BFLOAT16_BITS,
            "values": list(words),
            "shape": [len(words)],
        }
    if target == "directx":
        return {"dtype": "uint16", "values": list(words), "shape": [len(words)]}
    return {
        "dtype": "float32",
        "encoding": FLOAT32_BITS,
        "values": [word << 16 for word in words],
        "shape": [len(words)],
    }


def _case(root, target, form="scalar", words=WORDS):
    root.mkdir(parents=True, exist_ok=True)
    width = 2 if form in {"vector2", "struct"} else 4 if form == "vector4" else 1
    payload = "bfloat" + (str(width) if form.startswith("vector") else "")
    declarations = ""
    if form == "alias":
        declarations, payload = "using Value = bfloat;", "Value"
    elif form == "struct":
        declarations, payload = "struct Cell { bfloat first; bfloat second; };", "Cell"
    qualifier = "constant" if form in {"constant", "reference"} else "const device"
    pointer = "&" if form == "reference" else "*"
    expression = "values" if form == "reference" else "values[i]"
    source = f"""#include <metal_stdlib>
using namespace metal;
{declarations}
kernel void bfloat_storage({qualifier} {payload}{pointer} values [[buffer(0)]],
                           device {payload}* results [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {{
    results[i + 1u] = {expression};
}}
"""
    _, descriptor, package = _package(
        root, target, "uint", (1, 1, 1), source=source, software_subgroups=False
    )
    values = list(words) if form != "reference" else [0x8001]
    count = len(words) // width
    expected = (
        [0x3EAA] * width
        + (values * count if form == "reference" else values)
        + [0x3EAA] * width
    )
    inputs = {
        "values": _storage(target, values),
        "results": _storage(target, [0x3EAA] * len(expected)),
    }
    outputs = {"results": _storage(target, expected)}
    if target == "directx" and form == "reference":
        binding = next(
            binding for binding in descriptor["bindings"] if binding["access"] == "read"
        )
        inputs[binding["scalarLayout"]["memberName"]] = inputs.pop("values")
    request = _request(descriptor, package, inputs, outputs, count)
    return source, descriptor, package, inputs, outputs, request


@pytest.mark.parametrize(
    "target,form",
    [(target, form) for target, forms in TARGET_FORMS.items() for form in forms],
)
def test_bfloat_copy_packages_preserve_physical_storage(tmp_path, target, form):
    _, descriptor, _, inputs, _, request = _case(tmp_path, target, form)
    dtype = {"metal": "bfloat16", "directx": "uint16", "opengl": "float32"}[target]
    for binding in descriptor["bindings"]:
        layout = binding["scalarLayout"]
        assert layout["elementType"] == dtype
        assert layout["elementSizeBytes"] == layout["elementStrideBytes"]
        assert layout["elementSizeBytes"] == (
            4 if target == "opengl" else 2
        ) * layout.get("vectorWidth", layout.get("componentCount", 1))
    if target == "directx":
        bound = _bound_values(descriptor, inputs)
        for buffer in _directx_buffers(request):
            values = bound[buffer.name]["values"]
            assert buffer.payload == struct.pack("<" + "H" * len(values), *values)
            assert buffer.dtype == "uint16"


def test_bfloat_encoding_preserves_every_storage_word():
    words = list(range(65536))
    packed = _pack_values(
        words,
        "bfloat16",
        expected_count=len(words),
        target="Metal",
        encoding=BFLOAT16_BITS,
    )
    assert packed == struct.pack("<65536H", *words)
    assert (
        _unpack_values(packed, "bfloat16", target="Metal", encoding=BFLOAT16_BITS)
        == words
    )
    for alias in ("bfloat", "bfloat16", "bfloat16_t"):
        assert _normalize_dtype(alias, target="Metal") == "bfloat16"
    for target in ("OpenGL", "DirectX", "Vulkan"):
        with pytest.raises(RuntimeExecutorUnavailable):
            _normalize_dtype("bfloat16", target=target)


def test_bfloat_native_driver_requires_explicit_storage_encoding():
    with pytest.raises(RuntimeExecutorUnavailable, match="explicit bfloat16-bits"):
        _pack_values([1.5], "bfloat16", expected_count=1, target="Metal")


@pytest.mark.parametrize("encoding", ([], {}, True, 16))
def test_bfloat_malformed_encodings_fail_with_a_dispatch_diagnostic(tmp_path, encoding):
    _, descriptor, package, inputs, outputs, _ = _case(tmp_path, "metal")
    inputs["values"]["encoding"] = encoding
    with pytest.raises(NativeLoaderDispatchError, match="value-encoding-invalid"):
        _request(descriptor, package, inputs, outputs, len(WORDS))


@pytest.mark.parametrize("physical,dtype", (("short", "int16"), ("ushort", "uint16")))
@pytest.mark.parametrize("width", (1, 2, 4))
def test_metal_integer_16_reflection(tmp_path, physical, dtype, width):
    source = tmp_path / "storage.metal"
    physical += str(width) if width > 1 else ""
    source.write_text(
        f"#include <metal_stdlib>\nusing namespace metal;\nkernel void storage(device {physical}* result [[buffer(0)]], uint i [[thread_position_in_grid]]) {{ result[i] = 0; }}"
    )
    layout = reflect_target_host_interface(source, target="metal")["resources"][0][
        "scalarLayout"
    ]
    assert layout["elementType"] == dtype
    assert layout["physicalType"] == physical
    assert layout["elementSizeBytes"] == layout["elementStrideBytes"] == width * 2
    assert layout["alignmentBytes"] == width * 2


@pytest.mark.parametrize("dtype", ("int16", "uint16"))
@pytest.mark.parametrize("target", ("Metal", "DirectX"))
def test_integer_16_storage_preserves_every_word(dtype, target):
    values = list(range(-32768, 32768)) if dtype == "int16" else list(range(65536))
    assert _normalize_dtype(dtype, target=target) == dtype
    packed = _pack_values(values, dtype, expected_count=len(values), target=target)
    assert len(packed) == 131072
    assert _unpack_values(packed, dtype, target=target) == values


@pytest.mark.parametrize(
    "dtype,bad",
    (
        ("int16", -32769),
        ("int16", 32768),
        ("uint16", -1),
        ("uint16", 65536),
        ("int16", True),
        ("uint16", 1.0),
    ),
)
def test_integer_16_storage_rejects_invalid_values(dtype, bad):
    with pytest.raises(RuntimeExecutorUnavailable):
        _pack_values([bad], dtype, expected_count=1, target="DirectX")


@pytest.mark.parametrize(
    "dtype,physical", (("int16", "int16_t"), ("uint16", "uint16_t"))
)
@pytest.mark.parametrize("width", (1, 2, 3, 4))
def test_explicit_hlsl_integer_16_reflection(tmp_path, dtype, physical, width):
    source = tmp_path / "storage.hlsl"
    physical += str(width) if width > 1 else ""
    source.write_text(
        f"RWStructuredBuffer<{physical}> result : register(u0);\n[numthreads(1,1,1)] void main(uint i : SV_DispatchThreadID) {{ result[i] = 0; }}"
    )
    layout = reflect_target_host_interface(source, target="directx")["resources"][0][
        "scalarLayout"
    ]
    assert layout["physicalType"] == physical
    assert layout["elementType"] == dtype
    assert layout["elementSizeBytes"] == layout["elementStrideBytes"] == width * 2
    assert layout["alignmentBytes"] == 2


@pytest.mark.parametrize(
    "fault",
    (
        "no-encoding",
        "binary16",
        "binary32",
        "boolean",
        "negative",
        "oversized",
        "numeric",
    ),
)
def test_bfloat_dispatch_rejects_ambiguous_or_invalid_storage(tmp_path, fault):
    _, descriptor, package, inputs, outputs, _ = _case(tmp_path, "metal")
    value = inputs["values"]
    if fault == "no-encoding":
        value.pop("encoding")
    elif fault in {"binary16", "binary32"}:
        value["encoding"] = FLOAT16_BITS if fault == "binary16" else FLOAT32_BITS
    else:
        value["values"][0] = {
            "boolean": True,
            "negative": -1,
            "oversized": 65536,
            "numeric": 1.0,
        }[fault]
    with pytest.raises(NativeLoaderDispatchError, match="value-encoding-invalid"):
        _request(descriptor, package, inputs, outputs, len(WORDS))


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_bfloat_logical_values_cannot_bind_other_physical_storage(tmp_path, target):
    _, descriptor, package, inputs, outputs, _ = _case(tmp_path, target)
    inputs["values"] = _storage("metal", WORDS)
    with pytest.raises(NativeLoaderDispatchError, match="resource-layout-mismatch"):
        _request(descriptor, package, inputs, outputs, len(WORDS))


@pytest.mark.parametrize(
    "field,value",
    (
        ("elementSizeBytes", 4),
        ("elementStrideBytes", 4),
        ("alignmentBytes", 1),
        ("physicalType", "half"),
        ("memberOffsetBytes", 1),
        ("elementType", "float16"),
    ),
)
def test_bfloat_layout_reinterpretation_is_rejected(tmp_path, field, value):
    _, descriptor, package, inputs, outputs, _ = _case(tmp_path, "metal")
    descriptor = copy.deepcopy(descriptor)
    for binding in descriptor["bindings"]:
        if binding["name"] == "results":
            binding["scalarLayout"][field] = value
    for binding in descriptor["scalarLayout"]["bindings"]:
        if binding["binding"] == "results":
            binding["layout"][field] = value
    with pytest.raises(NativeLoaderDispatchError):
        _request(descriptor, package, inputs, outputs, len(WORDS))


@pytest.mark.parametrize("change", (None, "nan", "zero", "bit", "encoding"))
def test_bfloat_storage_requires_bitwise_equality(change):
    expected = RuntimeValue(
        name="out", **{**_storage("metal", WORDS), "shape": (len(WORDS),)}
    )
    actual = replace(expected, values=list(WORDS))
    if change == "nan":
        actual.values[-1] = 0x7FC0
    elif change == "zero":
        actual.values[0] = 0x8000
    elif change == "bit":
        actual.values[6] ^= 1
    elif change == "encoding":
        actual = replace(actual, encoding=FLOAT16_BITS)
    result = _compare_runtime_value(
        expected, actual, default_tolerance=RuntimeTolerance(1e30, 1e30)
    )
    assert result["status"] == ("passed" if change is None else "comparison-failed")
    assert result["comparison"] == "bitwise"
    assert result["tolerance"] == {"absolute": 0.0, "relative": 0.0}


@pytest.mark.parametrize("form", TARGET_FORMS.get(NATIVE_TARGET, ()))
def test_bfloat_copy_executes_natively(tmp_path, form):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required bfloat storage")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, descriptor, _, _, outputs, request = _case(tmp_path, target, form)
    _execute(
        request,
        _bound_values(descriptor, outputs),
        tmp_path,
        original_source=source,
        original_entry="bfloat_storage",
        validate=_validate_half,
    )


@pytest.mark.parametrize("start", (0, 16384, 32768, 49152))
def test_bfloat_native_storage_covers_every_payload(tmp_path, start):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required bfloat storage")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, descriptor, _, _, outputs, request = _case(
        tmp_path, target, words=range(start, start + 16384)
    )
    _execute(
        request,
        _bound_values(descriptor, outputs),
        tmp_path,
        original_source=source,
        original_entry="bfloat_storage",
        validate=_validate_half,
    )


def test_bfloat_native_gate_is_required_on_every_target():
    from tools import ci_coverage

    workflow = Path(".github/workflows/demo-project-testing.yml").read_text()
    for name in (
        "Validate indexed OpenGL gather and resource aggregates",
        "Validate indexed DirectX gather and resource aggregates",
        "Validate Metal byte and vector storage",
    ):
        step = ci_coverage.workflow_step_section(workflow, name)
        assert f'{REQUIRE_ENV}: "1"' in step
        assert "tests/test_translator/test_bfloat_buffer_runtime.py" in step
        assert (
            "-n auto" in step and "continue-on-error" not in step and "if:" not in step
        )
