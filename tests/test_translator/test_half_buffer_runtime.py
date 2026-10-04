"""Binary16 storage and conversion use explicit native buffer layouts."""

import base64
import copy
import os
import shutil
import struct
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from crosstl.project import (
    DirectXRuntimeParityAdapter,
    MetalComputeRuntime,
    NativeLoaderDispatchError,
)
from crosstl.project.native_runtime_drivers import (
    _normalize_dtype,
    _pack_values,
    _prepare_directx_buffers,
    _unpack_values,
)
from crosstl.project.runtime_value_encoding import FLOAT16_BITS, FLOAT32_BITS
from crosstl.project.runtime_verification import (
    RuntimeAdapterSetupError,
    RuntimeAllocationView,
    RuntimeExecutionState,
    RuntimeExecutorUnavailable,
    RuntimeTolerance,
    RuntimeValue,
    _compare_runtime_value,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values, _request
from tests.test_translator.test_directx_float_atomics import (
    _compile as _compile_directx,
)
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_native_runtime import _native_request
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_HALF_BUFFER_RUNTIME"
FORMS = (
    "scalar",
    "constant",
    "reference",
    "vector2",
    "vector4",
    "alias",
    "struct",
    "offset",
)
WORDS = (
    0,
    0x8000,
    1,
    0x8001,
    0x3FF,
    0x400,
    0x3C00,
    0xBC00,
    0x7BFF,
    0xFBFF,
    0x7C00,
    0xFC00,
    0x7E01,
    0xFE55,
    0x7D01,
    0xFD55,
)


def _validate_half(artifact, directory, target):
    source = artifact.read_text(encoding="utf-8")
    if target == "directx":
        assert shutil.which("dxc"), "DXC is required for native half storage"
        return _compile_directx(
            source, directory, profile="cs_6_6", flags=("-enable-16bit-types",)
        )[1]
    return _compile(source, target, directory)[1]


def test_half_directx_validation_requires_native_storage_flag(tmp_path, monkeypatch):
    source = tmp_path / "half.hlsl"
    source.write_text("StructuredBuffer<float16_t> values;", encoding="utf-8")
    observed = {}

    def compile_source(text, directory, **options):
        observed.update(text=text, directory=directory, **options)
        return source, directory / "half.dxil"

    monkeypatch.setattr(shutil, "which", lambda name: "dxc" if name == "dxc" else None)
    monkeypatch.setattr(sys.modules[__name__], "_compile_directx", compile_source)
    assert _validate_half(source, tmp_path, "directx") == tmp_path / "half.dxil"
    assert observed == {
        "text": source.read_text(encoding="utf-8"),
        "directory": tmp_path,
        "profile": "cs_6_6",
        "flags": ("-enable-16bit-types",),
    }
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(AssertionError, match="DXC is required"):
        _validate_half(source, tmp_path, "directx")


def _case(root, target, form="scalar", words=WORDS):
    root.mkdir(parents=True, exist_ok=True)
    width = 2 if form in {"vector2", "struct"} else 4 if form == "vector4" else 1
    payload = "half" + (str(width) if form.startswith("vector") else "")
    declarations = ""
    if form == "alias":
        declarations, payload = "using Value = half;", "Value"
    elif form == "struct":
        declarations, payload = "struct Cell { half first; half second; };", "Cell"
    qualifier = "constant" if form in {"constant", "reference"} else "const device"
    pointer = "&" if form == "reference" else "*"
    expression = "values" if form == "reference" else "values[i]"
    source = f"""#include <metal_stdlib>
using namespace metal;
{declarations}
kernel void half_storage({qualifier} {payload}{pointer} values [[buffer(0)]],
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
        [0x3555] * width
        + (values * count if form == "reference" else values)
        + [0x3555] * width
    )
    initial = [0x3555] * len(expected)
    inputs = {
        "values": {
            "dtype": "float16",
            "shape": [len(values)],
            "encoding": FLOAT16_BITS,
            "values": values,
        },
        "results": {
            "dtype": "float16",
            "shape": [len(initial)],
            "encoding": FLOAT16_BITS,
            "values": initial,
        },
    }
    outputs = {"results": {**inputs["results"], "values": expected}}
    if form == "offset":
        for collection in (inputs, outputs):
            for name, value in list(collection.items()):
                size = len(value["values"]) * 2
                collection[name] = RuntimeValue(
                    name=name,
                    dtype="float16",
                    shape=tuple(value["shape"]),
                    values=value["values"],
                    encoding=FLOAT16_BITS,
                    allocation=RuntimeAllocationView(name, 6, size, size + 14),
                )
    if target == "directx" and form == "reference":
        constant = next(
            binding for binding in descriptor["bindings"] if binding["access"] == "read"
        )
        inputs[constant["scalarLayout"]["memberName"]] = inputs.pop("values")
    request = _request(descriptor, package, inputs, outputs, count)
    return source, descriptor, package, inputs, outputs, request


def _directx_buffers(request):
    state = RuntimeExecutionState(request=request, plan=request.execution_plan)
    adapter = DirectXRuntimeParityAdapter(runtime=SimpleNamespace())
    native = adapter._prepare_dispatch_request(
        state, request.artifact_path, request.artifact_path.with_suffix(".dxil")
    )
    return _prepare_directx_buffers(native.buffers)


def _assert_directx_subview_rejected(request):
    with pytest.raises(RuntimeAdapterSetupError) as error:
        _directx_buffers(request)
    assert error.value.details["reasonKind"] == "unsupported-allocation-subview"
    assert error.value.details["byteOffset"] == 6
    assert error.value.details["targetConstraint"] == "compushady-buffer-view-range"


@pytest.mark.parametrize("target", ("metal", "directx"))
@pytest.mark.parametrize("form", FORMS)
def test_half_packages_retain_two_byte_storage(tmp_path, target, form):
    _, descriptor, _, inputs, _, request = _case(tmp_path, target, form)
    for binding in descriptor["bindings"]:
        layout = binding["scalarLayout"]
        assert layout["elementType"] == "float16"
        assert layout["elementSizeBytes"] == layout["elementStrideBytes"]
        assert layout["elementSizeBytes"] == 2 * layout.get(
            "vectorWidth", layout.get("componentCount", 1)
        )
    if target == "metal":
        _, native = _native_request(request)
        payload, readbacks = MetalComputeRuntime()._prepare_request(native)
        allocations = {item["id"]: item for item in payload["allocations"]}
        for binding in payload["buffers"]:
            name = descriptor["bindings"][binding["index"]]["name"]
            value = inputs[name]
            values = (
                value.values if isinstance(value, RuntimeValue) else value["values"]
            )
            allocation = allocations[binding["allocation"]]
            size = len(values) * 2
            assert binding["length"] == size
            assert binding["offset"] == (6 if form == "offset" else 0)
            assert allocation["length"] == size + (14 if form == "offset" else 0)
            assert allocation["uploads"] == [
                {
                    "offset": binding["offset"],
                    "data": (
                        base64.b64encode(
                            struct.pack("<" + "H" * len(values), *values)
                        ).decode("ascii")
                    ),
                }
            ]
        assert all(value[0] == "float16" for value in readbacks.values())
    elif form == "offset":
        _assert_directx_subview_rejected(request)
    else:
        bound = _bound_values(descriptor, inputs)
        for buffer in _directx_buffers(request):
            values = bound[buffer.name]["values"]
            layout = next(
                binding["scalarLayout"]
                for binding in descriptor["bindings"]
                if binding["name"] == buffer.name
            )
            assert buffer.payload == struct.pack("<" + "H" * len(values), *values)
            assert buffer.dtype == "float16" and buffer.byte_offset == 0
            if buffer.namespace == "cbv":
                assert buffer.byte_length == layout["blockSizeBytes"] == 16
                assert buffer.stride == 0 and buffer.allocation_size == 256
            else:
                assert buffer.byte_length == len(values) * 2
                assert buffer.stride == layout["elementStrideBytes"]
                assert buffer.allocation_size == len(values) * 2


def test_half_encoding_round_trips_every_storage_word():
    words = list(range(65536))
    for target in ("Metal", "DirectX"):
        for alias in ("half", "f16", "float16", "float16_t"):
            assert _normalize_dtype(alias, target=target) == "float16"
        payload = _pack_values(
            words,
            "float16",
            expected_count=len(words),
            target=target,
            encoding=FLOAT16_BITS,
        )
        assert payload == struct.pack("<65536H", *words)
        assert (
            _unpack_values(payload, "float16", target=target, encoding=FLOAT16_BITS)
            == words
        )
    for target in ("OpenGL", "Vulkan"):
        with pytest.raises(RuntimeExecutorUnavailable):
            _normalize_dtype("float16", target=target)


@pytest.mark.parametrize("bad", (True, -1, 65536, 1.0, "nan", None))
def test_half_raw_words_reject_invalid_payloads(bad):
    with pytest.raises(RuntimeExecutorUnavailable):
        _pack_values(
            [bad], "float16", expected_count=1, target="Metal", encoding=FLOAT16_BITS
        )


@pytest.mark.parametrize(
    "bad", (True, "1", None, 65520, -65520, float("nan"), float("inf"))
)
def test_half_numeric_values_do_not_silently_overflow(bad):
    with pytest.raises(RuntimeExecutorUnavailable):
        _pack_values([bad], "float16", expected_count=1, target="Metal")


def test_half_numeric_packing_keeps_signed_zero_subnormals_and_tokens():
    values = [0.0, -0.0, 2.0**-24, 65504.0, "nan", "+infinity", "-infinity"]
    assert _pack_values(
        values, "float16", expected_count=7, target="Metal"
    ) == struct.pack("<7H", 0, 0x8000, 1, 0x7BFF, 0x7E00, 0x7C00, 0xFC00)


@pytest.mark.parametrize(
    "field,value",
    (
        ("elementSizeBytes", 4),
        ("elementStrideBytes", 4),
        ("alignmentBytes", 1),
        ("physicalType", "float"),
        ("memberOffsetBytes", 1),
        ("runtimeSized", False),
    ),
)
def test_half_descriptor_rejects_layout_reinterpretation(tmp_path, field, value):
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


@pytest.mark.parametrize(
    "offset,length,backing", ((1, 32, 48), (6, 31, 48), (6, 32, 37))
)
def test_half_allocation_views_reject_misalignment_and_truncation(
    tmp_path, offset, length, backing
):
    _, descriptor, package, inputs, outputs, _ = _case(tmp_path, "metal", "offset")
    inputs["values"] = replace(
        inputs["values"],
        allocation=RuntimeAllocationView("values", offset, length, backing),
    )
    with pytest.raises(NativeLoaderDispatchError):
        _request(descriptor, package, inputs, outputs, len(WORDS))


def test_half_payload_cannot_bind_widened_opengl_storage(tmp_path):
    with pytest.raises(NativeLoaderDispatchError, match="resource-layout-mismatch"):
        _case(tmp_path, "opengl")


@pytest.mark.parametrize(
    "fault", ("width", "boolean", "negative", "oversized", "numeric", "dtype")
)
def test_half_public_dispatch_rejects_invalid_encoding(tmp_path, fault):
    _, descriptor, package, inputs, outputs, _ = _case(tmp_path, "metal")
    value = inputs["values"]
    if fault == "width":
        value["encoding"] = FLOAT32_BITS
    elif fault == "dtype":
        value["dtype"] = "float32"
    else:
        value["values"][0] = {
            "boolean": True,
            "negative": -1,
            "oversized": 65536,
            "numeric": 1.0,
        }[fault]
    with pytest.raises(NativeLoaderDispatchError, match="value-encoding-invalid"):
        _request(descriptor, package, inputs, outputs, len(WORDS))


@pytest.mark.parametrize("change", (None, "nan", "zero", "bit", "encoding"))
def test_half_storage_comparisons_cannot_use_numeric_tolerance(change):
    expected = RuntimeValue(
        name="out",
        dtype="float16",
        shape=(len(WORDS),),
        values=list(WORDS),
        encoding=FLOAT16_BITS,
    )
    actual = replace(expected, values=list(WORDS))
    if change == "nan":
        actual.values[-1] = 0x7E00
    elif change == "zero":
        actual.values[0] = 0x8000
    elif change == "bit":
        actual.values[6] ^= 1
    elif change == "encoding":
        actual = replace(actual, encoding=None)
    result = _compare_runtime_value(
        expected, actual, default_tolerance=RuntimeTolerance(1e30, 1e30)
    )
    assert result["status"] == ("passed" if change is None else "comparison-failed")
    assert result["comparison"] == "bitwise"
    assert result["tolerance"] == {"absolute": 0.0, "relative": 0.0}


def test_half_native_proofs_are_required_on_all_platforms():
    from tools import ci_coverage

    workflow = Path(".github/workflows/demo-project-testing.yml").read_text()
    for name in (
        "Validate indexed OpenGL gather and resource aggregates",
        "Validate indexed DirectX gather and resource aggregates",
        "Validate Metal byte and vector storage",
    ):
        step = ci_coverage.workflow_step_section(workflow, name)
        assert f'{REQUIRE_ENV}: "1"' in step
        assert "tests/test_translator/test_half_buffer_runtime.py" in step
        assert (
            "-n auto" in step and "continue-on-error" not in step and "if:" not in step
        )


@pytest.mark.parametrize("form", FORMS)
def test_half_buffers_execute_native_storage(tmp_path, form):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required half storage")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    if target == "opengl":
        test_half_payload_cannot_bind_widened_opengl_storage(tmp_path)
        return
    source, descriptor, _, _, outputs, request = _case(tmp_path, target, form)
    if target == "directx" and form == "offset":
        _assert_directx_subview_rejected(request)
        return
    expected = _bound_values(
        descriptor,
        {
            name: (
                {
                    "dtype": value.dtype,
                    "shape": list(value.shape),
                    "values": value.values,
                    "encoding": value.encoding,
                }
                if isinstance(value, RuntimeValue)
                else value
            )
            for name, value in outputs.items()
        },
    )
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="half_storage",
        validate=_validate_half,
    )


@pytest.mark.parametrize("start", (0, 16384, 32768, 49152))
def test_half_native_storage_covers_every_payload(tmp_path, start):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required half storage")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    if target == "opengl":
        test_half_payload_cannot_bind_widened_opengl_storage(tmp_path)
        return
    source, descriptor, _, _, outputs, request = _case(
        tmp_path, target, words=range(start, start + 16384)
    )
    _execute(
        request,
        _bound_values(descriptor, outputs),
        tmp_path,
        original_source=source,
        original_entry="half_storage",
        validate=_validate_half,
    )


def _conversion(root, target):
    source = """#include <metal_stdlib>
using namespace metal;
kernel void half_conversion(const device float* values [[buffer(0)]],
                            device half* results [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
    results[i + 1u] = half(values[i]);
}
"""
    _, descriptor, package = _package(
        root, target, "uint", (1, 1, 1), source=source, software_subgroups=False
    )
    values = [
        0.0,
        -0.0,
        2.0**-24,
        -(2.0**-24),
        2.0**-25,
        3 * 2.0**-25,
        1 + 2.0**-11,
        1 + 3 * 2.0**-11,
        65504.0,
        -65504.0,
        1.0 / 3.0,
        -2.5,
    ]
    bits = [struct.unpack("<H", struct.pack("<e", value))[0] for value in values]
    dtype, encoding, guard = "float16", FLOAT16_BITS, 0x3555
    if target == "opengl":
        bits = [
            struct.unpack(
                "<I",
                struct.pack("<f", struct.unpack("<e", struct.pack("<H", value))[0]),
            )[0]
            for value in bits
        ]
        dtype, encoding, guard = "float32", FLOAT32_BITS, 0x422A0000
    initial = [guard] * (len(values) + 2)
    inputs = {
        "values": {"dtype": "float32", "shape": [len(values)], "values": values},
        "results": {
            "dtype": dtype,
            "shape": [len(initial)],
            "encoding": encoding,
            "values": initial,
        },
    }
    outputs = {"results": {**inputs["results"], "values": [guard, *bits, guard]}}
    request = _request(descriptor, package, inputs, outputs, len(values))
    return source, descriptor, request, _bound_values(descriptor, outputs)


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_half_conversion_exposes_the_target_storage_width(tmp_path, target):
    _, descriptor, request, _ = _conversion(tmp_path, target)
    output = next(value for value in request.fixture.expected_outputs)
    assert output.dtype == ("float32" if target == "opengl" else "float16")
    layout = next(
        binding["scalarLayout"]
        for binding in descriptor["bindings"]
        if binding["access"] == "read_write"
    )
    assert layout["elementSizeBytes"] == (4 if target == "opengl" else 2)


def test_half_conversion_executes_on_each_native_target(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required half storage")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, _, request, expected = _conversion(tmp_path, target)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="half_conversion",
        validate=_validate_half,
    )
