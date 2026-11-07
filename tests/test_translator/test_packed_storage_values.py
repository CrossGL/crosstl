"""Packed little-endian storage is accepted, uploaded and read back exactly."""

import json
import struct

import pytest

from crosstl.project.native_loader_dispatch import NativeLoaderDispatchError
from crosstl.project.native_runtime_drivers import (
    RuntimeExecutorUnavailable,
    _buffer_readback,
    _pack_values,
    _prepare_directx_buffers,
    _prepare_opengl_buffers,
    _unpack_values,
)
from crosstl.project.packed_storage import (
    ELEMENT_SIZES,
    PackedStorageError,
    check_packed_storage,
    is_packed,
    packed_json,
    parse_packed_json,
    unpack_storage,
)
from crosstl.project.runtime_value_encoding import (
    BFLOAT16_BITS,
    FLOAT16_BITS,
    FLOAT32_BITS,
)
from crosstl.project.runtime_verification import (
    RuntimeAdapterDispatchError,
    RuntimeTolerance,
    RuntimeValue,
    RuntimeVerificationError,
    _compare_runtime_value,
    _parse_runtime_value,
)
from tests.test_translator.test_metal_native_runtime import _native_request
from tests.test_translator.test_native_loader_dispatch import _build

# One list of values per dtype, with each width's extremes, and its encoding.
CASES = {
    "bool": ([True, False, True, True], None),
    "int8": ([-128, -1, 0, 127], None),
    "uint8": ([0, 1, 128, 255], None),
    "int16": ([-32768, -1, 0, 32767], None),
    "uint16": ([0, 1, 0x8000, 0xFFFF], None),
    "int32": ([-(1 << 31), -1, 0, (1 << 31) - 1], None),
    "uint32": ([0, 1, 1 << 31, (1 << 32) - 1], None),
    "int64": ([-(1 << 63), -1, 0, (1 << 63) - 1], None),
    "uint64": ([0, 1, 1 << 63, (1 << 64) - 1], None),
    # NaN payloads, signed zero, subnormals and infinities as storage words.
    "float32": ([0x80000000, 0x00000001, 0x7FC12345, 0xFF800000], FLOAT32_BITS),
    "float16": ([0x8000, 0x0001, 0x7E01, 0xFC00], FLOAT16_BITS),
    "bfloat16": ([0x8000, 0x0001, 0x7FC1, 0xFF80], BFLOAT16_BITS),
}
CODES = {
    "bool": "?",
    "int8": "b",
    "uint8": "B",
    "int16": "h",
    "uint16": "H",
    "int32": "i",
    "uint32": "I",
    "int64": "q",
    "uint64": "Q",
    "float32": "I",
    "float16": "H",
    "bfloat16": "H",
}


def _storage(dtype):
    values, _ = CASES[dtype]
    return struct.pack(f"<{len(values)}{CODES[dtype]}", *values)


@pytest.mark.parametrize("dtype", sorted(CASES))
def test_packed_storage_round_trips_every_element_width(dtype):
    values, encoding = CASES[dtype]
    data = _storage(dtype)
    assert ELEMENT_SIZES[dtype] * len(values) == len(data)
    check_packed_storage(data, dtype, encoding, len(values))
    assert unpack_storage(data, dtype, encoding) == values
    assert parse_packed_json(packed_json(data)) == data
    assert is_packed(data) and not is_packed(values)


@pytest.mark.parametrize(
    "data,dtype,encoding,count,reason",
    [
        (bytearray(4), "uint32", None, 1, "value-data-invalid"),
        (memoryview(bytes(4)), "uint32", None, 1, "value-data-invalid"),
        (bytes(4), "float64", None, 1, "value-data-invalid"),
        (bytes(4), None, None, 1, "value-data-invalid"),
        (bytes(4), "float32", None, 1, "value-encoding-invalid"),
        (bytes(2), "float16", None, 1, "value-encoding-invalid"),
        (bytes(4), "uint32", FLOAT32_BITS, 1, "value-encoding-invalid"),
        (bytes(4), "float32", FLOAT16_BITS, 1, "value-encoding-invalid"),
        (bytes(2), "bfloat16", None, 1, "value-encoding-invalid"),
        (bytes(3), "uint32", None, 1, "value-size-mismatch"),
        (bytes(8), "uint32", None, 1, "value-size-mismatch"),
        (bytes(4), "uint32", None, -1, "value-size-mismatch"),
        (bytes(4), "uint32", None, True, "value-size-mismatch"),
        (b"\x00\x02", "bool", None, 2, "value-data-invalid"),
        (b"\xff", "bool", None, 1, "value-data-invalid"),
    ],
)
def test_packed_storage_refuses_what_it_cannot_hold(
    data, dtype, encoding, count, reason
):
    with pytest.raises(PackedStorageError) as caught:
        check_packed_storage(data, dtype, encoding, count)
    assert caught.value.reason == reason


@pytest.mark.parametrize("text", [None, 12, "AA=A", "A", "a b=", "\u00e9"])
def test_packed_json_requires_strict_base64(text):
    with pytest.raises(PackedStorageError):
        parse_packed_json(text)


@pytest.mark.parametrize("target", ["Metal", "OpenGL", "DirectX", "Vulkan"])
@pytest.mark.parametrize("dtype", sorted(CASES))
def test_packed_upload_and_readback_match_the_value_list_path(target, dtype):
    values, encoding = CASES[dtype]
    data = _storage(dtype)
    listed = _pack_values(
        values, dtype, expected_count=len(values), target=target, encoding=encoding
    )
    packed = _pack_values(
        data, dtype, expected_count=len(values), target=target, encoding=encoding
    )
    assert packed == listed == data
    assert _unpack_values(data, dtype, target=target, encoding=encoding) == values
    assert _unpack_values(
        data, dtype, target=target, encoding=encoding, packed=True
    ) == bytes(data)
    readback = _buffer_readback(
        data, dtype, [len(values)], target=target, encoding=encoding, packed=True
    )
    assert readback["values"] == data and readback["shape"] == [len(values)]
    assert readback.get("encoding") == encoding


@pytest.mark.parametrize(
    "data,dtype,encoding,count",
    [
        (bytes(8), "uint32", None, 1),
        (bytes(4), "float32", None, 1),
        (b"\x02", "bool", None, 1),
        (bytes(4), "float32", FLOAT16_BITS, 1),
    ],
)
def test_packed_upload_refuses_invalid_storage(data, dtype, encoding, count):
    with pytest.raises(RuntimeExecutorUnavailable, match="packed buffer storage"):
        _pack_values(data, dtype, expected_count=count, encoding=encoding)


@pytest.mark.parametrize("payload,dtype", [(b"\x00\x02", "bool"), (bytes(3), "uint32")])
def test_packed_readback_keeps_the_layout_checks(payload, dtype):
    with pytest.raises(RuntimeAdapterDispatchError):
        _buffer_readback(payload, dtype, [1], target="Metal", packed=True)


def _packed_inputs(data, **extra):
    return {
        "input_values": {
            "dtype": "float32",
            "shape": [4],
            "encoding": FLOAT32_BITS,
            "values": data,
            **extra,
        }
    }


WORDS = [0x3F800000, 0x7FC12345, 0x80000000, 0x00000001]
DATA = struct.pack("<4I", *WORDS)


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("form", ["bytes", "json", "runtime-value"])
def test_public_dispatch_uploads_packed_storage_and_reads_it_back_packed(
    tmp_path, target, form
):
    if form == "bytes":
        inputs = _packed_inputs(DATA)
    elif form == "json":
        inputs = {
            "input_values": {
                "dtype": "float32",
                "shape": [4],
                "encoding": FLOAT32_BITS,
                "packedValues": packed_json(DATA),
            }
        }
    else:
        inputs = {
            "input_values": RuntimeValue(
                name="input_values",
                dtype="float32",
                shape=(4,),
                values=DATA,
                encoding=FLOAT32_BITS,
            )
        }
    outputs = {
        "output_values": {
            "dtype": "float32",
            "shape": [4],
            "encoding": FLOAT32_BITS,
            "values": bytes(16),
        }
    }
    request = _build(tmp_path, target, input_values=inputs, output_values=outputs)
    (value,) = request.fixture.inputs
    assert value.values is DATA if form != "json" else value.values == DATA
    assert json.loads(json.dumps(value.to_json()))["packedValues"] == packed_json(DATA)
    _, native = _native_request(request)
    prepare = (
        _prepare_directx_buffers if target == "directx" else _prepare_opengl_buffers
    )
    buffers = {item.name: item for item in prepare(native.buffers)}
    assert buffers["input_values"].payload == DATA
    assert buffers["output_values"].readback_packed
    (tmp_path / "listed").mkdir()
    listed = _build(
        tmp_path / "listed",
        target,
        input_values=_packed_inputs(WORDS),
        output_values={
            "output_values": {
                "dtype": "float32",
                "shape": [4],
                "encoding": FLOAT32_BITS,
            }
        },
    )
    _, listed_native = _native_request(listed)
    listed_buffers = {item.name: item for item in prepare(listed_native.buffers)}
    assert listed_buffers["input_values"].payload == DATA
    assert not listed_buffers["output_values"].readback_packed


@pytest.mark.parametrize(
    "inputs,code",
    [
        (_packed_inputs(DATA[:12]), "value-size-mismatch"),
        (_packed_inputs(DATA + bytes(4)), "value-size-mismatch"),
        (
            {"input_values": {"dtype": "float32", "shape": [4], "values": DATA}},
            "value-encoding-invalid",
        ),
        (_packed_inputs(DATA, encoding=FLOAT16_BITS), "value-encoding-invalid"),
        (
            {
                "input_values": {
                    "dtype": "float32",
                    "shape": [4],
                    "encoding": FLOAT32_BITS,
                    "packedValues": packed_json(DATA),
                    "values": WORDS,
                }
            },
            "value-schema-invalid",
        ),
        (
            {
                "input_values": {
                    "dtype": "float32",
                    "shape": [4],
                    "encoding": FLOAT32_BITS,
                    "packedValues": "not base64!",
                }
            },
            "value-data-invalid",
        ),
        (_packed_inputs(bytearray(DATA)), "value-encoding-invalid"),
    ],
)
def test_public_dispatch_refuses_invalid_packed_storage(tmp_path, inputs, code):
    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(tmp_path, "directx", input_values=inputs)
    assert caught.value.code.endswith("." + code), caught.value.code


def test_packed_runtime_values_parse_and_compare_as_their_value_lists():
    value = RuntimeValue(
        name="result",
        dtype="float32",
        shape=(4,),
        values=DATA,
        encoding=FLOAT32_BITS,
    )
    payload = json.loads(json.dumps(value.to_json()))
    assert "values" not in payload
    assert _parse_runtime_value(payload, field_name="value") == value
    listed = RuntimeValue(
        name="result", dtype="float32", shape=(4,), values=WORDS, encoding=FLOAT32_BITS
    )
    tolerance = RuntimeTolerance(absolute=1.0, relative=1.0)
    for expected, actual in ((value, listed), (listed, value), (value, value)):
        assert (
            _compare_runtime_value(expected, actual, default_tolerance=tolerance)[
                "status"
            ]
            == "passed"
        )
    changed = RuntimeValue(
        name="result",
        dtype="float32",
        shape=(4,),
        values=DATA[:-1] + b"\x01",
        encoding=FLOAT32_BITS,
    )
    comparison = _compare_runtime_value(value, changed, default_tolerance=tolerance)
    assert comparison["status"] != "passed" and comparison["mismatchCount"] == 1
    for invalid in (
        {**payload, "packedValues": packed_json(DATA[:12])},
        {**payload, "values": WORDS},
        {key: item for key, item in payload.items() if key != "shape"},
        {key: item for key, item in payload.items() if key != "encoding"},
    ):
        with pytest.raises(RuntimeVerificationError, match="packedValues"):
            _parse_runtime_value(invalid, field_name="value")
