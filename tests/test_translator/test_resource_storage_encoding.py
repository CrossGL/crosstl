"""Logical binary16 values can use explicitly declared integer buffer storage."""

import copy
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from crosstl.project.host_reflection import reflect_target_host_interface
from crosstl.project.native_loader_abi import build_native_loader_abi_descriptor
from crosstl.project.native_loader_dispatch import (
    NativeLoaderDispatchError,
    _validated_scalar_layout,
    build_native_loader_dispatch_request,
)
from crosstl.project.native_runtime_drivers import _pack_values, _scalar_block_size
from crosstl.project.runtime_verification import RuntimeAdapterSetupError, RuntimeValue
from crosstl.translator.resource_storage import (
    BINARY16_STORAGE,
    MAX_RESOURCE_STORAGE_HEADER_BYTES,
    RESOURCE_STORAGE_PREFIX,
    apply_resource_storage,
    parse_resource_storage_header,
    resource_storage_header,
)
from tests.test_translator.test_half_buffer_runtime import _validate_half
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_native_loader_abi import _load_unit

CONTRACTS = {name: dict(BINARY16_STORAGE) for name in ("input_values", "output_values")}
FORMS = {"scalar": 1, "vector2": 2, "vector4": 4, "struct": 2}
REQUIRE_ENV = "CROSTL_REQUIRE_HALF_BUFFER_RUNTIME"


def _source(form="scalar", arithmetic=False):
    width = FORMS[form]
    declaration = ""
    physical = "uint16_t" + (str(width) if width != 1 else "")
    operation = " + float16_t(0.5)" if arithmetic else ""
    if form == "struct":
        declaration = "struct Storage { uint16_t first; uint16_t second; };\n"
        physical = "Storage"
        body = "\n".join(
            f"output_values[i + 1u].{member} = "
            f"asuint16(asfloat16(input_values[i].{member}){operation});"
            for member in ("first", "second")
        )
    else:
        body = (
            "output_values[i + 1u] = "
            f"asuint16(asfloat16(input_values[i]){operation});"
        )
    return resource_storage_header(CONTRACTS) + f"""{declaration}
StructuredBuffer<{physical}> input_values : register(t0);
RWStructuredBuffer<{physical}> output_values : register(u1);
[numthreads(1, 1, 1)] void CSMain(uint3 position : SV_DispatchThreadID) {{
    uint i = position.x;
    {body}
}}
"""


def _descriptor(root, source):
    path = root / "artifacts/directx/copy.hlsl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    reflected = reflect_target_host_interface(path, target="directx")
    assert reflected["status"] == "ready", reflected
    unit = _load_unit()
    unit.update(
        hostInterface=reflected,
        hash={
            "algorithm": "sha256",
            "value": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        sizeBytes=path.stat().st_size,
        source="copy.hlsl",
        sourcePath="copy.hlsl",
        sourceBackend="directx",
        sourceHash={
            "algorithm": "sha256",
            "value": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        sourceSizeBytes=path.stat().st_size,
        entryPoint={"source": "CSMain", "target": "CSMain", "stage": "compute"},
        provenance={"pipeline": "handwritten-storage-abi-control", "target": "directx"},
        sourceRemap=None,
        specializationConstants=[],
    )
    descriptor = build_native_loader_abi_descriptor(unit)
    (root / "descriptor.json").write_text(
        json.dumps(descriptor, indent=2), encoding="utf-8"
    )
    return descriptor


def _values(words):
    return {
        "dtype": "float16",
        "encoding": "ieee754-binary16",
        "shape": [len(words)],
        "values": list(words),
    }


def _request(root, form, words, *, arithmetic=False):
    descriptor = _descriptor(root, _source(form, arithmetic))
    width = FORMS[form]
    expected_words = list(words)
    if arithmetic:
        expected_words = [
            struct.unpack(
                "<H",
                struct.pack(
                    "<e", struct.unpack("<e", struct.pack("<H", word))[0] + 0.5
                ),
            )[0]
            for word in words
        ]
    guards = [0x3555] * width
    expected = {"output_values": _values(guards + expected_words + guards)}
    request = build_native_loader_dispatch_request(
        descriptor,
        root,
        {
            "input_values": _values(words),
            "output_values": _values([0x3555] * (len(words) + 2 * width)),
        },
        expected,
        {"workgroupCount": [len(words) // width, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target="directx",
    )
    return request, expected


def test_storage_header_is_canonical_and_detached():
    header = resource_storage_header(CONTRACTS)
    assert header == resource_storage_header(dict(reversed(list(CONTRACTS.items()))))
    decoded = parse_resource_storage_header(header + "void main() {}\n")
    assert decoded == CONTRACTS
    decoded["input_values"]["encoding"] = "changed"
    assert CONTRACTS["input_values"] == BINARY16_STORAGE
    assert parse_resource_storage_header("// ordinary comment\nvoid main() {}") == {}


@pytest.mark.parametrize(
    "payload",
    [
        "null",
        "[]",
        "{}",
        '{"schemaVersion":true,"resources":{}}',
        '{"schemaVersion":2,"resources":{}}',
        '{"schemaVersion":1,"resources":{},"extra":1}',
        '{"schemaVersion":1,"resources":{}}',
        '{"schemaVersion":1,"schemaVersion":1,"resources":{}}',
        '{"schemaVersion":1,"resources":{"input_values":null}}',
        '{"schemaVersion":1,"resources":{"input_values":NaN}}',
        '{"schemaVersion":1,"resources":{"input_values":{"logicalElementType":"float16","encoding":"ieee754-binary16","encoding":"ieee754-binary16"}}}',
        "[" * 2000,
    ],
)
def test_invalid_storage_json_is_rejected(payload):
    with pytest.raises(ValueError):
        parse_resource_storage_header(RESOURCE_STORAGE_PREFIX + payload + "\n")


@pytest.mark.parametrize(
    "resources",
    [
        {},
        {"bad name": BINARY16_STORAGE},
        {"\u03bb": BINARY16_STORAGE},
        {"input_values": {**BINARY16_STORAGE, "extra": True}},
        {"input_values": {**BINARY16_STORAGE, "encoding": "bfloat16-bits"}},
        {"input_values": {**BINARY16_STORAGE, "logicalElementType": "uint16"}},
        {f"value{i}": BINARY16_STORAGE for i in range(257)},
    ],
)
def test_invalid_storage_contracts_are_not_serialized(resources):
    with pytest.raises(ValueError):
        resource_storage_header(resources)


@pytest.mark.parametrize("prefix", ["\n", " ", "// comment\n", "/*\n", "void f() {}\n"])
def test_storage_contract_cannot_be_moved_into_shader_body(prefix):
    with pytest.raises(ValueError, match="first line"):
        parse_resource_storage_header(prefix + resource_storage_header(CONTRACTS))


def test_duplicate_and_oversized_headers_are_rejected():
    header = resource_storage_header(CONTRACTS)
    with pytest.raises(ValueError, match="once"):
        parse_resource_storage_header(header + header)
    with pytest.raises(ValueError, match="byte limit"):
        parse_resource_storage_header(
            RESOURCE_STORAGE_PREFIX + " " * MAX_RESOURCE_STORAGE_HEADER_BYTES
        )
    with pytest.raises(ValueError, match="byte limit"):
        resource_storage_header(
            {"x" * MAX_RESOURCE_STORAGE_HEADER_BYTES: BINARY16_STORAGE}
        )


@pytest.mark.parametrize("form", FORMS)
def test_storage_reflection_and_loader_preserve_both_types(tmp_path, form):
    request, _ = _request(tmp_path, form, range(16))
    descriptor = json.loads((tmp_path / "descriptor.json").read_text())
    for binding in descriptor["bindings"]:
        layout = binding["scalarLayout"]
        assert layout["elementType"] == "uint16"
        assert layout["storageEncoding"] == BINARY16_STORAGE
        assert layout["elementStrideBytes"] == 2 * FORMS[form]
        assert layout["alignmentBytes"] == 2
    assert all(value.dtype == "float16" for value in request.fixture.inputs)
    assert all(value.encoding == "ieee754-binary16" for value in request.fixture.inputs)


@pytest.mark.parametrize(
    "replacement",
    [
        "StructuredBuffer<float16_t>",
        "StructuredBuffer<int16_t>",
        "StructuredBuffer<uint>",
        "StructuredBuffer<uint16_t3x3>",
        "Buffer<uint16_t>",
        "Texture1D<uint16_t>",
        "ByteAddressBuffer",
    ],
)
def test_reflection_cannot_relabel_incompatible_physical_storage(tmp_path, replacement):
    source = _source().replace(
        "StructuredBuffer<uint16_t> input_values", f"{replacement} input_values"
    )
    path = tmp_path / "invalid.hlsl"
    path.write_text(source)
    result = reflect_target_host_interface(path, target="directx")
    assert result["status"] == "failed"
    assert result["resources"] == []
    assert result["diagnosticRecords"][0]["details"]["contract"] == "resource-storage"


@pytest.mark.parametrize(
    "source",
    [
        RESOURCE_STORAGE_PREFIX + "{}\n",
        resource_storage_header({"missing": BINARY16_STORAGE})
        + "[numthreads(1,1,1)] void CSMain() {}",
        _source() + "StructuredBuffer<uint16_t> input_values : register(t2);",
    ],
)
def test_invalid_header_or_missing_resource_fails_reflection(tmp_path, source):
    path = tmp_path / "invalid.hlsl"
    path.write_text(source)
    result = reflect_target_host_interface(path, target="directx")
    assert result["status"] == "failed"
    assert result["resources"] == []


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "encoding",
        "logical",
        "physical",
        "stride",
        "alignment",
        "offset",
        "width",
        "extra",
    ],
)
def test_dispatch_rejects_inconsistent_storage_contracts(tmp_path, form, mutation):
    descriptor = _descriptor(tmp_path, _source(form))
    layout = descriptor["bindings"][0]["scalarLayout"]
    if mutation == "missing":
        layout.pop("storageEncoding")
    elif mutation == "encoding":
        layout["storageEncoding"]["encoding"] = "bfloat16-bits"
    elif mutation == "logical":
        layout["storageEncoding"]["logicalElementType"] = "uint16"
    elif mutation == "extra":
        layout["storageEncoding"]["optional"] = True
    elif mutation == "physical":
        if form == "struct":
            layout["structMembers"][0]["physicalType"] = "float16_t"
        else:
            layout["physicalType"] = "float16_t"
    elif mutation == "width":
        layout["elementType"] = "uint32"
    else:
        layout[
            {
                "stride": "elementStrideBytes",
                "alignment": "alignmentBytes",
                "offset": "memberOffsetBytes",
            }[mutation]
        ] = 3
    descriptor["scalarLayout"]["bindings"] = [
        {"binding": binding["name"], "layout": copy.deepcopy(binding["scalarLayout"])}
        for binding in descriptor["bindings"]
    ]
    with pytest.raises(NativeLoaderDispatchError):
        build_native_loader_dispatch_request(
            descriptor,
            tmp_path,
            {"input_values": _values(range(16)), "output_values": _values(range(24))},
            {"output_values": _values(range(24))},
            {"workgroupCount": [1, 1, 1], "workgroupSize": [1, 1, 1]},
        )


def test_binary16_bit_storage_preserves_every_host_payload():
    words = list(range(65536))
    assert _pack_values(
        words, "float16", expected_count=len(words), encoding="ieee754-binary16"
    ) == struct.pack("<65536H", *words)


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("dtype", ["float16", "uint16", "float32"])
def test_storage_codec_requires_matching_logical_type_and_target(
    tmp_path, target, dtype
):
    descriptor = _descriptor(tmp_path, _source())
    layout = descriptor["bindings"][0]["scalarLayout"]
    value = RuntimeValue(
        name="input_values", kind="buffer", dtype=dtype, shape=(4,), values=[0] * 4
    )
    if target == "directx" and dtype == "float16":
        validated = _validated_scalar_layout(
            layout,
            runtime_value=value,
            target=target,
            resource_kind="buffer",
            path="$.layout",
        )
        assert validated == layout and validated is not layout
    else:
        with pytest.raises(NativeLoaderDispatchError):
            _validated_scalar_layout(
                layout,
                runtime_value=value,
                target=target,
                resource_kind="buffer",
                path="$.layout",
            )


def test_storage_contract_application_is_atomic(tmp_path):
    source = tmp_path / "physical.hlsl"
    source.write_text(_source().split("\n", 1)[1])
    resources = reflect_target_host_interface(source, target="directx")["resources"]
    before = copy.deepcopy(resources)
    with pytest.raises(ValueError):
        apply_resource_storage(
            resources, {"input_values": BINARY16_STORAGE, "missing": BINARY16_STORAGE}
        )
    assert resources == before


@pytest.mark.parametrize(
    "physical,logical", [("uint16_t", "uint16"), ("float16_t", "float16")]
)
def test_unannotated_storage_keeps_its_physical_contract(tmp_path, physical, logical):
    source = tmp_path / "physical.hlsl"
    source.write_text(_source().split("\n", 1)[1].replace("uint16_t", physical))
    reflected = reflect_target_host_interface(source, target="directx")
    assert reflected["status"] == "ready"
    for resource in reflected["resources"]:
        assert resource["scalarLayout"]["elementType"] == logical
        assert "storageEncoding" not in resource["scalarLayout"]


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("quarter", range(4))
def test_binary16_storage_executes_exact_payloads(tmp_path, form, quarter):
    if sys.platform != "win32" or os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip("required native DirectX storage test runs on Windows")
    words = range(quarter * 16384, (quarter + 1) * 16384)
    request, expected = _request(tmp_path, form, words)
    _execute(request, expected, tmp_path, validate=_validate_half)


@pytest.mark.parametrize("form", FORMS)
def test_binary16_storage_decodes_for_arithmetic(tmp_path, form):
    if sys.platform != "win32" or os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip("required native DirectX storage test runs on Windows")
    words = [
        struct.unpack("<H", struct.pack("<e", i / 32 - 16))[0] for i in range(1024)
    ]
    request, expected = _request(tmp_path, form, words, arithmetic=True)
    _execute(request, expected, tmp_path, validate=_validate_half)


def test_storage_contract_controls_join_existing_windows_job():
    from tools import ci_coverage

    workflow = Path(".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate indexed DirectX gather and resource aggregates"
    )
    assert "tests/test_translator/test_resource_storage_encoding.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "-n auto" in step
    assert "continue-on-error" not in step


def _constant_layout(root, width):
    physical = "uint16_t" + (str(width) if width != 1 else "")
    source = resource_storage_header({"Constants": BINARY16_STORAGE})
    source += f"cbuffer Constants : register(b0) {{ {physical} value; }};\n"
    source += "[numthreads(1, 1, 1)] void CSMain() {}\n"
    descriptor = _descriptor(root, source)
    return descriptor["bindings"][0]["scalarLayout"]


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_constant_storage_codec_retains_fixed_layout(tmp_path, width):
    layout = _constant_layout(tmp_path, width)
    assert layout["storageEncoding"] == BINARY16_STORAGE
    assert layout["elementType"] == "uint16" and layout["runtimeSized"] is False
    runtime_value = RuntimeValue(
        name="value", kind="buffer", dtype="float16", shape=(width,), values=[0] * width
    )
    assert (
        _validated_scalar_layout(
            layout,
            runtime_value=runtime_value,
            target="directx",
            resource_kind="constant-buffer",
            path="$.layout",
        )
        == layout
    )
    binding = SimpleNamespace(
        name="Constants", binding=SimpleNamespace(metadata={"scalarLayout": layout})
    )
    assert (
        _scalar_block_size(
            binding, target="directx", dtype="float16", payload_size=2 * width
        )
        == 16
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"storageEncoding": None},
        {"storageEncoding": {**BINARY16_STORAGE, "encoding": "unknown"}},
        {"storageEncoding": {**BINARY16_STORAGE, "extra": True}},
        {"elementType": "float16"},
        {"physicalType": "float16_t"},
        {"runtimeSized": True},
        {"storageLayout": "hlsl-structured-buffer"},
        {"alignmentBytes": 3},
        {"elementStrideBytes": 4},
        {"blockSizeBytes": 1},
        {"memberOffsetBytes": 16},
    ],
)
def test_constant_storage_codec_rejects_mismatches_in_loader_and_driver(
    tmp_path, mutation
):
    layout = {**_constant_layout(tmp_path, 1), **mutation}
    runtime_value = RuntimeValue(
        name="value", kind="buffer", dtype="float16", shape=(1,), values=[0]
    )
    with pytest.raises(NativeLoaderDispatchError):
        _validated_scalar_layout(
            layout,
            runtime_value=runtime_value,
            target="directx",
            resource_kind="constant-buffer",
            path="$.layout",
        )
    binding = SimpleNamespace(
        name="Constants", binding=SimpleNamespace(metadata={"scalarLayout": layout})
    )
    with pytest.raises(RuntimeAdapterSetupError):
        _scalar_block_size(binding, target="directx", dtype="float16", payload_size=2)
