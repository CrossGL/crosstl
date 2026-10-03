"""Preserve native floating atomics through the public Metal project pipeline."""

import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from crosstl.project import ProjectConfig, translate_project
from crosstl.translator import parse
from crosstl.translator.ast import (
    ArrayAccessNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
)
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from tests.test_translator.test_metal_builtin_ownership import _compile

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_FLOAT_ATOMICS"
ROOT = Path(__file__).resolve().parents[2]


def _source(operation, aggregate, contended):
    storage = "Counter" if aggregate else "atomic_float"
    member = ".value" if aggregate else ""
    index = "0u" if contended else "tid"
    return f"""#include <metal_stdlib>
    using namespace metal;
    struct Counter {{ uint before; atomic_float value; uint after; }};
    kernel void update_values(device {storage}* counters [[buffer(0)]],
                              device const uint* values [[buffer(1)]],
                              device uint* results [[buffer(2)]],
                              uint tid [[thread_position_in_grid]]) {{
        uint index = {index};
        uint value_index = tid;
        float previous = atomic_{operation}_explicit(
            &counters[index++]{member}, as_type<float>(values[value_index++]),
            memory_order_relaxed);
        results[tid * 3] = as_type<uint>(previous);
        results[tid * 3 + 1] = index;
        results[tid * 3 + 2] = value_index;
    }}
    """


def _translate(tmp_path, source):
    path = tmp_path / "kernel.metal"
    path.write_text(source, encoding="utf-8")
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            targets=("metal",),
            include_patterns=(path.name,),
            entry_points={path.name: ("update_values",)},
            workgroup_size=(1, 1, 1),
            output_dir="out",
        ),
        format_output=False,
    )
    report.write_json(tmp_path / "report.json")
    payload = report.to_json()
    assert payload["diagnostics"] == []
    assert payload["summary"]["translatedCount"] == 1
    return (tmp_path / payload["artifacts"][0]["path"]).read_text(encoding="utf-8")


@pytest.mark.parametrize("operation", ["fetch_add", "exchange"])
@pytest.mark.parametrize("aggregate", [False, True])
def test_project_preserves_native_float_atomic_operation(
    tmp_path, operation, aggregate
):
    generated = _translate(tmp_path, _source(operation, aggregate, False))
    assert f"atomic_{operation}_explicit(" in generated
    assert "reinterpret_cast<device atomic_float*>" in generated
    assert "unsupported" not in generated
    assert generated.count("counters[index++]") == 1
    assert generated.count("values[value_index++]") == 1


@pytest.mark.parametrize("element_type", ["float", "uint", "Counter"])
@pytest.mark.parametrize("access_kind", ["index", "load"])
def test_buffer_element_type_and_address_follow_storage(element_type, access_kind):
    codegen = MetalCodeGen()
    codegen.local_variable_types["values"] = f"RWStructuredBuffer<{element_type}>"
    codegen.current_address_space_variables["values"] = "device"
    resource = IdentifierNode("values")
    access = (
        ArrayAccessNode(resource, 0)
        if access_kind == "index"
        else FunctionCallNode("buffer_load", [resource, 0])
    )
    assert codegen.expression_result_type(access) == element_type
    assert codegen.argument_address_space(access) == "device"
    codegen.current_address_space_variables["values"] = "thread"
    assert codegen.argument_address_space(access) == "thread"


def test_buffer_load_member_type_uses_its_owner():
    codegen = MetalCodeGen()
    codegen.local_variable_types["values"] = "RWStructuredBuffer<Counter>"
    codegen.current_address_space_variables["values"] = "device"
    codegen.struct_member_types = {
        "Counter": {"value": "float"},
        "Other": {"value": "uint"},
    }
    access = MemberAccessNode(
        FunctionCallNode("buffer_load", [IdentifierNode("values"), 0]), "value"
    )
    assert codegen.expression_result_type(access) == "float"
    assert codegen.argument_address_space(access) == "device"


def test_user_buffer_load_does_not_inherit_resource_storage():
    codegen = MetalCodeGen()
    codegen.user_function_names.add("buffer_load")
    codegen.function_return_types["buffer_load"] = "float"
    codegen.local_variable_types["values"] = "RWStructuredBuffer<uint>"
    codegen.current_address_space_variables["values"] = "device"
    call = FunctionCallNode("buffer_load", [IdentifierNode("values"), 0])
    assert codegen.expression_result_type(call) == "float"
    assert codegen.argument_address_space(call) is None


@pytest.mark.parametrize("access_kind", ["index", "load"])
@pytest.mark.parametrize("storage", ["thread", "constant", "readonly"])
def test_atomic_storage_rejects_nonwritable_or_local_memory(access_kind, storage):
    codegen = MetalCodeGen()
    kind = "StructuredBuffer" if storage == "readonly" else "RWStructuredBuffer"
    codegen.local_variable_types["values"] = f"{kind}<float>"
    codegen.current_address_space_variables["values"] = (
        "device" if storage == "readonly" else storage
    )
    target = (
        ArrayAccessNode(IdentifierNode("values"), 0)
        if access_kind == "index"
        else FunctionCallNode("buffer_load", [IdentifierNode("values"), 0])
    )
    result = codegen.generate_metal_buffer_resource_atomic_call(
        "atomicAdd", [target, 1.5]
    )
    assert "unsupported Metal buffer atomic" in result
    assert "reinterpret_cast" not in result


@pytest.mark.parametrize("operation", ["atomicAdd", "atomicExchange"])
def test_glsl_storage_float_atomics_keep_native_metal_operation(operation):
    source = f"""shader AtomicBuffer {{
        layout(std430, binding=0) buffer Data {{ float values[]; }} data;
        compute {{ void main() {{ {operation}(data.values[0], 1.5); }} }}
    }}"""
    generated = MetalCodeGen().generate_stage(parse(source), "compute")
    assert "unsupported" not in generated
    assert "reinterpret_cast<device atomic_float*>" in generated


@pytest.mark.parametrize(
    "operation", ["atomicMin", "atomicMax", "atomicAnd", "atomicOr", "atomicXor"]
)
def test_float_atomics_reject_operations_without_native_metal_support(operation):
    source = f"""shader AtomicBuffer {{
        buffer float values[1];
        compute {{ void main() {{ {operation}(values[0], 1.5); }} }}
    }}"""
    generated = MetalCodeGen().generate_stage(parse(source), "compute")
    assert "unsupported Metal" in generated
    assert "atomic_float" not in generated


@pytest.fixture(scope="session")
def metal_word_runner(tmp_path_factory):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 to require native float atomic execution")
    assert sys.platform == "darwin", "Native Metal execution requires macOS"
    directory = tmp_path_factory.mktemp("metal-word-runner")
    runner = directory / "readback"
    subprocess.run(
        [
            "swiftc",
            str(
                ROOT / "tests/fixtures/runtime_verification/metal_uint32_buffers.swift"
            ),
            "-o",
            str(runner),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return runner


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


@pytest.mark.parametrize("operation", ["fetch_add", "exchange"])
@pytest.mark.parametrize("aggregate", [False, True])
@pytest.mark.parametrize("contended", [False, True])
def test_native_float_atomic_storage_returns_and_evaluation(
    tmp_path, metal_word_runner, operation, aggregate, contended
):
    source = _source(operation, aggregate, contended)
    generated = _translate(tmp_path, source)
    if contended:
        initial = [_bits(0.0)]
        increments = [
            _bits(1.5 if operation == "fetch_add" else (i + 1) / 2) for i in range(513)
        ]
    else:
        pairs = [
            (0, 0),
            (0x80000000, 0x80000000),
            (0, 0x80000000),
            (0x3F800000, 0x3FC00000),
            (0xBF800000, 0x3FC00000),
            (0x7F800000, 0xFF800000),
            (0xFF800000, 0x3F800000),
            (0x7FC12345, 0x3F800000),
            (0x3F800000, 0xFFC54321),
            (0x7F7FFFFF, 0x7F7FFFFF),
            (0x00800000, 0x80800000),
            (1, 0),
            (1, 1),
            (0xFF7FFFFF, 0xFF7FFFFF),
        ]
        initial, increments = map(list, zip(*pairs))
    storage = (
        [
            word
            for index, value in enumerate(initial)
            for word in (0x12340000 + index, value, 0x56780000 + index)
        ]
        if aggregate
        else initial
    )
    request = {
        "inputs": [storage, increments],
        "outputCount": len(increments) * 3,
        "threadCount": len(increments),
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    records = {}
    for label, shader in (("original", source), ("generated", generated)):
        directory = tmp_path / label
        directory.mkdir()
        artifact, module = _compile(shader, "metal", directory)
        assert module.is_file()
        result = subprocess.run(
            [str(metal_word_runner), str(module), "update_values", str(request_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        (directory / "readback.json").write_text(result.stdout, encoding="utf-8")
        data = json.loads(result.stdout)
        outputs = data["values"]
        assert outputs == data["buffers"][2]
        assert data["buffers"][1] == increments
        assert outputs[1::3] == (
            [1] * len(increments) if contended else list(range(1, len(increments) + 1))
        )
        assert outputs[2::3] == list(range(1, len(increments) + 1))
        previous = outputs[::3]
        final = data["buffers"][0]
        if aggregate:
            assert final[::3] == storage[::3]
            assert final[2::3] == storage[2::3]
            final = final[1::3]
        if contended:
            if operation == "fetch_add":
                assert sorted(previous) == [
                    _bits(i * 1.5) for i in range(len(increments))
                ]
                assert final == [_bits(len(increments) * 1.5)]
            else:
                assert sorted(previous + final) == sorted(initial + increments)
                assert final[0] in increments
        else:
            assert previous == initial
            if operation == "exchange":
                assert final == increments
        records[label] = {
            "previous": sorted(previous) if contended else previous,
            "final": final,
            "artifactSha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "moduleSha256": hashlib.sha256(module.read_bytes()).hexdigest(),
            "device": data["device"],
        }
    if not contended or operation == "fetch_add":
        assert records["original"]["previous"] == records["generated"]["previous"]
        assert records["original"]["final"] == records["generated"]["final"]
    (tmp_path / "evidence.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
