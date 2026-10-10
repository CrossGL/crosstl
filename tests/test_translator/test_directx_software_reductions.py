"""Exact-width reductions independent of the physical DirectX wave size."""

import hashlib
import json
import math
import os
import struct
import sys
from types import SimpleNamespace

import pytest

from crosstl.project import ProjectConfig, translate_project
from crosstl.project.native_runtime_drivers import DirectXComputeRuntime
from crosstl.project.runtime_verification import (
    NativeRuntimeBufferBinding,
    NativeRuntimeDispatchRequest,
    RuntimeAllocationView,
    RuntimeDispatchGeometry,
    RuntimeResourceBinding,
)
from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import (
    DirectXSoftwareSubgroupError,
    HLSLCodeGen,
)
from tests.test_translator.test_directx_float_atomics import _compile
from tools.run_directx_diagnostics import _loaded_runtime_libraries

REQUIRE_ENV = "CROSTL_REQUIRE_DIRECTX_SOFTWARE_REDUCTIONS"


def _codegen():
    return HLSLCodeGen(
        software_subgroup_width=32, relative_wave_shuffle_out_of_range="self"
    )


def _source(value_type, body=None, helpers=""):
    cast = {"float": "asfloat", "int": "asint", "uint": "uint"}[value_type]
    if body is None:
        body = f"""
            {value_type} value = {cast}(inputWords[index]);
            {value_type} total = WaveActiveSum(value);
            {value_type} smallest = WaveActiveMin(value);
            {value_type} largest = WaveActiveMax(value);
            {value_type} shifted = WaveShuffleDown(value, 1u);
            {value_type} repeated = WaveActiveSum(shifted);
            outputWords[index * 5u] = asuint(total);
            outputWords[index * 5u + 1u] = asuint(smallest);
            outputWords[index * 5u + 2u] = asuint(largest);
            outputWords[index * 5u + 3u] = asuint(shifted);
            outputWords[index * 5u + 4u] = asuint(repeated);
        """
    return f"""shader SoftwareReductions {{
        StructuredBuffer<uint> inputWords @register(t0);
        RWStructuredBuffer<uint> outputWords @register(u1);
        {helpers}
        compute {{
            @numthreads(32, 4, 1)
            void main(uint invocation @gl_LocalInvocationIndex,
                      uvec3 group @gl_WorkGroupID) {{
                uint index = group.x * 128u + invocation;
                {body}
            }}
        }}
    }}"""


@pytest.mark.parametrize("value_type", ["float", "int", "uint"])
def test_software_reductions_compile_and_share_scratch(tmp_path, value_type):
    source = _source(value_type)
    generated = _codegen().generate_stage(parse(source), "compute")
    assert generated.count(f"groupshared {value_type} ") == 1
    assert generated.count("GroupMemoryBarrierWithGroupSync();") == 8
    assert "WaveActive" not in generated and "WaveReadLane" not in generated
    assert "WaveSize" not in generated
    _compile(generated, tmp_path)


@pytest.mark.parametrize("operation", ["Sum", "Min", "Max"])
@pytest.mark.parametrize(
    "body",
    [
        "if (invocation < 16u) { outputWords[index] = WaveActiveOP(index); }",
        "if (invocation == 0u) { return; } outputWords[index] = WaveActiveOP(index);",
        "outputWords[index] = invocation == 0u ? WaveActiveOP(index) : 0u;",
        "bool selected = invocation == 0u && WaveActiveOP(index) > 0u;",
        "for (uint i = 0u; i < invocation; ++i) { outputWords[index] = WaveActiveOP(i); }",
    ],
)
def test_software_reductions_reject_divergent_barriers(operation, body):
    with pytest.raises(DirectXSoftwareSubgroupError):
        _codegen().generate_stage(
            parse(_source("uint", body.replace("OP", operation))), "compute"
        )


@pytest.mark.parametrize("value_type", ["float2", "double", "int64_t", "float16_t"])
def test_software_reductions_reject_unsupported_payloads(value_type):
    source = _source(
        "uint",
        f"{value_type} value = {value_type}(1); {value_type} total = WaveActiveSum(value);",
    )
    with pytest.raises(DirectXSoftwareSubgroupError, match="32-bit"):
        _codegen().generate_stage(parse(source), "compute")


def test_software_reductions_keep_helper_identity_and_single_evaluation(tmp_path):
    source = _source(
        "uint",
        """uint value = index;
        uint __crossgl_software_subgroup_sum_uint = 17u;
        outputWords[index] = reduce_value(value++) + __crossgl_software_subgroup_sum_uint;
        """,
        "uint reduce_value(uint value) { return WaveActiveSum(value); }",
    )
    generated = _codegen().generate_stage(parse(source), "compute")
    assert generated.count("value++") == 1
    assert "uint __crossgl_software_subgroup_sum_uint_1(" in generated
    _compile(generated, tmp_path)
    reused = _codegen()
    reused.generate_stage(parse(source), "compute")
    assert reused.generate_stage(
        parse(_source("float")), "compute"
    ) == _codegen().generate_stage(parse(_source("float")), "compute")


def test_native_wave_reductions_remain_available(tmp_path):
    generated = HLSLCodeGen(relative_wave_shuffle_out_of_range="self").generate_stage(
        parse(_source("float")), "compute"
    )
    assert "WaveActiveSum(" in generated and "WaveActiveMin(" in generated
    assert "WaveActiveMax(" in generated
    assert "__crossgl_software_subgroup" not in generated
    _compile(generated, tmp_path)


@pytest.mark.parametrize("value_type", ["float", "int", "uint"])
def test_metal_project_reductions_compile_as_software_hlsl(tmp_path, value_type):
    source = tmp_path / "kernel.metal"
    source.write_text(f"""#include <metal_stdlib>
    using namespace metal;
    kernel void reduce_values(device {value_type}* output [[buffer(0)]],
                              uint index [[thread_index_in_threadgroup]]) {{
        {value_type} value = {value_type}(index + 1u);
        {value_type} maximum = simd_max(value);
        {value_type} minimum = simd_min(value);
        {value_type} total = simd_sum(value);
        output[index] = maximum + minimum + total;
    }}""")
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=(source.name,),
            targets=("directx",),
            output_dir="out",
            entry_points={source.name: ("reduce_values",)},
            workgroup_size=(32, 4, 1),
            source_options={
                "metal": {
                    "target_options": {
                        "directx": {
                            "software_subgroup_width": 32,
                            "relative_wave_shuffle_out_of_range": "self",
                        }
                    }
                }
            },
        ),
        format_output=False,
    )
    report.write_json(tmp_path / "report.json")
    payload = report.to_json()
    assert payload["diagnostics"] == []
    assert payload["summary"]["translatedCount"] == 1
    generated = (tmp_path / payload["artifacts"][0]["path"]).read_text()
    assert "WaveActive" not in generated
    assert generated.count("GroupMemoryBarrierWithGroupSync();") == 6
    _compile(generated, tmp_path)


def _word(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _float(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]


def _inputs(value_type):
    if value_type == "uint":
        return [
            (0xFFFFFF00 + lane * 17 + group * 997) & 0xFFFFFFFF
            for group in range(12)
            for lane in range(32)
        ]
    if value_type == "int":
        return [0x7FFFFFFF] * 32 + [
            (group * 1009 - lane * 127) & 0xFFFFFFFF
            for group in range(1, 12)
            for lane in range(32)
        ]
    return [
        word
        for group in (
            [_word((lane - 16) / 4) for lane in range(32)],
            [_word(16777216.0), _word(1.0), _word(-16777216.0), _word(1.0)] * 8,
            [0x7FC00001, _word(3.0), _word(-7.0), 0xFFC00002] * 8,
            [0x7FC00001, 0xFFC00002] * 16,
            [0x7F800000] + [_word(3.0)] * 31,
            [0xFF800000] + [_word(-3.0)] * 31,
            [0x7F800000, 0xFF800000] + [_word(2.0)] * 30,
            [0x80000000, 0] * 16,
            [0x80000000] * 32,
            [0] * 32,
            [_word((lane % 2 * 2 - 1) * 8) for lane in range(32)],
            [_word(-lane - 1) for lane in range(32)],
        )
        for word in group
    ]


def _reduction_reference(words, value_type):
    if value_type != "float":
        values = [
            word - 0x100000000 if value_type == "int" and word >= 0x80000000 else word
            for word in words
        ]
        return [
            sum(values) & 0xFFFFFFFF,
            min(values) & 0xFFFFFFFF,
            max(values) & 0xFFFFFFFF,
        ]
    values = [_float(word) for word in words]
    total = values[0]
    for value in values[1:]:
        total = _float(_word(total + value))
    numeric = [value for value in values if not math.isnan(value)]
    limits = [min(numeric), max(numeric)] if numeric else [math.nan, math.nan]
    results = [_word(total), *map(_word, limits)]
    zeros = [word for word in words if word & 0x7FFFFFFF == 0]
    if numeric and min(numeric) == 0:
        results[1] = 0x80000000 if 0x80000000 in zeros else 0
    if numeric and max(numeric) == 0:
        results[2] = 0 if 0 in zeros else 0x80000000
    return results


def _expected(words, value_type):
    output = []
    for start in range(0, len(words), 32):
        lanes = words[start : start + 32]
        reductions = _reduction_reference(lanes, value_type)
        shifted = lanes[1:] + lanes[-1:]
        repeated = _reduction_reference(shifted, value_type)[0]
        for word in shifted:
            output.extend([*reductions, word, repeated])
    return output


@pytest.fixture
def directx_runtime(monkeypatch):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for DirectX numerical execution")
    assert sys.platform == "win32", "DirectX execution requires Windows"
    monkeypatch.setenv("CROSTL_REQUIRE_DIRECTX_FLOAT_ATOMICS", "1")


@pytest.mark.parametrize("value_type", ["float", "int", "uint"])
def test_software_reductions_execute(tmp_path, directx_runtime, value_type):
    generated = _codegen().generate_stage(parse(_source(value_type)), "compute")
    words = _inputs(value_type)
    expected = _expected(words, value_type)
    actual = _execute_words(tmp_path, generated, words, expected, value_type=value_type)
    for index, (got, want) in enumerate(zip(actual, expected)):
        if value_type == "float" and want & 0x7FFFFFFF > 0x7F800000:
            assert got & 0x7FFFFFFF > 0x7F800000, (index, got, want)
        else:
            assert got == want, (index, got, want)


def _execute_words(
    tmp_path,
    generated,
    words,
    expected,
    *,
    value_type="uint",
    workgroup_size=(32, 4, 1),
    workgroup_count=(3, 1, 1),
    compile_flags=(),
    buffer_byte_offset=None,
    buffer_view_executor=None,
    module_compiler=None,
):
    compiler = _compile if module_compiler is None else module_compiler
    artifact, module = compiler(generated, tmp_path, flags=compile_flags)
    guard = [0x6A15BEEF] * 32
    inputs = {
        "inputWords": words + guard,
        "outputWords": [0xDEADBEEF] * len(expected) + guard,
    }
    buffers = {
        name: NativeRuntimeBufferBinding(
            name=name,
            binding=RuntimeResourceBinding(
                name=name,
                kind="buffer",
                set=0,
                binding=slot,
                type_name=("RW" if slot else "") + "StructuredBuffer<uint>",
                access="read_write" if slot else "read",
                metadata={"byteStride": 4},
            ),
            source="expectedOutput" if slot else "input",
            dtype="uint32",
            shape=(len(values),),
            value=values,
            allocation=(
                RuntimeAllocationView(
                    allocation_id=name,
                    byte_offset=buffer_byte_offset,
                    byte_length=len(values) * 4,
                    allocation_byte_length=buffer_byte_offset + len(values) * 4 + 16,
                )
                if buffer_byte_offset is not None
                else None
            ),
        )
        for slot, (name, values) in enumerate(inputs.items())
    }
    request = NativeRuntimeDispatchRequest(
        target="directx",
        artifact={"target": "directx"},
        artifact_path=artifact,
        module_path=module,
        loaded_artifact=module.read_bytes(),
        buffers=buffers,
        constants={},
        entry_point="CSMain",
        dispatch=RuntimeDispatchGeometry(
            entry_point="CSMain",
            workgroup_size=workgroup_size,
            workgroup_count=workgroup_count,
        ),
    )
    (tmp_path / "inputs.json").write_text(json.dumps(inputs))
    (tmp_path / "expected.json").write_text(json.dumps(expected))
    state = SimpleNamespace(details={})
    runtime = DirectXComputeRuntime(
        **(
            {"buffer_view_executor": buffer_view_executor}
            if buffer_view_executor is not None
            else {}
        )
    )
    available = runtime.is_available(None, request)
    evidence = {
        "type": value_type,
        "logicalWidth": 32,
        "workgroupSize": list(workgroup_size),
        "workgroupCount": list(workgroup_count),
        "bufferByteOffset": buffer_byte_offset,
        "availability": {
            "available": available.available,
            "reason": available.reason,
            "details": dict(available.details),
        },
        "artifactSha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "moduleSha256": hashlib.sha256(module.read_bytes()).hexdigest(),
    }

    def retain_runtime():
        evidence["runtime"] = state.details
        try:
            evidence["runtimeLibraries"] = _loaded_runtime_libraries()
        except Exception as error:
            evidence["runtimeLibraryError"] = str(error)
        (tmp_path / "evidence.json").write_text(json.dumps(evidence, indent=2))

    retain_runtime()
    assert available.available, available
    try:
        outputs = runtime.dispatch(None, state, request)
        (tmp_path / "readback.json").write_text(json.dumps(outputs))
    except Exception as error:
        evidence["dispatchError"] = {
            "type": type(error).__name__,
            "message": str(error),
            "details": getattr(error, "details", {}),
        }
        raise
    finally:
        # Device creation can load additional DLLs. Retain identity on failures too.
        retain_runtime()
    if buffer_byte_offset is not None:
        assert state.details["directxRuntime"]["runtime"] == "native-buffer-views"
    actual = outputs["outputWords"]["values"]
    assert actual[len(expected) :] == guard
    assert len(actual) == len(expected) + len(guard)
    return actual[: len(expected)]


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("buffer_byte_offset", [None, 16])
@pytest.mark.parametrize("injected", [False, True])
@pytest.mark.parametrize("compiler_injected", [False, True])
def test_word_executor_retains_runtime_identity(
    tmp_path, monkeypatch, failure, buffer_byte_offset, injected, compiler_injected
):
    module = sys.modules[__name__]
    artifact = tmp_path / "translated.hlsl"
    artifact.write_text("source")
    binary = tmp_path / "translated.dxil"
    binary.write_bytes(b"compiled")
    flags = []

    def compile_source(source, work, **options):
        assert source == "source" and work == tmp_path
        flags.append(options["flags"])
        return artifact, binary

    class Runtime:
        def __init__(self, **kwargs):
            assert kwargs == ({"buffer_view_executor": executor} if injected else {})

        def is_available(self, adapter, request):
            return SimpleNamespace(
                available=True,
                reason=None,
                details={"device": "Test device", "isHardware": False},
            )

        def dispatch(self, adapter, state, request):
            before = json.loads((tmp_path / "evidence.json").read_text())
            assert before["availability"]["details"]["device"] == "Test device"
            state.details["dispatched"] = True
            assert request.loaded_artifact == b"compiled"
            for binding in request.buffers.values():
                if buffer_byte_offset is None:
                    assert binding.allocation is None
                else:
                    assert binding.allocation.byte_offset == buffer_byte_offset
                    assert binding.allocation.byte_length == len(binding.value) * 4
                    assert binding.allocation.allocation_byte_length == (
                        buffer_byte_offset + len(binding.value) * 4 + 16
                    )
            if buffer_byte_offset is not None:
                state.details["directxRuntime"] = {"runtime": "native-buffer-views"}
            if failure:
                raise RuntimeError("dispatch failed")
            return {"outputWords": {"values": [7] + [0x6A15BEEF] * 32}}

    monkeypatch.setattr(module, "_compile", compile_source)
    monkeypatch.setattr(module, "DirectXComputeRuntime", Runtime)
    libraries = [{"name": "d3d10warp.dll", "loaded": True, "sha256": "digest"}]
    monkeypatch.setattr(module, "_loaded_runtime_libraries", lambda: libraries)
    options = {"compile_flags": ("-Gis",), "buffer_byte_offset": buffer_byte_offset}
    if compiler_injected:
        options["module_compiler"] = compile_source
        monkeypatch.setattr(
            module,
            "_compile",
            lambda *args, **kwargs: pytest.fail("Unexpected DXC call"),
        )
    executor = object()
    if injected:
        options["buffer_view_executor"] = executor
    if failure:
        with pytest.raises(RuntimeError, match="dispatch failed"):
            _execute_words(tmp_path, "source", [1], [7], **options)
    else:
        assert _execute_words(tmp_path, "source", [1], [7], **options) == [7]
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    assert evidence["runtimeLibraries"] == libraries
    assert evidence["runtime"]["dispatched"] is True
    assert evidence["bufferByteOffset"] == buffer_byte_offset
    if failure:
        assert evidence["dispatchError"] == {
            "type": "RuntimeError",
            "message": "dispatch failed",
            "details": {},
        }
    else:
        assert "dispatchError" not in evidence
    assert evidence["moduleSha256"] == hashlib.sha256(b"compiled").hexdigest()
    assert flags == [("-Gis",)]


def test_software_reduction_oracle_checks_special_values():
    assert _reduction_reference([0x80000000, 0] * 16, "float") == [0, 0x80000000, 0]
    assert _reduction_reference([0x80000000] * 32, "float") == [0x80000000] * 3
    assert _reduction_reference(
        [0x7FC00000, _word(-7.0), _word(3.0)] + [_word(0)] * 29, "float"
    )[1:] == [_word(-7.0), _word(3.0)]
    assert _reduction_reference([0xFFFFFFFF] * 32, "uint") == [
        0xFFFFFFE0,
        0xFFFFFFFF,
        0xFFFFFFFF,
    ]
