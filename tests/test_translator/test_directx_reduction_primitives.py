"""Native checks that isolate the operations used by repository reductions."""

import json
import os
import shutil
import struct
import subprocess
import sys

import pytest

from crosstl.project.native_runtime_drivers import (
    DirectXComputeRuntime,
    _complete_directx_register_layout,
    _prepare_directx_buffers,
    _validate_directx_register_layout,
)
from crosstl.project.runtime_verification import (
    NativeRuntimeBufferBinding,
    NativeRuntimeDispatchRequest,
    RuntimeDispatchGeometry,
    RuntimeResourceBinding,
)
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

CASES = ("metadata", "division", "shuffle", "combined")
REQUIRE_ENV = "CROSTL_REQUIRE_DIRECTX_REDUCTION_PRIMITIVES"

DECLARATIONS = """
    StructuredBuffer<float> inputValues @ binding(0);
    RWStructuredBuffer<uint> outputValues @ binding(1);
    StructuredBuffer<int> shape @ binding(2);
    StructuredBuffer<int64_t> inputStrides @ binding(3);
    StructuredBuffer<int64_t> outputStrides @ binding(4);
    cbuffer Dimensions @ register(b5) { uint64_t ndim; };
    cbuffer AxisStride @ register(b6) { int64_t axisStride; };
    cbuffer AxisSize @ register(b7) { uint64_t axisSize; };
    cbuffer DispatchInfo @ register(b8) { uint3 groupCount; };

    int64_t locate(int64_t elem, int ndimValue) {
        int64_t loc = 0;
        for (int i = ndimValue - 1; i >= 0 && elem > 0; --i) {
            loc += (elem % shape[i]) * inputStrides[i];
            elem /= shape[i];
        }
        return loc;
    }

"""

SHUFFLE_HELPER = """
    struct Pair { uint index; float value; };
    Pair neighbor(Pair pair, uint delta) {
        Pair result;
        result.index = WaveShuffleDown(pair.index, delta);
        result.value = WaveShuffleDown(pair.value, delta);
        return result;
    }
"""


def _source(case):
    bodies = {
        "metadata": (
            """
            outputValues[gid.y * 32u + lid.x] = uint(ndim + axisSize
                + axisStride + shape[0] + inputStrides[0] + outputStrides[0]
                + groupCount.y) + uint(inputValues[gid.y * 32u + lid.x]);
        """
        ),
        "division": (
            """
            int64_t row = int64_t(gid.y) + int64_t(groupCount.y) * gid.z;
            int64_t loc = locate(row, int(ndim));
            outputValues[gid.y * 32u + lid.x] = uint(loc)
                + uint((axisSize + 127u) / 128u) + uint(row % shape[0]);
        """
        ),
        "shuffle": (
            """
            Pair best;
            best.index = lid.x;
            best.value = float(31u - lid.x + gid.y * 100u);
        """
        ),
        "combined": (
            """
            int64_t row = int64_t(gid.y) + int64_t(groupCount.y) * gid.z;
            int64_t loc = locate(row, int(ndim));
            Pair best;
            best.index = 0u;
            best.value = 1000000.0;
            for (uint r = 0u; r < (axisSize + 127u) / 128u; r++) {
                uint current = r * 128u + lid.x * 4u;
                for (uint i = 0u; i < 4u; i++) {
                    uint index = current + i;
                    if (index < axisSize) {
                        float value = inputValues[uint(loc + index * axisStride)];
                        if (value < best.value) {
                            best.value = value;
                            best.index = index;
                        }
                    }
                }
            }
        """
        ),
    }
    body = bodies[case]
    helpers = DECLARATIONS
    if case in {"shuffle", "combined"}:
        helpers += SHUFFLE_HELPER
        body += """
            for (uint offset = 16u; offset > 0u; offset /= 2u) {
                Pair other = neighbor(best, offset);
                if (other.value < best.value) { best = other; }
            }
            if (lid.x == 0u) {
                outputValues[gid.y] = best.index + uint(best.value);
            }
        """
    return f"""
        shader ReductionPrimitives {{
            {helpers}
            compute {{
                layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
                void main(uint3 gid @ gl_GlobalInvocationID,
                          uint3 lid @ gl_LocalInvocationID) @ WaveSize(32) {{
                    {body}
                }}
            }}
        }}
    """


def _compile(case, work):
    compiler = shutil.which("dxc")
    if compiler is None:
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail("DXC is required for the Direct3D reduction primitive checks")
        pytest.skip("DXC is unavailable")
    source = _source(case)
    (work / "input.cgl").write_text(source, encoding="utf-8")
    ast = Parser(Lexer(source).get_tokens()).parse()
    generated = HLSLCodeGen(
        software_subgroup_width=32 if case in {"shuffle", "combined"} else None,
        relative_wave_shuffle_out_of_range="self",
    ).generate(ast)
    artifact = work / "kernel.hlsl"
    module = work / "kernel.dxil"
    artifact.write_text(generated, encoding="utf-8")
    result = subprocess.run(
        [
            compiler,
            "-T",
            "cs_6_6",
            "-E",
            "CSMain",
            "-WX",
            str(artifact),
            "-Fo",
            str(module),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert module.stat().st_size > 0
    return artifact, module


def _buffers(case):
    result = {}
    specifications = (
        (
            "inputValues",
            0,
            "float32",
            "float",
            [float(31 - lane + row * 100) for row in range(2) for lane in range(32)],
        ),
        ("shape", 2, "int32", "int", [2]),
        ("inputStrides", 3, "int64", "int64_t", [32]),
        ("outputStrides", 4, "int64", "int64_t", [1]),
        ("Dimensions", 5, "uint64", "uint64_t", [1]),
        ("AxisStride", 6, "int64", "int64_t", [1]),
        ("AxisSize", 7, "uint64", "uint64_t", [32]),
        ("DispatchInfo", 8, "uint32", "uint", [1, 2, 1]),
        ("outputValues", 1, "uint32", "uint", None),
    )
    for name, index, dtype, physical_type, values in specifications:
        constant = index >= 5
        output = values is None
        type_name = (
            name
            if constant
            else f"{'RW' if output else ''}StructuredBuffer<{physical_type}>"
        )
        metadata = {}
        if constant:
            width = len(values)
            element_size = (8 if dtype.endswith("64") else 4) * width
            metadata["scalarLayout"] = {
                "physicalType": physical_type + (str(width) if width > 1 else ""),
                "elementType": dtype,
                "elementSizeBytes": element_size,
                "elementStrideBytes": element_size,
                "storageLayout": "hlsl-constant-buffer",
                "alignmentBytes": 16,
                "blockSizeBytes": 16,
                "memberOffsetBytes": 0,
                "runtimeSized": False,
                "vectorWidth": width,
            }
        result[name] = NativeRuntimeBufferBinding(
            name=name,
            binding=RuntimeResourceBinding(
                name=name,
                kind="constant-buffer" if constant else "buffer",
                type_name=type_name,
                set=0,
                binding=index,
                access="read_write" if output else "read",
                metadata=metadata,
            ),
            source="expectedOutput" if output else "input",
            dtype=dtype,
            shape=(
                (
                    len(values)
                    if values is not None
                    else (2 if case in {"shuffle", "combined"} else 64)
                ),
            ),
            value=values,
        )
    return result


def _expected(case):
    if case in {"shuffle", "combined"}:
        return [31, 131]
    if case == "division":
        return [1] * 32 + [34] * 32
    return [71 + 31 - lane + row * 100 for row in range(2) for lane in range(32)]


@pytest.mark.parametrize("case", CASES)
def test_directx_reduction_primitive_resource_layout(case):
    resources = _validate_directx_register_layout(
        _complete_directx_register_layout(_prepare_directx_buffers(_buffers(case)))
    )
    by_name = {resource.name: resource for resource in resources}
    for namespace, count in (("cbv", 9), ("srv", 5), ("uav", 2)):
        assert [r.binding_index for r in resources if r.namespace == namespace] == list(
            range(count)
        )
    for name, value in (("Dimensions", 1), ("AxisStride", 1), ("AxisSize", 32)):
        resource = by_name[name]
        assert resource.payload[:8] == struct.pack("<Q", value)
        assert resource.allocation_size == 256
    assert by_name["DispatchInfo"].payload[:12] == struct.pack("<3I", 1, 2, 1)
    assert by_name["inputStrides"].stride == 8
    assert by_name["inputStrides"].payload == struct.pack("<q", 32)
    assert by_name["outputValues"].upload is False
    assert by_name["outputValues"].shape == (len(_expected(case)),)


@pytest.mark.parametrize("case", CASES)
def test_directx_reduction_primitives_compile(tmp_path, case):
    _compile(case, tmp_path)


@pytest.mark.parametrize("case", CASES)
def test_directx_reduction_primitives_execute(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 to require native Direct3D execution")
    assert sys.platform == "win32", "Native Direct3D checks require Windows"
    artifact, module = _compile(case, tmp_path)
    request = NativeRuntimeDispatchRequest(
        target="directx",
        artifact={"target": "directx"},
        artifact_path=artifact,
        module_path=module,
        loaded_artifact=module.read_bytes(),
        buffers=_buffers(case),
        constants={},
        dispatch=RuntimeDispatchGeometry(
            entry_point="CSMain", workgroup_size=(32, 1, 1), workgroup_count=(1, 2, 1)
        ),
        entry_point="CSMain",
    )
    runtime = DirectXComputeRuntime()
    availability = runtime.is_available(None, None)
    assert availability.available, availability.reason
    print(f"[directx-primitive] dispatch {case}", flush=True)
    outputs = runtime.dispatch(None, None, request)
    (tmp_path / "outputs.json").write_text(
        json.dumps(outputs, indent=2), encoding="utf-8"
    )
    assert outputs["outputValues"]["values"] == _expected(case)
