"""HLSL structured-buffer parameters carry the layout of the buffers passed to them.

The HLSL importer marked global structured buffers ``@ buffer_layout(hlsl_structured)`` but not buffer parameters
of helper functions, so a helper read a 12-byte ``float3`` buffer with its target's own rules: GLSL and SPIR-V
refused to pass the word-lowered buffer to the helper, and Metal declared the parameter ``device float3*`` while the
argument was ``device packed_float3*``, which the Metal compiler rejects.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import sys

import pytest

import crosstl

REQUIRE_ENV = "CROSSTL_REQUIRE_BUFFER_PARAMETER_EXECUTION"

# Sample is 16 bytes and float3 12 under HLSL packing; std430 gives 32 and 16, MSL's float3 16.
HLSL = """
struct Sample { float weight; float3 value; };
StructuredBuffer<Sample> samples : register(t0);
RWStructuredBuffer<float3> data : register(u0);

float3 weighted(StructuredBuffer<Sample> s, uint i) { return s[i].value * s[i].weight; }
void accumulate(RWStructuredBuffer<float3> b, uint i, float3 v) { b[i] += v; }

[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    accumulate(data, id.x, weighted(samples, id.x));
}
"""

SAMPLES = [
    (2.0, (1.0, 2.0, 3.0)),
    (0.5, (4.0, -2.0, 8.0)),
    (1.0, (0.0, 1.0, 0.0)),
    (3.0, (1.0, 1.0, -1.0)),
]
DATA = [(10.0, 20.0, 30.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (-1.0, -2.0, -3.0)]


def translate(tmp_path, backend):
    path = tmp_path / "shader.hlsl"
    path.write_text(HLSL, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def run(command):
    result = subprocess.run(
        [str(part) for part in command], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_crossgl_parameters_carry_the_hlsl_layout(tmp_path):
    crossgl = translate(tmp_path, "cgl")
    for parameter in (
        "StructuredBuffer<Sample> s",
        "RWStructuredBuffer<float3> b",
    ):
        assert f"@ buffer_layout(hlsl_structured) {parameter}" in crossgl, crossgl
    assert "@ buffer_layout(hlsl_structured) uint i" not in crossgl


def test_opengl_helpers_read_the_callers_words(tmp_path):
    glsl = translate(tmp_path, "opengl")
    assert re.search(r"buffer samplesBuffer \{\s*uint samples\[\];", glsl)
    assert re.search(r"buffer dataBuffer \{\s*uint data\[\];", glsl)
    # Sample is four words, float3 three.
    assert "uint(i) * 4u" in glsl
    assert "uint(i) * 3u" in glsl
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")
    artifact = tmp_path / "shader.comp"
    artifact.write_text(glsl, encoding="utf-8")
    run([glslang, artifact])


def test_spirv_helpers_validate(tmp_path):
    assembly = translate(tmp_path, "vulkan")
    assert "ArrayStride 4" in assembly
    assert not re.search(r"ArrayStride (16|32)\b", assembly)
    spirv_as, spirv_val = shutil.which("spirv-as"), shutil.which("spirv-val")
    if spirv_as is None or spirv_val is None:
        pytest.skip("SPIRV-Tools is not installed")
    source, binary = tmp_path / "shader.spvasm", tmp_path / "shader.spv"
    source.write_text(assembly, encoding="utf-8")
    run([spirv_as, "--target-env", "vulkan1.1", source, "-o", binary])
    run([spirv_val, "--target-env", "vulkan1.1", binary])


def test_metal_parameters_match_their_arguments(tmp_path):
    metal = translate(tmp_path, "metal")
    assert re.search(r"void accumulate\(device packed_float3\s*\*\s*b,", metal), metal
    xcrun = shutil.which("xcrun")
    if xcrun is None or (
        subprocess.run(
            [xcrun, "-sdk", "macosx", "-f", "metal"], capture_output=True
        ).returncode
        != 0
    ):
        pytest.skip("the Metal toolchain is not installed")
    shader = tmp_path / "shader.metal"
    shader.write_text(metal, encoding="utf-8")
    run([xcrun, "-sdk", "macosx", "metal", "-c", shader, "-o", tmp_path / "shader.air"])


def test_helpers_execute_on_opengl(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1" or sys.platform == "darwin":
        pytest.skip(f"set {REQUIRE_ENV}=1 on Linux to require the OpenGL dispatch")
    import moderngl

    glsl = translate(tmp_path, "opengl")
    bindings = {
        name: int(binding)
        for binding, name in re.findall(
            r"layout\(std430, binding = (\d+)\)[^{]*\{\s*\w+ (\w+)\[\];", glsl
        )
    }
    context = moderngl.create_standalone_context(require=430, backend="egl")
    buffers = {}
    try:
        shader = context.compute_shader(glsl)
        contents = {
            "samples": b"".join(
                struct.pack("<4f", weight, *value) for weight, value in SAMPLES
            ),
            "data": b"".join(struct.pack("<3f", *value) for value in DATA),
        }
        for name, payload in contents.items():
            buffers[name] = context.buffer(payload)
            buffers[name].bind_to_storage_buffer(bindings[name])
        shader.run(group_x=1)
        context.finish()
        result = list(struct.iter_unpack("<3f", buffers["data"].read()))
        expected = [
            tuple(base + weight * component for base, component in zip(start, value))
            for start, (weight, value) in zip(DATA, SAMPLES)
        ]
        assert result == expected, glsl
    finally:
        for buffer in buffers.values():
            buffer.release()
        context.release()
