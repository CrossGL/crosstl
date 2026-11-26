"""GLSL output writes HLSL ``f32tof16`` and ``f16tof32`` with ``packHalf2x16`` and ``unpackHalf2x16``.

GLSL output refused both calls as functions that are neither GLSL built-ins nor declared by the program. See
:mod:`crosstl.translator.codegen.glsl_half_conversions`.
"""

from __future__ import annotations

import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.glsl_half_conversions import (
    emit_glsl_half_conversion_helpers,
)

GLSLANG = shutil.which("glslangValidator")

COMPUTE = """RWStructuredBuffer<uint> words : register(u0);
RWStructuredBuffer<float4> values : register(u1);

float4 load(uint i) { return values[i]; }

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    int k = int(id.x) - 3;
    words[0] = f32tof16(values[0].x);
    words[1] = f32tof16(k);
    uint3 packed = f32tof16(load(id.x).xyz);
    words[2] = packed.z;
    values[1].x = f16tof32(words[3]);
    values[1].y = f16tof32(k);
    values[2] = f16tof32(uint4(words[4], words[5], words[6], words[7]));
    values[3].xyz = f16tof32(f32tof16(values[3].xyz));
}
"""

FRAGMENT = """float4 main(float4 pos : SV_Position) : SV_Target {
    uint2 bits = f32tof16(pos.xy);
    return float4(f16tof32(bits), f16tof32(f32tof16(0.5)), 1.0);
}
"""

USER_FUNCTION = """RWStructuredBuffer<uint> words : register(u0);

uint f32tof16(float x) { return asuint(x) >> 16; }

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    words[id.x] = f32tof16(float(id.x));
}
"""


def _translate(tmp_path, source, backend, name="probe.hlsl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _glslang(tmp_path, glsl, stage):
    if GLSLANG is None:
        pytest.skip("glslangValidator is not on PATH")
    path = tmp_path / f"translated.{stage}"
    path.write_text(glsl)
    result = subprocess.run(
        [GLSLANG, str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr + glsl


def _collapsed(code):
    return " ".join(code.split())


def test_scalars_pack_into_the_low_half_and_unpack_from_it(tmp_path):
    glsl = _collapsed(_translate(tmp_path, COMPUTE, "opengl"))
    assert "words[0] = packHalf2x16(vec2(values[0].x, 0.0));" in glsl
    assert "values[1].x = unpackHalf2x16(words[3]).x;" in glsl


def test_operands_of_other_types_convert_as_hlsl_converts_them(tmp_path):
    glsl = _collapsed(_translate(tmp_path, COMPUTE, "opengl"))
    assert "words[1] = packHalf2x16(vec2(float(k), 0.0));" in glsl
    assert "values[1].y = unpackHalf2x16(uint(k)).x;" in glsl


def test_vectors_convert_componentwise_through_one_helper_per_shape(tmp_path):
    glsl = _translate(tmp_path, COMPUTE, "opengl")
    flat = _collapsed(glsl)
    assert (
        "uvec3 crossgl_f32tof16_vec3(vec3 v) { return uvec3(packHalf2x16(vec2(v.x, 0.0)), "
        "packHalf2x16(vec2(v.y, 0.0)), packHalf2x16(vec2(v.z, 0.0))); }" in flat
    )
    assert (
        "vec4 crossgl_f16tof32_uvec4(uvec4 v) { return vec4(unpackHalf2x16(v.x).x, "
        "unpackHalf2x16(v.y).x, unpackHalf2x16(v.z).x, unpackHalf2x16(v.w).x); }"
        in flat
    )
    # The operand, here a call, is evaluated once.
    assert "crossgl_f32tof16_vec3(load(gl_GlobalInvocationID.x).xyz)" in flat
    assert (
        "values[3].xyz = crossgl_f16tof32_uvec3(crossgl_f32tof16_vec3(values[3].xyz));"
        in flat
    )
    assert glsl.count("uvec3 crossgl_f32tof16_vec3(") == 1
    _glslang(tmp_path, glsl, "comp")


@pytest.mark.parametrize("backend", ["opengl", "gles"])
def test_compute_output_validates(tmp_path, backend):
    _glslang(tmp_path, _translate(tmp_path, COMPUTE, backend), "comp")


@pytest.mark.parametrize("backend", ["opengl", "gles", "webgl"])
def test_fragment_output_validates(tmp_path, backend):
    glsl = _translate(tmp_path, FRAGMENT, backend)
    assert "crossgl_f16tof32_uvec2(bits)" in glsl
    assert "unpackHalf2x16(packHalf2x16(vec2(0.5, 0.0))).x" in _collapsed(glsl)
    _glslang(tmp_path, glsl, "frag")


def test_a_function_the_program_declares_keeps_its_name(tmp_path):
    glsl = _translate(tmp_path, USER_FUNCTION, "opengl")
    assert "packHalf2x16" not in glsl
    assert "f32tof16(float(gl_GlobalInvocationID.x))" in glsl


def test_a_new_program_starts_without_the_previous_programs_helpers(tmp_path):
    _translate(tmp_path, COMPUTE, "opengl", "first.hlsl")
    glsl = _translate(tmp_path, FRAGMENT, "opengl", "second.hlsl")
    assert "crossgl_f32tof16_vec3" not in glsl
    assert "crossgl_f32tof16_vec2" in glsl


def test_emitted_helpers_are_empty_until_a_call_records_one():
    class Generator:
        pass

    assert emit_glsl_half_conversion_helpers(Generator()) == ""


def _half_bits(value):
    return struct.unpack("<H", struct.pack("<e", value))[0]


@pytest.mark.parametrize("value", [0.5, -2.0, 65504.0, 1.0e-7])
def test_the_high_half_holds_the_bits_of_positive_zero(value):
    # packHalf2x16(vec2(x, 0.0)) == bits(x) | bits(+0.0) << 16, and bits(+0.0) is 0, as HLSL f32tof16 requires.
    assert _half_bits(0.0) == 0
    assert (_half_bits(value) | _half_bits(0.0) << 16) == _half_bits(value)
