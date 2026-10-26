"""HLSL implicit vector truncation on assignment and initialization (DXC warning X3206)."""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from tests.conformance.corpus.checks import spirv_environment

SOURCE = """
struct Payload { float3 color; float weight; };
RWStructuredBuffer<Payload> results : register(u0);
RWStructuredBuffer<float4> inputs : register(u1);

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    float4 wide = inputs[id.x];
    Payload payload;
    payload.color.rgb = wide * 2.0;
    payload.weight = wide;
    float3 narrow = wide;
    float2 pair;
    pair = wide.xyz;
    pair += wide;
    float4 same = wide;
    float3 splat = 1.0;
    payload.color = narrow + float3(pair, splat.x) + same.xyz;
    results[id.x] = payload;
}
"""


def _translate(tmp_path, backend):
    path = tmp_path / "truncation.hlsl"
    path.write_text(textwrap.dedent(SOURCE).strip() + "\n", encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_truncation_keeps_the_leading_components(tmp_path):
    crossgl = _translate(tmp_path, "cgl")
    assert "payload.color.rgb = (wide * 2.0).xyz;" in crossgl
    assert "payload.weight = wide.x;" in crossgl
    assert "vec3 narrow = wide.xyz;" in crossgl
    assert "pair = wide.xyz.xy;" in crossgl
    assert "pair += wide.xy;" in crossgl


def test_values_of_the_target_width_are_unchanged(tmp_path):
    crossgl = _translate(tmp_path, "cgl")
    assert "vec4 same = wide;" in crossgl
    assert "vec3 splat = 1.0;" in crossgl
    assert "vec4 wide = inputs[id.x];" in crossgl


def test_truncated_glsl_compiles(tmp_path):
    glsl = _translate(tmp_path, "opengl")
    validator = shutil.which("glslangValidator")
    if validator is None:
        pytest.skip("glslangValidator is not installed")
    path = tmp_path / "truncation.comp"
    path.write_text(glsl, encoding="utf-8")
    completed = subprocess.run(
        [validator, "-S", "comp", str(path)], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_truncated_spirv_validates(tmp_path):
    spirv = _translate(tmp_path, "vulkan")
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if assembler is None or validator is None:
        pytest.skip("SPIRV-Tools are not installed")
    assembly = tmp_path / "truncation.spvasm"
    assembly.write_text(spirv, encoding="utf-8")
    binary = tmp_path / "truncation.spv"
    environment = spirv_environment(spirv)
    subprocess.run(
        [assembler, "--target-env", environment, str(assembly), "-o", str(binary)],
        check=True,
    )
    completed = subprocess.run(
        [validator, "--target-env", environment, str(binary)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
