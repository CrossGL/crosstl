"""HLSL scalar swizzles, intrinsic promotion and descriptor indexing reach SPIR-V.

* A component selection on a scalar (``s.xxx``, HLSL and GLSL 4.20 section 5.5) is the scalar or its splat.
* ``min``, ``max`` and ``clamp`` promote a scalar operand to the width of the vector operands, as HLSL's
  intrinsics do, so ``max(v, 0.0)`` is a ``float3``.
* ``NonUniformResourceIndex`` marks a divergent descriptor index, as GLSL's ``nonuniformEXT`` does
  (``NonUniform`` decorations and the ``ShaderNonUniform`` capability).
* An unsized array of textures or samplers is a runtime descriptor array (``RuntimeDescriptorArray``,
  SPV_EXT_descriptor_indexing); an unsized buffer element array is not.
"""

import re
import shutil
import subprocess

import crosstl

INTRINSICS_HLSL = """
Texture2D textures[] : register(t0, space1);
SamplerState samp : register(s0);
struct PSInput {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
    nointerpolation uint index : TEXCOORD1;
    float3 n : NORMAL;
};
float4 main(PSInput input) : SV_Target
{
    float ndotl = max(dot(normalize(input.n), float3(0.0, 0.0, 1.0)), 0.0);
    float3 lit = max(ndotl.xxx, 0.25);
    float3 clamped = clamp(lit, 0.0, float3(1.0, 1.0, 1.0));
    float4 c = textures[NonUniformResourceIndex(input.index)].Sample(samp, input.uv);
    return float4(c.rgb * clamped * min(0.5, ndotl.xxx), c.a * ndotl.x);
}
"""

BUFFER_HLSL = """
RWStructuredBuffer<float> values : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) { values[id.x] = (2.0).x; }
"""


def _translate(tmp_path, source, name):
    path = tmp_path / f"{name}.hlsl"
    path.write_text(source)
    return crosstl.translate(
        str(path), backend="vulkan", source_backend="directx", format_output=False
    )


def _spirv_val(tmp_path, code):
    if shutil.which("spirv-as") is None or shutil.which("spirv-val") is None:
        return
    asm = tmp_path / "out.spvasm"
    asm.write_text(code)
    binary = tmp_path / "out.spv"
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


def _type_ids(code, pattern):
    return set(re.findall(rf"^\s*%(\w+) = {pattern}$", code, re.M))


def test_scalar_swizzles_and_promoted_intrinsics(tmp_path):
    code = _translate(tmp_path, INTRINSICS_HLSL, "intrinsics.frag")
    float3 = _type_ids(code, r"OpTypeVector %\w+ 3")
    assert float3
    vector = "|".join(float3)
    # max(ndotl.xxx, 0.25), clamp(lit, 0.0, ...) and min(0.5, ndotl.xxx) are float3 extended instructions.
    for operation in ("FMax", "FClamp", "FMin"):
        assert re.search(rf"= OpExtInst %({vector}) %\w+ {operation} ", code), operation
    # ndotl.xxx is a splat of the scalar.
    assert re.search(rf"OpCompositeConstruct %({vector}) (%(\w+)) \2 \2$", code, re.M)
    _spirv_val(tmp_path, code)


def test_non_uniform_resource_index_decorates_the_access(tmp_path):
    code = _translate(tmp_path, INTRINSICS_HLSL, "intrinsics.frag")
    lines = [line.strip() for line in code.splitlines()]
    assert "OpCapability ShaderNonUniform" in lines
    assert re.search(r"OpDecorate %\w+ NonUniform$", code, re.M)
    assert "NonUniformResourceIndex" not in code


def test_unsized_texture_array_is_a_runtime_descriptor_array(tmp_path):
    code = _translate(tmp_path, INTRINSICS_HLSL, "intrinsics.frag")
    lines = [line.strip() for line in code.splitlines()]
    assert "OpCapability RuntimeDescriptorArray" in lines
    assert 'OpExtension "SPV_EXT_descriptor_indexing"' in lines
    assert lines.count("OpCapability RuntimeDescriptorArray") == 1


def test_unsized_buffer_elements_need_no_descriptor_indexing(tmp_path):
    code = _translate(tmp_path, BUFFER_HLSL, "buffer.comp")
    assert "OpTypeRuntimeArray" in code
    assert "RuntimeDescriptorArray" not in code
    assert "SPV_EXT_descriptor_indexing" not in code
    _spirv_val(tmp_path, code)
