"""HLSL interface declarations that GLSL spells differently: constant buffers named like their member, and
tessellation patch-constant functions and patch parameters, which GLSL has no declaration for.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import TessellationPatchUnsupportedError

NAMED_LIKE_MEMBER_HLSL = """
struct UBO { float4x4 projection; float4x4 model; };
cbuffer ubo : register(b0) { UBO ubo; }
struct VSInput { float3 Pos : POSITION0; };
struct VSOutput { float4 Pos : SV_POSITION; };
VSOutput main(VSInput input) {
    VSOutput output = (VSOutput)0;
    output.Pos = mul(ubo.projection, mul(ubo.model, float4(input.Pos, 1.0)));
    return output;
}
"""

DOMAIN_HLSL = """
struct HSOutput { float4 Pos : SV_POSITION; };
struct ConstantsHSOutput { float TessLevelOuter[3] : SV_TessFactor; float TessLevelInner : SV_InsideTessFactor; };
struct DSOutput { float4 Pos : SV_POSITION; };
[domain("tri")]
DSOutput main(ConstantsHSOutput input, float3 TessCoord : SV_DomainLocation,
              const OutputPatch<HSOutput, 3> patch) {
    DSOutput output = (DSOutput)0;
    output.Pos = TessCoord.x * patch[0].Pos + TessCoord.y * patch[1].Pos + TessCoord.z * patch[2].Pos;
    return output;
}
"""

HULL_HLSL = """
struct VSOutput { float4 Pos : SV_POSITION; };
struct HSOutput { float4 Pos : SV_POSITION; };
struct ConstantsHSOutput { float TessLevelOuter[3] : SV_TessFactor; float TessLevelInner : SV_InsideTessFactor; };
ConstantsHSOutput ConstantsHS(InputPatch<VSOutput, 3> patch) {
    ConstantsHSOutput output = (ConstantsHSOutput)0;
    output.TessLevelOuter[0] = 1.0;
    output.TessLevelOuter[1] = 1.0;
    output.TessLevelOuter[2] = 1.0;
    output.TessLevelInner = 1.0;
    return output;
}
[domain("tri")]
[partitioning("integer")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("ConstantsHS")]
HSOutput main(InputPatch<VSOutput, 3> patch, uint InvocationID : SV_OutputControlPointID) {
    HSOutput output = (HSOutput)0;
    output.Pos = patch[InvocationID].Pos;
    return output;
}
"""


def _translate(tmp_path, source, name, backend):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize("backend", ["opengl", "webgl"])
def test_constant_buffer_named_like_its_member_gets_another_block_name(
    tmp_path, backend
):
    glsl = _translate(tmp_path, NAMED_LIKE_MEMBER_HLSL, "shader.hlsl", backend)

    assert "uniform ubo_block {" in glsl
    assert "UBO ubo;" in glsl
    assert "ubo.projection" in glsl
    if shutil.which("glslangValidator") is not None:
        result = subprocess.run(
            ["glslangValidator", "--stdin", "-S", "vert"],
            input=glsl,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_hlsl_domain_without_its_hull_is_a_typed_opengl_refusal(tmp_path):
    # GLSL declares spacing and winding on the evaluation shader; HLSL only on the hull.
    with pytest.raises(TessellationPatchUnsupportedError) as refused:
        _translate(tmp_path, DOMAIN_HLSL, "shader.tese.hlsl", "opengl")
    assert refused.value.reason_code == "domain-without-hull"


def test_hlsl_hull_patch_constant_function_lowers_for_opengl(tmp_path):
    glsl = _translate(tmp_path, HULL_HLSL, "shader.tesc.hlsl", "opengl")

    assert "patchconstantfunc" not in glsl and "InputPatch" not in glsl
    assert "ConstantsHSOutput hs_patch = ConstantsHS();" in glsl
    assert "gl_TessLevelInner[0] = hs_patch.TessLevelInner;" in glsl
    if shutil.which("glslangValidator") is not None:
        result = subprocess.run(
            ["glslangValidator", "--stdin", "-S", "tesc"],
            input=glsl,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
