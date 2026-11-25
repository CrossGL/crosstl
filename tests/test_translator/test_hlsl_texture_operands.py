"""HLSL in both directions for CrossGL's compare bias, compare-gather component and sample position (#4796).

HLSL input used to refuse ``SampleCmpBias`` ("CrossGL has no operand for LOD bias"), ``GatherCmpGreen``,
``GatherCmpBlue`` and ``GatherCmpAlpha`` ("CrossGL only represents the default/red compare gather") and
``RWTexture2DMS[Array].GetSamplePosition``. CrossGL now carries each operand (:mod:`crosstl.translator.semantics.
texture_operands`): the bias is the last operand of ``textureCompare[Offset][Clamp]``, the component the last
operand of ``textureGatherCompare[Offset]``, and ``textureSamplePosition`` takes a storage multisample image too.
HLSL output writes them back as ``SampleCmpBias`` (Shader Model 6.8) and ``GatherCmp<Channel>``.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.DirectX.DirectxCrossGLCodeGen import DirectXImportRefused

DECLARATIONS = """Texture2D<float4> obj : register(t0);
TextureCubeArray<float4> cube : register(t1);
SamplerComparisonState cmp : register(s1);
RWStructuredBuffer<float4> outp : register(u1);
"""
FRAGMENT = "float4 main(float4 p : SV_Position) : SV_Target {{ return {call}; }}\n"
COMPUTE = (
    "[numthreads(1, 1, 1)] void main(uint3 id : SV_DispatchThreadID) "
    "{{ outp[id.x] = {call}; }}\n"
)

# (HLSL call, the CrossGL call it imports as, the HLSL call that CrossGL writes back)
COMPARE_BIAS = {
    "bias": (
        "obj.SampleCmpBias(cmp, p.xy, 0.5, 1.0)",
        "textureCompare(obj, cmp, p.xy, 0.5, 1.0)",
        "obj.SampleCmpBias(cmp, p.xy, 0.5, 1.0)",
    ),
    "offset": (
        "obj.SampleCmpBias(cmp, p.xy, 0.5, 1.0, int2(1, 0))",
        "textureCompareOffset(obj, cmp, p.xy, 0.5, ivec2(1, 0), 1.0)",
        "obj.SampleCmpBias(cmp, p.xy, 0.5, 1.0, int2(1, 0))",
    ),
    "offset-clamp": (
        "obj.SampleCmpBias(cmp, p.xy, 0.5, 1.0, int2(1, 0), 2.0)",
        "textureCompareOffsetClamp(obj, cmp, p.xy, 0.5, ivec2(1, 0), 2.0, 1.0)",
        "obj.SampleCmpBias(cmp, p.xy, 0.5, 1.0, int2(1, 0), 2.0)",
    ),
    # A cube has no offset operand, so its fifth operand is the clamp.
    "cube-clamp": (
        "cube.SampleCmpBias(cmp, p, 0.5, 1.0, 2.0)",
        "textureCompareClamp(cube, cmp, p, 0.5, 2.0, 1.0)",
        "cube.SampleCmpBias(cmp, p, 0.5, 1.0, 2.0)",
    ),
}
GATHER_COMPARE = {
    "green": (
        "obj.GatherCmpGreen(cmp, float2(0.5, 0.5), 0.5)",
        "textureGatherCompare(obj, cmp, vec2(0.5, 0.5), 0.5, 1)",
        "obj.GatherCmpGreen(cmp, float2(0.5, 0.5), 0.5)",
    ),
    "blue-offset": (
        "obj.GatherCmpBlue(cmp, float2(0.5, 0.5), 0.5, int2(1, 0))",
        "textureGatherCompareOffset(obj, cmp, vec2(0.5, 0.5), 0.5, ivec2(1, 0), 2)",
        "obj.GatherCmpBlue(cmp, float2(0.5, 0.5), 0.5, int2(1, 0))",
    ),
    "alpha-cube": (
        "cube.GatherCmpAlpha(cmp, float4(0.5, 0.5, 0.5, 1.0), 0.5)",
        "textureGatherCompare(cube, cmp, vec4(0.5, 0.5, 0.5, 1.0), 0.5, 3)",
        "cube.GatherCmpAlpha(cmp, float4(0.5, 0.5, 0.5, 1.0), 0.5)",
    ),
    # The red component is the default, which the call without a component gathers.
    "red": (
        "obj.GatherCmpRed(cmp, float2(0.5, 0.5), 0.5)",
        "textureGatherCompare(obj, cmp, vec2(0.5, 0.5), 0.5)",
        "obj.GatherCmp(cmp, float2(0.5, 0.5), 0.5)",
    ),
}

SAMPLE_POSITION = """RWTexture2DMS<float4> ms : register(u2);
RWTexture2DMSArray<float4> msArray : register(u3);
RWStructuredBuffer<float4> outp : register(u1);
[numthreads(1, 1, 1)] void main(uint3 id : SV_DispatchThreadID) {
    outp[id.x] = float4(ms.GetSamplePosition(1), msArray.GetSamplePosition(2));
}
"""


def _write(tmp_path, text, name="probe.hlsl"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def _source(template, call):
    return DECLARATIONS + template.format(call=call)


def _dxc(tmp_path, code, profile):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not on PATH")
    source = tmp_path / "translated.hlsl"
    source.write_text(code)
    result = subprocess.run(
        [dxc, "-T", profile, "-E", "PSMain" if profile.startswith("ps") else "CSMain"]
        + ["-HV", "2021", str(source), "-Fo", str(tmp_path / "translated.dxil")],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("form", sorted(COMPARE_BIAS))
def test_hlsl_input_keeps_the_compare_bias(tmp_path, form):
    call, crossgl_call, hlsl_call = COMPARE_BIAS[form]
    path = _write(tmp_path, _source(FRAGMENT, call))

    assert crossgl_call in crosstl.translate(path, backend="cgl")
    assert hlsl_call in crosstl.translate(path, backend="directx")


@pytest.mark.parametrize("form", sorted(GATHER_COMPARE))
def test_hlsl_input_keeps_the_gather_compare_component(tmp_path, form):
    call, crossgl_call, hlsl_call = GATHER_COMPARE[form]
    path = _write(tmp_path, _source(COMPUTE, call))

    crossgl = crosstl.translate(path, backend="cgl")
    assert crossgl_call + ";" in crossgl
    assert hlsl_call in crosstl.translate(path, backend="directx")


def test_hlsl_input_reads_sample_positions_of_storage_multisample_textures(tmp_path):
    path = _write(tmp_path, SAMPLE_POSITION)

    crossgl = crosstl.translate(path, backend="cgl")
    assert "textureSamplePosition(ms, 1)" in crossgl
    assert "textureSamplePosition(msArray, 2)" in crossgl
    hlsl = crosstl.translate(path, backend="directx")
    assert "ms.GetSamplePosition(1)" in hlsl
    assert "msArray.GetSamplePosition(2)" in hlsl


@pytest.mark.parametrize(
    "call, missing",
    [
        # SampleCmpBias's status output reports tiled-resource residency, which CrossGL has no value for.
        (
            "obj.SampleCmpBias(cmp, p.xy, 0.5, 1.0, int2(1, 0), 2.0, status)",
            "status output",
        ),
        (
            "obj.GatherCmpGreen(cmp, p.xy, 0.5, int2(1, 0), int2(0, 1), int2(1, 1), "
            "int2(0, 0))",
            "four-offset overload",
        ),
    ],
)
def test_hlsl_input_still_refuses_operands_crossgl_lacks(tmp_path, call, missing):
    body = (
        "float4 main(float4 p : SV_Position) : SV_Target "
        f"{{ uint status; return {call}; }}\n"
    )
    path = _write(tmp_path, DECLARATIONS + body)

    with pytest.raises(DirectXImportRefused, match=missing):
        crosstl.translate(path, backend="cgl")


CROSSGL = """shader Operands {
    fragment {
        uniform sampler2DShadow shadowMap;
        uniform texture2D depth;
        uniform comparison_sampler cmp;

        vec4 main(vec4 position @ gl_FragCoord) @ gl_FragColor {
            return %s;
        }
    }
}
"""


@pytest.mark.parametrize(
    "call, method",
    [
        (
            "vec4(texture(shadowMap, vec3(position.xy, 0.5), 1.0))",
            "shadowMap.SampleCmpBias(shadowMapSampler, position.xy, 0.5, 1.0)",
        ),
        (
            "vec4(textureCompareClamp(depth, cmp, position.xy, 0.5, 2.0, 1.0))",
            "depth.SampleCmpBias(cmp, position.xy, 0.5, 1.0, int2(0, 0), 2.0)",
        ),
        (
            "textureGatherCompare(depth, cmp, position.xy, 0.5, 3)",
            "depth.GatherCmpAlpha(cmp, position.xy, 0.5)",
        ),
        (
            "textureGatherCompareOffset(depth, cmp, position.xy, 0.5, ivec2(1, 0), 2)",
            "depth.GatherCmpBlue(cmp, position.xy, 0.5, int2(1, 0))",
        ),
    ],
)
def test_hlsl_output_writes_the_operands_dxc_compiles(tmp_path, call, method):
    path = _write(tmp_path, CROSSGL % call, "probe.cgl")
    hlsl = crosstl.translate(path, backend="directx")

    assert method in hlsl
    # SampleCmpBias is Shader Model 6.8.
    _dxc(tmp_path, hlsl, "ps_6_8")


def test_hlsl_round_trips_reach_dxil(tmp_path):
    source = DECLARATIONS + (
        "float4 main(float4 p : SV_Position) : SV_Target {\n"
        "    float4 g = obj.GatherCmpGreen(cmp, p.xy, 0.5)"
        " + cube.GatherCmpAlpha(cmp, p, 0.5);\n"
        "    return g + obj.SampleCmpBias(cmp, p.xy, 0.5, 1.0, int2(1, 0))"
        " + cube.SampleCmpBias(cmp, p, 0.5, 1.0, 2.0);\n"
        "}\n"
    )
    hlsl = crosstl.translate(_write(tmp_path, source), backend="directx")

    _dxc(tmp_path, hlsl, "ps_6_8")
