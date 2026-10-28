"""GLSL built-ins HLSL reads through another system value or an intrinsic, and the refusal of the rest.

The HLSL generator kept any ``gl_`` name its semantic table did not list as a user semantic, which DXC accepts, so a
built-in such as ``@gl_BaryCoordNV`` or ``@gl_HelperInvocation`` became a varying no stage writes.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
import crosstl.translator
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.hlsl_refusals import DirectXLoweringRefused
from crosstl.translator.semantics.builtin_names import CROSSGL_GL_BUILTIN_NAMES

FRAGMENT_PARAMETER = """shader Probe {{
    fragment {{
        vec4 main({type} value @{name}) @gl_FragColor {{
            return vec4(float(value{swizzle}));
        }}
    }}
}}
"""

VERTEX_MEMBER = """shader Probe {{
    struct VertexOutput {{
        vec4 position @gl_Position;
        {type} value @{name};
    }}
    vertex {{
        VertexOutput main() {{
            VertexOutput output;
            output.position = vec4(0.0);
            output.value = {type}(0);
            return output;
        }}
    }}
}}
"""

COMPUTE_PARAMETER = """shader Probe {{
    compute {{
        buffer float values[];
        @workgroup_size(64, 1, 1)
        void main({type} value @{name}) {{
            values[0] = float(value{swizzle});
        }}
    }}
}}
"""

FRAGMENT_BODY = """shader Probe {
    fragment {
        vec4 main() @gl_FragColor {
            uint lanes = gl_SubgroupSize + gl_SubgroupInvocationID;
            int samples = gl_NumSamples;
            return vec4(float(lanes), float(samples), 0.0, 1.0);
        }
    }
}
"""

VERTEX_BODY = """shader Probe {
    vertex {
        vec4 main() @gl_Position {
            return vec4(float(gl_SubgroupSize + gl_SubgroupInvocationID));
        }
    }
}
"""

STAGE_BODY_SAMPLES = {
    "vertex": (
        """shader Probe {
    vertex {
        vec4 main() @gl_Position {
            return vec4(float(gl_NumSamples));
        }
    }
}
"""
    ),
    "compute": (
        """shader Probe {
    compute {
        buffer int values[];
        @workgroup_size(1, 1, 1)
        void main() {
            values[0] = gl_NumSamples;
        }
    }
}
"""
    ),
}

USER_GL_SEMANTIC = re.compile(r":\s*gl_\w+")


def _translate(tmp_path, source):
    path = tmp_path / "probe.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend="directx", format_output=False)


def _compile(tmp_path, hlsl, profile, entry):
    if shutil.which("dxc") is None:
        return
    source = tmp_path / "out.hlsl"
    source.write_text(hlsl)
    result = subprocess.run(
        ["dxc", "-T", profile, "-E", entry, str(source)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _swizzle(type_name):
    return ".x" if type_name[-1] in "234" else ""


@pytest.mark.parametrize(
    "name, type_name, expected",
    [
        ("gl_SampleID", "uint", [r"uint value : SV_SampleIndex"]),
        (
            "gl_BaryCoordNV",
            "vec3",
            [r"(?<!noperspective )float3 value : SV_Barycentrics"],
        ),
        (
            "gl_BaryCoordNoPerspNV",
            "vec3",
            [r"noperspective float3 value : SV_Barycentrics"],
        ),
        ("gl_ShadingRateEXT", "uint", [r"uint value : SV_ShadingRate"]),
        (
            "gl_ShadingRateEXT",
            "int",
            [
                r"uint value_shadingRate : SV_ShadingRate",
                r"int value = int\(value_shadingRate\);",
            ],
        ),
        (
            "gl_FragSizeEXT",
            "ivec2",
            [
                r"uint value_shadingRate : SV_ShadingRate",
                r"uint2 CrossGLFragmentSizeFromShadingRate\(uint shadingRate\)",
                r"int2 value = int2\(CrossGLFragmentSizeFromShadingRate\(value_shadingRate\)\);",
            ],
        ),
        (
            "gl_FullyCoveredEXT",
            "bool",
            [
                r"uint value_innerCoverage : SV_InnerCoverage",
                r"bool value = \(value_innerCoverage != 0u\);",
            ],
        ),
        ("gl_HelperInvocation", "bool", [r"bool value = IsHelperLane\(\);"]),
        ("gl_SubgroupSize", "uint", [r"uint value = WaveGetLaneCount\(\);"]),
        ("gl_SubgroupInvocationID", "uint", [r"uint value = WaveGetLaneIndex\(\);"]),
        ("gl_SubgroupEqMask", "uvec4", [r"uint4 value = \(.*WaveGetLaneIndex\(\)"]),
        ("gl_SubgroupLtMask", "uvec4", [r"uint4 value = .*WaveGetLaneIndex\(\)"]),
        (
            "gl_NumSamples",
            "int",
            [r"int value = int\(GetRenderTargetSampleCount\(\)\);"],
        ),
    ],
)
def test_fragment_parameter_reads_the_hlsl_form(tmp_path, name, type_name, expected):
    source = FRAGMENT_PARAMETER.format(
        type=type_name, name=name, swizzle=_swizzle(type_name)
    )
    hlsl = _translate(tmp_path, source)
    for pattern in expected:
        assert re.search(pattern, hlsl), (pattern, hlsl)
    assert not USER_GL_SEMANTIC.search(hlsl), hlsl
    _compile(tmp_path, hlsl, "ps_6_6", "PSMain")


def test_intrinsic_inputs_leave_the_signature(tmp_path):
    source = FRAGMENT_PARAMETER.format(
        type="bool", name="gl_HelperInvocation", swizzle=""
    )
    hlsl = _translate(tmp_path, source)
    assert re.search(r"PSMain\(\s*\)", hlsl), hlsl


def test_primitive_shading_rate_member_is_a_uint_system_value(tmp_path):
    hlsl = _translate(
        tmp_path, VERTEX_MEMBER.format(type="int", name="gl_PrimitiveShadingRateEXT")
    )
    assert re.search(r"uint value\s*:\s*SV_ShadingRate;", hlsl), hlsl
    _compile(tmp_path, hlsl, "vs_6_6", "VSMain")


@pytest.mark.parametrize(
    "name", ["gl_SubgroupEqMask", "gl_SubgroupGeMask", "gl_SubgroupGtMask"]
)
def test_compute_lane_mask_parameter_is_a_local(tmp_path, name):
    hlsl = _translate(
        tmp_path, COMPUTE_PARAMETER.format(type="uvec4", name=name, swizzle=".x")
    )
    assert re.search(r"uint4 value = .*WaveGetLaneIndex\(\)", hlsl), hlsl
    assert not USER_GL_SEMANTIC.search(hlsl), hlsl
    _compile(tmp_path, hlsl, "cs_6_6", "CSMain")


def test_fragment_body_reads_take_intrinsics(tmp_path):
    hlsl = _translate(tmp_path, FRAGMENT_BODY)
    assert "WaveGetLaneCount() + WaveGetLaneIndex()" in hlsl
    assert "int(GetRenderTargetSampleCount())" in hlsl
    assert not re.search(r"\bgl_(SubgroupSize|SubgroupInvocationID|NumSamples)\b", hlsl)
    _compile(tmp_path, hlsl, "ps_6_6", "PSMain")


def test_vertex_body_reads_take_wave_intrinsics(tmp_path):
    hlsl = _translate(tmp_path, VERTEX_BODY)
    assert "WaveGetLaneCount() + WaveGetLaneIndex()" in hlsl
    assert "gl_Subgroup" not in hlsl
    _compile(tmp_path, hlsl, "vs_6_6", "VSMain")


@pytest.mark.parametrize("stage", sorted(STAGE_BODY_SAMPLES))
def test_sample_count_outside_pixel_shaders_is_refused(tmp_path, stage):
    with pytest.raises(DirectXLoweringRefused, match="only in pixel shaders"):
        _translate(tmp_path, STAGE_BODY_SAMPLES[stage])


def test_sample_count_on_dx10_is_refused():
    ast = crosstl.translator.parse(
        FRAGMENT_BODY.replace("gl_SubgroupSize + gl_SubgroupInvocationID", "1u")
    )
    with pytest.raises(DirectXLoweringRefused, match="Shader Model 4.1"):
        HLSLCodeGen(target_profile="dx10").generate(ast)


def test_generator_alone_maps_the_sample_index():
    ast = crosstl.translator.parse(
        FRAGMENT_PARAMETER.format(type="uint", name="gl_SampleID", swizzle="")
    )
    assert re.search(r"uint value : SV_SampleIndex", HLSLCodeGen().generate(ast))


@pytest.mark.parametrize(
    "template, name, type_name",
    [
        (FRAGMENT_PARAMETER, "gl_BaryCoordSmoothAMD", "vec2"),
        (FRAGMENT_PARAMETER, "gl_BaryCoordPullModelAMD", "vec3"),
        (FRAGMENT_PARAMETER, "gl_DeviceIndex", "int"),
        (FRAGMENT_PARAMETER, "gl_FragInvocationCountEXT", "int"),
        (FRAGMENT_PARAMETER, "gl_SMIDNV", "uint"),
        (FRAGMENT_PARAMETER, "gl_QuadGroupInvocationID", "uint"),
        (VERTEX_MEMBER, "gl_ViewportMaskNV", "int"),
        (VERTEX_MEMBER, "gl_SecondaryPositionNV", "vec4"),
        (COMPUTE_PARAMETER, "gl_WarpIDNV", "uint"),
    ],
)
def test_builtins_without_a_direct3d_counterpart_are_refused(
    tmp_path, template, name, type_name
):
    source = template.format(type=type_name, name=name, swizzle=_swizzle(type_name))
    with pytest.raises(
        DirectXLoweringRefused, match=f"no system value for the built-in @{name}"
    ):
        _translate(tmp_path, source)


def test_mismatched_parameter_type_is_refused(tmp_path):
    source = FRAGMENT_PARAMETER.format(
        type="vec2", name="gl_SubgroupSize", swizzle=".x"
    )
    with pytest.raises(
        DirectXLoweringRefused, match="HLSL uint value; a float2 parameter"
    ):
        _translate(tmp_path, source)


@pytest.mark.parametrize(
    "template",
    [FRAGMENT_PARAMETER, VERTEX_MEMBER, COMPUTE_PARAMETER],
    ids=["frag", "vert", "comp"],
)
def test_no_crossgl_builtin_becomes_a_user_semantic(tmp_path, template):
    leaked = []
    for name in sorted(CROSSGL_GL_BUILTIN_NAMES):
        source = template.format(type="uint", name=name, swizzle="")
        try:
            hlsl = _translate(tmp_path, source)
        except Exception:
            continue
        if USER_GL_SEMANTIC.search(hlsl):
            leaked.append(name)
    assert leaked == []
