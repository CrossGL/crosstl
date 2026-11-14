"""A fragment colour output may be a scalar or a narrower vector than vec4, as HLSL, MSL and Vulkan allow."""

import re
import shutil
import subprocess

import pytest

import crosstl

SCALAR_TARGET = """Texture2D depthTex : register(t0);
SamplerState samp : register(s0);
struct PSIn { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };
float main(PSIn input) : SV_TARGET
{
    return depthTex.Sample(samp, input.uv).r * 0.5;
}
"""

MIXED_TARGETS = """struct PSOut { float4 color : SV_TARGET0; float occlusion : SV_TARGET1; uint id : SV_TARGET2; };
PSOut main(float4 pos : SV_POSITION)
{
    PSOut o;
    o.color = float4(pos.xy, 0.0, 1.0);
    o.occlusion = pos.z;
    o.id = 7u;
    return o;
}
"""

BOOL_TARGET = """
shader main {
    fragment {
        bool main() @ gl_FragColor { return true; }
    }
}
"""


def _translate(tmp_path, source, name, backend):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


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


def _output_pointer_types(spirv):
    """Map each Location-decorated output variable to the type its pointer points at."""
    types = dict(re.findall(r"(%\w+) = (OpType(?:Float|Int|Vector)[^\n]*)", spirv))
    pointers = dict(re.findall(r"(%\w+) = OpTypePointer Output (%\w+)", spirv))
    variables = dict(re.findall(r"(%\w+) = OpVariable (%\w+) Output", spirv))
    locations = re.findall(r"OpDecorate (%\w+) Location (\d+)", spirv)
    return {
        int(location): types.get(pointers.get(variables.get(variable)))
        for variable, location in locations
        if variable in variables
    }


def test_hlsl_keeps_a_scalar_render_target_return(tmp_path):
    hlsl = _translate(tmp_path, SCALAR_TARGET, "occlusion.hlsl", "directx")

    assert re.search(r"\bfloat \w+\(PSIn input\)\s*:\s*SV_TARGET\b", hlsl), hlsl


def test_spirv_writes_a_scalar_render_target_return_to_a_float_output(tmp_path):
    spirv = _translate(tmp_path, SCALAR_TARGET, "occlusion.hlsl", "vulkan")

    assert _output_pointer_types(spirv)[0] == "OpTypeFloat 32", spirv
    _spirv_val(tmp_path, spirv)


def test_spirv_keeps_the_type_of_each_render_target_member(tmp_path):
    spirv = _translate(tmp_path, MIXED_TARGETS, "mixed.hlsl", "vulkan")

    outputs = _output_pointer_types(spirv)
    assert outputs[0].startswith("OpTypeVector"), spirv
    assert outputs[1] == "OpTypeFloat 32", spirv
    assert outputs[2] == "OpTypeInt 32 0", spirv
    _spirv_val(tmp_path, spirv)


def test_spirv_still_refuses_a_boolean_colour_output(tmp_path):
    with pytest.raises(ValueError, match="float, int or uint scalar or vector"):
        _translate(tmp_path, BOOL_TARGET, "flag.cgl", "vulkan")


@pytest.mark.parametrize(
    ("crossgl_type", "value", "hlsl_type"),
    (
        ("float", "0.5", "float"),
        ("vec3", "vec3(0.5)", "float3"),
        ("uint", "1u", "uint"),
    ),
)
def test_hlsl_accepts_a_narrow_colour_attachment_zero(
    tmp_path, crossgl_type, value, hlsl_type
):
    source = f"""
    shader main {{
        struct Output {{ {crossgl_type} color @ gl_FragColor; }};
        fragment {{
            {crossgl_type} main() @ gl_FragColor {{ return {value}; }}
        }}
    }}
    """
    hlsl = _translate(tmp_path, source, "narrow.cgl", "directx")

    # DXC accepts any numeric scalar or 1-4 component vector at SV_Target; only the legacy GLSL built-in is a vec4.
    assert re.search(rf"\b{hlsl_type} \w+\(\)\s*:\s*SV_TARGET\b", hlsl), hlsl
    assert re.search(rf"\b{hlsl_type} color\s*:\s*SV_TARGET\b", hlsl), hlsl
