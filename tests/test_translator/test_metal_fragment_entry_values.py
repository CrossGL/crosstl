"""Metal fragment entry points: scalar colour returns and the position a [[stage_in]] member already carries."""

import re
import shutil
import subprocess

import pytest

import crosstl


def _translate(tmp_path, source, suffix):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _compile(tmp_path, code):
    if shutil.which("xcrun") is None:
        return
    source = tmp_path / "out.metal"
    source.write_text(code)
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "out.air"),
        ],
        capture_output=True,
        text=True,
    )
    if "unable to find utility" in result.stderr:
        return
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "result_type, value, metal_type",
    [
        ("float", "length(input.Pos.xyz)", "float"),
        ("float2", "input.Pos.xy", "float2"),
        ("uint", "uint(input.Pos.x)", "uint"),
    ],
)
def test_hlsl_scalar_and_vector_colour_returns_reach_metal(
    tmp_path, result_type, value, metal_type
):
    source = f"""struct VSOutput {{
    float4 Pos : POSITION0;
}};

{result_type} main(VSOutput input) : SV_TARGET
{{
    return {value};
}}
"""
    code = _translate(tmp_path, source, ".hlsl")
    assert re.search(
        rf"fragment {metal_type} fragment_main\(VSOutput input \[\[stage_in\]\]\)", code
    ), code
    _compile(tmp_path, code)


def test_hlsl_bool_colour_return_is_still_refused(tmp_path):
    # A bool colour value has no Metal colour attachment format.
    with pytest.raises(Exception, match="color"):
        _translate(
            tmp_path,
            "bool main(float4 p : SV_Position) : SV_TARGET { return p.x > 0.0; }\n",
            ".hlsl",
        )


SUBPASS = """[[vk::input_attachment_index(0)]][[vk::binding(1)]] SubpassInput depthInput;

struct VSOutput
{
    float4 Pos : SV_POSITION;
[[vk::location(0)]] float2 UV : TEXCOORD0;
};

float4 main(VSOutput input) : SV_TARGET
{
    float depth = depthInput.SubpassLoad().a;
    return float4(input.UV, depth, input.Pos.z);
}
"""


def test_fragment_coordinate_reuses_the_stage_in_position_member(tmp_path):
    code = _translate(tmp_path, SUBPASS, ".hlsl")
    # The subpass load reads the fragment's pixel; the input structure already carries [[position]].
    assert code.count("[[position]]") == 1, code
    assert "_crossglFragCoord" not in code
    assert re.search(r"\.read\(uint2\(int2\(input\.Pos\.xy\)\)", code), code
    _compile(tmp_path, code)


def test_crossgl_fragment_coordinate_reads_reuse_the_input_member(tmp_path):
    source = """shader S {
    struct FIn {
        vec4 position @ gl_Position;
        vec2 uv @ TEXCOORD0;
    }
    fragment {
        vec4 main(FIn input) @ gl_FragColor {
            return vec4(input.uv, gl_FragCoord.xy);
        }
    }
}
"""
    code = _translate(tmp_path, source, ".cgl")
    assert code.count("[[position]]") == 1, code
    assert "input.position.xy" in code, code
    _compile(tmp_path, code)
