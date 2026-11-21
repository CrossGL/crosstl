"""HLSL input infers a pixel entry point that returns nothing from its discard or clip."""

import textwrap

import pytest

import crosstl
from tests.test_backend.test_directx.test_codegen import generate_crossgl, parse_crossgl

DEPTH_PASS = textwrap.dedent("""
    Texture2D colorMap : register(t0);
    SamplerState colorSampler : register(s0);

    void main(float2 uv : TEXCOORD0)
    {
        float alpha = colorMap.Sample(colorSampler, uv).a;
        if (alpha < 0.5) {
            clip(-1);
        }
    }
    """)


@pytest.mark.parametrize(
    "discarding", ["clip(-1);", "discard;", "{ if (uv.x > 0.5) { discard; } }"]
)
def test_a_void_entry_that_discards_is_a_pixel_entry(discarding):
    code = textwrap.dedent(f"""
        void main(float2 uv : TEXCOORD0)
        {{
            {discarding}
        }}
        """)

    output = generate_crossgl(code)

    assert "fragment {" in output
    assert "void main(vec2 uv @ TexCoord0) {" in output
    parse_crossgl(output)


def test_a_helper_without_a_stage_interface_keeps_no_stage():
    code = textwrap.dedent("""
        void reject(float alpha)
        {
            clip(alpha - 0.5);
        }

        float4 main(float2 uv : TEXCOORD0) : SV_Target
        {
            reject(uv.x);
            return float4(uv, 0.0, 1.0);
        }
        """)

    output = generate_crossgl(code)

    # The helper stays a module-scope function outside the one fragment stage block.
    assert output.count("fragment {") == 1
    assert output.index("void reject(float alpha)") < output.index("fragment {")
    parse_crossgl(output)


@pytest.mark.parametrize(
    ("backend", "spelling"),
    [
        ("metal", "fragment void fragment_main("),
        ("opengl", "discard;"),
        ("vulkan", "OpKill"),
    ],
)
def test_a_depth_pass_translates_as_a_pixel_shader(tmp_path, backend, spelling):
    source = tmp_path / "depthpass.frag"
    source.write_text(DEPTH_PASS, encoding="utf-8")

    output = crosstl.translate(str(source), backend=backend, source_backend="directx")

    assert spelling in output
