"""HLSL input keeps DXC's ``[[vk::location(N)]]`` as CrossGL ``@location(N)``.

DXC's SPIR-V output places a stage variable with ``[[vk::location(N)]]`` at ``Location N``. The importer used to drop
the attribute, so GLSL, SPIR-V and Metal placed the variable by its semantic alone and moved it away from the
location the host's vertex input or the other stage expects.
"""

from __future__ import annotations

import textwrap

import pytest

import crosstl
from crosstl.backend.import_refusals import DirectXImportRefused


def _translate(tmp_path, code, backend):
    path = tmp_path / "shader.hlsl"
    path.write_text(textwrap.dedent(code), encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


VERTEX = """
    struct VSInput {
        [[vk::location(0)]] float3 Pos : POSITION0;
        [[vk::location(2)]] float2 UV : TEXCOORD0;
    };
    struct VSOutput {
        [[vk::location(5)]] float4 Pos : SV_POSITION;
        [[vk::location(1)]] float2 UV : TEXCOORD0;
    };
    VSOutput main(VSInput input) {
        VSOutput o;
        o.Pos = float4(input.Pos, 1.0);
        o.UV = input.UV;
        return o;
    }
    """


def test_struct_members_keep_their_semantic_and_gain_the_location(tmp_path):
    crossgl = _translate(tmp_path, VERTEX, "cgl")

    assert "vec3 Pos @ Position @location(0);" in crossgl
    assert "vec2 UV @ TexCoord0 @location(2);" in crossgl
    assert "vec2 UV @ TexCoord0 @location(1);" in crossgl
    # DXC ignores the attribute on a system value that becomes a SPIR-V built-in.
    assert "vec4 Pos @ gl_Position;" in crossgl


def test_glsl_output_places_the_members_at_their_vulkan_locations(tmp_path):
    glsl = _translate(tmp_path, VERTEX, "opengl")

    assert "layout(location = 0) in vec3 Pos;" in glsl
    assert "layout(location = 2) in vec2 UV;" in glsl
    assert "layout(location = 1) out vec2" in glsl


def test_entry_parameters_gain_the_location(tmp_path):
    crossgl = _translate(
        tmp_path,
        """
        float4 main([[vk::location(1)]] float2 uv : TEXCOORD0,
                    [[vk::location(3)]] float4 col : COLOR0) : SV_TARGET
        {
            return float4(uv, 0.0, 1.0) * col;
        }
        """,
        "cgl",
    )

    assert "vec2 uv @ TexCoord0 @location(1)" in crossgl
    assert "vec4 col @ Color0 @location(3)" in crossgl


def test_a_render_target_at_its_own_location_needs_no_attribute(tmp_path):
    crossgl = _translate(
        tmp_path,
        """
        struct PSOut {
            [[vk::location(0)]] float4 color : SV_Target0;
            [[vk::location(1)]] float4 normal : SV_Target1;
        };
        PSOut main(float3 n : NORMAL0) {
            PSOut o;
            o.color = float4(n, 1.0);
            o.normal = float4(n * 0.5 + 0.5, 1.0);
            return o;
        }
        """,
        "cgl",
    )

    assert "@location" not in crossgl


def test_a_render_target_at_another_location_is_refused(tmp_path):
    # DXC's SPIR-V output writes render target 0 at Location 1, so Direct3D and Vulkan write different
    # attachments; a CrossGL output names one render target.
    with pytest.raises(DirectXImportRefused, match=r"vk::location\(1\)\]\] on 'color'"):
        _translate(
            tmp_path,
            """
            struct PSOut { [[vk::location(1)]] float4 color : SV_Target0; };
            PSOut main(float3 n : NORMAL0) { PSOut o; o.color = float4(n, 1.0); return o; }
            """,
            "cgl",
        )


def test_a_location_on_a_function_return_with_a_user_semantic_is_refused(tmp_path):
    with pytest.raises(DirectXImportRefused, match="return value of 'main'"):
        _translate(
            tmp_path,
            """
            [[vk::location(2)]] float4 main(float4 p : POSITION) : TEXCOORD0 { return p; }
            """,
            "cgl",
        )


def test_a_location_on_a_render_target_return_is_dropped(tmp_path):
    crossgl = _translate(
        tmp_path,
        """
        [[vk::location(0)]] float4 main(float2 uv : TEXCOORD0) : SV_Target0 {
            return float4(uv, 0.0, 1.0);
        }
        """,
        "cgl",
    )

    assert "vk::location" not in crossgl
    assert "@location" not in crossgl
