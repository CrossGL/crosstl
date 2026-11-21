"""HLSL output keeps a declared semantic next to ``@location(N)`` and writes the location as ``vk::location``.

The HLSL importer writes ``[[vk::location(N)]] float2 uv : TEXCOORD0`` as ``vec2 uv @ TexCoord0 @location(N)``. HLSL
output used to replace such a semantic with ``TEXCOORD<N>``, which renamed vertex inputs the host's input layout
binds by name and dropped the Vulkan location. The attribute sits inside ``#ifdef __spirv__`` because DXC's DXIL
output warns that it ignores it.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

VERTEX = """shader main {
    struct VSInput {
        vec3 Pos @ Position @location(0);
        vec2 UV @ TexCoord0 @location(2);
    }
    struct VSOutput {
        vec4 Pos @ gl_Position;
        vec2 UV @ TexCoord0 @location(1);
    }
    vertex {
    VSOutput main(VSInput input) {
        VSOutput o;
        o.Pos = vec4(input.Pos, 1.0);
        o.UV = input.UV;
        return o;
    }
    }
}
"""

FRAGMENT = """shader main {
    fragment {
    vec4 main(vec2 uv @ TexCoord0 @location(1), vec4 col @ Color0 @location(3)) @ gl_FragColor {
        return vec4(uv, 0.0, 1.0) * col;
    }
    }
}
"""

PLACED_ONLY = """shader main {
    struct VSOutput {
        vec4 Pos @ gl_Position;
        vec2 UV @location(2);
    }
    vertex {
    VSOutput main(vec3 pos @ Position) {
        VSOutput o;
        o.Pos = vec4(pos, 1.0);
        o.UV = pos.xy;
        return o;
    }
    }
}
"""


def _hlsl(tmp_path, program):
    path = tmp_path / "shader.cgl"
    path.write_text(program, encoding="utf-8")
    return crosstl.translate(str(path), backend="directx")


def _guarded(index):
    return re.compile(
        r"#ifdef __spirv__\n\s*\[\[vk::location\(" + str(index) + r"\)\]\]\n#endif\n"
    )


def _dxc(tmp_path, code, profile, entry, *flags):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not on PATH")
    path = tmp_path / "translated.hlsl"
    path.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [dxc, "-T", profile, "-E", entry, "-WX", *flags, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_struct_members_keep_their_semantic_and_write_the_location(tmp_path):
    code = _hlsl(tmp_path, VERTEX)

    assert re.search(_guarded(0).pattern + r"\s*float3 Pos : Position;", code)
    assert re.search(_guarded(2).pattern + r"\s*float2 UV : TexCoord0;", code)
    assert re.search(_guarded(1).pattern + r"\s*float2 UV : TexCoord0;", code)
    assert "TEXCOORD2" not in code


def test_struct_members_compile_to_dxil_and_to_their_spirv_locations(tmp_path):
    code = _hlsl(tmp_path, VERTEX)

    _dxc(tmp_path, code, "vs_6_0", "VSMain")
    spirv = _dxc(tmp_path, code, "vs_6_0", "VSMain", "-spirv")
    assert re.search(r"OpDecorate %in_var_Position Location 0\b", spirv)
    assert re.search(r"OpDecorate %in_var_TexCoord0 Location 2\b", spirv)
    assert re.search(r"OpDecorate %out_var_TexCoord0 Location 1\b", spirv)


def test_entry_parameters_compile_to_their_spirv_locations(tmp_path):
    code = _hlsl(tmp_path, FRAGMENT)

    assert re.search(_guarded(1).pattern + r"\s*float2 uv : TexCoord0", code)
    _dxc(tmp_path, code, "ps_6_0", "PSMain")
    spirv = _dxc(tmp_path, code, "ps_6_0", "PSMain", "-spirv")
    assert re.search(r"OpDecorate %in_var_TexCoord0 Location 1\b", spirv)
    assert re.search(r"OpDecorate %in_var_Color0 Location 3\b", spirv)


def test_a_location_without_a_semantic_still_chooses_texcoord(tmp_path):
    code = _hlsl(tmp_path, PLACED_ONLY)

    assert "float2 UV : TEXCOORD2;" in code
    assert "vk::location" not in code
