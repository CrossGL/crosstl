"""Metal fragment arguments that read colour attachments are not stage inputs.

A deferred-lighting fragment reads its G-buffer through a struct argument whose members are ``[[color(n)]]`` and
takes its interpolated inputs through a separate ``[[stage_in]]`` argument. The emitter gave both arguments
``[[stage_in]]``, and Metal rejects a second stage input (MSL specification section 5.2.3.4).
"""

import re
import shutil
import subprocess
import textwrap

import pytest

import crosstl

METAL_DEFERRED = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;
    struct QuadInOut { float4 position [[position]]; float2 uv; };
    struct GBufferData { float4 lighting [[color(0)]]; float4 albedo [[color(1)]]; };
    struct AccumOut { float4 lighting [[color(0)]]; };
    fragment AccumOut shade(QuadInOut in [[stage_in]],
                            constant float4 &tint [[buffer(0)]],
                            GBufferData gbuffer) {
        AccumOut out;
        out.lighting = gbuffer.lighting + gbuffer.albedo * tint * in.uv.x;
        return out;
    }
    """)

METAL_GBUFFER_ONLY = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;
    struct GBufferData { float4 lighting [[color(0)]]; float4 albedo [[color(1)]]; };
    fragment float4 resolve(GBufferData gbuffer) {
        return gbuffer.lighting + gbuffer.albedo;
    }
    """)

METAL_MIXED = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;
    struct Mixed { float4 lighting [[color(0)]]; float2 uv; };
    fragment float4 blend(Mixed in [[stage_in]]) {
        return in.lighting * in.uv.x;
    }
    """)

CGL = textwrap.dedent("""
    shader main {
        struct Varyings {
            vec4 position @gl_Position;
            vec2 uv;
        }
        struct GBuffer {
            vec4 lighting @gl_FragColor0;
            vec4 albedo @gl_FragColor1;
        }
        fragment {
            vec4 main(Varyings varyings, GBuffer gbuffer) @gl_FragColor {
                return gbuffer.lighting + gbuffer.albedo * varyings.uv.x;
            }
        }
    }
    """)


def _translate(tmp_path, source, suffix):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="metal")


def _parameter(code, type_name):
    match = re.search(rf"\b{type_name}\s+\w+(\s*\[\[[^\]]*\]\])?\s*[,)]", code)
    assert match, code
    return match.group(0)


def _compile_metal(tmp_path, code):
    if not shutil.which("xcrun"):
        pytest.skip("the Metal toolchain is not installed")
    (tmp_path / "a.metal").write_text(code, encoding="utf-8")
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", "a.metal", "-o", "a.air"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_metal_gbuffer_argument_beside_stage_input(tmp_path):
    code = _translate(tmp_path, METAL_DEFERRED, ".metal")
    assert "[[stage_in]]" in _parameter(code, "QuadInOut")
    assert "[[" not in _parameter(code, "GBufferData")
    assert code.count("[[stage_in]]") == 1
    _compile_metal(tmp_path, code)


def test_metal_gbuffer_only_argument(tmp_path):
    code = _translate(tmp_path, METAL_GBUFFER_ONLY, ".metal")
    assert "[[stage_in]]" not in code
    assert "[[" not in _parameter(code, "GBufferData")
    _compile_metal(tmp_path, code)


def test_metal_struct_mixing_colours_and_varyings_stays_a_stage_input(tmp_path):
    code = _translate(tmp_path, METAL_MIXED, ".metal")
    assert "[[stage_in]]" in _parameter(code, "Mixed")
    _compile_metal(tmp_path, code)


def test_crossgl_colour_struct_parameter(tmp_path):
    code = _translate(tmp_path, CGL, ".cgl")
    assert "[[stage_in]]" in _parameter(code, "Varyings")
    assert "[[" not in _parameter(code, "GBuffer")
    assert code.count("[[stage_in]]") == 1
    _compile_metal(tmp_path, code)
