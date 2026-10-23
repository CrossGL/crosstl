"""Swizzles of scalar values become vector constructors (or the value) before any generator sees them."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics.scalar_swizzles import expand_scalar_swizzles

HLSL = """
struct VSOutput
{
    float4 Pos : SV_POSITION;
    [[vk::location(0)]] float3 Color : COLOR0;
};

float4 main(VSOutput input) : SV_TARGET
{
    float3 color = lerp(input.Color, dot(float3(0.2126, 0.7152, 0.0722), input.Color).xxx, 0.65);
    float s = input.Color.x;
    float2 t = s.xx + (s * 2.0).rr;
    int i = int(s);
    int3 k = (i + 1).xxx;
    return float4(color + t.xyx + s.x + float3(k), 1.0);
}
"""

CROSSGL = """
shader Splats {
    struct Pair {
        float x;
        float y;
    }

    compute {
        buffer float values[];

        float bump(uint i) {
            values[i] = values[i] + 1.0;
            return values[i];
        }

        @compute @workgroup_size(1, 1, 1)
        void main() {
            vec2 t = bump(0u).xx;
            float u = bump(1u).x;
            vec3 v = vec3(1.0, 2.0, 3.0);
            vec2 w = v.xx;
            Pair p;
            p.x = 1.0;
            p.y = p.x;
            values[2] = t.x + t.y + u + w.y + p.y;
        }
    }
}
"""


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"shader.{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _scalar_swizzles(code):
    return re.findall(r"\)\.(?:x{1,4}|r{1,4})\b", code)


def test_the_pass_counts_scalar_swizzles_only():
    program = Parser(Lexer(CROSSGL).get_tokens()).parse()
    # bump(0u).xx and bump(1u).x; v.xx is a vector swizzle and p.x a structure member.
    assert expand_scalar_swizzles(program) == 2


def test_metal_spells_hlsl_scalar_swizzles_as_constructors(tmp_path):
    code = _translate(tmp_path, HLSL, "frag.hlsl", "metal")
    assert _scalar_swizzles(code) == []
    assert re.search(r"int3 k = int3\(", code)
    if shutil.which("xcrun") is None:
        pytest.skip("the Metal toolchain is not installed")
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    if found.returncode != 0:
        pytest.skip("the Metal toolchain is not installed")
    artifact = tmp_path / "out.metal"
    artifact.write_text(code)
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(artifact)]
        + ["-o", str(tmp_path / "out.air")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_glsl_es_output_has_no_scalar_swizzles(tmp_path):
    code = _translate(tmp_path, HLSL, "frag.hlsl", "webgl")
    assert _scalar_swizzles(code) == []
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")
    artifact = tmp_path / "out.frag"
    artifact.write_text(code)
    result = subprocess.run(
        [glslang, str(artifact)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", ["metal", "opengl", "directx"])
def test_a_swizzled_call_is_evaluated_once(tmp_path, backend):
    code = _translate(tmp_path, CROSSGL, "cgl", backend)
    assert len(re.findall(r"\bbump\(0u", code)) == 1
    assert len(re.findall(r"\bbump\(1u", code)) == 1
    assert _scalar_swizzles(code) == []


def test_vector_swizzles_and_members_are_kept(tmp_path):
    code = _translate(tmp_path, CROSSGL, "cgl", "opengl")
    assert "v.xx" in code
    assert "p.x" in code


IDENTITY = """
shader Identity {
    compute {
        buffer vec2 points[];

        @compute @workgroup_size(1, 1, 1)
        void main() {
            vec2 p = points[0].xy;
            vec3 q = vec3(p, 1.0);
            points[1].xy = q.xyz.xy + p.rg;
        }
    }
}
"""


def test_identity_swizzles_are_dropped():
    program = Parser(Lexer(IDENTITY).get_tokens()).parse()
    # points[0].xy, points[1].xy, q.xyz and p.rg name every component in order; q.xyz.xy does not.
    assert expand_scalar_swizzles(program) == 4


def test_identity_swizzle_stores_reach_metal_as_whole_stores(tmp_path):
    code = _translate(tmp_path, IDENTITY, "cgl", "metal")
    assert not re.search(r"\.xy\s*=", code)
    assert ".xyz" not in code
    assert "q.xy" in code
