"""Slang places ``@location(N)`` stage varyings at semantic index and SPIR-V location ``N``.

The Slang generator turned ``@location(N)`` into the bare semantic ``: location``, which loses ``N``, which DXC
rejects as a pixel shader output (``invalid semantic 'LOCATION' for ps``), and which slangc numbers in declaration
order. slangc also ignores the index of ``SV_Target<N>`` when it assigns SPIR-V locations, so an HLSL fragment
output at ``SV_Target2`` after ``SV_Target0`` reached attachment 1. The generator now emits ``TEXCOORD<N>`` or, for a
fragment colour output, ``SV_Target<N>``, with ``[[vk::location(N)]]`` (slang_varying_locations.py).
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

GLSL_VERTEX = """#version 450
layout(location = 3) in vec3 pos;
layout(location = 1) in vec2 tc;
layout(location = 2) out vec2 uv;
layout(location = 0) out vec3 normal;
void main() {
    uv = tc;
    normal = pos;
    gl_Position = vec4(pos, 1.0);
}
"""

GLSL_FRAGMENT = """#version 450
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
void main() { color = vec4(uv, 0.0, 1.0); }
"""

HLSL_TARGETS = """
struct PSOut { float4 a : SV_Target0; float4 b : SV_Target2; };
PSOut main(float4 p : SV_Position) { PSOut o; o.a = p; o.b = p * 2; return o; }
"""

CGL_TARGETS = """
shader main {
    struct Targets {
        vec4 first;
        vec4 second @location(2);
        vec4 third @location(0);
    }
    fragment {
        Targets main(vec4 pos @gl_FragCoord) {
            Targets t;
            t.first = pos;
            t.second = pos;
            t.third = pos;
            return t;
        }
    }
}
"""

CGL_MIXED = """
shader main {
    struct VSOut {
        vec2 uv @location(1);
        vec3 extra;
        vec4 pos @gl_Position;
    }
    vertex {
        VSOut main(vec3 p @location(2)) {
            VSOut o;
            o.uv = p.xy;
            o.extra = p;
            o.pos = vec4(p, 1.0);
            return o;
        }
    }
}
"""


def _slang(tmp_path, source, suffix):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="slang", format_output=False)


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _slangc(tmp_path, code, stage, target):
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    source = tmp_path / f"{stage}.slang"
    source.write_text(code, encoding="utf-8")
    output = tmp_path / f"{stage}.{target}.out"
    completed = subprocess.run(
        [slangc, str(source), "-entry", "main", "-stage", stage]
        + ["-target", target, "-o", str(output)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return output.read_text(encoding="utf-8")


def _spirv_locations(spirv):
    """Varying name (the part after slangc's last ``_`` prefix) to its SPIR-V Location."""
    locations = {}
    for name, location in re.findall(r"OpDecorate %(\S+) Location (\d+)", spirv):
        locations[name.rsplit("_", 1)[-1]] = int(location)
    return locations


def test_glsl_vertex_varyings_keep_their_locations(tmp_path):
    code = _flat(_slang(tmp_path, GLSL_VERTEX, ".vert"))
    assert ": location" not in code
    for location, name in ((3, "pos"), (1, "tc"), (2, "uv"), (0, "normal")):
        assert re.search(
            rf"\[\[vk::location\({location}\)\]\] float\d {name} : TEXCOORD{location};",
            code,
        ), code
    spirv = _slangc(
        tmp_path, _slang(tmp_path, GLSL_VERTEX, ".vert"), "vertex", "spirv-asm"
    )
    assert _spirv_locations(spirv) == {"pos": 3, "tc": 1, "uv": 2, "normal": 0}


def test_glsl_fragment_output_is_a_render_target(tmp_path):
    code = _flat(_slang(tmp_path, GLSL_FRAGMENT, ".frag"))
    assert ": location" not in code
    assert re.search(r"float4 main\(FragmentInput input\) : SV_Target0", code), code
    hlsl = _slangc(
        tmp_path, _slang(tmp_path, GLSL_FRAGMENT, ".frag"), "fragment", "hlsl"
    )
    assert re.search(r"\) : SV_TARGET0?\b", hlsl), hlsl


def test_hlsl_render_target_indices_are_spirv_locations(tmp_path):
    code = _flat(_slang(tmp_path, HLSL_TARGETS, ".hlsl"))
    assert "[[vk::location(0)]] float4 a : SV_Target0;" in code, code
    assert "[[vk::location(2)]] float4 b : SV_Target2;" in code, code
    spirv = _slangc(
        tmp_path, _slang(tmp_path, HLSL_TARGETS, ".hlsl"), "fragment", "spirv-asm"
    )
    assert _spirv_locations(spirv) == {"a": 0, "b": 2}


def test_unplaced_fragment_output_takes_the_free_target(tmp_path):
    code = _flat(_slang(tmp_path, CGL_TARGETS, ".cgl"))
    assert "[[vk::location(1)]] float4 first : SV_Target1;" in code, code
    assert "[[vk::location(2)]] float4 second : SV_Target2;" in code, code
    assert "[[vk::location(0)]] float4 third : SV_Target0;" in code, code
    spirv = _slangc(
        tmp_path, _slang(tmp_path, CGL_TARGETS, ".cgl"), "fragment", "spirv-asm"
    )
    assert _spirv_locations(spirv) == {"first": 1, "second": 2, "third": 0}


def test_placed_structure_gives_every_varying_a_location(tmp_path):
    code = _flat(_slang(tmp_path, CGL_MIXED, ".cgl"))
    assert "[[vk::location(1)]] float2 uv : TEXCOORD1;" in code, code
    # slangc refuses explicit and implicit locations among one direction's varyings (E39023).
    assert "[[vk::location(0)]] float3 extra : TEXCOORD0;" in code, code
    # A system value is no user varying and takes no location.
    assert "; float4 pos : SV_Position;" in code, code
    assert "main([[vk::location(2)]] float3 p : TEXCOORD2)" in code, code
    spirv = _slangc(
        tmp_path, _slang(tmp_path, CGL_MIXED, ".cgl"), "vertex", "spirv-asm"
    )
    assert _spirv_locations(spirv) == {"p": 2, "uv": 1, "extra": 0}


GLSL_POINT_SIZE = """#version 450
layout(location = 0) in vec3 pos;
layout(location = 0) out vec2 uv;
void main() { uv = pos.xy; gl_Position = vec4(pos, 1.0); gl_PointSize = 2.0; }
"""


def test_unnumbered_user_semantic_takes_a_free_location(tmp_path):
    # gl_PointSize becomes the user semantic PSIZE; left implicit, slangc refuses the mix with uv's location.
    code = _flat(_slang(tmp_path, GLSL_POINT_SIZE, ".vert"))
    assert "[[vk::location(0)]] float2 uv : TEXCOORD0;" in code, code
    assert "[[vk::location(1)]] float gl_PointSize : PSIZE;" in code, code
    spirv = _slangc(
        tmp_path, _slang(tmp_path, GLSL_POINT_SIZE, ".vert"), "vertex", "spirv-asm"
    )
    assert _spirv_locations(spirv)["uv"] == 0


GLSL_INTEGER_OUTPUT = """#version 450
layout(location = 0) flat in int id;
layout(location = 0) out uvec4 o;
void main() { o = uvec4(id); }
"""


def test_integer_render_target_is_accepted(tmp_path):
    # SV_Target takes an integer vector for an integer attachment, as in HLSL.
    code = _flat(_slang(tmp_path, GLSL_INTEGER_OUTPUT, ".frag"))
    assert "uint4 main(FragmentInput input) : SV_Target0" in code, code
    spirv = _slangc(
        tmp_path,
        _slang(tmp_path, GLSL_INTEGER_OUTPUT, ".frag"),
        "fragment",
        "spirv-asm",
    )
    assert "OpTypeVector %uint 4" in spirv
