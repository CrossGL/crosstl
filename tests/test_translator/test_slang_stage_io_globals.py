"""Slang carries module-scope ``in`` and ``out`` stage variables as entry-point parameters.

CrossGL keeps GLSL's ``layout(location = N) out vec4 color;`` as a module-scope ``out`` variable, which the GLSL import
does for every fragment shader with more than one colour output. The Slang generator declared it as a plain global,
which Slang reads as a uniform, so a write to it did not compile (slangc E30011) and an input came from a constant
buffer. Each such variable is now a ``static`` global that the entry point copies from an ``in`` parameter, or into
an ``out`` parameter in a ``defer`` block (slang_stage_io_globals.py).
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

GLSL_GBUFFER = """#version 450
layout(location = 0) in vec2 uv;
layout(location = 1) flat in int id;
layout(location = 0) out vec4 albedo;
layout(location = 1) out vec4 normal;
layout(location = 3) out uvec4 ids;
void write_normal(vec3 n) { normal = vec4(n, 1.0); }
void main() {
    albedo = vec4(uv, 0.0, 1.0);
    write_normal(vec3(uv, 1.0));
    ids = uvec4(id);
    gl_FragDepth = 0.75;
}
"""

CGL_EARLY_RETURN = """
shader main {
    struct FragmentInput {
        vec2 uv @location(0);
    };
    out vec4 color @location(0);
    out vec4 extra @location(2);
    void helper() {
        extra = vec4(1.0);
    }
    fragment {
        void main(FragmentInput input) {
            color = vec4(input.uv, 0.0, 1.0);
            helper();
            if (input.uv.x > 0.5) {
                return;
            }
            extra = vec4(0.0);
        }
    }
}
"""

CGL_VERTEX = """
shader main {
    in vec3 pos @location(0);
    out vec2 uv @location(1);
    vertex {
        void main() {
            uv = pos.xy;
            gl_Position = vec4(pos, 1.0);
        }
    }
}
"""

CGL_UNPLACED = """
shader main {
    in vec3 n @location(1) @flat;
    in vec2 uv @location(0);
    out vec4 a @location(1);
    out vec4 b;
    fragment {
        void main() {
            a = vec4(n, 1.0);
            b = vec4(uv, 0.0, 1.0);
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
    """Varying name, without slangc's ``input_`` or ``crossgl_in_``/``crossgl_out_`` prefix, to its Location."""
    locations = {}
    for name, location in re.findall(r"OpDecorate %(\S+) Location (\d+)", spirv):
        name = re.sub(r"^(input_|crossgl_(in|out)_)", "", name)
        locations[name] = int(location)
    return locations


def test_glsl_fragment_outputs_become_entry_parameters(tmp_path):
    code = _flat(_slang(tmp_path, GLSL_GBUFFER, ".frag"))
    for declaration in (
        "static float4 albedo;",
        "static float4 normal;",
        "static uint4 ids;",
    ):
        assert declaration in code, code
    for location, kind, name in (
        (0, "float4", "albedo"),
        (1, "float4", "normal"),
        (3, "uint4", "ids"),
    ):
        assert (
            f"[[vk::location({location})]] out {kind} crossgl_out_{name} : SV_Target{location}"
            in code
        ), code
    assert re.search(
        r"defer \{ crossgl_out_albedo = albedo; crossgl_out_normal = normal; crossgl_out_ids = ids; \}",
        code,
    ), code
    # A write in a called function reaches the output through the static global.
    assert "void write_normal(float3 n) { normal = float4(n, 1.0); }" in code, code
    assert ": SV_Depth" in code
    spirv = _slangc(
        tmp_path, _slang(tmp_path, GLSL_GBUFFER, ".frag"), "fragment", "spirv-asm"
    )
    assert _spirv_locations(spirv) == {
        "uv": 0,
        "id": 1,
        "albedo": 0,
        "normal": 1,
        "ids": 3,
    }


def test_outputs_are_copied_on_every_return_path(tmp_path):
    code = _flat(_slang(tmp_path, CGL_EARLY_RETURN, ".cgl"))
    assert (
        "defer { crossgl_out_color = color; crossgl_out_extra = extra; }" in code
    ), code
    hlsl = _slangc(
        tmp_path, _slang(tmp_path, CGL_EARLY_RETURN, ".cgl"), "fragment", "hlsl"
    )
    # slangc places the deferred copy before the early return and before the end of the body.
    assert len(re.findall(r"crossgl_out_extra_\d+ = extra_\d+;", hlsl)) == 2, hlsl


def test_vertex_inputs_and_outputs_become_entry_parameters(tmp_path):
    code = _flat(_slang(tmp_path, CGL_VERTEX, ".cgl"))
    assert "static float3 pos;" in code and "static float2 uv;" in code, code
    assert "[[vk::location(0)]] in float3 crossgl_in_pos : TEXCOORD0" in code, code
    assert "[[vk::location(1)]] out float2 crossgl_out_uv : TEXCOORD1" in code, code
    assert "pos = crossgl_in_pos; defer { crossgl_out_uv = uv; }" in code, code
    spirv = _slangc(
        tmp_path, _slang(tmp_path, CGL_VERTEX, ".cgl"), "vertex", "spirv-asm"
    )
    assert _spirv_locations(spirv) == {"pos": 0, "uv": 1}


def test_unplaced_output_takes_a_free_location_and_inputs_keep_interpolation(tmp_path):
    code = _flat(_slang(tmp_path, CGL_UNPLACED, ".cgl"))
    assert (
        "[[vk::location(1)]] in nointerpolation float3 crossgl_in_n : TEXCOORD1" in code
    ), code
    assert "[[vk::location(0)]] out float4 crossgl_out_b : SV_Target0" in code, code
    spirv = _slangc(
        tmp_path, _slang(tmp_path, CGL_UNPLACED, ".cgl"), "fragment", "spirv-asm"
    )
    assert _spirv_locations(spirv) == {"n": 1, "uv": 0, "a": 1, "b": 0}
    assert re.search(r"OpDecorate %crossgl_in_n Flat", spirv), spirv


CGL_STAGE_ATTRIBUTES = """
shader main {
    fragment {
        float4 color @output @location(0);
        float2 uv @input @location(0);
        float4 extra @output @location(1);
        float3 normal @input @location(1);
        void main() {
            color = float4(uv[0], uv[1], 0.0, 1.0);
            extra = float4(normal[0], normal[1], normal[2], 1.0);
            return;
        }
    }
}
"""


def test_spirv_import_stage_variables_become_entry_parameters(tmp_path):
    # The SPIR-V import declares a stage's interface variables in its stage block with @input and @output.
    code = _flat(_slang(tmp_path, CGL_STAGE_ATTRIBUTES, ".cgl"))
    assert "static float4 color;" in code and "static float2 uv;" in code, code
    assert "[[vk::location(1)]] out float4 crossgl_out_extra : SV_Target1" in code, code
    assert "[[vk::location(1)]] in float3 crossgl_in_normal : TEXCOORD1" in code, code
    spirv = _slangc(
        tmp_path,
        _slang(tmp_path, CGL_STAGE_ATTRIBUTES, ".cgl"),
        "fragment",
        "spirv-asm",
    )
    assert _spirv_locations(spirv) == {"color": 0, "uv": 0, "extra": 1, "normal": 1}


CGL_SPIRV_VERTEX = """
shader main {
    vertex {
        float2 uv @output @location(1);
        float3 pos @input @location(0);
        int gl_VertexID @input @gl_VertexID;
        float4 gl_Position @output @gl_Position;
        float gl_ClipDistance[1] @output @gl_ClipDistance;
        void main() {
            uv = pos.xy + float2(float(gl_VertexID), 0.0);
            gl_Position = float4(pos[0], pos[1], pos[2], 1.0);
            return;
        }
    }
}
"""

CGL_SPIRV_FRAGMENT = """
shader main {
    fragment {
        float4 color @output @location(0);
        float2 uv @input @location(0);
        float4 gl_FragCoord @input @gl_FragCoord;
        float gl_FragDepth @output @gl_FragDepth;
        void main() {
            color = float4(uv[0], uv[1], gl_FragCoord[0], 1.0);
            gl_FragDepth = 0.5;
        }
    }
}
"""


def test_spirv_import_builtin_outputs_keep_their_system_values(tmp_path):
    code = _flat(_slang(tmp_path, CGL_SPIRV_VERTEX, ".cgl"))
    assert "out float4 crossgl_out_gl_Position : SV_Position" in code, code
    assert "crossgl_out_gl_Position = gl_Position;" in code, code
    # A built-in input is read through its system-value parameter, not declared as a uniform.
    assert "int gl_VertexID;" not in code and ": SV_VertexID" in code, code
    # An unused output is static rather than a shader parameter.
    assert "static float gl_ClipDistance[1];" in code, code
    spirv = _slangc(
        tmp_path, _slang(tmp_path, CGL_SPIRV_VERTEX, ".cgl"), "vertex", "spirv-asm"
    )
    assert re.search(r"OpDecorate %\S+ BuiltIn Position", spirv), spirv
    assert _spirv_locations(spirv) == {"pos": 0, "uv": 1}


def test_declared_builtin_output_is_not_also_returned(tmp_path):
    # Without a return statement the generator would otherwise also return gl_FragDepth as SV_Depth.
    code = _flat(_slang(tmp_path, CGL_SPIRV_FRAGMENT, ".cgl"))
    assert "void main(" in code, code
    assert code.count("SV_Depth") == 1, code
    assert "crossgl_out_gl_FragDepth = gl_FragDepth;" in code, code
    spirv = _slangc(
        tmp_path, _slang(tmp_path, CGL_SPIRV_FRAGMENT, ".cgl"), "fragment", "spirv-asm"
    )
    assert re.search(r"OpDecorate %\S+ BuiltIn FragDepth", spirv), spirv
    assert re.search(r"OpDecorate %\S+ BuiltIn FragCoord", spirv), spirv
    assert _spirv_locations(spirv) == {"color": 0, "uv": 0}
