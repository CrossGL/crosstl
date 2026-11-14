"""A multi-stage GLSL artifact declares one ``main`` per stage, also when a stage has several entry points.

A Metal library with two vertex functions and a fragment function became GLSL with stage guards in which both vertex
functions were ``void main()``, which glslang rejects ("function already has a body"). As for a single-stage
program, the stage's first entry point is its ``main`` and later ones stay ordinary functions.
"""

import re
import shutil
import subprocess

import crosstl

LIBRARY = """
#include <metal_stdlib>
using namespace metal;

struct Varyings {
    float4 position [[position]];
    float2 uv;
};

vertex Varyings plain(uint vid [[vertex_id]]) {
    Varyings out;
    out.position = float4(float(vid), 0.0, 0.0, 1.0);
    out.uv = float2(0.0);
    return out;
}

vertex Varyings scaled(uint vid [[vertex_id]]) {
    Varyings out;
    out.position = float4(float(vid) * 0.5, 0.0, 0.0, 1.0);
    out.uv = float2(1.0);
    return out;
}

fragment float4 shade(Varyings in [[stage_in]]) {
    return float4(in.uv, 0.0, 1.0);
}
"""


def _sections(glsl, guard):
    return "\n".join(re.findall(rf"#ifdef {guard}\n(.*?)#endif", glsl, re.S))


def test_each_stage_section_declares_one_main(tmp_path):
    path = tmp_path / "library.metal"
    path.write_text(LIBRARY, encoding="utf-8")
    glsl = crosstl.translate(str(path), backend="opengl")

    vertex = _sections(glsl, "GL_VERTEX_SHADER")
    fragment = _sections(glsl, "GL_FRAGMENT_SHADER")
    assert vertex.count("void main()") == 1
    assert "void vertex_main()" in vertex
    assert fragment.count("void main()") == 1

    validator = shutil.which("glslangValidator")
    if validator is None:
        return
    shader = tmp_path / "library.glsl"
    shader.write_text(glsl, encoding="utf-8")
    # glslang defines the stage's GL_<STAGE>_SHADER guard itself, as the corpus checks rely on.
    for stage in ("vert", "frag"):
        completed = subprocess.run(
            [validator, "-S", stage, str(shader)], capture_output=True, text=True
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
