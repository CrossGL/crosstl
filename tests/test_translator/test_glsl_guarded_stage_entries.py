"""A GLSL artifact holds at most one entry point per stage, and refuses a program with two of one stage.

A Metal library with two vertex functions and a fragment function became GLSL with stage guards in which both vertex
functions were ``void main()``, which glslang rejects ("function already has a body"). A later fix kept the first
vertex function as ``main`` and emitted the second as a function nothing calls, which silently dropped it as an
entry point. GLSL has one ``main`` per stage, so the whole-file translation is refused, and each entry point is
translated on its own with ``entry_point``.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.glsl_stage_entries import OpenGLRepeatedStageEntryError

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


def test_two_entries_of_one_stage_are_refused(tmp_path):
    path = tmp_path / "library.metal"
    path.write_text(LIBRARY, encoding="utf-8")

    with pytest.raises(OpenGLRepeatedStageEntryError) as raised:
        crosstl.translate(str(path), backend="opengl")

    assert raised.value.stage == "vertex"
    assert raised.value.entry_points == ("plain", "scaled")
    assert raised.value.code == "project.translate.opengl-repeated-stage-entry"


@pytest.mark.parametrize(
    ("entry_point", "stage", "marker"),
    [
        ("plain", "vert", "float(vid)"),
        ("scaled", "vert", "* 0.5"),
        ("shade", "frag", "vec4("),
    ],
)
def test_each_entry_translates_to_its_own_artifact(
    tmp_path, entry_point, stage, marker
):
    path = tmp_path / "library.metal"
    path.write_text(LIBRARY, encoding="utf-8")
    glsl = crosstl.translate(str(path), backend="opengl", entry_point=entry_point)

    assert glsl.count("void main()") == 1
    assert marker in glsl
    assert "vertex_main" not in glsl

    validator = shutil.which("glslangValidator")
    if validator is None:
        return
    shader = tmp_path / f"{entry_point}.{stage}"
    shader.write_text(glsl, encoding="utf-8")
    completed = subprocess.run([validator, str(shader)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
