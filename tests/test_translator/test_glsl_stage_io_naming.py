"""GLSL stage inputs and outputs: the default clip position and the names varyings are declared under.

GLSL declares stage inputs and outputs as globals, in one namespace with variables, functions and the members of
unnamed blocks (GLSL 4.60 section 4.2.7), and links a varying without a location to the output of the same name
(section 4.3.4). Four rules follow, each found on a real shader (grid.vert, showcase_pbr.cgl):

* a vertex output member named ``position`` that is a float4 without a semantic is the clip position, as the HLSL
  (``SV_Position``) and Metal (``[[position]]``) generators already treat it, so GLSL writes ``gl_Position``;
* a stage input or output that would redeclare a global, such as a member of an unnamed uniform block, takes
  another name, and the fragment input that reads it by name takes the same one;
* each guarded stage section of a combined output is compiled on its own, so a fragment input keeps the name of the
  vertex output it links to;
* a local variable never hides the stage output it is written to.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest

import crosstl.translator
from crosstl.translator.codegen.clip_position_defaults import (
    default_clip_position_member,
    is_default_clip_position_name,
)
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen

GLSLANG = shutil.which("glslangValidator")

CAMERA_SHADER = """
shader CameraVaryings {
    struct VertexInput {
        vec3 position;
        vec2 view;
    }
    struct VertexOutput {
        vec3 worldPosition;
        vec4 view;
        vec2 uv;
        vec4 position;
    }
    struct FragmentInput {
        vec3 worldPosition;
        vec4 view;
        vec2 uv;
    }
    struct FragmentOutput {
        vec4 color;
    }
    cbuffer Camera {
        mat4 view;
        mat4 projection;
    }
    vertex {
        VertexOutput main(VertexInput input) {
            VertexOutput output;
            vec4 worldPosition = vec4(input.position, 1.0);
            output.worldPosition = worldPosition.xyz;
            output.view = vec4(input.view, 0.0, 1.0);
            output.uv = input.view;
            output.position = projection * worldPosition;
            return output;
        }
    }
    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            output.color = vec4(input.worldPosition, 1.0) + input.view + vec4(input.uv, 0.0, 0.0);
            return output;
        }
    }
}
"""


def stage_section(code, macro):
    """The concatenated text of every ``#ifdef <macro>`` section of a combined GLSL output."""
    return "".join(re.findall(rf"#ifdef {macro}\n(.*?)#endif\n", code, flags=re.DOTALL))


def declarations(code, storage):
    return {
        name: type_name
        for type_name, name in re.findall(
            rf"^(?:layout\([^)]*\) )?{storage} (\w+) (\w+);$", code, flags=re.MULTILINE
        )
    }


def member(name, type_name="vec4", semantic=None):
    return SimpleNamespace(name=name, type_name=type_name, semantic=semantic)


def pick(members):
    return default_clip_position_member(
        members,
        has_semantic=lambda node: node.semantic is not None,
        declares_position=lambda node: node.semantic == "gl_Position",
        is_float4=lambda node: node.type_name == "vec4",
    )


@pytest.mark.parametrize(
    "name", ["position", "Position", "clipPosition", "clip_position", "gl_Position"]
)
def test_default_clip_position_names(name):
    assert is_default_clip_position_name(name)


@pytest.mark.parametrize("name", ["pos", "worldPosition", "positions", None])
def test_other_names_do_not_carry_the_clip_position(name):
    assert not is_default_clip_position_name(name)


def test_default_clip_position_member_selection():
    first = member("position")
    assert pick([member("uv", "vec2"), first, member("clipPosition")]) is first
    # A member that declares the position keeps it, and so do members with any other semantic.
    assert pick([member("position"), member("pos", semantic="gl_Position")]) is None
    assert pick([member("position", semantic="TEXCOORD0")]) is None
    # Only a four-component float vector is a clip position.
    assert pick([member("position", "vec3")]) is None
    assert pick([]) is None and pick(None) is None


def test_vertex_output_position_without_semantic_writes_gl_position():
    ast = crosstl.translator.parse(CAMERA_SHADER)
    vertex = GLSLCodeGen().generate_stage(ast, "vertex")
    assert "gl_Position = (projection * worldPosition_2);" in vertex
    assert "out vec4 position;" not in vertex
    assert "out vec4 out_position;" not in vertex

    # The HLSL and Metal generators give the same member the clip position.
    assert "SV_Position" in HLSLCodeGen().generate(ast)
    assert "[[position]]" in MetalCodeGen().generate(ast)


def test_a_declared_position_semantic_keeps_a_member_named_position_a_varying():
    ast = crosstl.translator.parse("""
    shader DeclaredPosition {
        struct VertexOutput {
            vec4 clip @ gl_Position;
            vec4 position;
        }
        vertex {
            VertexOutput main() {
                VertexOutput output;
                output.clip = vec4(0.0, 0.0, 0.0, 1.0);
                output.position = vec4(1.0);
                return output;
            }
        }
    }
    """)
    vertex = GLSLCodeGen().generate_stage(ast, "vertex")
    assert "gl_Position = vec4(0.0, 0.0, 0.0, 1.0);" in vertex
    assert re.search(r"out vec4 \w*position;", vertex)
    assert "gl_Position = vec4(1.0);" not in vertex


def test_a_vec3_position_is_not_the_clip_position():
    ast = crosstl.translator.parse("""
    shader Vec3Position {
        struct VertexOutput {
            vec3 position;
            vec4 clip @ gl_Position;
        }
        vertex {
            VertexOutput main() {
                VertexOutput output;
                output.position = vec3(1.0);
                output.clip = vec4(1.0);
                return output;
            }
        }
    }
    """)
    vertex = GLSLCodeGen().generate_stage(ast, "vertex")
    assert "gl_Position = vec4(1.0);" in vertex
    assert "gl_Position = vec3(1.0);" not in vertex


def test_stage_io_that_would_redeclare_an_unnamed_block_member_is_renamed():
    ast = crosstl.translator.parse(CAMERA_SHADER)
    vertex = GLSLCodeGen().generate_stage(ast, "vertex")
    fragment = GLSLCodeGen().generate_stage(ast, "fragment")

    # ``view`` is a member of the unnamed block ``Camera``, so neither the vertex input nor the varying may use it.
    assert "uniform Camera {\n    mat4 view;" in vertex
    assert declarations(vertex, "in") == {"position": "vec3", "in_view": "vec2"}
    assert declarations(vertex, "out") == {
        "worldPosition": "vec3",
        "out_view": "vec4",
        "uv": "vec2",
    }
    assert "out_view = vec4(in_view, 0.0, 1.0);" in vertex
    assert "uv = in_view;" in vertex

    # The fragment input is another structure's member; it still links to the renamed vertex output.
    assert declarations(fragment, "in") == declarations(vertex, "out")
    assert "+ out_view)" in fragment


def test_combined_output_declares_the_same_varyings_as_the_separate_stages():
    ast = crosstl.translator.parse(CAMERA_SHADER)
    combined = GLSLCodeGen().generate(ast)
    vertex = stage_section(combined, "GL_VERTEX_SHADER")
    fragment = stage_section(combined, "GL_FRAGMENT_SHADER")

    assert declarations(vertex, "out") == declarations(
        GLSLCodeGen().generate_stage(ast, "vertex"), "out"
    )
    assert declarations(fragment, "in") == declarations(vertex, "out")
    assert "in_worldPosition" not in combined and "in_uv" not in combined


def test_a_local_never_hides_the_stage_output_it_is_written_to():
    ast = crosstl.translator.parse(CAMERA_SHADER)
    vertex = GLSLCodeGen().generate_stage(ast, "vertex")
    assert "vec4 worldPosition_2 = vec4(position, 1.0);" in vertex
    assert "worldPosition = worldPosition_2.xyz;" in vertex
    assert "vec4 worldPosition =" not in vertex


def link(tmp_path, vertex, fragment):
    (tmp_path / "shader.vert").write_text(vertex)
    (tmp_path / "shader.frag").write_text(fragment)
    return subprocess.run(
        [GLSLANG, "-l", "shader.vert", "shader.frag"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )


@pytest.mark.skipif(GLSLANG is None, reason="glslangValidator is not installed")
def test_separate_stages_compile_and_link(tmp_path):
    ast = crosstl.translator.parse(CAMERA_SHADER)
    result = link(
        tmp_path,
        GLSLCodeGen().generate_stage(ast, "vertex"),
        GLSLCodeGen().generate_stage(ast, "fragment"),
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(GLSLANG is None, reason="glslangValidator is not installed")
def test_combined_stage_sections_compile_and_link(tmp_path):
    # glslang defines GL_VERTEX_SHADER or GL_FRAGMENT_SHADER from the file's stage, as hosts do for this output.
    combined = GLSLCodeGen().generate(crosstl.translator.parse(CAMERA_SHADER))
    result = link(tmp_path, combined, combined)
    assert result.returncode == 0, result.stdout + result.stderr
