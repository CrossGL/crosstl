"""Locations of the user varyings of a GLSL mesh pipeline.

GL_EXT_mesh_shader compiles only under Vulkan rules, where every user input and output needs a location. Mesh
outputs and fragment inputs without a classic semantic used to be declared without one, which glslang rejects.
They now take the locations SPIR-V gives them: vertex members first, then primitive members, skipping the fixed
locations of classic semantics.
"""

import re
import shutil
import subprocess

import pytest

import crosstl.translator
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen

SHADER = """
shader MeshPipelineLocations {
    struct VOut {
        vec4 position @gl_Position;
        vec3 normal;
        vec2 uv @TEXCOORD0;
        mat2 basis;
    }

    struct POut {
        vec3 color;
        uint material;
    }

    struct FIn {
        vec4 position @gl_Position;
        vec3 normal;
        vec2 uv @TEXCOORD0;
        mat2 basis;
        vec3 color;
        uint material;
    }

    mesh {
        void main(out VOut verts[3] @vertices, out POut prims[1] @primitives,
                  out uvec3 tris[1] @indices) @numthreads(1, 1, 1) @outputtopology(triangle) {
            SetMeshOutputCounts(3u, 1u);
            for (uint i = 0u; i < 3u; i++) {
                verts[i].position = vec4(float(i), 0.0, 0.0, 1.0);
                verts[i].normal = vec3(0.0, 0.0, 1.0);
                verts[i].uv = vec2(float(i), 0.0);
                verts[i].basis = mat2(1.0);
            }
            prims[0].color = vec3(1.0, 0.0, 0.0);
            prims[0].material = 3u;
            tris[0] = uvec3(0u, 1u, 2u);
        }
    }

    fragment {
        vec4 main(FIn input) @gl_FragColor {
            vec2 turned = input.basis * input.uv;
            return vec4(input.normal + input.color, float(input.material) + turned.x);
        }
    }
}
"""


def _locations(code, direction):
    return {
        name: int(location)
        for location, name in re.findall(
            rf"layout\(location = (\d+)\)[^;]*\b{direction} \w+ (\w+)(?:\[\d+\])?;",
            code,
        )
    }


def test_mesh_outputs_and_fragment_inputs_take_matching_locations():
    shader = crosstl.translator.parse(SHADER)
    mesh = GLSLCodeGen().generate_stage(shader, "mesh")
    fragment = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(SHADER), "fragment"
    )

    expected = {"normal": 0, "uv": 5, "basis": 1, "color": 3, "material": 4}
    assert _locations(mesh, "out") == expected
    assert _locations(fragment, "in") == expected


def test_mesh_pipeline_locations_compile_and_link_with_glslang(tmp_path):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not available")
    files = []
    for stage, extension in (("mesh", "mesh"), ("fragment", "frag")):
        code = GLSLCodeGen().generate_stage(crosstl.translator.parse(SHADER), stage)
        path = tmp_path / f"pipeline.{extension}"
        path.write_text(code, encoding="utf-8")
        files.append(str(path))
    result = subprocess.run(
        [glslang, "--target-env", "vulkan1.3", "-l", *files],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
