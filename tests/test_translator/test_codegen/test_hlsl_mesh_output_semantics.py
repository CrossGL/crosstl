"""HLSL mesh output structs whose user members carry no semantic.

Metal and other sources that link stages by member name leave mesh output members without semantics. The HLSL
generator used to refuse them; it now gives them the ``TEXCOORD<n>`` defaults that vertex outputs and pixel inputs
use, starting the primitive struct after the vertex struct so the pixel stage sees one consistent signature.
"""

import re
import shutil
import subprocess

import pytest

import crosstl.translator
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen

SHADER = """
shader MeshDefaultSemantics {
    struct VOut {
        vec4 position @gl_Position;
        vec3 normal;
        vec2 uv;
    }

    struct POut {
        vec3 color;
        uint material;
    }

    struct FIn {
        vec4 position @gl_Position;
        vec3 normal;
        vec2 uv;
        vec3 color;
        uint material;
    }

    mesh {
        void main(out VOut verts[3] @vertices, out POut prims[1] @primitives,
                  out uvec3 tris[1] @indices) @numthreads(1, 1, 1) @outputtopology(triangle)
                  @max_total_threads_per_threadgroup(32) {
            SetMeshOutputCounts(3u, 1u);
            for (uint i = 0u; i < 3u; i++) {
                verts[i].position = vec4(float(i), 0.0, 0.0, 1.0);
                verts[i].normal = vec3(0.0, 0.0, 1.0);
                verts[i].uv = vec2(float(i), 0.0);
            }
            prims[0].color = vec3(1.0, 0.0, 0.0);
            prims[0].material = 3u;
            tris[0] = uvec3(0u, 1u, 2u);
        }
    }

    fragment {
        vec4 main(FIn input) @gl_FragColor {
            return vec4(input.normal + input.color, float(input.material) + input.uv.x);
        }
    }
}
"""


def _member_semantics(code, struct):
    body = re.search(rf"struct {struct}\s*\{{(.*?)\}};", code, re.DOTALL).group(1)
    return dict(re.findall(r"\s(\w+)\s*:\s*(\w+);", body))


def test_mesh_output_members_take_texcoord_defaults_after_the_vertex_struct():
    code = HLSLCodeGen().generate(crosstl.translator.parse(SHADER))

    assert _member_semantics(code, "VOut") == {
        "position": "SV_POSITION",
        "normal": "TEXCOORD0",
        "uv": "TEXCOORD1",
    }
    assert _member_semantics(code, "POut") == {
        "color": "TEXCOORD2",
        "material": "TEXCOORD3",
    }
    assert _member_semantics(code, "FIn") == {
        "position": "SV_POSITION",
        "normal": "TEXCOORD0",
        "uv": "TEXCOORD1",
        "color": "TEXCOORD2",
        "material": "TEXCOORD3",
    }
    assert "max_total_threads_per_threadgroup" not in code


def test_mesh_output_defaults_compile_and_link_with_dxc(tmp_path):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not available")
    code = HLSLCodeGen().generate(crosstl.translator.parse(SHADER))
    source = tmp_path / "mesh.hlsl"
    source.write_text(code, encoding="utf-8")
    for profile, entry in (("ms_6_5", "MSMain"), ("ps_6_5", "PSMain")):
        result = subprocess.run(
            [dxc, "-T", profile, "-E", entry, str(source)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
