"""Structure-typed GLSL vertex outputs and fragment inputs.

The GLSL generator declared ``out Inner inner;`` before ``struct Inner``, so a shader with a structure-typed
varying never compiled. The declarations now follow the data structures. A structure-typed varying with a
built-in field (Metal's nested ``[[stage_in]]`` with ``[[position]]``) is refused rather than turned into a user
varying.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.glsl_nested_stage_io import GLSLNestedStageBuiltinError

NESTED = """
shader NestedVaryings {
    struct Inner {
        vec3 a;
        float b;
    }

    struct VOut {
        vec4 position @gl_Position;
        Inner inner;
        vec2 uv;
    }

    vertex {
        VOut main(vec3 position @POSITION) {
            VOut o;
            o.position = vec4(position, 1.0);
            o.inner.a = position;
            o.inner.b = 1.0;
            o.uv = position.xy;
            return o;
        }
    }

    fragment {
        vec4 main(VOut input) @gl_FragColor {
            return vec4(input.inner.a * input.inner.b, input.uv.x);
        }
    }
}
"""

METAL = """#include <metal_stdlib>
using namespace metal;
struct VOut { float4 position [[position]]; float3 normal; };
struct POut { float3 color; };
struct FIn { VOut v; POut p; };
fragment float4 fs(FIn in [[stage_in]]) { return float4(in.v.normal + in.p.color, 1.0); }
"""


def _translate(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="opengl")


def test_structure_varyings_follow_their_structure_and_link(tmp_path):
    code = _translate(tmp_path, "nested.cgl", NESTED)

    assert code.index("struct Inner {") < code.index("out Inner inner;")
    assert code.index("struct Inner {") < code.index("in Inner inner;")

    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not available")
    vertex = tmp_path / "nested.vert"
    fragment = tmp_path / "nested.frag"
    vertex.write_text(code, encoding="utf-8")
    fragment.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [glslang, "-l", str(vertex), str(fragment)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_structure_varying_with_a_builtin_field_is_refused(tmp_path):
    with pytest.raises(GLSLNestedStageBuiltinError, match="'v.position'.*gl_Position"):
        _translate(tmp_path, "nested.metal", METAL)

    vertex_builtin = NESTED.replace(
        "struct Inner {\n        vec3 a;",
        "struct Inner {\n        vec4 clip @gl_Position;\n        vec3 a;",
    )
    with pytest.raises(GLSLNestedStageBuiltinError, match="vertex output 'inner'"):
        _translate(tmp_path, "builtin.cgl", vertex_builtin)
