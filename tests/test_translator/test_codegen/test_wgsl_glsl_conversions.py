"""WGSL output of GLSL conversions WGSL does not make itself.

- A matrix constructed from a matrix of another shape keeps the overlapping elements and takes the identity's
  elsewhere (GLSL 4.60 section 5.4.2); WGSL converts only between matrices of one shape, so naga rejected
  ``mat2x2<f32>(m)`` of a ``mat3x3<f32>``.
- An integer operand of a floating-point arithmetic or comparison operator converts to the floating-point type
  (GLSL 4.60 section 4.1.10); WGSL has no implicit conversions, so ``a * 0.5`` of an ``i32`` was rejected.
- A store to a multi-component swizzle spelt with colour names (``c.ga = v``) was printed as is; WGSL assigns one
  component at a time.
- A ``bvecN`` uniform block member was declared ``vecN<bool>``, which is not host-shareable (WGSL section 6.2.1).
"""

import shutil
import subprocess

import pytest

from crosstl.translator.codegen.wgsl_codegen import WGSLCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def generate_wgsl(source):
    ast = Parser(Lexer(source).get_tokens()).parse()
    return WGSLCodeGen().generate(ast)


def validate_with_naga(tmp_path, code):
    naga = shutil.which("naga")
    if naga is None:
        pytest.skip("naga is not installed")
    path = tmp_path / "shader.wgsl"
    path.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [naga, str(path)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


COMPUTE = """
shader Probe {
    UNIFORMS
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer int* iin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            BODY
        }
    }
}
"""


def compute(body, uniforms=""):
    return generate_wgsl(COMPUTE.replace("UNIFORMS", uniforms).replace("BODY", body))


def test_matrix_resize_calls_a_helper_per_shape_pair(tmp_path):
    code = compute("""
            mat3 m = mat3(fin[i], 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
            mat2 a = mat2(m);
            mat3 b = mat3(a * 2.0);
            mat4 c = mat4(m);
            mat2x3 n = mat2x3(m);
            mat2 d = mat2(m);
            outp[i] = a[1][1] + b[2][2] + c[3][3] + n[1][2] + d[0][0];
""")
    assert "var a: mat2x2<f32> = crosstl_resize_mat3x3_f32_to_mat2x2_f32(m);" in code
    assert (
        "var b: mat3x3<f32> = crosstl_resize_mat2x2_f32_to_mat3x3_f32((a * 2.0));"
        in code
    )
    assert code.count("fn crosstl_resize_mat3x3_f32_to_mat2x2_f32(") == 1
    assert "    return mat2x2<f32>(m[0].xy, m[1].xy);\n" in code
    assert (
        "    return mat3x3<f32>(vec3<f32>(m[0], 0.0), vec3<f32>(m[1], 0.0), "
        "vec3<f32>(0.0, 0.0, 1.0));\n" in code
    )
    assert (
        "    return mat4x4<f32>(vec4<f32>(m[0], 0.0), vec4<f32>(m[1], 0.0), "
        "vec4<f32>(m[2], 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));\n" in code
    )
    assert "    return mat2x3<f32>(m[0], m[1]);\n" in code
    validate_with_naga(tmp_path, code)


def test_same_shape_matrix_constructor_is_unchanged(tmp_path):
    code = compute("""
            mat3 m = mat3(fin[i]);
            mat3 k = mat3(m);
            outp[i] = k[0][0];
""")
    assert "var k: mat3x3<f32> = mat3x3<f32>(m);" in code
    assert "crosstl_resize" not in code
    validate_with_naga(tmp_path, code)


def test_integer_operands_of_float_operators_are_converted(tmp_path):
    code = compute("""
            int a = iin[i];
            uint u = 3u;
            float x = fin[i];
            float y = a * 0.5 + 1;
            float z = x / 4 + u;
            bool w = a < x;
            vec3 v = vec3(x) * ivec3(a, 2, 3);
            outp[i] = y + z + v.y + (w ? 1.0 : 0.0);
""")
    assert "var y: f32 = ((f32(a) * 0.5) + 1);" in code
    assert "var z: f32 = ((x / 4) + f32(u));" in code
    assert "var w: bool = (f32(a) < x);" in code
    assert "vec3<f32>(vec3<i32>(a, 2, 3))" in code
    validate_with_naga(tmp_path, code)


@pytest.mark.parametrize("names", ["ga", "tq", "yw"])
def test_colour_and_texture_swizzle_stores_are_component_wise(tmp_path, names):
    code = compute(f"""
            vec4 c = vec4(fin[i], 1.0, 2.0, 3.0);
            c.{names} = vec2(float(iin[i]), 5.0);
            outp[i] = c.x + c.y + c.z + c.w;
""")
    assert f"c.{names} =" not in code
    assert "    c.y = (c_yw).x;\n" in code
    assert "    c.w = (c_yw).y;\n" in code
    validate_with_naga(tmp_path, code)


def test_uniform_bool_vectors_are_u32_vectors(tmp_path):
    code = compute(
        """
            bvec4 c = f;
            c.y = !g.x;
            outp[i] = (c.x ? s : 1.0) + (c.w && g.y ? 4.0 : 0.0) + (h ? 8.0 : 0.0);
""",
        uniforms="""cbuffer Params @binding(3) {
        bvec4 f;
        bvec2 g;
        bool h;
        float s;
    };""",
    )
    assert "    f: vec4<u32>,\n    g: vec2<u32>,\n    h: u32,\n" in code
    assert "var c: vec4<bool> = vec4<bool>(_Params.f);" in code
    assert "c.y = !(vec2<bool>(_Params.g).x);" in code
    assert "(_Params.h != 0u)" in code
    validate_with_naga(tmp_path, code)
