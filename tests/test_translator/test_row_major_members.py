"""Row-major struct members are stored as their column-major transpose where structs cannot carry the layout."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import RowMajorMemberError

STRUCT = """shader P {
    struct S {
        mat3x2 a;
        @ row_major
        mat3x2 b;
        @ row_major
        mat2 c[2];
    }
    buffer S* sb;
    buffer float* outp;
    S make(mat3x2 m) {
        return S(m, m, {mat2(1.0), mat2(2.0)});
    }
    compute {
    @ numthreads(1, 1, 1)
    void main(uvec3 id @ gl_GlobalInvocationID) {
        S s = sb[0];
        uint i = id.x;
        mat3x2 whole = s.b;
        vec2 column = s.b[2];
        float element = s.b[1][0];
        float picked = s.b[i][1];
        float listed = s.c[1][0][1];
        vec2 product = s.b * vec3(1.0, 2.0, 3.0);
        sb[0].b = whole;
        sb[0].b[2][1] = element;
        sb[0].b += whole;
        sb[0].c[0][1][0] = listed;
        S t = {whole, whole, {mat2(1.0), mat2(3.0)}};
        S u = make(whole);
        outp[i] = column.x + picked + product.y + t.b[0][1] + u.c[1][0][0];
    }
    }
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return " ".join(code.split())


def test_glsl_declares_the_transpose_and_rewrites_every_use(tmp_path):
    code = _flat(_translate(tmp_path, STRUCT, "opengl"))
    assert re.search(r"struct S \{ mat3x2 a; mat2x3 b; mat2 c\[2\]; \};", code)
    assert "row_major" not in code
    assert "mat3x2 whole = transpose(s.b);" in code
    assert "vec2 column = transpose(s.b)[2];" in code
    assert "float element = s.b[0][1];" in code
    assert "float picked = s.b[1][i];" in code
    assert "float listed = s.c[1][1][0];" in code
    assert "transpose(s.b) * vec3(1.0, 2.0, 3.0)" in code
    assert "sb[0].b = transpose(whole);" in code
    assert "sb[0].b[1][2] = element;" in code
    assert "sb[0].b = transpose((transpose(sb[0].b) + whole));" in code
    assert "sb[0].c[0][0][1] = listed;" in code
    assert (
        "return S(m, transpose(m), mat2[2](transpose(mat2(1.0)), transpose(mat2(2.0))));"
        in code
    )
    assert (
        "S t = S(whole, transpose(whole), mat2[2](transpose(mat2(1.0)), transpose(mat2(3.0))));"
        in code
    )
    assert "t.b[1][0]" in code


def test_metal_declares_the_transpose(tmp_path):
    code = _flat(_translate(tmp_path, STRUCT, "metal"))
    assert "float2x3 b;" in code
    assert "row_major" not in code
    assert "float element = s.b[0][1];" in code
    assert "transpose(s.b)[2]" in code


@pytest.mark.parametrize("backend", ["vulkan", "wgsl"])
def test_spirv_and_wgsl_declare_the_transpose(tmp_path, backend):
    code = _translate(tmp_path, STRUCT, backend)
    assert "RowMajor" not in code
    assert "row_major" not in code
    assert "Transpose" in code if backend == "vulkan" else "transpose(" in code


def test_hlsl_keeps_row_major_struct_members(tmp_path):
    code = _flat(_translate(tmp_path, STRUCT, "directx"))
    assert "row_major float2x3 b;" in code
    assert "transpose(s.b)" not in code


BLOCK = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer B { layout(row_major) mat3x2 m; float o[]; } b;
void main() { b.o[0] = b.m[2][1] + b.m[0][1]; }
"""


@pytest.mark.parametrize("backend", ["opengl", "metal", "directx", "slang"])
def test_glsl_buffer_block_members_are_transposed_everywhere(tmp_path, backend):
    path = tmp_path / "block.comp"
    path.write_text(BLOCK)
    code = _flat(crosstl.translate(str(path), backend=backend, format_output=False))
    if backend == "opengl":
        assert "mat2x3 m;" in code
        assert "b.m[1][2] + b.m[1][0]" in code
    if backend == "metal":
        # The row-major mat3x2 takes two 16-byte rows, so `o` starts at byte 32.
        assert "(b + 32)" in code
    if backend in ("directx", "slang"):
        assert "Load3" in code and "(32)" not in code.replace(" ", "")
        assert "row_major" not in code


def test_untouched_without_row_major_members(tmp_path):
    source = STRUCT.replace("        @ row_major\n", "")
    code = _flat(_translate(tmp_path, source, "opengl"))
    assert "transpose(" not in code
    assert "mat3x2 b;" in code


@pytest.mark.parametrize(
    "statement, message",
    [
        ("sb[0].b[1] = vec2(1.0);", "an assignment of a column"),
        ("sb[0].b[1].y = 1.0;", "an assignment of components of a column"),
        ("sb[0].b[1]++;", "an increment or decrement of a column"),
        ("bump(sb[0].b);", "passing to an out or inout parameter of a whole matrix"),
        ("bumpc(sb[0].b[1]);", "passing to an out or inout parameter of a column"),
        ("sb[i].b[i++][0] = 1.0;", "through indices with side effects"),
        ("sb[i++].b += mat3x2(1.0);", "through an object with side effects"),
        ("sb[0].c = sb[1].c;", "without selecting one matrix"),
    ],
)
def test_refuses_uses_the_transposed_storage_cannot_keep(tmp_path, statement, message):
    source = STRUCT.replace(
        "    compute {\n",
        "    void bump(inout mat3x2 m) { m = m * 2.0; }\n"
        "    void bumpc(inout vec2 v) { v = v * 2.0; }\n    compute {\n",
    ).replace(
        "        S u = make(whole);\n",
        f"        S u = make(whole);\n        {statement}\n",
    )
    with pytest.raises(RowMajorMemberError, match=re.escape(message)):
        _translate(tmp_path, source, "opengl")


@pytest.mark.skipif(
    shutil.which("glslangValidator") is None, reason="glslangValidator unavailable"
)
def test_glsl_output_compiles(tmp_path):
    output = tmp_path / "out.comp"
    output.write_text(_translate(tmp_path, STRUCT, "opengl"))
    result = subprocess.run(
        ["glslangValidator", "-S", "comp", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
