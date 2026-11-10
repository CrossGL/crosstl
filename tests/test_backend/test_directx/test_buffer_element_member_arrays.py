"""Array members of structured-buffer elements in HLSL input.

``e[1].v[i]`` with ``RWStructuredBuffer<E> e`` and ``float3 v[2]`` in ``E`` subscripts the array member ``v``. The
importer looked the member's array dimensions up under the buffer's declared type, ``RWStructuredBuffer<E>``, found
none, and took the subscript for a component selection of ``float3``. An assignment of a whole ``float3`` to the
element was then truncated to its first component (``e[1].v[i] = (vec3(4.0, 5.0, 6.0)).x``), and an array of
matrices was written and read as rows of one matrix. The struct is the element type of the indexed buffer, also
when the array is a member of a member, ``e[1].n.c[i]``.
"""

import shutil
import subprocess

import pytest

import crosstl

SOURCE = """struct E {
    float3 v[2];
    uint u[3];
    float2x2 m[2];
    float w;
};
RWStructuredBuffer<E> e : register(u0);
StructuredBuffer<E> r : register(t1);
RWStructuredBuffer<float> o : register(u2);
[numthreads(1, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint i = id.x % 2u;
    e[1].v[i] = float3(4.0, 5.0, 6.0);
    e[1].v[0] += r[0].v[1];
    float3 x = r[1].v[i];
    e[0].u[2] = r[0].u[i] + 1u;
    e[0].m[1] = r[1].m[0];
    o[0] = x.y + e[0].v[1].z + float(r[1].u[2] >> 1) + r[0].m[1][1].x;
    o[1] = dot(r[0].v[0], r[1].v[1]);
}
"""

NESTED_SOURCE = """struct Inner {
    float a;
    float2 c[2];
};
struct Item {
    float s;
    Inner n;
};
RWStructuredBuffer<Item> e : register(u0);
StructuredBuffer<Item> r : register(t1);
[numthreads(1, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint i = id.x % 2u;
    e[1].n.c[0] = float2(3.0, 4.0);
    e[0].n.c[i] += r[1].n.c[1];
    float2 y = r[0].n.c[i];
    e[1].s = y.y + r[1].n.c[0].x;
}
"""


def _translate(tmp_path, backend, source=SOURCE):
    path = tmp_path / "elements.hlsl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_a_vector_element_of_a_buffer_struct_array_keeps_its_width(tmp_path):
    crossgl = _translate(tmp_path, "cgl")
    assert "e[1].v[i] = vec3(4.0, 5.0, 6.0);" in crossgl, crossgl
    assert "e[1].v[0] += r[0].v[1];" in crossgl, crossgl
    assert "vec3 x = r[1].v[i];" in crossgl, crossgl
    assert ".x;" not in crossgl, crossgl


def test_an_array_member_of_a_member_keeps_its_width(tmp_path):
    crossgl = _translate(tmp_path, "cgl", NESTED_SOURCE)
    assert "e[1].n.c[0] = vec2(3.0, 4.0);" in crossgl, crossgl
    assert "e[0].n.c[i] += r[1].n.c[1];" in crossgl, crossgl
    assert "vec2 y = r[0].n.c[i];" in crossgl, crossgl
    assert "e[1].s = y.y + r[1].n.c[0].x;" in crossgl, crossgl


def test_a_matrix_element_of_a_buffer_struct_array_is_one_matrix(tmp_path):
    crossgl = _translate(tmp_path, "cgl")
    assert "e[0].m[1] = r[1].m[0];" in crossgl, crossgl
    assert "cgl_hlsl_set_row" not in crossgl, crossgl
    # HLSL m[1][1].x is row 1, column 0 of matrix 1; CrossGL subscripts the column first.
    assert "r[0].m[1][0][1]" in crossgl, crossgl


def test_glslang_compiles_the_glsl_output(tmp_path):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not available")
    shader = tmp_path / "elements.comp"
    shader.write_text(_translate(tmp_path, "opengl", NESTED_SOURCE), encoding="utf-8")
    result = subprocess.run(
        [glslang, "-V", str(shader), "-o", str(tmp_path / "elements.spv")],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
