"""Whole arrays read from and written to GLSL buffer blocks lowered to byte offsets.

A GLSL array is a value (GLSL 4.60 sections 5.8 and 6.1.1), so ``float c[4] = data.row;``, ``f(data.row)`` and
``data.grid[1] = r;`` move every element. The HLSL, Slang and Metal emitters reach a std140 or std430 block one
element at a time, and refused such whole-array uses.

HLSL and Slang output now read a whole array as ``crossgl_array_value_float_4(e0, e1, e2, e3).v``, a member of a
helper-constructed struct whose arguments are the element loads, and write one through a temporary array and one
store per element (``glsl_buffer_array_values``). Metal output initializes from a brace list of the element loads
and copies assignments element by element (``metal_array_copies``). Its array value is therefore no longer rendered
before the copy is chosen.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.buffer_block_diagnostics import (
    BufferBlockUnsupportedError,
)

SOURCE = """#version 450
layout(local_size_x = 1) in;
struct Cell { vec2 uv; float w; };
layout(std430, binding = 1) buffer Data {
    float head;
    float row[4];
    float grid[2][3];
    vec3 dirs[2];
    Cell cells[2];
    float out4[4];
    float outg[2][3];
    Cell outc[2];
} data;
float sum4(float a[4]) { return a[0] + a[1] + a[2] + a[3]; }
void main() {
    int i = int(gl_GlobalInvocationID.x);
    float copy[4] = data.row;
    float part[3];
    part = data.grid[i + 1];
    float g[2][3] = data.grid;
    vec3 d[2] = data.dirs;
    Cell cs[2] = data.cells;
    data.head = SUM + copy[1] + part[2] + g[0][1] + d[1].z + cs[1].w;
    data.out4 = copy;
    data.outg[1] = part;
    data.outg[0] = float[3](7.0, 8.0, 9.0);
    data.outc = cs;
    data.out4[0] += 1.0;
}
"""

# std430 offsets: head 0, row 4, grid 20 (12-byte rows), dirs 48 (16-byte stride), cells 80 (16-byte Cell),
# out4 112, outg 128, outc 152.
WITH_ARGUMENT = SOURCE.replace("SUM", "sum4(data.row)")
WITHOUT_ARGUMENT = SOURCE.replace("SUM", "copy[0]")

SIDE_EFFECT_SOURCE = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 1) buffer Data { float grid[2][3]; float n; } data;
void main() {
    int i = 0;
    float part[3] = data.grid[i++];
    data.n = part[0];
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "values.comp"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _compact(text):
    return "".join(text.split())


@pytest.mark.parametrize("backend", ["directx", "slang"])
def test_byte_address_output_reads_whole_arrays_as_values(tmp_path, backend):
    code = _compact(_translate(tmp_path, WITH_ARGUMENT, backend))

    assert (
        "floatcopy[4]=crossgl_array_value_float_4(asfloat(data.Load(4)),asfloat(data.Load(8)),"
        "asfloat(data.Load(12)),asfloat(data.Load(16))).v;"
    ) in code
    assert "part=crossgl_array_value_float_3(asfloat(data.Load((20+" in code
    assert "*12+8)))).v;" in code
    assert (
        "floatg[2][3]=crossgl_array_value_float_2_3(crossgl_array_value_float_3(asfloat(data.Load(20)),"
    ) in code
    assert (
        "Cellcs[2]=crossgl_array_value_Cell_2(__crossgl_load_rw_glsl_buffer_Cell_"
        in code
    )
    assert "sum4(crossgl_array_value_float_4(asfloat(data.Load(4))," in code
    assert "structCrossGLArrayValue_float_2_3{floatv[2][3];};" in code


@pytest.mark.parametrize("backend", ["directx", "slang"])
def test_byte_address_output_writes_whole_arrays_element_by_element(tmp_path, backend):
    code = _compact(_translate(tmp_path, WITH_ARGUMENT, backend))

    assert "float__crossgl_array_store_0[4]=copy;" in code
    assert "data.Store(124,asuint(__crossgl_array_store_0[3]));" in code
    assert "data.Store(148,asuint(__crossgl_array_store_1[2]));" in code
    literal = {
        "directx": (
            "float__crossgl_array_store_2[3]=crossgl_array_value_float_3(7.0,8.0,9.0).v;"
        ),
        "slang": "float__crossgl_array_store_2[3]={7.0,8.0,9.0};",
    }[backend]
    assert literal in code
    assert "data.Store(136,asuint(__crossgl_array_store_2[2]));" in code
    assert "Cell__crossgl_array_store_3[2]=cs;" in code
    assert "data.Store(176,asuint(__crossgl_array_store_3[1].w));" in code
    # An element write is still one store.
    assert "data.Store(112,asuint((asfloat(data.Load(112))+1.0)));" in code


def test_metal_output_initializes_and_copies_element_by_element(tmp_path):
    code = _compact(_translate(tmp_path, WITHOUT_ARGUMENT, "metal"))

    assert "floatcopy[4]={(*reinterpret_cast<constdevicefloat*>(data+4))," in code
    assert "floatg[2][3]={{(*reinterpret_cast<constdevicefloat*>(data+20))," in code
    assert "__crossgl_array_index_0<3;" in code
    assert (
        "part[__crossgl_array_index_0]=(*reinterpret_cast<constdevicefloat*>(" in code
    )
    assert "(*reinterpret_cast<devicefloat*>(data+(112+__crossgl_array_index_" in code


def test_hlsl_refuses_subscripts_with_side_effects(tmp_path):
    # Each element load repeats the byte offset, so 'i++' would step once per element.
    with pytest.raises(BufferBlockUnsupportedError, match="side effects"):
        _translate(tmp_path, SIDE_EFFECT_SOURCE, "directx")


@pytest.mark.parametrize("backend", ["slang", "metal"])
def test_other_backends_refuse_subscripts_with_side_effects(tmp_path, backend):
    with pytest.raises(Exception, match="effects"):
        _translate(tmp_path, SIDE_EFFECT_SOURCE, backend)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("dxc") is None, reason="dxc is not available")
def test_dxc_compiles_the_hlsl_output(tmp_path):
    shader = tmp_path / "values.hlsl"
    shader.write_text(_translate(tmp_path, WITH_ARGUMENT, "directx"))

    code, log = _run([shutil.which("dxc"), "-T", "cs_6_0", "-E", "CSMain", str(shader)])

    assert code == 0, log


@pytest.mark.skipif(shutil.which("slangc") is None, reason="slangc is not available")
def test_slangc_compiles_the_slang_output(tmp_path):
    shader = tmp_path / "values.slang"
    shader.write_text(_translate(tmp_path, WITH_ARGUMENT, "slang"))

    code, log = _run(
        [
            shutil.which("slangc"),
            str(shader),
            "-target",
            "spirv",
            "-entry",
            "main",
            "-stage",
            "compute",
            "-o",
            str(tmp_path / "values.spv"),
        ]
    )

    assert code == 0, log


@pytest.mark.skipif(shutil.which("xcrun") is None, reason="requires the Metal compiler")
def test_metal_output_compiles(tmp_path):
    shader = tmp_path / "values.metal"
    shader.write_text(_translate(tmp_path, WITHOUT_ARGUMENT, "metal"))

    code, log = _run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader),
            "-o",
            str(tmp_path / "values.air"),
        ]
    )
    if "unable to find utility" in log:
        pytest.skip("the Metal toolchain is not installed")

    assert code == 0, log


RETURN_SOURCE = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 1) buffer Data {
    float head;
    float row[4];
    float tail[4];
    int same;
} data;
float[4] getRow() { return data.row; }
void main() {
    float r[4] = getRow();
    data.head = r[2];
    data.same = (data.row == data.tail) ? 1 : 0;
}
"""


def test_metal_returns_a_whole_array_as_a_brace_list(tmp_path):
    code = _compact(_translate(tmp_path, RETURN_SOURCE, "metal"))

    assert (
        "array<float,4>getRow(deviceuchar*data){return{(*reinterpret_cast<constdevicefloat*>(data+4)),"
    ) in code
    assert "(*reinterpret_cast<constdevicefloat*>(data+16))};}" in code


@pytest.mark.parametrize("backend", ["directx", "slang"])
def test_byte_address_output_returns_and_compares_whole_arrays(tmp_path, backend):
    code = _compact(_translate(tmp_path, RETURN_SOURCE, backend))

    assert "crossgl_array_value_float_4(asfloat(data.Load(4))," in code
    assert "asfloat(data.Load(20))" in code and "asfloat(data.Load(32))" in code
