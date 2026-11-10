"""Arrays of arrays in GLSL buffer blocks lowered to byte offsets (HLSL, Slang and Metal output).

The byte-offset lowering of std140 and std430 buffer blocks refused every array of arrays, so a block member such as
``float grid[2][3]`` stopped the whole block. Each array dimension's stride is now the next dimension's extent times
its stride (an array of arrays is an array of its inner array type), and ``grid[i][j]`` lies ``i * 12 + j * 4``
bytes past ``grid`` in std430.

An expression that still denotes an array, such as ``block.row`` or ``block.grid[1]``, used to be lowered as if it
were its first element (HLSL and Slang) or printed unchanged (Metal); test_glsl_buffer_array_values.py covers
reading and writing such arrays whole. A component write to a vector element of a block array,
``block.vecs[1].y = s``, also used to assign to a load in HLSL and Slang; it now stores the one component.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.glsl_buffer_block_arrays import (
    glsl_buffer_array_elements,
)
from crosstl.translator.codegen.glsl_buffer_layout import glsl_buffer_array_dimensions

GLSL_SOURCE = """#version 450
layout(local_size_x = 1) in;
struct Cell {
    vec2 uv;
    float w[2][3];
};
layout(std140, binding = 0) readonly buffer Params {
    float scale[2][3];
    vec3 dirs[2][2];
    mat2 rot[2][2];
} params;
layout(std430, binding = 1) buffer Data {
    float grid[2][3];
    vec3 vgrid[3][2];
    Cell cells[2][2];
    ivec2 tail[][2];
} data;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    float acc = 0.0;
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 3; ++b) {
            acc += params.scale[a][b] * float(b + 1);
            data.grid[a][b] = acc;
        }
    }
    vec3 d = params.dirs[1][i] + params.dirs[0][1];
    data.vgrid[2][1] = d;
    data.vgrid[1][0].y = acc;
    data.vgrid[0][1] += vec3(1.0);
    vec2 r = params.rot[1][0] * vec2(1.0, 2.0) + params.rot[0][1][1];
    Cell c = data.cells[1][0];
    c.w[1][2] = r.x + c.w[0][1];
    data.cells[0][1] = c;
    data.cells[1][1].w[1][0] = r.y;
    data.tail[i + 1][1] = ivec2(data.tail[i][0].y, int(data.grid[1][2]));
}
"""

WHOLE_ARRAY_TEMPLATE = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 1) buffer Data {{
    float head;
    float row[4];
    float grid[2][3];
}} data;
void main() {{
    float copy[4];
    float part[3];
    {statement}
    data.head = copy[1] + part[2];
}}
"""

COMPONENT_STORE_SOURCE = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 1) buffer Data {
    vec3 vecs[3];
    vec4 vgrid[2][2];
} data;
void main() {
    data.vecs[1].y = 2.0;
    data.vecs[2].xz = vec2(3.0);
    data.vgrid[1][0].w = 5.0;
    data.vgrid[0][1][2] = 7.0;
}
"""


def _translate(tmp_path, source, backend, name="blocks"):
    path = tmp_path / f"{name}.comp"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _compact(text):
    return "".join(text.split())


def _method_call(match):
    return f"{match.group(2)}.{match.group(1).capitalize()}3("


def test_inner_dimensions_take_the_next_extent_times_its_stride():
    member = {"size": 12, "align": 16, "array_stride": 16, "inner_array_sizes": (2, 3)}

    std430 = glsl_buffer_array_dimensions(member, "std430", int)
    std140 = glsl_buffer_array_dimensions(
        {"size": 4, "align": 4, "inner_array_sizes": (3,)}, "std140", int
    )

    assert std430 == (96, ((2, 48), (3, 16)))
    assert std140 == (48, ((3, 16),))
    assert glsl_buffer_array_dimensions(member, "std430", lambda size: None) is None


def test_array_elements_are_listed_in_memory_order():
    member = {"array_count": 2, "stride": 12, "inner_arrays": ((3, 4),)}

    elements = list(glsl_buffer_array_elements(member, 8))

    assert elements[0] == ("[0][0]", "8")
    assert elements[2] == ("[0][2]", "16")
    assert elements[3] == ("[1][0]", "20")
    assert elements[-1] == ("[1][2]", "28")
    symbolic = dict(glsl_buffer_array_elements(member, "o"))
    assert symbolic["[0][0]"] == "o"
    assert symbolic["[1][2]"] == "(o + 20)"


@pytest.mark.parametrize("backend", ["directx", "slang"])
def test_byte_address_output_indexes_every_dimension(tmp_path, backend):
    output = _compact(_translate(tmp_path, GLSL_SOURCE, backend))
    if backend == "slang":
        # Slang moves three words through helpers (slang_byte_address_words); compare in the method form.
        output = re.sub(r"__crossgl_(load|store)3_uint\((\w+),", _method_call, output)

    # std430 float[2][3]: 12-byte rows of 4-byte elements.
    assert "data.Store(((a*12)+b*4),asuint(acc));" in output
    # std140 float[2][3]: every element starts a 16-byte register.
    assert "asfloat(params.Load(((a*48)+b*16)))" in output
    # std140 vec3[2][2] at 96: rows of two 16-byte elements.
    assert "asfloat(params.Load3((128+i*16)))+asfloat(params.Load3(112))" in output
    assert "data.Store3(112,asuint(d));" in output
    assert "data.Store3(48,asuint((asfloat(data.Load3(48))+float3(1.0" in output
    assert "data.Store(68,asuint((acc)));" in output
    # std140 mat2[2][2] at 160: 32-byte matrices.
    assert "float2x2(asfloat(params.Load2(224)),asfloat(params.Load2(240)))" in output
    # Cell[2][2] at 128 with 32-byte cells, and Cell.w[2][3] at 8 in each cell.
    assert "result.w[1][2]=asfloat(buffer.Load((offset+8+20)));" in output
    assert "(data,192);" in output
    assert "data.Store(188,asuint(__crossgl_aggregate_store_0.w[1][2]));" in output
    assert "data.Store(244,asuint(r.y));" in output
    # The runtime ivec2[][2] tail at 256: 16-byte rows of 8-byte elements.
    assert "(256+i*16)" in output
    assert "*16+8)," in output
    assert "asfloat(data.Load(20))" in output


def test_metal_output_indexes_every_dimension(tmp_path):
    output = _compact(_translate(tmp_path, GLSL_SOURCE, "metal"))

    assert "(*reinterpret_cast<devicefloat*>(data+((a*12)+b*4)))=acc;" in output
    assert "params+((a*48)+b*16)" in output
    assert (
        "(*reinterpret_cast<devicefloat*>(data+112))=__crossgl_buffer_store_0.x;"
        in output
    )
    assert "(*reinterpret_cast<devicefloat*>(data+68))=(acc);" in output
    assert "(*reinterpret_cast<constdevicefloat*>(params+244))" in output
    assert "(*reinterpret_cast<devicefloat*>(data+244))=r.y;" in output
    assert "result.w[1][2]=" in output


def test_metal_copies_a_sub_array_write_element_by_element(tmp_path):
    # Metal writes every C-array assignment as a loop over its elements, and a sub-array of a block array now has
    # its array type, so each element is stored at its own offset: grid starts at 20, after head and row[4].
    source = WHOLE_ARRAY_TEMPLATE.format(statement="data.grid[0] = part;")

    output = _compact(_translate(tmp_path, source, "metal"))

    assert "__crossgl_array_index_0<3;" in output
    assert (
        "(*reinterpret_cast<devicefloat*>(data+(20+__crossgl_array_index_0*4)))="
        "part[__crossgl_array_index_0];"
    ) in output


@pytest.mark.parametrize("backend", ["directx", "slang"])
def test_component_writes_to_vector_elements_store_the_components(tmp_path, backend):
    output = _compact(_translate(tmp_path, COMPONENT_STORE_SOURCE, backend))

    assert "data.Store(20,asuint((2.0)));" in output
    assert "data.Store(32,asuint(__crossgl_component_store_0.x));" in output
    assert "data.Store(40,asuint(__crossgl_component_store_0.y));" in output
    # vec4[2][2] at 48: 32-byte rows of 16-byte elements.
    assert "data.Store(92,asuint((5.0)));" in output
    assert "data.Store(72,asuint((7.0)));" in output
    assert "Load3(16)).y=" not in output


def test_metal_component_writes_to_vector_elements(tmp_path):
    output = _compact(_translate(tmp_path, COMPONENT_STORE_SOURCE, "metal"))

    assert "(*reinterpret_cast<devicefloat*>(data+20))=(2.0);" in output
    assert "(*reinterpret_cast<devicefloat*>(data+92))=(5.0);" in output
    assert "(*reinterpret_cast<devicefloat*>(data+72))=(7.0);" in output


def _run_tool(command):
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("dxc") is None, reason="dxc is not available")
def test_dxc_compiles_the_hlsl_output(tmp_path):
    shader = tmp_path / "blocks.hlsl"
    shader.write_text(_translate(tmp_path, GLSL_SOURCE, "directx"))

    code, log = _run_tool(
        [shutil.which("dxc"), "-T", "cs_6_0", "-E", "CSMain", str(shader)]
    )

    assert code == 0, log


@pytest.mark.skipif(shutil.which("slangc") is None, reason="slangc is not available")
def test_slangc_compiles_the_slang_output(tmp_path):
    shader = tmp_path / "blocks.slang"
    shader.write_text(_translate(tmp_path, GLSL_SOURCE, "slang"))

    code, log = _run_tool(
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
            str(tmp_path / "blocks.spv"),
        ]
    )

    assert code == 0, log


@pytest.mark.skipif(shutil.which("xcrun") is None, reason="requires the Metal compiler")
def test_metal_output_compiles(tmp_path):
    shader = tmp_path / "blocks.metal"
    shader.write_text(_translate(tmp_path, GLSL_SOURCE, "metal"))

    code, log = _run_tool(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader),
            "-o",
            str(tmp_path / "blocks.air"),
        ]
    )
    if "unable to find utility" in log:
        pytest.skip("the Metal toolchain is not installed")

    assert code == 0, log
