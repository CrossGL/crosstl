"""Writes to a column of a matrix in a GLSL buffer block lowered to byte offsets (HLSL, Slang and Metal output).

A subscript of a matrix selects a column (GLSL 4.60 section 5.6), and std140 and std430 store each column as one
vector every ``column_stride`` bytes. HLSL, Slang and Metal output load such a matrix by value, so a write to
``block.m[c]``, to one of its components (``block.m[c].y``) or to one of its elements (``block.m[c][r]``) used to
assign to that loaded copy: DXC and slangc refused the assignment to an rvalue, and the Metal compiler accepted it
and dropped the write. The write now stores the column's own bytes, for a constant or dynamic column, for matrices
that are members, struct fields or array elements, and for compound assignments.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl

GLSL_SOURCE = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 1) buffer Data {
    mat3x2 mats[2];
    mat3x2 one;
    mat3 m3;
    mat2 mgrid[2][2];
    float n;
} data;
void main() {
    int c = int(gl_GlobalInvocationID.x) + 1;
    data.mats[1][2] = vec2(1.0, 4.0);
    data.one[c] = vec2(2.0);
    data.mats[0][1].y = 6.0;
    data.m3[c].zx = vec2(7.0, 8.0);
    data.m3[2][1] += 0.5;
    data.one[0] *= 3.0;
    data.mgrid[1][0][1] = vec2(9.0, 10.0);
    data.mgrid[0][1][0][1] = 11.0;
    data.n = data.m3[1][2] + data.mats[1][2].y + data.mgrid[1][0][1].x;
}
"""

# std430 offsets: mats at 0 (24-byte mat3x2, 8-byte columns), one at 48, m3 at 80 (16-byte columns), mgrid at 128
# (16-byte mat2, rows of two), n at 192.
BYTE_ADDRESS_STORES = [
    "data.Store2(40,asuint(float2(1.0,4.0)));",
    "data.Store2((48+c*8),asuint(float2(2.0",
    "data.Store(12,asuint((6.0)));",
    "data.Store((80+c*16+8),asuint(__crossgl_component_store_0.x));",
    "data.Store((80+c*16),asuint(__crossgl_component_store_0.y));",
    "data.Store(116,asuint((asfloat(data.Load(116))+(0.5))));",
    "data.Store2(48,asuint((asfloat(data.Load2(48))*3.0)));",
    "data.Store2(168,asuint(float2(9.0,10.0)));",
    "data.Store(148,asuint((11.0)));",
]

METAL_STORES = [
    "(*reinterpret_cast<devicefloat*>(data+40))=__crossgl_buffer_store_0.x;",
    "(*reinterpret_cast<devicefloat*>(data+44))=__crossgl_buffer_store_0.y;",
    "(*reinterpret_cast<devicefloat*>(data+(48+c*8+4)))=__crossgl_buffer_store_1.y;",
    "(*reinterpret_cast<devicefloat*>(data+12))=(6.0);",
    "(*reinterpret_cast<devicefloat*>(data+(80+c*16+8)))=__crossgl_component_store_2.x;",
    "(*reinterpret_cast<devicefloat*>(data+116))=((*reinterpret_cast<constdevicefloat*>(data+116))+(0.5));",
    "(*reinterpret_cast<devicefloat*>(data+52))=__crossgl_buffer_store_3.y;",
    "(*reinterpret_cast<devicefloat*>(data+172))=__crossgl_buffer_store_4.y;",
    "(*reinterpret_cast<devicefloat*>(data+148))=(11.0);",
]


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    path = tmp_path_factory.mktemp("columns") / "columns.comp"
    path.write_text(GLSL_SOURCE)
    return {
        backend: crosstl.translate(str(path), backend=backend)
        for backend in ("directx", "slang", "metal")
    }


def _compact(text):
    return "".join(text.split())


@pytest.mark.parametrize("backend", ["directx", "slang"])
def test_byte_address_output_stores_the_column_bytes(outputs, backend):
    code = _compact(outputs[backend])

    for store in BYTE_ADDRESS_STORES:
        assert store in code
    assert "__crossgl_set_column" not in code
    assert ")))=" not in code and "._m01_m11.y=" not in code


def test_metal_output_stores_the_column_bytes(outputs):
    code = _compact(outputs["metal"])

    for store in METAL_STORES:
        assert store in code
    assert ")[2]=" not in code and ")[1].y=" not in code


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("dxc") is None, reason="dxc is not available")
def test_dxc_compiles_the_hlsl_output(outputs, tmp_path):
    shader = tmp_path / "columns.hlsl"
    shader.write_text(outputs["directx"])

    code, log = _run([shutil.which("dxc"), "-T", "cs_6_0", "-E", "CSMain", str(shader)])

    assert code == 0, log


@pytest.mark.skipif(shutil.which("slangc") is None, reason="slangc is not available")
def test_slangc_compiles_the_slang_output(outputs, tmp_path):
    shader = tmp_path / "columns.slang"
    shader.write_text(outputs["slang"])

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
            str(tmp_path / "columns.spv"),
        ]
    )

    assert code == 0, log


@pytest.mark.skipif(shutil.which("xcrun") is None, reason="requires the Metal compiler")
def test_metal_output_compiles(outputs, tmp_path):
    shader = tmp_path / "columns.metal"
    shader.write_text(outputs["metal"])

    code, log = _run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader),
            "-o",
            str(tmp_path / "columns.air"),
        ]
    )
    if "unable to find utility" in log:
        pytest.skip("the Metal toolchain is not installed")

    assert code == 0, log
