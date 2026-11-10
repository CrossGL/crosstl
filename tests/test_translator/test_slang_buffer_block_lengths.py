"""``length()`` of a GLSL buffer block's array member in Slang output.

Slang output lowers std140 and std430 buffer blocks to byte-address buffers but printed ``block.member.length()``
unchanged, which slangc refused. A runtime array's length is now the number of whole elements that the bound buffer
holds after its offset (GLSL 4.60 section 4.1.9), computed from ``GetDimensions`` as in DirectX output, and a sized
array's length is its size.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl

GLSL_SOURCE = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Plain { float vals[]; };
layout(std430, binding = 1) buffer Data {
    float n;
    vec3 rt[];
} data;
layout(std430, binding = 2) readonly buffer Ro {
    uint k;
    uint rows[];
} ro;
void main() {
    vals[0] = float(vals.length());
    data.n = float(data.rt.length()) + float(ro.rows.length());
}
"""


@pytest.fixture(scope="module")
def slang(tmp_path_factory):
    path = tmp_path_factory.mktemp("lengths") / "lengths.comp"
    path.write_text(GLSL_SOURCE)
    return crosstl.translate(str(path), backend="slang")


def _compact(text):
    return "".join(text.split())


def test_runtime_array_lengths_count_the_bound_bytes(slang):
    code = _compact(slang)

    assert ".length()" not in code
    # vec3 rt[] starts at 16 with a 16-byte stride; uint rows[] at 4 with a 4-byte stride.
    assert "int((cgl_buffer_bytes(plain)-0u)/4u)" in code
    assert "int((cgl_buffer_bytes(data)-16u)/16u)" in code
    assert "int((cgl_buffer_bytes(ro)-4u)/4u)" in code


def test_the_helpers_query_both_buffer_types(slang):
    code = _compact(slang)

    for buffer_type in ("ByteAddressBuffer", "RWByteAddressBuffer"):
        assert (
            f"uintcgl_buffer_bytes({buffer_type}b){{uintbytes;b.GetDimensions(bytes);returnbytes;}}"
            in code
        )


def test_slang_matches_the_directx_lengths(slang, tmp_path):
    path = tmp_path / "lengths.comp"
    path.write_text(GLSL_SOURCE)
    hlsl = _compact(crosstl.translate(str(path), backend="directx"))

    for length in (
        "int((cgl_buffer_bytes(data)-16u)/16u)",
        "int((cgl_buffer_bytes(ro)-4u)/4u)",
    ):
        assert length in hlsl and length in _compact(slang)


def test_slangc_compiles_the_output(slang, tmp_path):
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    shader = tmp_path / "lengths.slang"
    shader.write_text(slang)

    result = subprocess.run(
        [
            slangc,
            str(shader),
            "-target",
            "spirv",
            "-entry",
            "main",
            "-stage",
            "compute",
            "-o",
            str(tmp_path / "lengths.spv"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
