"""Array members of a GLSL buffer block held as ``device uchar*`` pass to by-value array parameters as copies.

The Metal emitter lowers a GLSL ``buffer`` block whose layout MSL cannot reproduce to a ``device uchar*`` and reads
each member through byte offsets. An array member passed to a function was printed as ``outp.v``, a member access
on a pointer that has no members, so the Metal compiler rejected the program. It is now built as a thread copy from
its elements, ``crosstl_array_value<float3, 2>{{...}}.v``, as other arrays outside thread storage are.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

SOURCE = """#version 450
layout(local_size_x = 1) in;
struct P { vec3 a[2]; float w; };
layout(std430, binding = 0) buffer Outp { float pad; vec3 v[2]; P p[2]; float tail[3]; } outp;
float sumv(vec3 a[2]) { a[0].x += 1.0; return a[0].x + a[1].y; }
float sumt(float a[3]) { a[1] = 0.0; return a[0] * 2.0 + a[1] + a[2]; }
void main() {
    outp.pad = sumv(outp.v) + sumv(outp.p[1].a) + sumt(outp.tail) + outp.v[1].y;
}
"""


@pytest.fixture(scope="module")
def metal(tmp_path_factory):
    path = tmp_path_factory.mktemp("bytes") / "block.comp"
    path.write_text(SOURCE, encoding="utf-8")
    return crosstl.translate(str(path), backend="metal")


def _compact(text):
    return re.sub(r"\s+", "", text)


def test_the_block_is_byte_addressed(metal):
    # Compared without whitespace: the pointer spacing depends on whether clang-format is installed.
    assert "deviceuchar*outp[[buffer(0)]]" in _compact(metal)


def test_array_members_are_copied_element_by_element(metal):
    code = _compact(metal)
    assert "outp.v" not in code and "outp.tail" not in code and ".a)" not in code
    # vec3 v[2] lies at offset 16 with a 16-byte stride (std430), its second element at 32.
    assert (
        "sumv(crosstl_array_value<float3,2>{{float3((*reinterpret_cast<constdevicefloat*>(outp+16)),"
        in code
    )
    assert "float3((*reinterpret_cast<constdevicefloat*>(outp+32))," in code
    # P p[2] starts at 48 with a 48-byte stride, so p[1].a is at 96 and 112.
    assert "float3((*reinterpret_cast<constdevicefloat*>(outp+96))," in code
    assert "float3((*reinterpret_cast<constdevicefloat*>(outp+112))," in code
    # float tail[3] follows at 144.
    assert (
        "sumt(crosstl_array_value<float,3>{{(*reinterpret_cast<constdevicefloat*>(outp+144)),"
        "(*reinterpret_cast<constdevicefloat*>(outp+148)),"
        "(*reinterpret_cast<constdevicefloat*>(outp+152))}}.v)"
    ) in code


def test_metal_compiles_the_copies(metal, tmp_path):
    if shutil.which("xcrun") is None:
        pytest.skip("xcrun is not installed")
    (tmp_path / "block.metal").write_text(metal, encoding="utf-8")
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", "block.metal", "-o", "block.air"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if "unable to find utility" in result.stderr:
        pytest.skip("the Metal compiler is not installed")
    assert result.returncode == 0, result.stderr
