"""Constant-buffer members keep the offsets and matrix orientation they declare on every target.

HLSL ``packoffset(cN.c)`` puts a member at byte ``16 * N + 4 * c``, and ``row_major`` (or ``#pragma
pack_matrix(row_major)``) stores a matrix as its rows, one register each. The host writes those bytes, so a target
that reproduces the HLSL layout must read every member from them, and a ``row_major`` matrix must be read as rows,
not as columns.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.buffer_member_layout import (
    packoffset_bytes,
    packoffset_text,
)

ROW_MAJOR = """
struct S { row_major float2x3 q; float k; };
cbuffer C : register(b0) {
    row_major float4x4 r;
    float4x4 c;
    row_major float2x3 q;
    float tail;
    S s;
    row_major float3x4 arr[2];
};
float4 main(float4 p : POSITION) : SV_Position {
    return mul(r, p) + mul(c, p) + float4(mul(q, p.xyz), tail) + float4(mul(s.q, p.xyz), s.k, 0)
         + float4(mul(arr[1], p), 1);
}
"""

# HLSL offsets: r 0, c 64, q rows at 128/144 (28 bytes), tail packs into q's last register at 156,
# s at 160 (s.q 28 bytes, s.k at 188), arr at 192 with 48-byte elements.
ROW_MAJOR_OFFSETS = {"r": 0, "c": 64, "q": 128, "tail": 156, "s": 160, "arr": 192}

PACKOFFSET = """
struct T { float4 v; float w; };
cbuffer P : register(b1) {
    float a : packoffset(c0.y);
    float4 b : packoffset(c2);
    float2 d : packoffset(c1.z);
    T t : packoffset(c4);
};
float4 main(float4 p : POSITION) : SV_Position { return b * a + float4(d, t.w, 0) + t.v; }
"""

PACKOFFSET_OFFSETS = {"a": 4, "d": 24, "b": 32, "t": 64}

PRAGMA = """
#pragma pack_matrix(row_major)
cbuffer R : register(b0) { float4x4 m; float2x3 n; };
#pragma pack_matrix(column_major)
cbuffer K : register(b1) { float4x4 k; };
float4 main(float4 p : POSITION) : SV_Position { return mul(m, p) + mul(k, p) + float4(mul(n, p.xyz), 0, 0); }
"""


def _translate(tmp_path, source, backend, name="shader.hlsl"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, source_backend="directx")


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]


def _metal_offsets(tmp_path, code, struct, offsets):
    """Compile ``code`` with static assertions of the byte offsets of ``struct``'s members."""
    if not shutil.which("xcrun"):
        pytest.skip("the Metal toolchain is not available")
    checks = "".join(
        f'static_assert(__builtin_offsetof({struct}, {member}) == {offset}, "{member}");\n'
        for member, offset in offsets.items()
    )
    path = tmp_path / "offsets.metal"
    path.write_text(code + "\n" + checks, encoding="utf-8")
    probe = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True, text=True
    )
    if probe.returncode != 0:
        pytest.skip("the Metal compiler is not available")
    _run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(path),
            "-o",
            str(tmp_path / "offsets.air"),
        ]
    )


@pytest.mark.parametrize(
    ("text", "offset"), (("c0", 0), ("c0.y", 4), ("c2.w", 44), ("c10.z", 168))
)
def test_packoffset_names_bytes(text, offset):
    assert packoffset_bytes(text) == offset
    assert packoffset_bytes(packoffset_text(offset)) == offset


def test_metal_reads_row_major_matrices_as_rows(tmp_path):
    code = _translate(tmp_path, ROW_MAJOR, "metal")
    flat = _flat(code)
    # A row-major float4x4 holds four rows of four floats: Metal's float4x4 over those bytes is the transpose.
    assert "transpose(c.r) * p" in flat
    assert "c.c * p" in flat
    # float2x3 rows have three floats; the register words are read back as rows.
    assert re.search(r"result\[0\] = float2\(m\.v\[0\], m\.v\[4\]\);", flat)
    assert re.search(r"result\[2\] = float2\(m\.v\[2\], m\.v\[6\]\);", flat)
    _metal_offsets(tmp_path, code, "C", ROW_MAJOR_OFFSETS)
    _metal_offsets(tmp_path, code, "S", {"q": 0, "k": 28})


def test_metal_places_members_at_their_packoffsets(tmp_path):
    code = _translate(tmp_path, PACKOFFSET, "metal")
    _metal_offsets(tmp_path, code, "P", PACKOFFSET_OFFSETS)


def test_pack_matrix_pragma_orients_the_matrices_after_it(tmp_path):
    crossgl = _flat(_translate(tmp_path, PRAGMA, "cgl"))
    assert "pack_matrix" not in crossgl
    assert re.search(
        r"cbuffer R \{ @ row_major mat4 m; @ row_major mat3x2 n; \}", crossgl
    )
    assert re.search(r"cbuffer K \{ @ column_major mat4 k; \}", crossgl)
    metal = _flat(_translate(tmp_path, PRAGMA, "metal"))
    assert "transpose(r.m) * p" in metal
    assert "k.k * p" in metal


def test_opengl_loads_row_major_and_packoffset_members_from_their_bytes(tmp_path):
    code = _translate(tmp_path, ROW_MAJOR, "opengl")
    flat = _flat(code)
    # q's rows sit in registers 8 and 9, so column 2 takes the third component of each.
    assert re.search(r"q\[2\]\[0\] = uintBitsToFloat\(\w+\[8\]\.z\);", flat)
    assert re.search(r"q\[2\]\[1\] = uintBitsToFloat\(\w+\[9\]\.z\);", flat)
    assert re.search(r"tail = uintBitsToFloat\(\w+\[9\]\.w\);", flat)
    packed = _flat(_translate(tmp_path, PACKOFFSET, "opengl"))
    assert re.search(r"a = uintBitsToFloat\(\w+\[0\]\.y\);", packed)
    assert re.search(r"d = uintBitsToFloat\(\w+\[1\]\.zw\);", packed)
    assert re.search(r"b = uintBitsToFloat\(\w+\[2\]\.xyzw\);", packed)
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not on PATH")
    for name, text in (
        ("row.vert", code),
        ("pack.vert", _translate(tmp_path, PACKOFFSET, "opengl")),
    ):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        _run(["glslangValidator", str(path)])


def test_hlsl_keeps_struct_member_orientation_as_a_qualifier(tmp_path):
    code = _flat(_translate(tmp_path, ROW_MAJOR, "directx"))
    assert "row_major float2x3 q;" in code
    assert ": row_major" not in code


@pytest.mark.parametrize(
    ("body", "reason"),
    (
        ("float a : packoffset(c0.y); float b;", "mixes packoffset"),
        ("float3 a : packoffset(c0.z);", "packoffset of 'a'"),
        ("float4x4 m : packoffset(c0.y);", "packoffset of 'm'"),
        ("float a : packoffset(c0); float b : packoffset(c0);", "overlaps"),
    ),
)
def test_metal_refuses_packoffsets_hlsl_rejects(tmp_path, body, reason):
    source = f"cbuffer B : register(b0) {{ {body} }};\nfloat4 main() : SV_Target {{ return 0; }}\n"
    with pytest.raises(ValueError, match=re.escape(reason)):
        _translate(tmp_path, source, "metal")
