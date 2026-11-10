"""Three-component byte-address loads and stores in Slang output.

slangc 2026.18.3 lowers ``Load3``/``Store3`` (and ``Load<T3>``/``Store<T3>``) at a constant offset that is a nonzero
multiple of the vector's size to the wrong bytes in SPIR-V: ``Store3(48, v)`` writes bytes 64..75. A std430 ``vec3``
at byte 48 or a ``mat3`` column at byte 96 hits this, so Slang output now moves three components through helpers that
move two and then one (``slang_byte_address_words``). CrossGL ``buffer_load3``/``buffer_store3``, which the HLSL
importer makes of ``Load3``/``Store3`` calls, also reached Slang output unchanged; they are now lowered as well.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.slang_byte_address_words import (
    SlangThreeComponentAccess,
)

GLSL_BLOCK = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 1) buffer Data {
    vec3 a;
    vec3 b[3];
    mat3 m;
    ivec3 k;
    float n;
} data;
void main() {
    data.b[2] += vec3(1.0);
    data.m = data.m * 2.0;
    data.k = ivec3(data.m[2] + data.b[2]);
    data.a = data.b[2] + data.m[1];
    data.n = data.m[2].y;
}
"""

GLSL_WIDE_BLOCK = """#version 450
#extension GL_EXT_shader_explicit_arithmetic_types : require
layout(local_size_x = 1) in;
layout(std430, binding = 1) buffer Data {
    u16vec3 h0;
    u16vec3 h1;
    u16vec3 h2;
    u16vec3 h3;
    u64vec3 w0;
    u64vec3 w1;
    u64vec3 w2;
} data;
void main() {
    data.h3 = data.h1 + data.h2 + u16vec3(1);
    data.w2 = data.w1 + data.w0 + u64vec3(5);
}
"""

HLSL_SOURCE = """RWByteAddressBuffer buf : register(u0);
ByteAddressBuffer src : register(t1);
[numthreads(1, 1, 1)] void CSMain(uint3 id : SV_DispatchThreadID) {
    uint3 v = buf.Load3(48) + src.Load3(12);
    buf.Store3(96, v + uint3(1, 2, 3));
    buf.Store2(0, src.Load2(8));
    buf.Store4(16, src.Load4(32));
}
"""


def _translate(tmp_path, source, suffix, backend="slang"):
    path = tmp_path / f"words{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _compact(text):
    return "".join(text.split())


def test_helpers_move_two_components_then_one():
    access = SlangThreeComponentAccess()

    assert access.load("data", "48") == "__crossgl_load3_uint(data, 48)"
    assert access.store("data", "48", "v") == "__crossgl_store3_uint(data, 48, v)"
    assert access.load("data", "24", storage="uint16_t") == (
        "__crossgl_load3_uint16_t(data, 24)"
    )
    code = _compact(access.declarations())

    for buffer_type in ("ByteAddressBuffer", "RWByteAddressBuffer"):
        assert (
            f"uint3__crossgl_load3_uint({buffer_type}buffer,uintoffset)"
            "{returnuint3(buffer.Load2(offset),buffer.Load(offset+8u));}"
        ) in code
    assert (
        "void__crossgl_store3_uint(RWByteAddressBufferbuffer,uintoffset,uint3value)"
        "{buffer.Store2(offset,value.xy);buffer.Store(offset+8u,value.z);}"
    ) in code
    assert (
        "uint16_t3__crossgl_load3_uint16_t(RWByteAddressBufferbuffer,uintoffset)"
        "{returnuint16_t3(buffer.Load<uint16_t2>(offset),buffer.Load<uint16_t>(offset+4u));}"
    ) in code


def test_unknown_storage_is_refused():
    with pytest.raises(ValueError, match="no three-component"):
        SlangThreeComponentAccess().load("data", "0", storage="half")


def test_glsl_block_vectors_and_matrix_columns_use_the_helpers(tmp_path):
    code = _compact(_translate(tmp_path, GLSL_BLOCK, ".comp"))

    assert "Load3(" not in code and "Store3(" not in code
    # vec3 b[2] at 48 and the mat3 columns at 64, 80 and 96 (std430).
    assert (
        "__crossgl_store3_uint(data,48,asuint((asfloat(__crossgl_load3_uint(data,48))"
        in code
    )
    assert "asfloat(__crossgl_load3_uint(data,96))" in code
    assert "__crossgl_store3_uint(data,112,asuint(int3(" in code


def test_hlsl_output_keeps_the_methods(tmp_path):
    code = _compact(_translate(tmp_path, GLSL_BLOCK, ".comp", backend="directx"))

    assert "data.Store3(48," in code
    assert "__crossgl_load3" not in code


def test_wide_components_use_typed_helpers(tmp_path):
    code = _compact(_translate(tmp_path, GLSL_WIDE_BLOCK, ".comp"))

    assert "Load<uint16_t3>" not in code and "Store<uint16_t3>" not in code
    assert "Load<uint64_t3>" not in code and "Store<uint64_t3>" not in code
    assert (
        "__crossgl_store3_uint16_t(data,24,uint16_t3(__crossgl_load3_uint16_t(data,8)"
        in code
    )
    assert "__crossgl_store3_uint64_t(data,96," in code
    assert (
        "returnuint64_t3(buffer.Load<uint64_t2>(offset),buffer.Load<uint64_t>(offset+16u));"
        in code
    )


def test_hlsl_vector_byte_address_calls_reach_slang(tmp_path):
    code = _compact(_translate(tmp_path, HLSL_SOURCE, ".hlsl"))

    assert "buffer_load" not in code and "buffer_store" not in code
    assert "uint3v=__crossgl_load3_uint(buf,48)+__crossgl_load3_uint(src,12);" in code
    assert "__crossgl_store3_uint(buf,96,v+uint3(1,2,3));" in code
    assert "buf.Store2(0,src.Load2(8));" in code
    assert "buf.Store4(16,src.Load4(32));" in code


def _slangc(tmp_path, code, entry):
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    shader = tmp_path / "words.slang"
    shader.write_text(code)
    result = subprocess.run(
        [
            slangc,
            str(shader),
            "-target",
            "spirv",
            "-entry",
            entry,
            "-stage",
            "compute",
            "-o",
            str(tmp_path / "words.spv"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "source, suffix, entry",
    [
        (GLSL_BLOCK, ".comp", "main"),
        (GLSL_WIDE_BLOCK, ".comp", "main"),
        (HLSL_SOURCE, ".hlsl", "CSMain"),
    ],
)
def test_slangc_compiles_the_output(tmp_path, source, suffix, entry):
    _slangc(tmp_path, _translate(tmp_path, source, suffix), entry)
