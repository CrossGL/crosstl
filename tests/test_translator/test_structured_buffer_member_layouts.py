"""HLSL structured buffers whose struct elements hold arrays or matrices, on OpenGL.

An HLSL structured buffer packs its elements tightly (D3D11.3 functional specification, section 5.3.7; DXC's
``-fvk-use-dx-layout``): ``struct { float s; float3 b[2]; uint k; }`` puts ``b`` at byte 4 with a 12-byte stride
and ``k`` at byte 28, where std430 puts ``b`` at byte 16 with a 16-byte stride (GLSL 4.60 section 7.6.2.2). The
OpenGL generator copies structs of scalars and vectors into packed storage itself; any other struct element was
declared unchanged under std430, so the shader read and wrote other bytes than the host's. Such elements are now
held as 32-bit words, as SPIR-V output holds them, and an element whose layout needs a matrix split is refused.

The expected values of the native check are written out by hand from the HLSL rules above, so they do not depend
on the planner under test.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import sys

import pytest

import crosstl
from crosstl.translator.semantics.structured_layouts import (
    StructuredLayoutUnsupportedError,
)

REQUIRE_ENV = "CROSSTL_REQUIRE_STRUCTURED_MEMBER_LAYOUT_EXECUTION"

HLSL = """
struct Inner { float a; float2 c[2]; };
struct Item { float s; float3 b[2]; uint k; Inner n; };
StructuredBuffer<Item> items : register(t0);
RWStructuredBuffer<Item> dst : register(u1);
RWStructuredBuffer<float> outv : register(u2);

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    Item it = items[1];
    outv[0] = it.b[1].y;
    outv[1] = items[0].b[0].z;
    outv[2] = float(it.k);
    outv[3] = items[1].n.c[1].x;
    outv[4] = it.n.a;
    outv[5] = it.s;
    outv[6] = items[0].n.c[0].y;
    dst[0] = items[1];
    dst[1].k = 5u;
    dst[1].s = items[0].b[1].x;
    dst[1].n.a = it.b[0].x + 0.5;
}
"""

# Word offsets of Item's leaves under HLSL structured-buffer rules: s, b[0].xyz, b[1].xyz, k, n.a, n.c[0].xy, n.c[1].xy.
ITEM_WORDS = 13
K_WORD = 7


def item_words(index):
    """The host's words for ``items[index]``: floats, with ``k`` an integer."""
    words = [
        struct.unpack("<I", struct.pack("<f", 1.0 + index * 20.0 + word * 0.5))[0]
        for word in range(ITEM_WORDS)
    ]
    words[K_WORD] = 11 * (index + 1)
    return words


def as_float(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]


def expected():
    first, second = item_words(0), item_words(1)
    outputs = [
        as_float(second[5]),  # it.b[1].y
        as_float(first[3]),  # items[0].b[0].z
        float(second[K_WORD]),  # it.k
        as_float(second[11]),  # items[1].n.c[1].x
        as_float(second[8]),  # it.n.a
        as_float(second[0]),  # it.s
        as_float(first[10]),  # items[0].n.c[0].y
    ]
    untouched = struct.unpack("<I", struct.pack("<f", -1.0))[0]
    written = list(second) + [untouched] * ITEM_WORDS
    written[ITEM_WORDS + K_WORD] = 5
    written[ITEM_WORDS + 0] = first[4]  # dst[1].s = items[0].b[1].x
    written[ITEM_WORDS + 8] = struct.unpack(
        "<I", struct.pack("<f", as_float(second[1]) + 0.5)
    )[0]
    return outputs, written


def translate(tmp_path, backend, source=HLSL):
    path = tmp_path / "shader.hlsl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def run(command):
    result = subprocess.run(
        [str(part) for part in command], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_opengl_holds_struct_elements_with_arrays_as_words(tmp_path):
    glsl = translate(tmp_path, "opengl")
    assert re.search(r"buffer itemsBuffer \{\s*uint items\[\];", glsl), glsl
    assert re.search(r"buffer dstBuffer \{\s*uint dst\[\];", glsl), glsl
    # b[1].y is word 5 and n.c[1].x word 11 of a 13-word element.
    assert "items[((uint(1) * 13u) + 11u)]" in glsl
    assert "dst[((uint(1) * 13u) + 7u)] = uint(5u);" in glsl
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")
    artifact = tmp_path / "shader.comp"
    artifact.write_text(glsl, encoding="utf-8")
    run([glslang, artifact])


def test_opengl_keeps_struct_elements_with_arrays_whose_layout_matches(tmp_path):
    source = """
    struct Item { float4 a; float4 b[2]; };
    RWStructuredBuffer<Item> items : register(u0);
    [numthreads(1, 1, 1)]
    void main() { items[1].b[0] = items[0].b[1]; }
    """
    glsl = translate(tmp_path, "opengl", source)
    assert re.search(r"buffer itemsBuffer \{\s*Item items\[\];", glsl), glsl


@pytest.mark.parametrize(
    "member",
    ["float3x3 m;", "float2x3 m;", "float3 v; float3x2 m;"],
)
@pytest.mark.parametrize("backend", ["opengl", "vulkan"])
def test_struct_elements_whose_matrices_change_layout_are_refused(
    tmp_path, member, backend
):
    source = f"""
    struct Item {{ float s; {member} }};
    StructuredBuffer<Item> items : register(t0);
    RWStructuredBuffer<float> outv : register(u0);
    [numthreads(1, 1, 1)]
    void main() {{ outv[0] = items[1].s; }}
    """
    with pytest.raises(StructuredLayoutUnsupportedError, match="matrix"):
        translate(tmp_path, backend, source)


def test_struct_elements_with_arrays_execute_on_opengl(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1" or sys.platform == "darwin":
        pytest.skip(f"set {REQUIRE_ENV}=1 on Linux to require the OpenGL dispatch")
    import moderngl

    glsl = translate(tmp_path, "opengl")
    bindings = {
        name: int(binding)
        for binding, name in re.findall(
            r"layout\(std430, binding = (\d+)\)[^{]*\{\s*\w+ (\w+)\[\];", glsl
        )
    }
    assert sorted(bindings) == ["dst", "items", "outv"], glsl
    outputs, written = expected()
    context = moderngl.create_standalone_context(require=430, backend="egl")
    buffers = []

    def bind(payload, binding):
        buffer = context.buffer(payload)
        buffer.bind_to_storage_buffer(binding)
        buffers.append(buffer)
        return buffer

    try:
        shader = context.compute_shader(glsl)
        bind(
            struct.pack(f"<{2 * ITEM_WORDS}I", *item_words(0), *item_words(1)),
            bindings["items"],
        )
        dst = bind(struct.pack("<f", -1.0) * (2 * ITEM_WORDS), bindings["dst"])
        outv = bind(struct.pack("<f", -9.0) * len(outputs), bindings["outv"])
        shader.run(group_x=1)
        context.finish()
        assert list(struct.unpack(f"<{len(outputs)}f", outv.read())) == outputs, glsl
        assert list(struct.unpack(f"<{2 * ITEM_WORDS}I", dst.read())) == written, glsl
    finally:
        for buffer in buffers:
            buffer.release()
        context.release()
