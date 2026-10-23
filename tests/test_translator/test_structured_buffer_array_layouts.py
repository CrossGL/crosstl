"""Arrays of HLSL structured buffers keep the HLSL element layout on GLSL and SPIR-V.

``RWStructuredBuffer<T> b[N]`` declares N buffers whose elements HLSL packs tightly: ``struct { float4 a; float b; }``
takes 20 bytes and ``float3`` 12. The structured-layout pass lowered single buffers only, so the GLSL and SPIR-V
generators declared arrays of storage blocks with the std430 strides 32 and 16 and read other bytes than the host
wrote.
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

REQUIRE_ENV = "CROSSTL_REQUIRE_BUFFER_ARRAY_LAYOUT_EXECUTION"

HLSL = """
struct Item { float4 a; float b; };
StructuredBuffer<Item> items[2] : register(t0);
RWStructuredBuffer<float3> vecs[2] : register(u2);
RWStructuredBuffer<uint> results : register(u4);

[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    for (uint k = 0u; k < 2u; k++) {
        Item it = items[k][id.x];
        vecs[k][id.x] = it.a.xyz * it.b;
        vecs[k][id.x].y += items[1u - k][id.x].b;
    }
    if (id.x == 0u) {
        uint count, stride;
        items[1].GetDimensions(count, stride);
        results[0] = count;
        results[1] = stride;
        vecs[0].GetDimensions(count, stride);
        results[2] = count;
        results[3] = stride;
    }
}
"""

ITEMS = [
    [((1.0, 2.0, 3.0, 9.0), 2.0), ((0.5, 1.0, -1.0, 9.0), 4.0)] * 2,
    [((3.0, 0.0, 1.0, 9.0), 0.5), ((2.0, 2.0, 2.0, 9.0), 1.0)] * 2,
]
THREADS = 4


def translate(tmp_path, backend, source=HLSL, suffix="hlsl"):
    path = tmp_path / f"shader.{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def run(command):
    result = subprocess.run(
        [str(part) for part in command], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def expected_vectors(k):
    vectors = []
    for (a, b), (_other_a, other_b) in zip(ITEMS[k], ITEMS[1 - k]):
        x, y, z = (component * b for component in a[:3])
        vectors.append((x, y + other_b, z))
    return vectors


def test_opengl_arrays_hold_hlsl_packed_words(tmp_path):
    glsl = translate(tmp_path, "opengl")
    assert re.search(r"buffer itemsBuffer \{\s*uint data\[\];\s*\}\s*items\[2\];", glsl)
    assert re.search(r"buffer vecsBuffer \{\s*uint data\[\];\s*\}\s*vecs\[2\];", glsl)
    # Helpers take the buffer's index before the element's first word.
    assert "Item cgl_items_load_Item(uint selector, uint base)" in glsl
    assert "items[selector].data[(base + 4u)]" in glsl
    assert re.findall(r"\bstride = (\d+)u;", glsl) == ["20", "12"]
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")
    artifact = tmp_path / "shader.comp"
    artifact.write_text(glsl, encoding="utf-8")
    run([glslang, artifact])


def test_spirv_arrays_hold_words(tmp_path):
    from tests.spirv_friendly_names import friendly_spirv

    assembly = translate(tmp_path, "vulkan")
    friendly = friendly_spirv(assembly)
    assert re.search(r"OpTypeArray %itemsBuffer\w* %\w+", friendly), friendly
    assert set(re.findall(r"ArrayStride (\d+)", friendly)) == {"4"}
    spirv_as, spirv_val = shutil.which("spirv-as"), shutil.which("spirv-val")
    if spirv_as is None or spirv_val is None:
        pytest.skip("SPIRV-Tools is not installed")
    source, binary = tmp_path / "shader.spvasm", tmp_path / "shader.spv"
    source.write_text(assembly, encoding="utf-8")
    run([spirv_as, "--target-env", "vulkan1.1", source, "-o", binary])
    run([spirv_val, "--target-env", "vulkan1.1", binary])


def test_arrays_whose_layout_matches_are_kept(tmp_path):
    source = """
    RWStructuredBuffer<float4> wide[2] : register(u0);
    [numthreads(1, 1, 1)]
    void main() { wide[1][0] = wide[0][0]; }
    """
    glsl = translate(tmp_path, "opengl", source)
    assert re.search(r"buffer wideBuffer \{\s*vec4 data\[\];", glsl)


@pytest.mark.parametrize(
    "body",
    [
        "consume(vecs[0]);",
        "consume_all(vecs);",
    ],
)
def test_passing_buffers_of_an_array_is_refused(tmp_path, body):
    source = f"""
    RWStructuredBuffer<float3> vecs[2] : register(u0);
    void consume(RWStructuredBuffer<float3> b) {{ b[0] = float3(1.0, 2.0, 3.0); }}
    void consume_all(RWStructuredBuffer<float3> b[2]) {{ b[1][0] = float3(1.0, 2.0, 3.0); }}
    [numthreads(1, 1, 1)]
    void main() {{ {body} }}
    """
    with pytest.raises(StructuredLayoutUnsupportedError):
        translate(tmp_path, "opengl", source)


def test_arrays_execute_on_opengl(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1" or sys.platform == "darwin":
        pytest.skip(f"set {REQUIRE_ENV}=1 on Linux to require the OpenGL dispatch")
    import moderngl

    glsl = translate(tmp_path, "opengl")
    bases = {
        name: int(binding)
        for binding, name in re.findall(
            r"layout\(std430, binding = (\d+)\)[^{]*\{[^}]*\}\s*(\w+)\[2\];", glsl
        )
    }
    results_binding = int(
        re.search(r"binding = (\d+)\)[^{]*\{\s*uint results\[\];", glsl).group(1)
    )
    context = moderngl.create_standalone_context(require=430, backend="egl")
    buffers = []

    def bind(payload, binding):
        buffer = context.buffer(payload)
        buffer.bind_to_storage_buffer(binding)
        buffers.append(buffer)
        return buffer

    try:
        shader = context.compute_shader(glsl)
        for k in range(2):
            bind(
                b"".join(struct.pack("<5f", *a, b) for a, b in ITEMS[k]),
                bases["items"] + k,
            )
        vecs = [bind(bytes(12 * THREADS), bases["vecs"] + k) for k in range(2)]
        results = bind(bytes(16), results_binding)
        shader.run(group_x=1)
        context.finish()
        for k in range(2):
            assert list(struct.iter_unpack("<3f", vecs[k].read())) == expected_vectors(
                k
            ), glsl
        assert list(struct.unpack("<4I", results.read())) == [THREADS, 20, THREADS, 12]
    finally:
        for buffer in buffers:
            buffer.release()
        context.release()
