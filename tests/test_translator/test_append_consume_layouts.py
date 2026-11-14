"""Append and consume buffers keep the HLSL element layout on GLSL and SPIR-V.

An HLSL ``AppendStructuredBuffer<float3>`` packs its elements in 12 bytes and ``struct { float m; float3 v; }`` in
16, but std430 gives them a 16- and a 32-byte stride. The structured-layout pass lowered indexed buffers only, so the
generators declared ``vec3 b[]`` (and, for OpenGL, assigned a struct to the generator's own differently named struct
copy). Queues now hold a struct of the element's words that every rule set packs tightly, and append and consume pack
and unpack it.
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

REQUIRE_ENV = "CROSSTL_REQUIRE_APPEND_CONSUME_EXECUTION"

HLSL = """
struct Particle { float mass; float3 velocity; };
ConsumeStructuredBuffer<Particle> pending : register(u0);
AppendStructuredBuffer<float3> moved : register(u1);
AppendStructuredBuffer<Particle> kept : register(u2);
RWStructuredBuffer<uint> results : register(u3);

[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    Particle p = pending.Consume();
    moved.Append(p.velocity * p.mass);
    p.mass += 1.0;
    kept.Append(p);
    if (id.x == 0) {
        uint count, stride;
        moved.GetDimensions(count, stride);
        results[0] = count;
        results[1] = stride;
        kept.GetDimensions(count, stride);
        results[2] = count;
        results[3] = stride;
    }
}
"""

PARTICLES = [
    (1.0, (1.0, 2.0, 3.0)),
    (2.0, (0.5, -1.0, 4.0)),
    (0.5, (8.0, 6.0, -2.0)),
    (3.0, (0.0, 1.0, 1.0)),
]
CAPACITY = 6

CGL_QUEUES = """
shader queues {
    @ buffer_layout(hlsl_structured)
    ConsumeStructuredBuffer<vec3> pending;
    @ buffer_layout(hlsl_structured)
    AppendStructuredBuffer<vec3> moved;
    compute {
        @ numthreads(1, 1, 1)
        void main() {
            BODY
        }
    }
}
"""


def translate(tmp_path, backend, source=HLSL, suffix="hlsl"):
    path = tmp_path / f"shader.{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def glsl_bindings(glsl):
    return {
        name: int(binding)
        for binding, name in re.findall(
            r"layout\(std430, binding = (\d+)\)[^{]*\{\s*\w+ (\w+)\[\];", glsl
        )
    }


def glsl_elements(glsl):
    return {
        name: element
        for element, name in re.findall(
            r"layout\(std430, binding = \d+\)[^{]*\{\s*(\w+) (\w+)\[\];", glsl
        )
    }


def validate_glsl(tmp_path, glsl):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")
    artifact = tmp_path / "shader.comp"
    artifact.write_text(glsl, encoding="utf-8")
    result = subprocess.run(
        [glslang, str(artifact)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_opengl_queues_hold_hlsl_packed_words(tmp_path):
    glsl = translate(tmp_path, "opengl")
    elements = glsl_elements(glsl)
    assert elements["pending"] == elements["kept"] == "cgl_words4"
    assert elements["moved"] == "cgl_words3"
    assert re.search(r"struct cgl_words3 \{\s*uint w\[3\];", glsl)
    assert re.search(r"struct cgl_words4 \{\s*uint w\[4\];", glsl)
    # The counters keep their own buffers and count elements.
    assert "atomicAdd(movedCounter[0], 1u)" in glsl
    assert "crossgl_hlsl_Particle" not in glsl
    assert re.findall(r"\b(stride) = (\d+)u;", glsl) == [
        ("stride", "16"),
        ("stride", "12"),
    ]
    validate_glsl(tmp_path, glsl)


def test_spirv_queue_strides_follow_hlsl(tmp_path):
    from tests.spirv_friendly_names import friendly_spirv

    assembly = translate(tmp_path, "vulkan")
    strides = re.findall(
        r"OpDecorate %_runtimearr___cgl_words(\d)\w* ArrayStride (\d+)",
        friendly_spirv(assembly),
    )
    assert sorted(set(strides)) == [("3", "12"), ("4", "16")]
    spirv_as, spirv_val = shutil.which("spirv-as"), shutil.which("spirv-val")
    if spirv_as is None or spirv_val is None:
        pytest.skip("SPIRV-Tools is not installed")
    source = tmp_path / "shader.spvasm"
    binary = tmp_path / "shader.spv"
    source.write_text(assembly, encoding="utf-8")
    subprocess.run(
        [spirv_as, "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
        check=True,
    )
    result = subprocess.run(
        [spirv_val, "--target-env", "vulkan1.1", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_queues_whose_layout_matches_are_kept(tmp_path):
    source = """
    AppendStructuredBuffer<float4> wide : register(u0);
    ConsumeStructuredBuffer<uint> words : register(u1);
    [numthreads(1, 1, 1)]
    void main() { wide.Append(float4(words.Consume(), 0.0, 0.0, 1.0)); }
    """
    glsl = translate(tmp_path, "opengl", source)
    assert glsl_elements(glsl) == {
        "wide": "vec4",
        "wideCounter": "uint",
        "words": "uint",
        "wordsCounter": "uint",
    }
    assert "cgl_words" not in glsl


def test_hlsl_keeps_its_own_queues(tmp_path):
    hlsl = translate(tmp_path, "directx")
    assert "AppendStructuredBuffer<float3> moved" in hlsl
    assert "ConsumeStructuredBuffer<Particle> pending" in hlsl
    assert "__cgl_words" not in hlsl


@pytest.mark.parametrize(
    "body, message",
    [
        ("vec3 v = pending[0];", "cannot be indexed"),
        ("buffer_append(pending, vec3(1.0));", "declared as ConsumeStructuredBuffer"),
        ("vec3 v = buffer_consume(moved);", "declared as AppendStructuredBuffer"),
    ],
)
def test_queue_misuse_is_refused(tmp_path, body, message):
    source = CGL_QUEUES.replace("BODY", body)
    with pytest.raises(StructuredLayoutUnsupportedError, match=message):
        translate(tmp_path, "opengl", source, "cgl")


def test_queues_execute_on_opengl(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1" or sys.platform == "darwin":
        pytest.skip(f"set {REQUIRE_ENV}=1 on Linux to require the OpenGL dispatch")
    import moderngl

    glsl = translate(tmp_path, "opengl")
    bindings = glsl_bindings(glsl)
    context = moderngl.create_standalone_context(require=430, backend="egl")
    buffers = {}
    try:
        shader = context.compute_shader(glsl)
        contents = {
            "pending": b"".join(
                struct.pack("<4f", mass, *velocity) for mass, velocity in PARTICLES
            ),
            "pendingCounter": struct.pack("<I", len(PARTICLES)),
            "moved": bytes(12 * CAPACITY),
            "movedCounter": bytes(4),
            "kept": bytes(16 * CAPACITY),
            "keptCounter": bytes(4),
            "results": bytes(16),
        }
        for name, data in contents.items():
            buffers[name] = context.buffer(data)
            buffers[name].bind_to_storage_buffer(bindings[name])
        shader.run(group_x=1)
        context.finish()

        def counter(name):
            return struct.unpack("<I", buffers[name].read())[0]

        assert [counter(f"{name}Counter") for name in ("pending", "moved", "kept")] == [
            0,
            4,
            4,
        ]
        moved = sorted(struct.iter_unpack("<3f", buffers["moved"].read()[: 12 * 4]))
        assert moved == sorted(
            tuple(component * mass for component in velocity)
            for mass, velocity in PARTICLES
        ), glsl
        kept = sorted(struct.iter_unpack("<4f", buffers["kept"].read()[: 16 * 4]))
        assert kept == sorted((mass + 1.0, *velocity) for mass, velocity in PARTICLES)
        results = list(struct.unpack("<4I", buffers["results"].read()))
        assert results == [CAPACITY, 12, CAPACITY, 16]
    finally:
        for buffer in buffers.values():
            buffer.release()
        context.release()
