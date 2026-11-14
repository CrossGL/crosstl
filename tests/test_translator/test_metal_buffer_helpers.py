"""Helper functions that take Metal buffer pointers keep the caller's buffer layout on every target.

A helper with a ``device`` or ``constant`` pointer parameter reads and writes the buffer its caller passes, so the
parameter has that buffer's element layout (MSL 3.2 section 4.2). The Metal importer marks such parameters with the
buffer's ``@buffer_layout``; the Metal generator gives the pointee the same storage as the buffer, and targets that
address the buffer in 32-bit words pass the word buffer to the helper with the element offset scaled to words.
Execution is covered by the ``m_packed_helper`` and ``m_float3_helper`` conformance cases.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics.structured_layouts import (
    StructuredLayoutUnsupportedError,
)
from tests.spirv_friendly_names import friendly_spirv

PRELUDE = "#include <metal_stdlib>\nusing namespace metal;\n"

# S is 32 bytes in MSL (b at byte 16) against 16 bytes in HLSL structured buffers and std430.
NATURAL = PRELUDE + """
struct S { float3 a; float b; };
float sum(device const S* p, uint i) { return p[i].a.y + p[i].b; }
void put(device S* q, uint i, float v) { q[i].b = v; q[i].a = float3(v, 0.0, 1.0); }
kernel void k(device const S* src [[buffer(0)]], device S* dst [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    uint odd = i & 1u;
    put(dst + odd, i - odd, sum(src + odd, i - odd));
}
"""

# packed_float3 is 12 bytes in MSL against 16 in std430.
PACKED = PRELUDE + """
float3 load3(device const packed_float3* p, uint i) { return float3(p[i]); }
void store3(device packed_float3* q, uint i, float3 v) { q[i] = v; q[i].y += 0.5; }
kernel void k(device const packed_float3* src [[buffer(0)]], device packed_float3* dst [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
    uint odd = i & 1u;
    store3(dst + odd, i - odd, load3(src + odd, i - odd).zyx);
}
"""


def _translate(tmp_path, backend, source, suffix="metal", format_output=False):
    path = tmp_path / f"shader.{suffix}"
    path.write_text(source, encoding="utf-8")
    if backend == "vulkan" and format_output:
        return friendly_spirv(
            crosstl.translate(str(path), backend=backend, format_output=False)
        )
    return crosstl.translate(str(path), backend=backend, format_output=format_output)


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("source", "marks"),
    (
        (NATURAL, ["metal"] * 4),
        (PACKED, ["hlsl_structured"] * 4),
    ),
)
def test_import_marks_helper_pointer_parameters(tmp_path, source, marks):
    code = _flat(_translate(tmp_path, "cgl", source))
    assert re.findall(r"@buffer_layout\((\w+)\)", code) == marks
    assert re.search(r"@buffer_layout\(\w+\) const device \w+\* p", code)
    assert re.search(r"@buffer_layout\(\w+\) device \w+\* q", code)


def test_import_leaves_thread_pointers_unmarked(tmp_path):
    source = PRELUDE + (
        "float first(thread float3* p) { return p[0].x; }\n"
        "kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) "
        "{ float3 v = float3(1.0); o[i] = first(&v); }\n"
    )
    assert "buffer_layout" not in _translate(tmp_path, "cgl", source)


def test_metal_gives_helper_pointees_the_buffer_storage(tmp_path):
    natural = _flat(_translate(tmp_path, "metal", NATURAL)).replace(" *", "* ")
    assert "float sum(const device S* p, uint i)" in natural
    assert "void put(device S* q, uint i, float v)" in natural
    packed = _flat(_translate(tmp_path, "metal", PACKED)).replace(" *", "* ")
    assert "float3 load3(const device packed_float3* p, uint i)" in packed
    assert "void store3(device packed_float3* q, uint i, float3 v)" in packed
    assert "buffer_layout" not in natural + packed


def test_hlsl_passes_word_buffers_with_word_offsets(tmp_path):
    code = _flat(_translate(tmp_path, "directx", NATURAL))
    assert "StructuredBuffer<uint> src : register(t0);" in code
    assert "RWStructuredBuffer<uint> dst : register(u1);" in code
    assert re.search(
        r"float sum\(StructuredBuffer<uint> p, \w+ p_offset, uint i\)", code
    )
    assert re.search(r"void put\(RWStructuredBuffer<uint> q, \w+ q_offset", code)
    # b is word 4 of an 8-word element, and one element of offset moves 8 words.
    assert "asfloat(p[uint((p_offset + ((uint(i) * 8u) + 4u)))])" in code
    assert code.count("(uint(odd) * 8u)") == 2
    # Padding words 3 and 5-7 are never written.
    assert not re.search(r"q\[uint\(\(q_offset \+ \(__cgl_base \+ [3-7]u\)\)\)\]", code)
    assert "buffer_layout" not in code


def test_hlsl_keeps_packed_elements_and_element_offsets(tmp_path):
    # packed_float3 has the HLSL structured-buffer layout, so HLSL passes the buffers and element offsets as they are.
    code = _flat(_translate(tmp_path, "directx", PACKED))
    assert "StructuredBuffer<float3> src : register(t0);" in code
    assert re.search(
        r"float3 load3\(StructuredBuffer<float3> p, \w+ p_offset, uint i\)", code
    )
    assert "buffer_layout" not in code


def test_opengl_scales_packed_offsets_by_three_words(tmp_path):
    code = _flat(_translate(tmp_path, "opengl", PACKED))
    assert re.search(r"readonly buffer \w+ { uint src\[\]; }", code)
    assert "uintBitsToFloat(src[(p_offset + int(((uint(i) * 3u) + 2u)))])" in code
    assert code.count("int((uint(odd) * 3u))") == 2


def test_opengl_reads_the_caller_words(tmp_path):
    source = _translate(tmp_path, "opengl", NATURAL)
    code = _flat(source)
    assert re.search(r"readonly buffer \w+ { uint src\[\]; }", code)
    assert re.search(r"buffer \w+ { uint dst\[\]; }", code)
    assert "uintBitsToFloat(src[(p_offset + int(((uint(i) * 8u) + 4u)))])" in code
    assert "dst[(q_offset + int(((uint(i) * 8u) + 4u)))] = floatBitsToUint(v);" in code
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not on PATH")
    path = tmp_path / "shader.comp"
    path.write_text(source, encoding="utf-8")
    _run(["glslangValidator", "-S", "comp", str(path)])


@pytest.mark.parametrize("source", (NATURAL, PACKED), ids=("natural", "packed"))
def test_spirv_addresses_helper_buffers_in_words(tmp_path, source):
    assembly = _translate(tmp_path, "vulkan", source, format_output=True)
    assert "OpTypeRuntimeArray %uint" in assembly
    assert "buffer_layout" not in assembly
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    path = tmp_path / "shader.spvasm"
    path.write_text(assembly, encoding="utf-8")
    binary = tmp_path / "shader.spv"
    _run(["spirv-as", "--target-env", "vulkan1.1", str(path), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])


GLOBALS = """
shader main {
    struct T { vec3 a; float b; }
    @buffer_layout(metal) StructuredBuffer<T> s;
    @buffer_layout(metal) RWStructuredBuffer<T> t;
    float get(@buffer_layout(metal) StructuredBuffer<T> b, uint i) { return b[i].b; }
    void put(@buffer_layout(metal) RWStructuredBuffer<T> b, uint i, float v) { b[i].b = v; }
    compute {
        @ stage_entry
        void k(uint id @gl_GlobalInvocationID) {
            put(t, id, get(s, id) + get(t, id));
        }
    }
}
"""


@pytest.mark.parametrize("backend", ("directx", "opengl"))
def test_global_buffers_passed_to_helpers_are_lowered(tmp_path, backend):
    code = _flat(_translate(tmp_path, backend, GLOBALS, suffix="cgl"))
    if backend == "directx":
        assert "StructuredBuffer<uint> s" in code
        assert "float get(StructuredBuffer<uint> b, uint i)" in code
        assert "put(t, id, (get(s, id) + get(t, id)));" in code
    else:
        # GLSL cannot pass buffers, so each call site gets a copy of the helper bound to its buffer.
        assert re.search(r"readonly buffer \w+ { uint s\[\]; }", code)
        assert "float get_glsl_b_s_uint(uint i, int b_offset)" in code
        assert "float get_glsl_b_t_uint(uint i, int b_offset)" in code
    assert "buffer_layout" not in code


@pytest.mark.parametrize("backend", ("directx", "opengl", "vulkan"))
@pytest.mark.parametrize(
    ("source", "reason"),
    (
        # The parameter has no layout, so the helper would read the words as whole elements.
        (
            GLOBALS.replace(
                "float get(@buffer_layout(metal) StructuredBuffer<T> b",
                "float get(StructuredBuffer<T> b",
            ),
            "the structured buffer s is passed to get, whose parameter does not take its layout",
        ),
        # Overloads whose parameters differ cannot share the caller's words.
        (
            GLOBALS.replace(
                "    void put(",
                "    float get(StructuredBuffer<T> b, int i) { return b[i].b; }\n    void put(",
            ),
            "the structured buffer s is passed to get, whose parameter does not take its layout",
        ),
        (
            GLOBALS.replace("put(t, id,", "put(s, id,"),
            "the read-only structured buffer s is passed to a writable parameter of put",
        ),
    ),
    ids=("unmarked", "overloaded", "read-only"),
)
def test_refuses_buffers_passed_to_parameters_without_their_words(
    tmp_path, backend, source, reason
):
    with pytest.raises(StructuredLayoutUnsupportedError, match=re.escape(reason)):
        _translate(tmp_path, backend, source, suffix="cgl")
