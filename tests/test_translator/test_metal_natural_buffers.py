"""Metal buffers in the natural MSL layout keep it on HLSL, OpenGL and SPIR-V.

MSL sizes and aligns ``float3`` like ``float4``, so ``device float3*`` has a 16-byte stride where an HLSL structured
buffer of ``float3`` has 12, and ``struct { float3 a; float b; }`` takes 32 bytes where HLSL structured buffers and
std430 take 16. The Metal importer marks such buffers ``@buffer_layout(metal)``; the Metal generator keeps the
natural layout and the other targets address the buffer in 32-bit words where their rules differ. Execution is
covered by the ``m_float3_stride``, ``m_float3_struct`` and ``m_float3x3_stride`` conformance cases.
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

PRELUDE = "#include <metal_stdlib>\nusing namespace metal;\n"

NATURAL = PRELUDE + """
struct T { float3 a; float b; float2 c; };
struct I { float4x4 m; float3 c; float3x2 n; };
kernel void k(device const float3* v [[buffer(0)]], device T* t [[buffer(1)]], device I* s [[buffer(2)]],
              device float* o [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    o[i] = v[i].y + t[i].c.x + s[i].m[2].w + s[i].n[2].y;
    t[i].a = v[i];
    t[i].c.y += 1.0;
    s[i].m[1] = float4(v[i], 1.0);
    T u = t[i + 1u];
    t[i] = u;
}
"""


def _translate(tmp_path, backend, source, suffix="metal", format_output=False):
    path = tmp_path / f"shader.{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=format_output)


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def _kernel(element, prelude=""):
    return (
        PRELUDE
        + prelude
        + f"kernel void k(device {element}* b [[buffer(0)]], uint i [[thread_position_in_grid]]) {{ b[i + 1u] = b[i]; }}\n"
    )


@pytest.mark.parametrize(
    ("element", "prelude", "layout"),
    (
        # 16 bytes against HLSL's 12.
        ("float3", "", "metal"),
        ("int3", "", "metal"),
        # half3 is 8 bytes against HLSL's 6.
        ("half3", "", "metal"),
        ("float2", "", None),
        ("float4", "", None),
        # b at 16 and 32 bytes, against b at 12 and 16 bytes.
        ("T", "struct T { float3 a; float b; };", "metal"),
        # 16 bytes against HLSL's 12.
        ("U", "struct U { float2 a; float b; };", "metal"),
        ("S", "struct S { float a; float4 b; };", "metal"),
        ("W", "struct W { float a; float b; };", None),
        # Three 16-byte columns against HLSL's 36 bytes; std430 has the MSL layout.
        ("float3x3", "", "metal"),
        ("float4x4", "", None),
        ("float2x2", "", None),
        # MSL bool is one byte; the host binding contract reports that storage.
        ("bool", "", None),
    ),
)
def test_import_marks_elements_whose_natural_layout_differs(
    tmp_path, element, prelude, layout
):
    code = _flat(_translate(tmp_path, "cgl", _kernel(element, prelude)))
    marks = re.findall(r"@buffer_layout\((\w+)\) RWStructuredBuffer", code)
    assert marks == ([layout] if layout else [])


def test_metal_keeps_the_natural_layout(tmp_path):
    code = _flat(_translate(tmp_path, "metal", NATURAL))
    assert "const device float3* v [[buffer(0)]]" in code
    assert "device T* t [[buffer(1)]]" in code
    assert "struct T { float3 a; float b; float2 c; };" in code
    assert "buffer_layout" not in code and "packed_" not in code


def test_hlsl_addresses_natural_buffers_in_words(tmp_path):
    code = _flat(_translate(tmp_path, "directx", NATURAL))
    for declaration in (
        "StructuredBuffer<uint> v",
        "RWStructuredBuffer<uint> t",
        "RWStructuredBuffer<uint> s",
    ):
        assert declaration in code
    # float3: 4 words per element; T: a at 0, b at 4, c at 6, 8 words; I: m 0-15, c 16-18, n 20-25, 28 words.
    assert "asfloat(v[((uint(i) * 4u) + 1u)])" in code
    assert "asfloat(t[((uint(i) * 8u) + 6u)])" in code
    assert "asfloat(s[((uint(i) * 28u) + 11u)])" in code
    assert "asfloat(s[((uint(i) * 28u) + 25u)])" in code
    assert "uint __cgl_base = ((uint(i) * 28u) + 4u);" in code
    # Padding words are never written.
    assert not re.search(r"t\[\(__cgl_base \+ [35]u\)\]", code)


def test_opengl_keeps_layouts_std430_shares_and_lowers_the_others(tmp_path):
    code = _flat(_translate(tmp_path, "opengl", NATURAL))
    # std430 gives vec3 a 16-byte stride and lays I out as MSL does.
    assert re.search(r"buffer \w+ { vec3 v\[\]; }", code)
    assert re.search(r"buffer \w+ { I s\[\]; }", code)
    assert re.search(r"buffer \w+ { uint t\[\]; }", code)
    assert "uintBitsToFloat(t[((uint(i) * 8u) + 6u)])" in code
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not on PATH")
    path = tmp_path / "shader.comp"
    path.write_text(_translate(tmp_path, "opengl", NATURAL), encoding="utf-8")
    _run(["glslangValidator", "-S", "comp", str(path)])


def test_spirv_lays_out_matrix_buffers_and_lowers_the_others(tmp_path):
    source = NATURAL.replace(
        "device float* o [[buffer(3)]]",
        "device float* o [[buffer(3)]], device const float3x3* r [[buffer(4)]]",
    ).replace("o[i] = v[i].y", "o[i] = r[i][1].z + v[i].y")
    assembly = _translate(tmp_path, "vulkan", source, format_output=True)
    assert "OpTypeRuntimeArray %uint" in assembly
    assert re.search(r"OpMemberDecorate %\w+ 0 ColMajor", assembly)
    assert re.search(r"OpMemberDecorate %\w+ 0 MatrixStride 16", assembly)
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    path = tmp_path / "shader.spvasm"
    path.write_text(assembly, encoding="utf-8")
    binary = tmp_path / "shader.spv"
    _run(["spirv-as", "--target-env", "vulkan1.1", str(path), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])


@pytest.mark.parametrize(
    ("source", "reason"),
    (
        # half3: 8 bytes in MSL, 6 in an HLSL structured buffer, and 16-bit words cannot be split into 32-bit ones.
        (_kernel("half3"), "cannot be split into 32-bit words"),
        # The buffer escapes into a local pointer, which the word rewrite cannot follow.
        (
            PRELUDE
            + "kernel void k(device const float3* v [[buffer(0)]], device float* o [[buffer(1)]], "
            + "uint i [[thread_position_in_grid]]) { device const float3* w = v + 1u; o[i] = w[i].y; }\n",
            "used other than through an element",
        ),
    ),
)
def test_hlsl_refuses_natural_layouts_it_cannot_reproduce(tmp_path, source, reason):
    with pytest.raises(StructuredLayoutUnsupportedError, match=re.escape(reason)):
        _translate(tmp_path, "directx", source)


GLOBAL = """
shader main {
    struct T { vec3 a; float b; }
    @buffer_layout(metal) RWStructuredBuffer<T> t;
    compute {
        @ stage_entry
        void k(uint id @gl_GlobalInvocationID) {
            T u = t[id + 1u];
            t[id] = u;
            t[id].b += 1.0;
        }
    }
}
"""


@pytest.mark.parametrize("backend", ("directx", "opengl", "vulkan"))
def test_global_natural_buffers_use_word_helpers(tmp_path, backend):
    code = _flat(
        _translate(tmp_path, backend, GLOBAL, suffix="cgl", format_output=True)
    )
    if backend == "vulkan":
        assert "OpTypeRuntimeArray %uint" in code
        return
    # GLSL reserves the double underscore, so the OpenGL generator renames the helpers.
    assert re.search(r"cgl_t_load_T\(\(?uint\(\(?id \+ 1u\)?\) \* 8u\)?\)", code)
    assert "cgl_t_store_T" in code
    assert "(uint(id) * 8u) + 4u" in code
