"""HLSL structured buffers keep their element layout on OpenGL and SPIR-V.

An HLSL structured buffer packs ``float3`` in 12 bytes and places struct members at 4-byte offsets; std430 aligns
three-component vectors to 16 bytes. Buffers whose layouts differ become ``uint`` word buffers with every access at
the HLSL offsets. Execution is covered by the ``h_float3_stride`` and ``h_struct_stride`` conformance cases.
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

HLSL = """
struct P {
    float a;
    float3 b;
    uint n;
    float3 v[2];
};
StructuredBuffer<float3> src : register(t0);
RWStructuredBuffer<P> ps : register(u1);
RWStructuredBuffer<uint3> counts : register(u2);
RWStructuredBuffer<float4> wide : register(u3);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint n, stride;
    ps.GetDimensions(n, stride);
    float3 s = src[id.x];
    P p = ps[id.x];
    p.a += s.x;
    ps[id.x] = p;
    ps[id.x].b.zx = s.xy;
    ps[id.x].v[id.y].y = float(n) + float(stride);
    ps[id.x].n++;
    uint before;
    InterlockedAdd(ps[id.x].n, 2u, before);
    InterlockedOr(counts[id.x].y, before);
    wide[id.x] = float4(p.v[1], 1.0);
}
"""


def _translate(tmp_path, backend, source=HLSL, format_output=False):
    path = tmp_path / "shader.hlsl"
    path.write_text(source, encoding="utf-8")
    if backend == "vulkan" and format_output:
        return friendly_spirv(
            crosstl.translate(str(path), backend=backend, format_output=False)
        )
    return crosstl.translate(str(path), backend=backend, format_output=format_output)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def _flat(code):
    return re.sub(r"\s+", " ", code)


def test_opengl_reads_and_writes_the_hlsl_offsets(tmp_path):
    code = _flat(_translate(tmp_path, "opengl"))
    # Vector elements become words; the OpenGL generator stores struct elements as scalarized copies itself.
    for buffer in ("src", "counts"):
        assert re.search(rf"buffer \w+ {{ uint {buffer}\[\]; }}", code), buffer
    assert re.search(r"buffer \w+ { vec4 wide\[\]; }", code)
    assert "vec3 s = cgl_src_load_vec3((uint(gl_GlobalInvocationID.x) * 3u));" in code
    assert (
        "vec3 cgl_src_load_vec3(uint base) { return vec3(uintBitsToFloat(src[(base + 0u)]), "
        "uintBitsToFloat(src[(base + 1u)]), uintBitsToFloat(src[(base + 2u)])); }"
    ) in code
    assert re.search(
        r"atomicOr\(counts\[\(\(uint\(gl_GlobalInvocationID\.x\) \* 3u\) \+ 1u\)\], before\)",
        code,
    )
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not on PATH")
    path = tmp_path / "shader.comp"
    path.write_text(_translate(tmp_path, "opengl"), encoding="utf-8")
    _run(["glslangValidator", "-S", "comp", str(path)])


def test_spirv_output_uses_word_strides_and_validates(tmp_path):
    assembly = _translate(tmp_path, "vulkan", format_output=True)
    strides = set(re.findall(r"ArrayStride (\d+)", assembly))
    assert "16" in strides  # the float4 buffer
    # Every other buffer, P's included, is a runtime array of words: no struct or vec3 elements remain.
    assert not re.search(r"OpTypeRuntimeArray %(v3float|v3uint|P)\b", assembly)
    assert not re.search(r"OpMemberDecorate %P \d+ Offset", assembly)
    assert "OpTypeRuntimeArray %uint" in assembly
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    path = tmp_path / "shader.spvasm"
    path.write_text(assembly, encoding="utf-8")
    binary = tmp_path / "shader.spv"
    _run(["spirv-as", "--target-env", "vulkan1.1", str(path), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])


def test_hlsl_and_metal_keep_their_own_layouts(tmp_path):
    assert "RWStructuredBuffer<P> ps" in _flat(_translate(tmp_path, "directx"))
    assert "uint ps" not in _flat(_translate(tmp_path, "metal"))


SIMPLE = """
struct Same {{
    float4 a;
    float2 b;
    float2 c;
}};
RWStructuredBuffer<{element}> b : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {{
    {body}
}}
"""


@pytest.mark.parametrize("element", ("float", "float2", "float4", "int2", "Same"))
def test_elements_with_the_same_layout_are_unchanged(tmp_path, element):
    source = SIMPLE.format(element=element, body="b[1] = b[0];")
    code = _flat(_translate(tmp_path, "opengl", source))
    assert " uint b[]" not in code
    assert "b[1] = b[0];" in code


@pytest.mark.parametrize(
    ("element", "prelude", "body", "reason"),
    (
        ("float3x3", "", "b[1] = b[0];", "matrix, boolean, 16-bit or 64-bit"),
        ("M", "struct M { float a; float3x3 m; };", "b[1] = b[0];", "matrix"),
        (
            "int3",
            "",
            "int o; InterlockedMin(b[id.x].x, 4, o);",
            "compare the words unsigned",
        ),
        (
            "float3",
            "",
            "b[id.x].x = 1.0; InterlockedAdd(b[id.x].y, 1);",
            "int or uint scalar",
        ),
        (
            "float3",
            "uint next() { return 1u; }",
            "b[next()].x += 1.0;",
            "index has side effects",
        ),
        (
            "float3",
            "void bump(inout float x) { x += 1.0; }",
            "bump(b[id.x].y);",
            "out parameter of bump",
        ),
    ),
)
def test_unsupported_layouts_and_accesses_are_refused(
    tmp_path, element, prelude, body, reason
):
    source = prelude + SIMPLE.format(element=element, body=body).replace(
        "struct Same", "struct Same_"
    )
    with pytest.raises(StructuredLayoutUnsupportedError, match=re.escape(reason)):
        _translate(tmp_path, "vulkan", source)
    if element != "M":
        with pytest.raises(StructuredLayoutUnsupportedError, match=re.escape(reason)):
            _translate(tmp_path, "opengl", source)


def test_bool_members_are_stored_as_hlsl_words(tmp_path):
    source = """
        struct Flags { bool on; float3 v; };
        RWStructuredBuffer<Flags> f : register(u0);
        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) {
            Flags x = f[id.x];
            f[id.x].on = !x.on;
        }
    """
    assembly = _translate(tmp_path, "vulkan", source, format_output=True)
    assert not re.search(r"OpTypeRuntimeArray %Flags\b", assembly)
    assert "OpTypeRuntimeArray %uint" in assembly
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    path = tmp_path / "shader.spvasm"
    path.write_text(assembly, encoding="utf-8")
    binary = tmp_path / "shader.spv"
    _run(["spirv-as", "--target-env", "vulkan1.1", str(path), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])
