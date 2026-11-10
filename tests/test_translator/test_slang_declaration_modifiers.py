"""Slang keeps the declaration modifiers CrossGL records as attributes.

The Slang generator turned the first attribute of a struct member into its semantic, so ``row_major``, ``invariant``
and ``highp`` became ``: row_major``, ``: invariant`` and ``: highp``. An ``invariant gl_Position`` output therefore
lost ``SV_Position``. On constant-buffer members it dropped ``row_major``, ``column_major`` and ``packoffset``, and it
dropped ``precise`` on variables, so slangc laid out the matrices transposed, placed packed members at other offsets
and allowed contraction. The generator now spells them as HLSL does (slang_declaration_modifiers.py).
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

HLSL = """
struct S { row_major float3x3 m; float4 v; };
cbuffer CB : register(b0) {
    row_major float4x4 world;
    column_major float4x4 view;
    float4 tint : packoffset(c12);
    S s;
};
RWStructuredBuffer<float4> outb : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    precise float x = tint.x * 2.0;
    outb[0] = mul(world, float4(x, s.v.y, 0, 1)) + mul(view, tint) + s.m[0].xxxy;
}
"""

GLSL_VERTEX = """#version 450
layout(location = 0) in highp vec3 pos;
layout(location = 0) out mediump vec2 uv;
layout(std140, binding = 0) uniform U { layout(row_major) mat4 mvp; highp vec4 tint; };
invariant gl_Position;
void main() {
    precise float k = tint.x * 2.0;
    uv = pos.xy * k;
    gl_Position = mvp * vec4(pos, 1.0);
}
"""

GLSL_NESTED = """#version 450
layout(local_size_x = 1) in;
struct L { mat2 n; vec4 c; };
layout(std140, row_major, binding = 0) uniform I { L l; } src;
layout(std430, binding = 1) buffer O { vec4 o[2]; } dst;
void main() { dst.o[0] = vec4(src.l.n[0], src.l.n[1]); dst.o[1] = src.l.c; }
"""

GLSL_VECTOR_ORIENTATION = """#version 450
layout(local_size_x = 1) in;
struct L { vec4 c; };
layout(std140, row_major, binding = 0) uniform I { vec4 a; L l; mat2 m; } src;
layout(std430, binding = 1) buffer O { vec4 o[3]; } dst;
void main() { dst.o[0] = src.a; dst.o[1] = src.l.c; dst.o[2] = vec4(src.m[0], src.m[1]); }
"""


def _slang(tmp_path, source, suffix):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="slang", format_output=False)


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _slangc_spirv(tmp_path, name, code, stage):
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    source = tmp_path / f"{name}.slang"
    source.write_text(code, encoding="utf-8")
    output = tmp_path / f"{name}.spvasm"
    completed = subprocess.run(
        [slangc, str(source), "-entry", "main", "-stage", stage]
        + ["-target", "spirv-asm", "-o", str(output)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return output.read_text(encoding="utf-8")


def _layout_decorations(spirv):
    """Member offsets, matrix orders and NoContraction, with block names erased."""
    decorations = []
    for line in spirv.splitlines():
        line = line.strip()
        if re.match(r"OpMemberDecorate %\S+ \d+ (Offset|RowMajor|ColMajor)", line):
            decorations.append(re.sub(r"^OpMemberDecorate %\S+", "member", line))
        elif re.match(r"OpDecorate %\S+ NoContraction", line):
            decorations.append(line)
    return sorted(decorations)


def test_hlsl_member_modifiers_are_declaration_qualifiers(tmp_path):
    code = _flat(_slang(tmp_path, HLSL, ".hlsl"))
    assert "row_major float3x3 m;" in code
    assert "row_major float4x4 world;" in code
    assert "column_major float4x4 view;" in code
    assert "float4 tint : packoffset(c12);" in code
    assert "precise float x =" in code
    assert ": row_major" not in code
    assert ": column_major" not in code


def test_hlsl_member_layout_matches_slangc_on_the_source(tmp_path):
    reference = _layout_decorations(
        _slangc_spirv(tmp_path, "reference", HLSL, "compute")
    )
    generated = _layout_decorations(
        _slangc_spirv(tmp_path, "generated", _slang(tmp_path, HLSL, ".hlsl"), "compute")
    )
    assert generated == reference
    assert any("Offset 192" in line for line in generated)
    assert any("NoContraction" in line for line in generated)


def test_glsl_invariant_position_keeps_its_system_value(tmp_path):
    code = _flat(_slang(tmp_path, GLSL_VERTEX, ".vert"))
    assert re.search(r"float4 gl_Position : SV_Position;", code), code
    for leaked in (": invariant", ": highp", ": mediump", ": buffer_layout"):
        assert leaked not in code
    assert "row_major float4x4 mvp;" in code
    assert "precise float k =" in code
    _slangc_spirv(tmp_path, "vertex", _slang(tmp_path, GLSL_VERTEX, ".vert"), "vertex")


def test_glsl_row_major_uniform_matrix_reaches_spirv(tmp_path):
    spirv = _slangc_spirv(
        tmp_path, "vertex", _slang(tmp_path, GLSL_VERTEX, ".vert"), "vertex"
    )
    # slangc flips HLSL's row_major to SPIR-V ColMajor because it transposes the matrix type as DXC does; the
    # memory order is the GLSL row_major one.
    assert re.search(r"OpMemberDecorate %\S+ 0 ColMajor", spirv), spirv[-2000:]


def test_orientation_on_vectors_is_dropped_and_matrices_keep_it(tmp_path):
    generated = _slang(tmp_path, GLSL_VECTOR_ORIENTATION, ".comp")
    code = _flat(generated)
    assert "row_major float2x2 m;" in code
    assert "row_major float4" not in code
    assert "row_major L" not in code
    _slangc_spirv(tmp_path, "vectors", generated, "compute")


CGL_STRUCT_ORIENTATION = """
shader main {
    struct L {
        mat2 n;
        vec4 c;
    };
    struct S {
        L l @row_major;
    };
    compute {
        layout(local_size_x = 1) in;
        void main() {
            S s;
        }
    }
}
"""


def test_block_orientation_reaches_a_struct_holding_a_matrix(tmp_path):
    # semantics/row_major_struct_members.py gives the block member a row-major copy of its struct.
    code = _flat(_slang(tmp_path, GLSL_NESTED, ".comp"))
    assert "struct L_row_major { row_major float2x2 n; float4 c; };" in code, code
    assert "L_row_major l;" in code, code


def test_orientation_on_a_struct_holding_a_matrix_is_refused(tmp_path):
    # Outside a constant block, row_major on a struct-typed member has no Slang spelling.
    with pytest.raises(
        Exception, match=r"cannot apply row_major to the matrices inside struct 'L'"
    ):
        _slang(tmp_path, CGL_STRUCT_ORIENTATION, ".cgl")
