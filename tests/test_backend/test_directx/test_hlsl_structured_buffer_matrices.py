"""Matrices in HLSL structured buffers, translated to CrossGL and Metal.

A structured buffer packs a matrix's columns (its rows, for ``row_major``) one
after another at four-byte alignment: ``float3x3`` takes 36 bytes and
``float2x3`` 24 (D3D11.3 functional specification, section 5.3.7; DXC's
``-fvk-use-dx-layout``). Metal's matrices align their columns to 8 or 16
bytes, so Metal output holds such a matrix as its words and reads and writes
it through helpers. The expected values of the native check are written out
by hand from those rules, so they do not depend on the planner under test.
"""

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from crosstl.backend.DirectX import DirectxCrossGLCodeGen
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from crosstl.backend.import_refusals import ImportRefused
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

REQUIRE_ENV = "CROSSTL_REQUIRE_HLSL_STRUCTURED_MATRIX_EXECUTION"
ROOT = Path(__file__).resolve().parents[3]

MATRICES_HLSL = """
    struct Item
    {
        float s;
        float3x3 m;
        float2 t;
        row_major float2x3 r;
    };
    StructuredBuffer<Item> items : register(t0);
    StructuredBuffer<float4x4> mats : register(t1);
    RWStructuredBuffer<float> outv : register(u0);
    RWStructuredBuffer<float2x3> rows : register(u1);
    [numthreads(1, 1, 1)]
    void main(uint3 id : SV_DispatchThreadID)
    {
        Item it = items[1];
        float3 v = mul(it.m, float3(1.0, 2.0, 3.0));
        float3 w = mul(float2(1.0, -1.0), it.r);
        outv[0] = v.x;
        outv[1] = v.y;
        outv[2] = v.z;
        outv[3] = w.x;
        outv[4] = w.y;
        outv[5] = w.z;
        outv[6] = items[0].m[2][1];
        outv[7] = mats[1][3][2];
        outv[8] = it.t.y + it.s;
        float2x3 rr = rows[0];
        rr[1] = float3(-1.0, -2.0, -3.0);
        rows[1] = rr;
    }
"""

# Words of one ``Item``: s, m (3 columns of 3), t, r (2 rows of 3).
ITEM_WORDS = 1 + 9 + 2 + 6


def item_word(item, index):
    return 100.0 * item + index


def mats_word(index):
    return 0.25 * index - 3.0


def expected_values():
    """The HLSL results, from the structured-buffer layout of the inputs."""
    m = [[item_word(1, 1 + 3 * c + r) for c in range(3)] for r in range(3)]
    v = [sum(m[r][c] * x for c, x in enumerate((1.0, 2.0, 3.0))) for r in range(3)]
    rows = [[item_word(1, 12 + 3 * r + c) for c in range(3)] for r in range(2)]
    w = [rows[0][c] - rows[1][c] for c in range(3)]
    return [
        *v,
        *w,
        item_word(0, 1 + 3 * 1 + 2),  # items[0].m[2][1]: column 1, row 2
        mats_word(16 + 4 * 2 + 3),  # mats[1][3][2]: column 2, row 3
        item_word(1, 11) + item_word(1, 0),
    ]


def expected_rows(initial):
    """``rows`` after the kernel: rows[1] is rows[0] with its second row replaced."""
    # A column-major float2x3 holds 3 columns of 2 rows: word 2 * column + row.
    second = list(initial[:6])
    for column, value in enumerate((-1.0, -2.0, -3.0)):
        second[2 * column + 1] = value
    return initial[:6] + second


def generate_crossgl(code, **options):
    ast = HLSLParser(HLSLLexer(textwrap.dedent(code).strip()).tokenize()).parse()
    return DirectxCrossGLCodeGen.HLSLToCrossGLConverter(**options).generate(ast)


def generate_metal(code, **options):
    crossgl = generate_crossgl(code, **options)
    return MetalCodeGen().generate(
        CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse()
    )


def run(command):
    completed = subprocess.run(
        [str(part) for part in command], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def metal_compiler():
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return None
    probe = subprocess.run(
        [xcrun, "-sdk", "macosx", "-f", "metal"], capture_output=True, text=True
    )
    return xcrun if probe.returncode == 0 else None


def compile_metal(source, tmp_path):
    xcrun = metal_compiler()
    if xcrun is None:
        return None
    shader = tmp_path / "shader.metal"
    shader.write_text(source, encoding="utf-8")
    air = tmp_path / "shader.air"
    library = tmp_path / "shader.metallib"
    run([xcrun, "-sdk", "macosx", "metal", "-Werror", "-c", shader, "-o", air])
    run([xcrun, "-sdk", "macosx", "metallib", air, "-o", library])
    return library


def test_crossgl_spells_matrix_elements_by_their_shape():
    crossgl = generate_crossgl(MATRICES_HLSL)
    # HLSL float2x3 has 2 rows of 3 columns: CrossGL mat3x2, not the ambiguous float2x3.
    assert "RWStructuredBuffer<mat3x2> rows;" in crossgl
    assert "StructuredBuffer<mat4> mats;" in crossgl
    row_major = generate_crossgl(MATRICES_HLSL, matrix_packing="row_major")
    assert "RWStructuredBuffer<mat2x3> rows;" in row_major


def test_orientation_qualifiers_on_matrix_elements_are_refused():
    with pytest.raises(ImportRefused, match="orientation qualifier"):
        generate_crossgl("""
            StructuredBuffer<row_major float4x4> mats : register(t0);
            RWStructuredBuffer<float> o : register(u0);
            [numthreads(1, 1, 1)]
            void main(uint3 id : SV_DispatchThreadID) { o[0] = mats[0][3][2]; }
            """)


def test_row_major_packing_inverts_explicit_orientations():
    # With -Zpr packing CrossGL holds the transpose of every HLSL matrix, so the
    # stored rows of a row_major matrix are CrossGL columns.
    crossgl = generate_crossgl(
        """
        cbuffer Params : register(b0) { row_major float2x3 r; column_major float3x2 c; };
        RWStructuredBuffer<float> o : register(u0);
        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) { o[0] = r[1][2] + c[2][0]; }
        """,
        matrix_packing="row_major",
    )
    assert re.search(r"@ column_major\s+mat2x3 r;", crossgl), crossgl
    assert re.search(r"@ row_major\s+mat3x2 c;", crossgl), crossgl
    default = generate_crossgl("""
        cbuffer Params : register(b0) { row_major float2x3 r; };
        RWStructuredBuffer<float> o : register(u0);
        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) { o[0] = r[1][2]; }
        """)
    assert re.search(r"@ row_major\s+mat3x2 r;", default), default


def test_metal_holds_structured_buffer_matrices_as_words(tmp_path):
    metal = generate_metal(MATRICES_HLSL)
    assert re.search(
        r"struct crosstl_hlsl_structured_float3x3 \{\s*float v\[9\];\s*\};", metal
    )
    assert "struct crosstl_hlsl_structured_float3x2_row_major {" in metal
    assert "const device crosstl_hlsl_structured_float4x4* mats" in metal
    assert "device crosstl_hlsl_structured_float3x2* rows" in metal
    # HLSL m[r][c] is Metal column c, row r; the element is loaded once.
    assert "crosstl_hlsl_structured_float3x3_load(items[0].m)[1][2]" in metal
    assert "crosstl_hlsl_structured_float4x4_load(mats[1])[2][3]" in metal
    assert "rows[1] = crosstl_hlsl_structured_float3x2_store(" in metal
    compile_metal(metal, tmp_path)


def test_metal_converts_a_subscripted_vector_element_once(tmp_path):
    metal = generate_metal("""
        struct S { float a; float3 p; bool2 q; };
        StructuredBuffer<float3> pos : register(t0);
        StructuredBuffer<float3> bufs[2] : register(t1);
        RWStructuredBuffer<float> o : register(u0);
        RWStructuredBuffer<uint3> cnt : register(u1);
        RWStructuredBuffer<bool2> flags : register(u2);
        RWStructuredBuffer<S> s : register(u3);
        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID)
        {
            o[0] = pos[1][2] + bufs[1][0][2];
            cnt[1][2] = cnt[0][1] + 4u;
            cnt[0][2] += 3u;
            flags[1][0] = pos[0].x > 1.0;
            s[0].p[1] = s[1].p[2] + 4.0;
            s[1].q[0] = s[0].q[1];
        }
        """)
    assert "float3(pos[1])[2] + float3(bufs[1][0])[2]" in metal
    assert "cnt[1][2] = uint3(cnt[0])[1] + 4u;" in metal
    assert "cnt[0][2] += 3u;" in metal
    assert "flags[1][0] = uint(float3(pos[0]).x > 1.0);" in metal
    assert "s[0].p[1] = float3(s[1].p)[2] + 4.0;" in metal
    assert "s[1].q[0] = uint(bool2(s[0].q)[1]);" in metal
    compile_metal(metal, tmp_path)


def test_metal_refuses_partial_stores_to_structured_buffer_matrices():
    with pytest.raises(ValueError, match="part of HLSL structured-buffer matrix 'm'"):
        generate_metal("""
            struct Item { float s; float3x3 m; };
            RWStructuredBuffer<Item> items : register(u0);
            [numthreads(1, 1, 1)]
            void main(uint3 id : SV_DispatchThreadID) { items[0].m[2][0] = 5.0; }
            """)
    with pytest.raises(ValueError, match="matrix element of HLSL structured buffer"):
        generate_metal("""
            RWStructuredBuffer<float2x2> mats : register(u0);
            [numthreads(1, 1, 1)]
            void main(uint3 id : SV_DispatchThreadID) { mats[1][0].y += 2.0; }
            """)


def test_structured_buffer_matrices_execute_on_metal(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1" or sys.platform != "darwin":
        pytest.skip(f"set {REQUIRE_ENV}=1 on macOS to require the Metal dispatch")
    metal = generate_metal(MATRICES_HLSL)
    library = compile_metal(metal, tmp_path)
    assert library is not None, "the Metal toolchain is required"
    kernel = re.search(r"kernel void (\w+)\((.*?)\) \{", metal, re.S)
    slots = {
        name: int(index)
        for name, index in re.findall(
            r"(\w+)\s+\[\[buffer\((\d+)\)\]\]", kernel.group(2)
        )
    }
    assert sorted(slots.values()) == [0, 1, 2, 3], slots
    expected = expected_values()
    rows = [10.0 + index for index in range(12)]
    payloads = {
        "items": struct.pack(
            f"<{2 * ITEM_WORDS}f",
            *(item_word(k, i) for k in range(2) for i in range(ITEM_WORDS)),
        ),
        "mats": struct.pack("<32f", *(mats_word(i) for i in range(32))),
        "outv": struct.pack("<f", -1234.0) * len(expected),
        "rows": struct.pack("<12f", *rows),
    }
    files = []
    for name, index in sorted(slots.items(), key=lambda item: item[1]):
        path = tmp_path / f"buffer-{index}.in"
        path.write_bytes(payloads[name])
        files.append(str(path))
    runner = tmp_path / "runner"
    run(
        [
            "swiftc",
            ROOT / "tests/fixtures/runtime_verification/metal_raw_buffers.swift",
            "-o",
            runner,
        ]
    )
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": files,
                "workgroupCount": [1, 1, 1],
                "workgroupSize": [1, 1, 1],
                "simdWidth": 32,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    run([runner, library, kernel.group(1), request, output])
    values = struct.unpack(
        f"<{len(expected)}f", (output / f"buffer-{slots['outv']}.bin").read_bytes()
    )
    assert list(values) == pytest.approx(expected), values
    written = struct.unpack(
        "<12f", (output / f"buffer-{slots['rows']}.bin").read_bytes()
    )
    assert list(written) == expected_rows(rows)
