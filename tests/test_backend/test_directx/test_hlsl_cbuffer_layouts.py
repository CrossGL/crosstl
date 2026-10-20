"""HLSL constant-buffer layouts on OpenGL and Metal.

DXC packs constant buffers into 16-byte registers: a value only moves to the
next register when it would straddle one, every row of a row-major matrix and
every array element starts a register, and the next member may pack directly
after the last row or element. The offsets below are written out by hand from
those rules (``-Zpr`` row-major packing, as ComputeSharp compiles), so they do
not depend on the planners under test. The native checks fill every byte the
host does not write with ``0xCD``, so a value read from any other offset
changes the result.
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

import crosstl

REQUIRE_ENV = "CROSSTL_REQUIRE_HLSL_CBUFFER_EXECUTION"
ROOT = Path(__file__).resolve().parents[3]
SOURCE_OPTIONS = {"matrix_packing": "row_major"}
FILL = 0xCD

MATRICES_HLSL = """
    struct Inner
    {
        float2x3 m;
        int k;
        bool2 flags;
    };

    cbuffer Constants : register(b0)
    {
        uint first;
        int2 pair;
        float3 triple;
        int single;
        float list[2];
        float2x3 rows;
        int packed;
        int2x2 imat;
        bool flag;
        float3x3 square;
        float tail;
        Inner inner;
        float afterInner;
        uint2x3 umat;
        float2x3 mats[2];
        float4 last;
    }

    RWStructuredBuffer<uint> results : register(u0);

    [numthreads(1, 1, 1)]
    void CSMain(uint3 id : SV_DispatchThreadID)
    {
        uint i = id.x;
        uint n = 0;
        results[n++] = first;
        results[n++] = asuint(pair.x);
        results[n++] = asuint(pair.y);
        results[n++] = asuint(triple.x);
        results[n++] = asuint(triple.y);
        results[n++] = asuint(triple.z);
        results[n++] = asuint(single);
        results[n++] = asuint(list[i]);
        results[n++] = asuint(list[i + 1]);
        for (int r = 0; r < 2; r++)
            for (int c = 0; c < 3; c++)
                results[n++] = asuint(rows[r][c]);
        results[n++] = asuint(packed);
        for (int r2 = 0; r2 < 2; r2++)
            for (int c2 = 0; c2 < 2; c2++)
                results[n++] = asuint(imat[r2][c2]);
        results[n++] = flag ? 1u : 0u;
        for (int r3 = 0; r3 < 3; r3++)
            for (int c3 = 0; c3 < 3; c3++)
                results[n++] = asuint(square[r3][c3]);
        results[n++] = asuint(tail);
        Inner copy = inner;
        for (int r4 = 0; r4 < 2; r4++)
            for (int c4 = 0; c4 < 3; c4++)
                results[n++] = asuint(copy.m[r4][c4]);
        results[n++] = asuint(inner.k);
        results[n++] = inner.flags.x ? 1u : 0u;
        results[n++] = inner.flags.y ? 1u : 0u;
        results[n++] = asuint(afterInner);
        for (int r5 = 0; r5 < 2; r5++)
            for (int c5 = 0; c5 < 3; c5++)
                results[n++] = umat[r5][c5];
        for (int e = 0; e < 2; e++)
            for (int r6 = 0; r6 < 2; r6++)
                for (int c6 = 0; c6 < 3; c6++)
                    results[n++] = asuint(mats[i + e][r6][c6]);
        results[n++] = asuint(last.x);
        results[n++] = asuint(last.w);
    }
"""

# (kind, HLSL byte offset) of every value the kernel writes, in its order.
MATRICES_FIELDS = (
    [("u", 0), ("i", 4), ("i", 8), ("f", 16), ("f", 20), ("f", 24), ("i", 28)]
    + [("f", 32), ("f", 48)]
    + [("f", 64 + 16 * r + 4 * c) for r in range(2) for c in range(3)]
    + [("i", 92)]
    + [("i", 96 + 16 * r + 4 * c) for r in range(2) for c in range(2)]
    + [("b", 120)]
    + [("f", 128 + 16 * r + 4 * c) for r in range(3) for c in range(3)]
    + [("f", 172)]
    + [("f", 176 + 16 * r + 4 * c) for r in range(2) for c in range(3)]
    + [("i", 204), ("b", 208), ("b", 212), ("f", 216)]
    + [("u", 224 + 16 * r + 4 * c) for r in range(2) for c in range(3)]
    + [
        ("f", 256 + 32 * e + 16 * r + 4 * c)
        for e in range(2)
        for r in range(2)
        for c in range(3)
    ]
    + [("f", 320), ("f", 332)]
)
MATRICES_SIZE = 336

# Arrays of scalars end after their last element, doubles pack at eight
# bytes and a straddling bool3 moves to the next register.
PACKED_HLSL = """
    cbuffer Packed : register(b0)
    {
        float values[3];
        float next;
        double wide;
        int2 tailPair;
        bool3 bits;
        float end;
    }

    RWStructuredBuffer<uint> results : register(u0);

    [numthreads(1, 1, 1)]
    void CSMain(uint3 id : SV_DispatchThreadID)
    {
        uint i = id.x;
        results[0] = asuint(values[i]);
        results[1] = asuint(values[i + 1]);
        results[2] = asuint(values[i + 2]);
        results[3] = asuint(next);
        results[4] = asuint((float)wide);
        results[5] = asuint(tailPair.x);
        results[6] = asuint(tailPair.y);
        results[7] = bits.x ? 1u : 0u;
        results[8] = bits.y ? 1u : 0u;
        results[9] = bits.z ? 1u : 0u;
        results[10] = asuint(end);
    }
"""

PACKED_FIELDS = [
    ("f", 0),
    ("f", 16),
    ("f", 32),
    ("f", 36),
    ("d", 40),
    ("i", 48),
    ("i", 52),
    ("b", 64),
    ("b", 68),
    ("b", 72),
    ("f", 76),
]
PACKED_SIZE = 80


def constants_and_expected(fields, size):
    """The constant-buffer bytes and the words the kernel must write."""
    data = bytearray([FILL] * size)
    expected = []
    for index, (kind, offset) in enumerate(fields):
        if kind == "f":
            value = (index + 1) * 0.375 * (-1) ** index
            struct.pack_into("<f", data, offset, value)
            expected.append(struct.unpack("<I", struct.pack("<f", value))[0])
        elif kind == "d":
            value = -(index + 1) * 1.5
            struct.pack_into("<d", data, offset, value)
            expected.append(struct.unpack("<I", struct.pack("<f", value))[0])
        elif kind == "i":
            value = -(index + 1) * 7
            struct.pack_into("<i", data, offset, value)
            expected.append(value & 0xFFFFFFFF)
        elif kind == "u":
            value = (0x9E3779B9 * (index + 1)) & 0xFFFFFFFF
            struct.pack_into("<I", data, offset, value)
            expected.append(value)
        else:
            value = index % 2
            struct.pack_into("<I", data, offset, value)
            expected.append(value)
    return bytes(data), expected


def translate(tmp_path, source, backend):
    path = tmp_path / "constants.hlsl"
    path.write_text(textwrap.dedent(source).strip() + "\n", encoding="utf-8")
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_options=SOURCE_OPTIONS
    )


def run(command):
    completed = subprocess.run(
        [str(item) for item in command], capture_output=True, text=True, timeout=180
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def validate_glsl(source, tmp_path):
    validator = shutil.which("glslangValidator")
    if validator is None:
        return
    shader = tmp_path / "constants.comp"
    shader.write_text(source, encoding="utf-8")
    run([validator, "-S", "comp", shader])


def metal_compiler():
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return None
    probe = subprocess.run(
        [xcrun, "-sdk", "macosx", "-f", "metal"], capture_output=True, text=True
    )
    return xcrun if probe.returncode == 0 else None


def compile_metal(source, tmp_path, checks=()):
    """Compile with static assertions; returns the library, or ``None``."""
    xcrun = metal_compiler()
    if xcrun is None:
        return None
    shader = tmp_path / "constants.metal"
    shader.write_text(source + "\n" + "\n".join(checks) + "\n", encoding="utf-8")
    air, library = tmp_path / "constants.air", tmp_path / "constants.metallib"
    run([xcrun, "-sdk", "macosx", "metal", "-Werror", "-c", shader, "-o", air])
    run([xcrun, "-sdk", "macosx", "metallib", air, "-o", library])
    return library


def required():
    return os.environ.get(REQUIRE_ENV) == "1"


def test_opengl_keeps_std140_blocks_that_match_hlsl(tmp_path):
    glsl = translate(
        tmp_path,
        """
        cbuffer Params : register(b0) { uint x; uint y; uint z; float4 v; float2 w; }
        RWStructuredBuffer<float> o : register(u0);
        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) { o[id.x] = v.y + w.x + x; }
        """,
        "opengl",
    )

    assert "crossgl_cbuffer_registers" not in glsl
    assert re.search(r"uniform Params \{\n    uint x;\n    uint y;", glsl)
    validate_glsl(glsl, tmp_path)


def test_opengl_reads_hlsl_cbuffer_values_from_their_registers(tmp_path):
    glsl = translate(tmp_path, MATRICES_HLSL, "opengl")

    registers = "crossgl_cbuffer_registers"
    assert f"uniform Constants {{\n    uvec4 {registers}[21];\n}};" in glsl
    loader = re.search(r"void (crossgl_load\w*)\(\) \{\n(.*?)\n\}\n", glsl, re.S)
    assert loader is not None
    body = loader.group(2)
    for line in (
        f"pair = ivec2({registers}[0].yz);",
        f"triple = uintBitsToFloat({registers}[1].xyz);",
        f"single = int({registers}[1].w);",
        f"list[crossgl_index] = uintBitsToFloat({registers}[crossgl_index + 2].x);",
        f"rows[1] = uintBitsToFloat({registers}[5].xyz);",
        # Mesa reserves ``packed``.
        f"packed_ = int({registers}[5].w);",
        f"imat.c[1] = ivec2({registers}[7].xy);",
        f"flag = ({registers}[7].z != 0u);",
        f"tail = uintBitsToFloat({registers}[10].w);",
        f"inner.k = int({registers}[12].w);",
        f"inner.flags = notEqual({registers}[13].xy, uvec2(0u));",
        f"umat.c[1] = {registers}[15].xyz;",
        f"mats[crossgl_index][1] = uintBitsToFloat({registers}[crossgl_index * 2 + 17].xyz);",
        f"last = uintBitsToFloat({registers}[20].xyzw);",
    ):
        assert line in body, line
    # Members are private globals, assigned before the entry body runs.
    assert re.search(r"\nmat2x3 rows;\n", glsl)
    main = glsl[glsl.index("void main() {") :]
    assert main.split("\n")[1].strip() == f"{loader.group(1)}();"
    validate_glsl(glsl, tmp_path)


def test_opengl_reads_packed_scalars_and_doubles_from_registers(tmp_path):
    glsl = translate(tmp_path, PACKED_HLSL, "opengl")

    registers = "crossgl_cbuffer_registers"
    assert f"uvec4 {registers}[5];" in glsl
    for line in (
        f"next = uintBitsToFloat({registers}[2].y);",
        f"wide = packDouble2x32({registers}[2].zw);",
        f"tailPair = ivec2({registers}[3].xy);",
        f"bits = notEqual({registers}[4].xyz, uvec3(0u));",
        f"end = uintBitsToFloat({registers}[4].w);",
    ):
        assert line in glsl, line
    validate_glsl(glsl, tmp_path)


def test_opengl_rejects_hlsl_cbuffers_without_a_layout(tmp_path):
    with pytest.raises(ValueError, match="HLSL layout of cbuffer 'Params' in OpenGL"):
        translate(
            tmp_path,
            """
            cbuffer Params : register(b0) { float grid[2][2]; float after; }
            RWStructuredBuffer<float> o : register(u0);
            [numthreads(1, 1, 1)]
            void main(uint3 id : SV_DispatchThreadID) { o[id.x] = grid[1][0] + after; }
            """,
            "opengl",
        )


def test_metal_reproduces_hlsl_cbuffer_matrices(tmp_path):
    metal = translate(tmp_path, MATRICES_HLSL, "metal")

    # Matrices without a matching native layout are register words.
    assert "    crosstl_hlsl_cbuffer_float2x3 rows;\n" in metal
    assert "    crosstl_hlsl_cbuffer_int2x2 imat;\n" in metal
    assert "    crosstl_hlsl_cbuffer_float3x3 square;\n" in metal
    assert "    crosstl_hlsl_cbuffer_float2x3_padded mats[2];\n" in metal
    assert "struct crosstl_hlsl_cbuffer_float2x3 {\n    float v[7];\n};" in metal
    assert "result[1] = float3(m.v[4], m.v[5], m.v[6]);" in metal
    assert "crosstl_hlsl_cbuffer_float2x3_load(constants.rows)" in metal
    offsets = {
        "first": 0,
        "pair": 4,
        "triple": 16,
        "single": 28,
        "list": 32,
        "rows": 64,
        "packed": 92,
        "imat": 96,
        "flag": 120,
        "square": 128,
        "tail": 172,
        "inner": 176,
        "afterInner": 216,
        "umat": 224,
        "mats": 256,
        "last": 320,
    }
    checks = [
        f"static_assert(__builtin_offsetof(Constants, {name}) == {offset}, "
        f'"{name}");'
        for name, offset in offsets.items()
    ]
    checks += [
        'static_assert(__builtin_offsetof(Inner, k) == 28, "Inner.k");',
        'static_assert(__builtin_offsetof(Inner, flags) == 32, "Inner.flags");',
        'static_assert(sizeof(crosstl_hlsl_cbuffer_float2x3_padded) == 32, "padded");',
    ]
    compile_metal(metal, tmp_path, checks)


def test_metal_rejects_partial_stores_to_register_matrices(tmp_path):
    with pytest.raises(ValueError, match="part of HLSL constant-buffer matrix 'm'"):
        translate(
            tmp_path,
            """
            struct Inner { float2x3 m; float k; };
            cbuffer Params : register(b0) { Inner inner; }
            RWStructuredBuffer<float> o : register(u0);
            [numthreads(1, 1, 1)]
            void main(uint3 id : SV_DispatchThreadID)
            {
                Inner copy = inner;
                copy.m[1][2] = 4.0;
                o[id.x] = copy.m[1][2] + copy.k;
            }
            """,
            "metal",
        )


def test_hlsl_cbuffer_layouts_execute_on_metal(tmp_path):
    if not required() or sys.platform != "darwin":
        pytest.skip(f"set {REQUIRE_ENV}=1 on macOS to require the Metal dispatch")
    metal = translate(tmp_path, MATRICES_HLSL, "metal")
    library = compile_metal(metal, tmp_path)
    assert library is not None, "the Metal toolchain is required"
    constants, expected = constants_and_expected(MATRICES_FIELDS, MATRICES_SIZE)
    kernel = re.search(r"kernel void (\w+)\((.*?)\) \{", metal, re.S)
    slots = {
        name: int(index)
        for name, index in re.findall(r"(\w+) \[\[buffer\((\d+)\)\]\]", kernel.group(2))
    }
    assert sorted(slots.values()) == [0, 1], slots
    payloads = {
        "constants": constants,
        "results": struct.pack("<I", 0xDEADBEEF) * len(expected),
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
    data = (output / f"buffer-{slots['results']}.bin").read_bytes()
    assert list(struct.unpack(f"<{len(expected)}I", data)) == expected
    assert (output / f"buffer-{slots['constants']}.bin").read_bytes() == constants


@pytest.mark.parametrize(
    "source,fields,size",
    [
        (MATRICES_HLSL, MATRICES_FIELDS, MATRICES_SIZE),
        (PACKED_HLSL, PACKED_FIELDS, PACKED_SIZE),
    ],
    ids=["matrices", "packed"],
)
def test_hlsl_cbuffer_layouts_execute_on_opengl(tmp_path, source, fields, size):
    if not required() or not sys.platform.startswith("linux"):
        pytest.skip(f"set {REQUIRE_ENV}=1 on Linux to require the OpenGL dispatch")
    import moderngl  # noqa: PLC0415 - required only for the native dispatch

    glsl = translate(tmp_path, source, "opengl")
    constants, expected = constants_and_expected(fields, size)
    context = moderngl.create_standalone_context(backend="egl", require=430)
    try:
        shader = context.compute_shader(glsl)
        uniforms = context.buffer(constants)
        uniforms.bind_to_uniform_block(0)
        results = context.buffer(struct.pack("<I", 0xDEADBEEF) * len(expected))
        results.bind_to_storage_buffer(0)
        shader.run(1, 1, 1)
        context.finish()
        data = results.read()
    finally:
        context.release()
    assert list(struct.unpack(f"<{len(expected)}I", data)) == expected
