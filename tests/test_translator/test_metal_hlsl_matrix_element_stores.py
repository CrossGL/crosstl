"""HLSL structured-buffer matrices stored one element at a time in Metal.

An HLSL structured buffer packs a matrix's columns, or its rows when ``row_major``, at four-byte alignment, so the
Metal output holds the matrix as its words and reads and writes it through ``_load`` and ``_store`` helpers. A store
to one element, ``m[r][c] += v`` or ``m[r].y = v``, used to be refused ("it is held as words"). The element is one of
those words, so the Metal output now writes ``storage.v[index]`` in place, at the index the ``_store`` helper gives
that element, with any index expression evaluated once.

The kernel below stores elements of a column-major ``RWStructuredBuffer<float2x2>``, of a ``row_major float3x2``
struct member, of an element of a ``row_major float2x2`` array member and of a column-major ``float2x3`` member,
with constant and run-time indices and with ``=``, ``+=``, ``-=``, ``*=`` and ``++``. The expected words follow the
HLSL packing rules (column-major by default) and are computed here independently of CrossTL.
"""

import re
import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.metal_hlsl_buffer_layout import MetalHlslBufferLayouts
from crosstl.translator.refusals import MetalRefusal

HLSL = """
struct Rows {
    row_major float3x2 m;
    float2 w;
};
struct Pairs {
    row_major float2x2 m[2];
    float2x3 c;
};
RWStructuredBuffer<float> fin : register(u0);
RWStructuredBuffer<int> iin : register(u1);
RWStructuredBuffer<float2x2> mats : register(u2);
RWStructuredBuffer<Rows> rows : register(u3);
RWStructuredBuffer<Pairs> pairs : register(u4);

[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    int k = iin[i] & 1;
    float x = fin[i];
    mats[i] = float2x2(x, 2.0, 3.0, 4.0);
    mats[i][k][1 - k] += 0.5;
    mats[i][1][0] = mats[i][0][1] * 2.0;
    rows[i].m = float3x2(x, 1.0, 2.0, 3.0, 4.0, 5.0);
    rows[i].m[2][k] -= 0.25;
    rows[i].m[0].y = 7.0;
    rows[i].w = rows[i].m[1];
    pairs[i].m[k] = float2x2(1.0, 2.0, 3.0, x);
    pairs[i].m[1 - k] = float2x2(5.0, 6.0, 7.0, 8.0);
    pairs[i].m[k][1][0] *= 3.0;
    pairs[i].c = float2x3(x, 1.0, 2.0, 3.0, 4.0, 5.0);
    pairs[i].c[1][2] = pairs[i].c[0][0] + 1.0;
    pairs[i].c[0].z++;
}
"""
FIN = (1.5, -2.0, 0.25, 4.0)
IIN = (0, 1, 2, 3)
# Bytes per element: float2x2 16; Rows 24 + float2 8; Pairs two 16-byte matrices + a 24-byte float2x3.
SIZES = {"mats": 16, "rows": 32, "pairs": 56}


def _column_major(rows):
    """Words of an HLSL matrix given by its rows, packed column by column."""
    return [rows[r][c] for c in range(len(rows[0])) for r in range(len(rows))]


def _row_major(rows):
    return [value for row in rows for value in row]


def _expected():
    """The words the HLSL kernel writes to ``mats``, ``rows`` and ``pairs``, per HLSL semantics."""
    words = {"mats": [], "rows": [], "pairs": []}
    for x, n in zip(FIN, IIN):
        k = n & 1
        m = [[x, 2.0], [3.0, 4.0]]
        m[k][1 - k] += 0.5
        m[1][0] = m[0][1] * 2.0
        words["mats"] += _column_major(m)
        r = [[x, 1.0], [2.0, 3.0], [4.0, 5.0]]
        r[2][k] -= 0.25
        r[0][1] = 7.0
        words["rows"] += _row_major(r) + r[1]
        pair = [None, None]
        pair[k] = [[1.0, 2.0], [3.0, x]]
        pair[1 - k] = [[5.0, 6.0], [7.0, 8.0]]
        pair[k][1][0] *= 3.0
        c = [[x, 1.0, 2.0], [3.0, 4.0, 5.0]]
        c[1][2] = c[0][0] + 1.0
        c[0][2] += 1.0
        words["pairs"] += _row_major(pair[0]) + _row_major(pair[1]) + _column_major(c)
    return words


@pytest.fixture(scope="module")
def metal_source(tmp_path_factory):
    path = tmp_path_factory.mktemp("element_stores") / "probe.hlsl"
    path.write_text(HLSL)
    return crosstl.translate(str(path), backend="metal")


def test_element_stores_write_the_element_word_in_place(metal_source):
    assert "Cannot store" not in metal_source
    # Column-major float2x2: element [r][c] is word 2 * c + r, here with run-time indices evaluated once.
    assert re.search(r"mats\[i\]\.v\[2 \* \(.*k.*\) \+ \(k\)\] \+= 0\.5;", metal_source)
    assert "mats[i].v[1] = " in metal_source
    # row_major float3x2: element [r][c] is word 2 * r + c.
    assert re.search(r"rows\[i\]\.m\.v\[2 \* \(2\) \+ \(k\)\] -= 0\.25;", metal_source)
    assert "rows[i].m.v[1] = 7.0;" in metal_source
    assert "pairs[i].m[k].v[2] *= 3.0;" in metal_source
    # Column-major float2x3 (three columns of two rows): element [1][2] is word 5, [0][2] word 4.
    assert "pairs[i].c.v[5] =" in metal_source
    assert "pairs[i].c.v[4]++;" in metal_source


def test_element_stores_compile_with_the_metal_compiler(metal_source, tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")
    source = tmp_path / "probe.metal"
    source.write_text(metal_source)
    result = subprocess.run(
        [xcrun, "--sdk", "macosx", "metal", "-Werror", "-c", str(source)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode and "unable to find utility" in result.stderr:
        pytest.skip("the Metal toolchain is not installed")
    assert result.returncode == 0, result.stderr


def _dispatch(source):
    metal = pytest.importorskip("Metal")
    device = metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    library, error = device.newLibraryWithSource_options_error_(
        source, metal.MTLCompileOptions.new(), None
    )
    assert library is not None, error
    entry = re.search(r"\bkernel\s+void\s+(\w+)\s*\(", source).group(1)
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_(entry), None
    )
    assert pipeline is not None, error
    inputs = {"fin": struct.pack("<4f", *FIN), "iin": struct.pack("<4i", *IIN)}
    lengths = {name: 4 * size for name, size in SIZES.items()}
    lengths.update({name: len(data) for name, data in inputs.items()})
    buffers = {}
    commands = device.newCommandQueue().commandBuffer()
    encoder = commands.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    for name, index in re.findall(r"\*\s*(\w+)\s*\[\[buffer\((\d+)\)\]\]", source):
        buffer = device.newBufferWithLength_options_(lengths[name], 0)
        if name in inputs:
            buffer.contents().as_buffer(lengths[name])[:] = inputs[name]
        buffers[name] = buffer
        encoder.setBuffer_offset_atIndex_(buffer, 0, int(index))
    assert set(buffers) == set(lengths), source
    encoder.dispatchThreads_threadsPerThreadgroup_(
        metal.MTLSizeMake(4, 1, 1), metal.MTLSizeMake(4, 1, 1)
    )
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    return {
        name: list(
            struct.unpack(
                f"<{SIZES[name]}f", buffers[name].contents().as_buffer(4 * SIZES[name])
            )
        )
        for name in SIZES
    }


def test_element_stores_compute_the_hlsl_results_on_the_gpu(metal_source):
    assert _dispatch(metal_source) == _expected()


@pytest.mark.parametrize(
    ("row_major", "vector_words", "column", "row", "index"),
    [
        (False, 3, "1", "2", "5"),
        (True, 2, "1", "2", "5"),
        (False, 4, "2u", "1", "9"),
        (False, 2, "k", "1", "2 * (k) + (1)"),
        (True, 2, "1", "j + 1", "2 * (j + 1) + (1)"),
    ],
)
def test_matrix_element_word_follows_the_store_helper(
    row_major, vector_words, column, row, index
):
    planner = MetalHlslBufferLayouts(lambda name: name)
    columns, rows = (2, 3) if row_major else (3, 3)
    planner.add_matrix_storage(
        "storage", "float", columns, rows, 9, row_major, vector_words=vector_words
    )
    assert planner.matrix_element_word("storage", column, row) == (index, None)
    assert planner.matrix_element_word("missing", column, row) is None
    if index.isdigit():
        # The store helper writes that element to the same word.
        element = f"value[{column.rstrip('u')}][{row}]"
        assert f"m.v[{index}] = {element};" in planner.matrix_storages["storage"]


def test_bool_matrix_element_words_store_converted_values():
    planner = MetalHlslBufferLayouts(lambda name: name)
    planner.add_matrix_storage("flags", "bool", 2, 2, 4, False, vector_words=2)
    assert planner.matrix_element_word("flags", "1", "0") == ("2", "uint")
    definition = planner.matrix_storages["flags"]
    # The store helper places element [1][0] at word 2, the word the element store writes.
    assert "m.v[2] = uint(value[1][0]);" in definition


def test_column_stores_to_word_held_matrices_still_fail_closed(tmp_path):
    # A column spans several words, so a store to it, whole or through a swizzle, stays a typed refusal.
    source = tmp_path / "columns.cgl"
    source.write_text("""
shader main {
    @ register(u0)
    @ buffer_layout(hlsl_structured)
    RWStructuredBuffer<mat2> mats;
    compute {
    @ numthreads(1, 1, 1)
    void main(uvec3 id @ gl_GlobalInvocationID) {
        mats[id.x][1] = vec2(2.0, 3.0);
    }
    }
}
""")
    with pytest.raises(
        MetalRefusal, match="part of a matrix element of HLSL structured buffer"
    ):
        crosstl.translate(str(source), backend="metal")
