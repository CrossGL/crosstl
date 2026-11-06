"""HLSL byte-address buffers: typed accesses import as word accesses, and GLSL and Metal bind uint word buffers.

``Load<T>``/``Store<T>`` and the value-deduced ``Store(a, v)`` move the bytes of ``T`` as they are, so the importer
writes them as word accesses plus a bit cast. GLSL and Metal have no byte-addressed view and lower every access to
indexing a uint storage buffer at ``address >> 2``; the Metal kernel runs on the GPU against values computed here.
"""

from __future__ import annotations

import re
import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.translator.diagnostics import TranslationDiagnostic
from crosstl.translator.semantics.atomic_forms import canonicalize_atomic_calls
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources

TYPED = """
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint base = id.x * 64;
    float4 f = src.Load<float4>(base);
    int2 i = src.Load<int2>(base + 16);
    bool b = src.Load<bool>(base + 24);
    uint3 u = src.Load3(base + 28);
    dst.Store<float2>(0, f.zw);
    dst.Store(8, f.x + f.y);
    dst.Store(12, i.x * 2);
    dst.Store(16, b);
    dst.Store3(20, u + 1);
    dst.Store<int>(32, i.y);
    int previous;
    dst.InterlockedAdd(36, -3, previous);
    dst.Store(40, previous);
    uint bytes;
    dst.GetDimensions(bytes);
    dst.Store(44, bytes);
    dst.InterlockedMax(48, 7u);
    dst.Store(52, asuint(f.w) + countbits(u.z));
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.hlsl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _kernel(body, declarations=""):
    return (
        "ByteAddressBuffer src : register(t0);\n"
        "RWByteAddressBuffer dst : register(u0);\n"
        f"{declarations}\n"
        "[numthreads(1, 1, 1)]\n"
        "void main(uint3 id : SV_DispatchThreadID)\n"
        f"{{\n{body}\n}}\n"
    )


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _validate_glsl(tmp_path, glsl):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")
    artifact = tmp_path / "shader.comp"
    artifact.write_text(glsl, encoding="utf-8")
    result = subprocess.run(
        [glslang, str(artifact)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr + glsl


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        ("float4 v = src.Load<float4>(4);", "uintBitsToFloat(buffer_load4(src, 4))"),
        ("float v = src.Load<float>(4);", "uintBitsToFloat(buffer_load(src, 4))"),
        ("int3 v = src.Load<int3>(4);", "ivec3(buffer_load3(src, 4))"),
        ("bool2 v = src.Load<bool2>(4);", "bvec2(buffer_load2(src, 4))"),
        ("uint v = src.Load<uint>(4);", "buffer_load(src, 4)"),
        (
            "float2 v = src.Load<vector<float, 2> >(4);",
            "uintBitsToFloat(buffer_load2(src, 4))",
        ),
        (
            "dst.Store<float2>(8, float2(1.0, 2.0));",
            "buffer_store2(dst, 8, floatBitsToUint(vec2(1.0, 2.0)))",
        ),
        ("dst.Store(8, 1.5f);", "buffer_store(dst, 8, floatBitsToUint(1.5))"),
        ("dst.Store(8, id.x > 0);", "buffer_store(dst, 8, uint(id.x > 0))"),
        ("dst.Store(8, -1);", "buffer_store(dst, 8, uint(-1))"),
        ("dst.Store(8, asuint(2.0f));", "buffer_store(dst, 8, floatBitsToUint(2.0))"),
        ("dst.Store(8, asuint(-2));", "buffer_store(dst, 8, asuint(-2))"),
        ("dst.Store<int>(8, 3u);", "buffer_store(dst, 8, uint(int(3u)))"),
        ("dst.Store2(8, uint2(1, 2));", "buffer_store2(dst, 8, uvec2(1, 2))"),
    ],
)
def test_typed_accesses_import_as_word_accesses(tmp_path, statement, expected):
    crossgl = _translate(tmp_path, _kernel(statement), "cgl")

    assert expected in crossgl


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        ("half v = src.Load<half>(0);", "Load<half>"),
        ("float4x4 v = src.Load<float4x4>(0);", "Load<float4x4>"),
        ("Item item = src.Load<Item>(0);", "Load<Item>"),
        ("Item item; item.value = 1.0; dst.Store(0, item);", "Store of a Item"),
        ("Item item; item.value = 1.0; dst.Store<Item>(0, item);", "Store<Item>"),
        ("double v = 1.0; dst.Store(0, v);", "Store of a double"),
    ],
)
def test_other_byte_address_types_are_refused(tmp_path, statement, message):
    source = _kernel(statement, "struct Item { float value; };")

    with pytest.raises(TranslationDiagnostic, match=re.escape(message)):
        _translate(tmp_path, source, "cgl")


def test_byte_address_atomics_keep_their_three_operand_form():
    # atomicMax(buffer, address, value) is the byte-address form, not atomicMax(dest, value, original).
    register_default_sources()
    program = SOURCE_REGISTRY.get("cgl").parse("""
        shader main {
            RWByteAddressBuffer dst;
            compute {
                void main() {
                    uint seen = 0u;
                    atomicMax(dst, 4u, seen);
                }
            }
        }
        """)

    assert canonicalize_atomic_calls(program, "opengl") == 0


def test_opengl_binds_uint_word_buffers(tmp_path):
    glsl = _translate(tmp_path, TYPED, "opengl")

    assert re.search(r"readonly buffer \w+ \{\s*uint src\[\];", glsl)
    assert re.search(r"buffer \w+ \{\s*uint dst\[\];", glsl)
    assert "ByteAddressBuffer" not in glsl
    assert "buffer_load" not in glsl and "buffer_store" not in glsl
    assert "uintBitsToFloat(uvec4(src[(base >> 2u)]" in glsl
    assert "previous = int(atomicAdd(dst[9u], uint((-3))));" in glsl
    assert "bytes = (dst.length() * 4u);" in glsl
    _validate_glsl(tmp_path, glsl)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            "uint k = id.x; uint2 v = src.Load2(k++); dst.Store(0, v.x);",
            "buffer_load2 on a byte-address buffer with a byte address that has side effects",
        ),
        (
            "dst.InterlockedMin(0, 1.5f);",
            "atomicMin on a byte-address buffer with a value that is neither int nor uint",
        ),
    ],
)
@pytest.mark.parametrize("backend", ["opengl", "metal"])
def test_unlowerable_accesses_are_refused(tmp_path, backend, body, message):
    with pytest.raises(TranslationDiagnostic, match=re.escape(message)) as raised:
        _translate(tmp_path, _kernel(body), backend)

    assert "passed to" not in str(raised.value)


def _run_metal(source, buffers, threads=1):
    Metal = pytest.importorskip("Metal")
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    library, error = device.newLibraryWithSource_options_error_(
        source, Metal.MTLCompileOptions.new(), None
    )
    assert library is not None, error
    entry = re.search(r"kernel void (\w+)\(", source).group(1)
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_(entry), None
    )
    assert pipeline is not None, error
    indices = {
        name: int(index)
        for name, index in re.findall(r"(\w+)\s*\[\[buffer\((\d+)\)\]\]", source)
    }
    shared = Metal.MTLResourceStorageModeShared
    native = {
        name: device.newBufferWithBytes_length_options_(data, len(data), shared)
        for name, data in buffers.items()
    }
    assert set(native) == set(indices), source
    command = device.newCommandQueue().commandBuffer()
    encoder = command.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    for name, buffer in native.items():
        encoder.setBuffer_offset_atIndex_(buffer, 0, indices[name])
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(threads, 1, 1), Metal.MTLSizeMake(threads, 1, 1)
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    assert command.error() is None, command.error()
    return {
        name: bytes(buffer.contents().as_buffer(len(buffers[name])))
        for name, buffer in native.items()
    }


def test_metal_moves_typed_words_on_the_gpu(tmp_path):
    metal = _translate(tmp_path, TYPED, "metal")
    floats = (1.5, -2.25, 3.0, 0.125)
    # Word 6 is 2: Load<bool> compares the word with zero, so an even value still loads as true.
    source_words = [*(_bits(value) for value in floats), (-7) & 0xFFFFFFFF, 123456, 2]
    source_words += [10, 20, 0xFFFFFFFF]
    destination = [0] * 16
    destination[9] = 10
    destination[12] = 3

    results = _run_metal(
        metal,
        {
            "src": struct.pack(f"<{len(source_words)}I", *source_words),
            "dst": struct.pack("<16I", *destination),
            "dstLength": struct.pack("<I", 16),
        },
    )

    assert list(struct.unpack("<16I", results["dst"])) == [
        _bits(3.0),
        _bits(0.125),
        _bits(1.5 + -2.25),
        (-14) & 0xFFFFFFFF,
        1,
        11,
        21,
        0,
        123456,
        7,
        10,
        64,
        7,
        _bits(0.125) + 32,
        0,
        0,
    ], metal


SIGNED_ORDERED = """
RWByteAddressBuffer dst : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    int lane = int(id.x);
    dst.InterlockedMax(0, lane - 40);
    dst.InterlockedMin(4, 20 - lane);
    if (lane == 0) {
        int low = -2;
        dst.InterlockedMin(8, low);
        int before;
        dst.InterlockedMax(12, low, before);
        dst.Store(16, before);
        uint word;
        dst.InterlockedMax(20, -9, word);
        dst.Store(24, word);
        float converted;
        dst.InterlockedMin(28, -1, converted);
        dst.Store(32, asuint(converted));
    }
}
"""


def _cgl_kernel(body):
    return (
        "shader main {\n    @ register(u0)\n    RWByteAddressBuffer dst;\n    compute {\n"
        "    @ numthreads(1, 1, 1)\n    void main(uvec3 id @ gl_GlobalInvocationID) {\n"
        f"{body}\n    }}\n    }}\n}}\n"
    )


def _translate_cgl(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_signed_min_max_become_compare_exchange_loops(tmp_path):
    glsl = _translate(tmp_path, SIGNED_ORDERED, "opengl")

    assert "atomicMin(" not in glsl and "atomicMax(" not in glsl
    assert (
        "uint(max(int(crosstl_byte_address_expected), crosstl_byte_address_value))"
        in glsl
    )
    assert (
        "uint(min(int(crosstl_byte_address_expected_1), crosstl_byte_address_value_1))"
        in glsl
    )
    assert "before = int(crosstl_byte_address_expected_3);" in glsl
    assert "word = crosstl_byte_address_expected_4;" in glsl
    # The signed overload's original value is an int, so -1 converts to -1.0 rather than 4294967295.0.
    assert "converted = float(int(crosstl_byte_address_expected_5));" in glsl
    _validate_glsl(tmp_path, glsl)


def test_metal_signed_min_max_compare_as_signed_on_the_gpu(tmp_path):
    metal = _translate(tmp_path, SIGNED_ORDERED, "metal")
    initial = [(-100) & 0xFFFFFFFF, 100, 5, (-7) & 0xFFFFFFFF, 0, 3, 0, 0xFFFFFFFF, 0]

    results = _run_metal(
        metal,
        {"dst": struct.pack("<9I", *initial)},
        threads=64,
    )

    # An unsigned comparison would keep 0xFFFFFF9C in word 0, 100 in word 1, 5 in word 2 and choose 0xFFFFFFF7
    # for word 5; word 7 already holds the value, so the first exchange succeeds and leaves it.
    assert list(struct.unpack("<9i", results["dst"])) == [
        23,
        20 - 63,
        -2,
        -2,
        -7,
        3,
        3,
        -1,
        _signed(_bits(-1.0)),
    ], metal


def _signed(word):
    return struct.unpack("<i", struct.pack("<I", word))[0]


@pytest.mark.parametrize("backend", ["opengl", "metal"])
def test_signed_min_max_declarations_splice_the_loop_into_their_block(
    tmp_path, backend
):
    body = (
        "        int base = 4;\n"
        "        const int first = atomicMax(dst, base, -3);\n"
        "        int second = atomicMin(dst, base + 4, first);\n"
        "        buffer_store(dst, 16, first + second);"
    )

    code = _translate_cgl(tmp_path, _cgl_kernel(body), backend)

    assert "atomicMax" not in code and "atomicMin" not in code
    assert re.search(r"const int first = int\(crosstl_byte_address_expected\);", code)
    assert re.search(r"int second = int\(crosstl_byte_address_expected_1\);", code)
    assert re.search(r"crosstl_byte_address_word = \(?uint\(base\) >> 2u\)?;", code)
    if backend == "opengl":
        _validate_glsl(tmp_path, code)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            "        int total = 1 + atomicMin(dst, 0, -2);",
            "atomicMin on a byte-address buffer with an int value inside an expression",
        ),
        (
            "        for (int seen = atomicMax(dst, 0, -2); seen < 0; seen++) { }",
            "atomicMax on a byte-address buffer with an int value inside an expression",
        ),
    ],
)
@pytest.mark.parametrize("backend", ["opengl", "metal"])
def test_signed_min_max_outside_a_statement_are_refused(
    tmp_path, backend, body, message
):
    with pytest.raises(TranslationDiagnostic, match=re.escape(message)):
        _translate_cgl(tmp_path, _cgl_kernel(body), backend)
