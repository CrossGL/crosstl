"""Metal 8- and 16-bit integer vectors: native storage reads, carriers and wrapping.

The Metal emitter keeps narrow vector values in 32-bit carriers (``uint2`` for ``u16vec2``) while struct fields and
buffer elements keep native ``ushort2`` storage. Metal rejects implicit conversions between vector types, and
CrossGL vector arithmetic keeps its narrow type (vectors do not promote), so it must wrap at the native width.
"""

import shutil
import struct
import subprocess

import pytest

import crosstl.translator
from crosstl.translator.codegen.metal_codegen import MetalCodeGen


def _compute(body):
    return f"""
shader NarrowVectors {{
    struct Pair {{
        u16vec2 narrow;
        i16vec2 signedNarrow;
        u8vec4 bytes;
        uint pad;
    }}
    StructuredBuffer<Pair> pairs;
    StructuredBuffer<u16vec2> vecs;
    RWStructuredBuffer<uint> output;

    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {{
            {body}
        }}
    }}
}}
"""


CONDITIONAL_BODY = """
            uvec2 wide = uvec2(70000u, 3u);
            bool pick = output[0] > 1u;
            uvec2 r = pick ? pairs[0].narrow : wide;
            uvec2 q = pick ? wide : vecs[1];
            u16vec2 n = pairs[0].narrow;
            uvec2 t = pick ? n : wide;
            ivec2 signedWide = ivec2(-1, 2);
            uvec2 mixed = pick ? wide : signedWide;
            vec2 floats = pick ? vec2(0.5) : wide;
            output[0] = r.x + q.y + t.x + mixed.y + uint(floats.x);
"""

ARITHMETIC_BODY = """
            u16vec2 n = pairs[0].narrow;
            u16vec2 m = vecs[0];
            uint16_t s = n.x;
            uvec2 w = uvec2(1u);
            bool sum = (n + m).x > 70000u;
            bool literal = (n + 1).x > 70000u;
            bool scaled = (n * s).x > 70000u;
            bool storage = (pairs[0].narrow + vecs[1]).x > 70000u;
            bool promoted = (n + w).x > 70000u;
            bool scalar = (s + s) > 70000;
            bool negated = (-pairs[0].signedNarrow).x > 0;
            bool inverted = (~n).x > 70000u;
            bool shifted = (n << 4).y > 70000u;
            bool bytes = (pairs[0].bytes * pairs[0].bytes).w > 255u;
            uvec2 d = pairs[0].narrow + w;
            bvec2 equal = pairs[0].narrow == w;
            u16vec2 stored = n + m;
            output[0] = uint(sum) + uint(literal) + uint(scaled) + uint(storage)
                + uint(promoted) + uint(scalar) + uint(negated) + uint(inverted)
                + uint(shifted) + uint(bytes) + d.x + uint(equal.y) + uint(stored.x);
"""


def _generate(body):
    return MetalCodeGen().generate(crosstl.translator.parse(_compute(body)))


def test_conditional_arms_widen_native_storage_reads_to_carriers():
    metal = _generate(CONDITIONAL_BODY)

    assert "ushort2 narrow;" in metal
    assert "const device ushort2* vecs" in metal
    assert "uint2 r = pick ? uint2(pairs[0].narrow) : wide;" in metal
    assert "uint2 q = pick ? wide : uint2(vecs[1]);" in metal
    assert "uint2 t = pick ? n : wide;" in metal


def test_conditional_arms_of_mixed_components_take_the_common_type():
    metal = _generate(CONDITIONAL_BODY)

    assert "uint2 mixed = pick ? wide : uint2(signedWide);" in metal
    assert "float2 floats = pick ? float2(0.5) : float2(wide);" in metal


def test_narrow_vector_arithmetic_wraps_at_the_native_width():
    metal = _generate(ARITHMETIC_BODY)

    assert "bool sum = (uint2(ushort2(n + m))).x > 70000u;" in metal
    assert "bool literal = (uint2(ushort2(n + 1))).x > 70000u;" in metal
    assert "bool scaled = (uint2(ushort2(n * s))).x > 70000u;" in metal
    assert (
        "bool storage = (uint2(ushort2(uint2(pairs[0].narrow) + uint2(vecs[1]))))"
        ".x > 70000u;"
    ) in metal
    assert "bool negated = (int2(short2(-int2(pairs[0].signedNarrow)))).x > 0;" in metal
    assert "bool inverted = (uint2(ushort2(~n))).x > 70000u;" in metal
    assert "bool shifted = (uint2(ushort2(n << 4))).y > 70000u;" in metal
    assert (
        "bool bytes = (uint4(uchar4(uint4(pairs[0].bytes) * uint4(pairs[0].bytes))))"
        ".w > 255u;"
    ) in metal


def test_promoting_operations_do_not_wrap():
    metal = _generate(ARITHMETIC_BODY)

    # A 32-bit operand converts the narrow vector; scalars promote to int.
    assert "bool promoted = (n + w).x > 70000u;" in metal
    assert "bool scalar = s + s > 70000;" in metal
    assert "uint2 d = uint2(pairs[0].narrow) + w;" in metal
    assert "bool2 equal = uint2(pairs[0].narrow) == w;" in metal


def test_narrow_stores_keep_a_single_normalization():
    metal = _generate(ARITHMETIC_BODY)

    assert "uint2 stored = uint2(ushort2(n + m));" in metal
    assert "uint2(ushort2(uint2(ushort2(" not in metal


@pytest.mark.parametrize(
    "body", [CONDITIONAL_BODY, ARITHMETIC_BODY], ids=["conditional", "arithmetic"]
)
def test_narrow_vector_lowering_compiles_with_xcrun_when_available(tmp_path, body):
    if shutil.which("xcrun") is None:
        pytest.skip("xcrun is not installed")
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"],
        capture_output=True,
        check=False,
    )
    if found.returncode != 0:
        pytest.skip("the Metal compiler is not installed")

    source = tmp_path / "narrow.metal"
    source.write_text(_generate(body), encoding="utf-8")
    compiled = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "narrow.air"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert compiled.returncode == 0, compiled.stdout + compiled.stderr


WRAP_PROBE = """
shader NarrowWrap {
    StructuredBuffer<u16vec2> vecs;
    StructuredBuffer<i16vec2> signedVecs;
    RWStructuredBuffer<uint> output;

    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            u16vec2 n = vecs[0];
            u16vec2 m = vecs[1];
            i16vec2 s = signedVecs[0];
            uvec2 w = uvec2(1u);
            output[0] = (n + m).x;
            output[1] = (n * m).y;
            output[2] = (n + w).x;
            output[3] = (~n).y;
            output[4] = (vecs[0] + vecs[1]).x;
            output[5] = uint((s + s).x + 40000);
        }
    }
}
"""

# vecs = [(65535, 300), (1, 300)], signedVecs = [(20000, 0)]. 16-bit vector arithmetic wraps; a 32-bit operand
# promotes the narrow vector first, so (n + w).x does not.
WRAP_EXPECTED = (0, 90000 % 65536, 65536, 0xFFFF & ~300, 0, 40000 - 25536)


def _run_wrap_probe(metal_source):
    Metal = pytest.importorskip("Metal")
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    library, error = device.newLibraryWithSource_options_error_(
        metal_source, Metal.MTLCompileOptions.new(), None
    )
    assert library is not None, error
    function = library.newFunctionWithName_("kernel_main")
    pipeline, error = device.newComputePipelineStateWithFunction_error_(function, None)
    assert pipeline is not None, error

    shared = Metal.MTLResourceStorageModeShared
    inputs = [struct.pack("<4H", 65535, 300, 1, 300), struct.pack("<2h", 20000, 0)]
    buffers = [
        device.newBufferWithBytes_length_options_(data, len(data), shared)
        for data in inputs
    ]
    result_size = 4 * len(WRAP_EXPECTED)
    buffers.append(device.newBufferWithLength_options_(result_size, shared))
    command = device.newCommandQueue().commandBuffer()
    encoder = command.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    for index, buffer in enumerate(buffers):
        encoder.setBuffer_offset_atIndex_(buffer, 0, index)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1), Metal.MTLSizeMake(1, 1, 1)
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    assert command.error() is None, command.error()
    data = bytes(buffers[-1].contents().as_buffer(result_size))
    return struct.unpack(f"<{len(WRAP_EXPECTED)}I", data)


def test_narrow_vector_arithmetic_wraps_on_the_gpu():
    metal = MetalCodeGen().generate(crosstl.translator.parse(WRAP_PROBE))
    assert "const device ushort2* vecs [[buffer(0)]]" in metal
    assert "const device short2* signedVecs [[buffer(1)]]" in metal
    assert "device uint* output [[buffer(2)]]" in metal

    assert _run_wrap_probe(metal) == WRAP_EXPECTED
