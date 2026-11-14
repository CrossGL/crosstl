"""GLSL buffer blocks with 8-, 16- and 64-bit members keep their std430 layout in byte-address lowerings.

GL_EXT_shader_16bit_storage and GL_EXT_shader_8bit_storage give a scalar the base alignment of its own size
(GLSL 4.60 section 7.6.2.2 rule 1), so a ``uint16_t`` occupies 2 bytes and an ``int8_t`` array has a stride of 1.
DirectX, Metal and Slang lower such blocks to byte-addressed memory; each member is read and written at its own
offset and width, and a target that cannot address a width refuses the block.
"""

import re
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

import crosstl
from crosstl.project.directx_toolchain import dxc_compiler_arguments_for_source

HEADER = """#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = 1) in;
"""

# std430: tag@0, weight@2, uv@4 (align 4), big@8, rgb@16 (align 8, 6 bytes); struct align 8, size 24.
# Out: count@0, delta@4, wide@16 (align 16), colour@32 (align 8).
WIDE_AND_HALF = HEADER + """
struct Item { uint16_t tag; float16_t weight; f16vec2 uv; int64_t big; u16vec3 rgb; };
layout(std430, binding = 0) buffer Items { Item items[]; };
layout(std430, binding = 1) buffer Out { uint count; int16_t delta; u64vec2 wide; f16vec4 colour; } outb;
void main() {
    uint index = gl_GlobalInvocationID.x;
    items[index].tag = uint16_t(index);
    items[index].weight += float16_t(0.5);
    items[index].uv.y = items[index].weight;
    items[index].big = items[index].big * 3l;
    items[index].rgb.yz = u16vec2(1us, 2us);
    outb.delta = int16_t(items[index].tag);
    outb.wide.x += uint64_t(items[index].big);
    outb.colour = f16vec4(items[index].uv, items[index].weight, float16_t(1.0));
    outb.count = uint(items[index].rgb.x);
}
"""

# std430: head@0, tail[3]@1 (stride 1), pair@4 (align 2), after@8.
BYTES = HEADER + """
layout(std430, binding = 0) buffer Bytes { uint8_t head; int8_t tail[3]; i8vec2 pair; uint after; } bytes;
void main() {
    bytes.after = uint(bytes.head) + uint(bytes.tail[1]) + uint(bytes.pair.y);
    bytes.tail[2] = int8_t(-5);
}
"""


def _translate(tmp_path, source, backend, name="shader.comp"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _buffer(output, binding, register_prefix="u"):
    match = re.search(
        rf"RWByteAddressBuffer (\w+) : register\({register_prefix}{binding}\)", output
    )
    assert match, output
    return match.group(1)


def test_directx_moves_each_member_at_its_own_width(tmp_path):
    hlsl = _translate(tmp_path, WIDE_AND_HALF, "directx")
    items = _buffer(hlsl, 0)

    # 16-bit members sit 2 bytes apart; a binary16 moves as its uint16_t bits.
    assert f"{items}.Store<uint16_t>((index * 24), uint16_t(uint16_t(index)));" in hlsl
    assert f"asfloat16({items}.Load<uint16_t>((index * 24 + 2)))" in hlsl
    assert f"{items}.Store<uint16_t>((index * 24 + 4 + 2), asuint16(" in hlsl
    # 64-bit members load and store 8 bytes, not one 32-bit word.
    assert (
        f"{items}.Store<int64_t>((index * 24 + 8), int64_t(({items}.Load<int64_t>((index * 24 + 8))"
        in hlsl
    )
    assert "outb.Store<uint64_t>(16, uint64_t((outb.Load<uint64_t>(16) + (" in hlsl
    # A component store of a u16vec3 writes components 1 and 2 at +2 and +4.
    temp = re.search(r"uint16_t2 (\w+) = uint16_t2\(1u, 2u\);", hlsl)
    assert temp, hlsl
    assert (
        f"{items}.Store<uint16_t>((index * 24 + 16 + 2), uint16_t({temp.group(1)}.x));"
        in hlsl
    )
    assert (
        f"{items}.Store<uint16_t>((index * 24 + 16 + 4), uint16_t({temp.group(1)}.y));"
        in hlsl
    )
    assert "outb.Store<int16_t>(4, " in hlsl
    assert "outb.Store<uint16_t4>(32, asuint16(float16_t4(" in hlsl
    assert f"outb.Store(0, uint({items}.Load<uint16_t3>((index * 24 + 16)).x));" in hlsl
    # No narrow or wide member goes through the 32-bit word methods.
    assert not re.search(r"\.(Load|Store)[234]?\((index \* 24|4|16|32)\b", hlsl), hlsl


def test_slang_moves_each_member_at_its_own_width(tmp_path):
    slang = _translate(tmp_path, WIDE_AND_HALF, "slang")

    assert "items_2.Store<uint16_t>((index * 24), uint16_t(uint16_t(index)));" in slang
    assert (
        "items_2.Store<int64_t>((index * 24 + 8), int64_t(items_2.Load<int64_t>((index * 24 + 8)) * 3));"
        in slang
    )
    assert "outb.Store<uint64_t>(16, uint64_t((outb.Load<uint64_t>(16) + (" in slang
    assert "items_2.Store<uint16_t>((index * 24 + 16 + 4), uint16_t(" in slang
    assert "outb.Store<uint16_t4>(32, asuint16(" in slang
    # Component writes are stores, never assignments to a load.
    assert not re.search(r"Load[^;]*\)\)?\.\w+\s*[-+*/]?=", slang), slang


def test_metal_moves_each_member_at_its_own_width(tmp_path):
    metal = _translate(tmp_path, WIDE_AND_HALF, "metal")

    def store(buffer, offset, scalar):
        return f"(*reinterpret_cast<device {scalar}*>({buffer} + {offset}))"

    assert store("items_2", "(index * 24)", "ushort") in metal
    assert store("items_2", "(index * 24 + 2)", "half") in metal
    assert store("items_2", "(index * 24 + 4 + 2)", "half") in metal
    assert store("items_2", "(index * 24 + 8)", "long") in metal
    assert store("items_2", "(index * 24 + 16 + 4)", "ushort") in metal
    assert store("outb", "4", "short") in metal
    assert store("outb", "16", "ulong") in metal
    for offset in (32, 34, 36, 38):
        assert store("outb", offset, "half") in metal
    # The component-store temporary keeps the generator's widened carrier for 16-bit integers.
    assert re.search(
        r"uint2 __crossgl_component_store_\d+ = uint2\(ushort2\(", metal
    ), metal
    assert "float16_t(" not in metal


def test_metal_addresses_8_bit_members_by_the_byte(tmp_path):
    metal = _translate(tmp_path, BYTES, "metal")

    assert "(*reinterpret_cast<const device uchar*>(bytes + 0))" in metal
    assert "(*reinterpret_cast<const device char*>(bytes + 2))" in metal
    assert "(*reinterpret_cast<const device char*>(bytes + 5))" in metal
    assert "(*reinterpret_cast<device char*>(bytes + 3)) = " in metal
    assert "(*reinterpret_cast<device uint*>(bytes + 8)) = " in metal


@pytest.mark.parametrize("backend", ["directx", "slang"])
def test_8_bit_members_are_refused_without_byte_access(tmp_path, backend):
    with pytest.raises(
        Exception, match=r"8-bit uint8_t components have no ByteAddressBuffer"
    ):
        _translate(tmp_path, BYTES, backend)


def test_slang_refuses_a_block_it_cannot_lay_out(tmp_path):
    with pytest.raises(
        Exception, match=r"unsupported Slang GLSL buffer block Bytes bytes"
    ):
        _translate(tmp_path, BYTES, "slang")


def test_float16_members_refuse_widened_binary16(tmp_path):
    from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
    from crosstl.translator.codegen.hlsl_byteaddress_typed_access import (
        hlsl_byteaddress_component_refusal,
    )

    codegen = HLSLCodeGen(widen_native_float16=True)
    member = {"component_type": "float16_t", "component_size": 2}
    assert "widen_native_float16" in hlsl_byteaddress_component_refusal(codegen, member)
    member = {"component_type": "uint16_t", "component_size": 2}
    assert hlsl_byteaddress_component_refusal(codegen, member) is None


def _dxc():
    return shutil.which("dxc")


def test_directx_output_compiles_with_dxc(tmp_path):
    compiler = _dxc()
    if compiler is None:
        pytest.skip("DXC is not installed")
    hlsl = _translate(tmp_path, WIDE_AND_HALF, "directx")
    artifact = tmp_path / "out.hlsl"
    artifact.write_text(hlsl)
    flags = dxc_compiler_arguments_for_source(hlsl)
    assert "-enable-16bit-types" in flags
    result = subprocess.run(
        [compiler, "-T", "cs_6_2", "-E", "CSMain", "-WX", *flags, str(artifact)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_slang_output_compiles_with_slangc(tmp_path):
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    for name, source in (("wide", WIDE_AND_HALF),):
        artifact = tmp_path / f"{name}.slang"
        artifact.write_text(_translate(tmp_path, source, "slang"))
        result = subprocess.run(
            [
                slangc,
                str(artifact),
                "-target",
                "spirv",
                "-o",
                str(tmp_path / f"{name}.spv"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr


RUNTIME = HEADER + """
struct Item { uint8_t tag; int16_t delta; float16_t weight; u16vec3 rgb; int64_t big; f16vec2 uv; };
layout(std430, binding = 0) buffer Items { Item items[]; };
layout(std430, binding = 1) buffer Result { uint words[8]; u64vec2 wide; } result;
void main() {
    result.words[0] = uint(items[1].tag);
    result.words[1] = uint(int(items[1].delta) + 100000);
    result.words[2] = uint(items[1].rgb.z);
    result.words[3] = floatBitsToUint(float(items[1].weight));
    result.words[4] = uint(uint64_t(items[1].big) >> 32);
    result.words[5] = uint(uint64_t(items[1].big) & 0xffffffffUL);
    result.words[6] = floatBitsToUint(float(items[1].uv.y));
    result.words[7] = uint(items[1].rgb.x) + uint(items[1].rgb.y);
    result.wide = u64vec2(uint64_t(items[0].big), uint64_t(items[1].big));
    items[0].tag = items[1].tag;
    items[0].delta = items[1].delta;
    items[0].rgb.zx = u16vec2(7us, 9us);
    items[0].weight = items[1].uv.x;
    items[0].big = items[1].big * 2l;
}
"""
# std430 Item: tag@0, delta@2, weight@4, rgb@8 (6 bytes), big@16, uv@24; align 8, size 32.
ITEM = struct.Struct("<B x h e x x H H H x x q e e x x x x")
SENTINEL = 0xAB


def _item(tag, delta, weight, rgb, big, uv):
    data = bytearray(ITEM.pack(tag, delta, weight, *rgb, big, *uv))
    for start, end in ((1, 2), (6, 8), (14, 16), (28, 32)):
        data[start:end] = bytes([SENTINEL]) * (end - start)
    return bytes(data)


def _run_metal(source, buffers, sizes):
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
    shared = Metal.MTLResourceStorageModeShared
    native = [
        device.newBufferWithBytes_length_options_(data, len(data), shared)
        for data in buffers
    ]
    command = device.newCommandQueue().commandBuffer()
    encoder = command.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    for index, buffer in enumerate(native):
        encoder.setBuffer_offset_atIndex_(buffer, 0, index)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1), Metal.MTLSizeMake(1, 1, 1)
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    assert command.error() is None, command.error()
    return [
        bytes(buffer.contents().as_buffer(size)) for buffer, size in zip(native, sizes)
    ]


def test_metal_reads_and_writes_std430_small_scalars_on_the_gpu(tmp_path):
    metal = _translate(tmp_path, RUNTIME, "metal")
    big = -0x123456789A
    first = _item(1, 2, 3.0, (4, 5, 6), 0x0102030405060708, (0.25, 0.5))
    second = _item(200, -3, 1.5, (40000, 11, 65535), big, (-2.0, 0.75))
    result = bytes([SENTINEL]) * 48

    items, words = _run_metal(metal, [first + second, result], [64, 48])

    unsigned_big = big & 0xFFFFFFFFFFFFFFFF
    assert struct.unpack_from("<8I", words) == (
        200,
        100000 - 3,
        65535,
        struct.unpack("<I", struct.pack("<f", 1.5))[0],
        unsigned_big >> 32,
        unsigned_big & 0xFFFFFFFF,
        struct.unpack("<I", struct.pack("<f", 0.75))[0],
        40011,
    )
    # wide is a u64vec2 at offset 32 (base alignment 16).
    assert struct.unpack_from("<2Q", words, 32) == (0x0102030405060708, unsigned_big)
    # items[0] holds the written members; padding bytes and items[1] are untouched.
    assert items[32:] == second
    assert items[:32] == _item(200, -3, -2.0, (9, 5, 7), big * 2, (0.25, 0.5))


def test_runtime_source_is_valid_glsl(tmp_path):
    validator = shutil.which("glslangValidator")
    if validator is None:
        pytest.skip("glslangValidator is not installed")
    for name, source in (
        ("runtime", RUNTIME),
        ("wide", WIDE_AND_HALF),
        ("bytes", BYTES),
    ):
        path = Path(tmp_path) / f"{name}.comp"
        path.write_text(source)
        result = subprocess.run(
            [
                validator,
                "-V",
                "--target-env",
                "vulkan1.2",
                str(path),
                "-o",
                str(tmp_path / f"{name}.spv"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
