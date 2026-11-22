"""CrossGL ``Buffer<T>``/``RWBuffer<T>`` in Metal output are texture buffers (#3642).

Metal output wrote ``constant Buffer<float> values = Buffer<float>(0);``, which the Metal compiler rejects. The typed
buffers now lower to ``texture_buffer<T, access>`` arguments, element reads to ``read`` narrowed to ``T``, stores to
``write`` and element atomics to the texture atomics; the kernel runs on the Apple GPU against values computed here.
"""

from __future__ import annotations

import re
import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator import parse
from crosstl.translator.codegen.metal_typed_buffers import lower_metal_typed_buffers

COUNT = 128

KERNEL = """
shader TypedBuffers {
    Buffer<float> values;
    Buffer<int2> pairs;
    RWBuffer<float4> rgba;
    RWBuffer<uint> counts;
    RWBuffer<uint> total;
    RWBuffer<half2> halves;
    RWStructuredBuffer<float> outp;
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = values[i] * 2.0 + float(pairs[i].y - pairs[i].x) + float(buffer_dimensions(values));
            rgba[i] = vec4(values[i], float(pairs[i].x), 2.0, 3.0);
            rgba[i].y += 4.0;
            rgba[i] *= 2.0;
            counts[i]++;
            atomicAdd(total[0], 1u);
            halves[i] = halves[i] + half2(1.0, 2.0);
        }
    }
}
"""


def _translate(tmp_path, source, backend="metal", name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def _compile(tmp_path, metal):
    if shutil.which("xcrun") is None:
        pytest.skip("no Metal compiler")
    path = tmp_path / "shader.metal"
    path.write_text(metal, encoding="utf-8")
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-c", str(path)]
        + ["-o", str(tmp_path / "shader.air")],
        capture_output=True,
        text=True,
        check=False,
    )
    if "unable to find utility" in result.stderr:
        pytest.skip("no Metal compiler")
    assert result.returncode == 0, result.stderr[-2000:]


def _slot(metal, name, kind):
    match = re.search(rf"\b{name}\s*\[\[{kind}\((\d+)\)\]\]", metal)
    assert match, (name, metal)
    return int(match.group(1))


def test_typed_buffers_are_texture_buffers(tmp_path):
    metal = _translate(tmp_path, KERNEL)

    assert "Buffer<" not in metal.replace("texture_buffer<", "")
    for declaration in (
        "texture_buffer<float, access::read> values",
        "texture_buffer<int, access::read> pairs",
        "texture_buffer<float, access::read_write> rgba",
        "texture_buffer<uint, access::read_write> counts",
        "texture_buffer<float, access::read_write> halves",
    ):
        assert declaration in metal
    assert "values.read(uint(i)).x" in metal
    assert "pairs.read(uint(i)).xy" in metal
    assert "total.atomic_fetch_add(uint(0), 1u)" in metal
    assert "half2(halves.read(uint(i)).xy)" in metal
    _compile(tmp_path, metal)


def _texture_buffer(Metal, device, pixel_format, data, width):
    alignment = device.minimumLinearTextureAlignmentForPixelFormat_(pixel_format)
    row = len(data) // width * width
    row = (row + alignment - 1) // alignment * alignment
    backing = device.newBufferWithBytes_length_options_(
        data + bytes(row - len(data)), row, Metal.MTLResourceStorageModeShared
    )
    descriptor = Metal.MTLTextureDescriptor.textureBufferDescriptorWithPixelFormat_width_resourceOptions_usage_(
        pixel_format,
        width,
        Metal.MTLResourceStorageModeShared,
        Metal.MTLTextureUsageShaderRead | Metal.MTLTextureUsageShaderWrite,
    )
    texture = backing.newTextureWithDescriptor_offset_bytesPerRow_(descriptor, 0, row)
    assert texture is not None
    return texture, backing


def test_typed_buffer_kernel_computes_the_hlsl_results_on_the_apple_gpu(tmp_path):
    Metal = pytest.importorskip("Metal")
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    metal = _translate(tmp_path, KERNEL)
    library, error = device.newLibraryWithSource_options_error_(metal, None, None)
    assert library is not None, error
    entry = re.search(r"kernel void\s+(\w+)\(", metal).group(1)
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_(entry), None
    )
    assert pipeline is not None, error

    n = COUNT
    inputs = {
        "values": (
            Metal.MTLPixelFormatR32Float,
            struct.pack(f"<{n}f", *(0.5 * i for i in range(n))),
        ),
        "pairs": (
            Metal.MTLPixelFormatRG32Sint,
            struct.pack(f"<{2 * n}i", *(v for i in range(n) for v in (i, 3 * i))),
        ),
        "rgba": (Metal.MTLPixelFormatRGBA32Float, bytes(16 * n)),
        "counts": (
            Metal.MTLPixelFormatR32Uint,
            struct.pack(f"<{n}I", *(10 + i for i in range(n))),
        ),
        "total": (Metal.MTLPixelFormatR32Uint, bytes(4 * n)),
        "halves": (
            Metal.MTLPixelFormatRG16Float,
            struct.pack(
                f"<{2 * n}e", *(v for i in range(n) for v in (0.25 * i, 0.5 * i))
            ),
        ),
    }
    textures = {
        name: _texture_buffer(Metal, device, pixel_format, data, n)
        for name, (pixel_format, data) in inputs.items()
    }
    output = device.newBufferWithLength_options_(4 * n, 0)
    commands = device.newCommandQueue().commandBuffer()
    encoder = commands.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    for name, (texture, _backing) in textures.items():
        encoder.setTexture_atIndex_(texture, _slot(metal, name, "texture"))
    encoder.setBuffer_offset_atIndex_(output, 0, _slot(metal, "outp", "buffer"))
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n, 1, 1), Metal.MTLSizeMake(64, 1, 1)
    )
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    assert commands.error() is None

    def read(name, form, length):
        backing = textures[name][1]
        return struct.unpack(form, backing.contents().as_buffer(length))

    assert read("rgba", f"<{4 * n}f", 16 * n) == tuple(
        v for i in range(n) for v in (float(i), 2.0 * (i + 4), 4.0, 6.0)
    )
    assert struct.unpack(f"<{n}f", output.contents().as_buffer(4 * n)) == tuple(
        float(3 * i + n) for i in range(n)
    )
    assert read("counts", f"<{n}I", 4 * n) == tuple(11 + i for i in range(n))
    assert read("total", "<I", 4) == (n,)
    assert read("halves", f"<{2 * n}e", 4 * n) == tuple(
        v for i in range(n) for v in (0.25 * i + 1.0, 0.5 * i + 2.0)
    )


ELEMENT_FORMS = {
    "array": (
        "Buffer<float> vs[2];",
        "outp[i] = vs[1][i];",
        "array<texture_buffer<float, access::read>, 2> vs",
    ),
    "function": (
        "RWBuffer<int> v;",
        "outp[i] = float(put(v, i));",
        "texture_buffer<int, access::read_write> b",
    ),
    "swizzle": (
        "RWBuffer<uint3> v;",
        "v[i].zx = uvec2(1u, 2u); outp[i] = float(v[i].z);",
        "uint3 crosstl_texel = v.read(uint(i)).xyz;",
    ),
    "texel_calls": (
        "RWBuffer<float> v;",
        "imageStore(v, int(i), vec4(1.0)); outp[i] = imageLoad(v, int(i)).x;",
        "v.write(float4(1.0), uint(int(i)));",
    ),
    "size_statement": (
        "RWBuffer<float> v;",
        "uint n; buffer_dimensions(v, n); outp[i] = float(n);",
        "n = uint(int(v.get_width()));",
    ),
}


@pytest.mark.parametrize("form", sorted(ELEMENT_FORMS))
def test_typed_buffer_uses_compile_as_metal(tmp_path, form):
    declaration, body, expected = ELEMENT_FORMS[form]
    source = f"""
shader P {{
    {declaration}
    RWStructuredBuffer<float> outp;
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        int put(RWBuffer<int> b, uint k) {{ b[k] = 3; return b[k]; }}
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            {body}
        }}
    }}
}}
"""
    if form != "function":
        source = source.replace(
            "int put(RWBuffer<int> b, uint k) { b[k] = 3; return b[k]; }", ""
        )
    metal = _translate(tmp_path, source)
    assert expected in metal
    _compile(tmp_path, metal)


def test_rasterizer_ordered_typed_buffer_is_in_raster_order_group_zero(tmp_path):
    source = """
shader P {
    RasterizerOrderedBuffer<uint> counts;
    fragment {
        vec4 main(vec4 pos @ gl_FragCoord) @ gl_FragColor {
            uint k = uint(pos.x);
            counts[k] += 1u;
            return vec4(float(counts[k]));
        }
    }
}
"""
    metal = _translate(tmp_path, source)
    assert re.search(
        r"texture_buffer<uint, access::read_write> counts\s*\[\[texture\(0\)\]\]\s*"
        r"\[\[raster_order_group\(0\)\]\]",
        metal,
    )
    _compile(tmp_path, metal)


REFUSED = {
    "read_only_store": ("Buffer<float> v;", "v[i] = 1.0;", "store to the read-only"),
    "double": (
        "Buffer<double> v;",
        "outp[i] = float(v[i]);",
        "Metal texture buffers hold texels",
    ),
    "side_effects": (
        "RWBuffer<float> v;",
        "v[i++] += 1.0;",
        "whose buffer or index has side effects",
    ),
    "nested_store": (
        "RWBuffer<float> v;",
        "outp[i] = (v[i] = 2.0);",
        "inside a larger expression",
    ),
    "local_alias": (
        "RWBuffer<float> v;",
        "RWBuffer<float> w = v; w[i] = 1.0;",
        "used other than through",
    ),
    "builtin_argument": (
        "RWBuffer<float> v;",
        "outp[i] = length(v);",
        "passed to 'length'",
    ),
    "structure_member": (
        "struct S { Buffer<float> b; };",
        "outp[i] = 1.0;",
        "as a structure member",
    ),
}


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_typed_buffer_forms_without_a_metal_lowering_are_refused(tmp_path, case):
    declaration, body, reason = REFUSED[case]
    source = f"""
shader P {{
    {declaration}
    RWStructuredBuffer<float> outp;
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            {body}
        }}
    }}
}}
"""
    with pytest.raises(Exception, match=re.escape(reason)):
        _translate(tmp_path, source)


def test_programs_without_typed_buffers_are_returned_unchanged():
    program = parse(
        "shader P { compute { layout(local_size_x = 1) in; void main() { } } }"
    )
    assert lower_metal_typed_buffers(program, None) is program
    assert isinstance(program, A.ShaderNode)
