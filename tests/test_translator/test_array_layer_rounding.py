"""Sampled array textures select the layer by rounding, on every target.

GLSL (4.60 section 8.9), Vulkan ("Texel Coordinate Systems") and Direct3D round the floating-point layer coordinate of
a sampled array texture to the nearest integer and clamp it to the layer range. Metal takes the layer of ``sample``,
``gather`` and ``sample_compare`` as an integer, and the implicit conversion truncates, so CrossGL must emit
``uint(rint(layer))``. Integer texel fetches already carry an integer layer and stay unrounded. In the other
direction, a floating-point Metal layer argument is truncated by Metal, so the Metal import keeps an explicit
``uint`` conversion in the CrossGL coordinate.
"""

from __future__ import annotations

import re
import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics.array_layers import (
    metal_integer_scalar_type,
    metal_sampled_array_layer,
    metal_sampled_layer_source_argument,
)

SAMPLED_ARRAYS = """
shader SampledArrays {
    sampler2DArray colorArray;
    samplerCubeArray cubeArray;
    sampler2DArrayShadow shadowArray;
    samplerCubeArrayShadow cubeShadowArray;
    sampler1DArray lineArray;

    struct FSInput {
        vec3 uvw @ TEXCOORD0;
        vec4 dirLayer @ TEXCOORD1;
        vec2 uLayer @ TEXCOORD2;
    };

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            vec4 a = texture(colorArray, input.uvw);
            vec4 b = textureLod(colorArray, input.uvw, 1.0);
            vec4 c = textureGrad(colorArray, input.uvw, vec2(0.1), vec2(0.1));
            vec4 d = textureOffset(colorArray, input.uvw, ivec2(1, 0));
            vec4 e = textureLodOffset(colorArray, input.uvw, 0.0, ivec2(1, 1));
            vec4 f = textureGather(colorArray, input.uvw);
            vec4 g = textureGatherOffset(colorArray, input.uvw, ivec2(1, 0), 1);
            vec4 h = texture(cubeArray, input.dirLayer);
            vec4 i = textureLod(cubeArray, input.dirLayer, 2.0);
            float j = texture(shadowArray, input.dirLayer);
            float k = texture(cubeShadowArray, input.dirLayer, 0.5);
            vec4 l = textureGather(shadowArray, input.uvw, 0.5);
            vec4 m = texture(lineArray, input.uLayer);
            vec4 n = texelFetch(colorArray, ivec3(1, 2, 1), 0);
            vec4 o = texelFetchOffset(lineArray, ivec2(3, 1), 0, 1);
            return a + b + c + d + e + f + g + h + i + vec4(j + k) + l + m + n + o;
        }
    }
}
"""

# Each sampled form becomes a Metal call whose layer argument is rounded.
METAL_SAMPLED_LAYERS = [
    "input.uvw.xy, uint(rint(clamp(float(input.uvw.z), 0.0, 2047.0)))); float4 b = colorArray.sample(",
    "input.uvw.xy, uint(rint(clamp(float(input.uvw.z), 0.0, 2047.0))), level(1.0));",
    "input.uvw.xy, uint(rint(clamp(float(input.uvw.z), 0.0, 2047.0))), gradient2d(float2(0.1), float2(0.1)));",
    "input.uvw.xy, uint(rint(clamp(float(input.uvw.z), 0.0, 2047.0))), int2(1, 0));",
    "input.uvw.xy, uint(rint(clamp(float(input.uvw.z), 0.0, 2047.0))), level(0.0), int2(1, 1));",
    "input.uvw.xy, uint(rint(clamp(float(input.uvw.z), 0.0, 2047.0)))); float4 g = colorArray.gather(",
    "input.uvw.xy, uint(rint(clamp(float(input.uvw.z), 0.0, 2047.0))), int2(1, 0), component::y);",
    "input.dirLayer.xyz, uint(rint(clamp(float(input.dirLayer.w), 0.0, 2047.0))));",
    "input.dirLayer.xyz, uint(rint(clamp(float(input.dirLayer.w), 0.0, 2047.0))), level(2.0));",
    "uint(rint(clamp(float(input.dirLayer.xyz.z), 0.0, 2047.0))), input.dirLayer.w);",
    "input.dirLayer.xyz, uint(rint(clamp(float(input.dirLayer.w), 0.0, 2047.0))), 0.5);",
    "input.uvw.xy, uint(rint(clamp(float(input.uvw.z), 0.0, 2047.0))), 0.5);",
    "input.uLayer.x, uint(rint(clamp(float(input.uLayer.y), 0.0, 2047.0))));",
]

# Integer texel fetches address the layer with an integer: no rounding.
METAL_FETCHED_LAYERS = [
    "colorArray.read(uint2((int3(1, 2, 1)).xy), uint((int3(1, 2, 1)).z), uint(0));",
    "lineArray.read(uint(((int2(3, 1)).x + 1)), uint((int2(3, 1)).y), uint(0));",
]

METAL_FLOAT_LAYERS = """
#include <metal_stdlib>
using namespace metal;

struct FragIn {
    float4 position [[position]];
    float3 uvw;
    float4 dirLayer;
};

fragment float4 frag_main(FragIn in [[stage_in]],
                          texture2d_array<float> colorArray [[texture(0)]],
                          depth2d_array<float> shadowArray [[texture(1)]],
                          texturecube_array<float> cubeArray [[texture(2)]],
                          depthcube_array<float> cubeShadowArray [[texture(3)]],
                          sampler s [[sampler(0)]],
                          constant uint& layerIndex [[buffer(0)]]) {
    float4 a = colorArray.sample(s, in.uvw.xy, in.uvw.z);
    float4 b = colorArray.sample(s, in.uvw.xy, layerIndex, level(1.0));
    float4 c = colorArray.gather(s, in.uvw.xy, in.uvw.z, int2(1, 0), component::y);
    float d = shadowArray.sample_compare(s, in.uvw.xy, in.uvw.z, 0.5);
    float4 e = shadowArray.gather_compare(s, in.uvw.xy, in.uvw.z, 0.5);
    float4 f = cubeArray.sample(s, in.dirLayer.xyz, in.dirLayer.w);
    float g = cubeShadowArray.sample_compare(s, in.dirLayer.xyz, in.dirLayer.w, 0.5);
    float4 h = colorArray.sample(s, in.uvw.xy, 2);
    return a + b + c + float4(d) + e + f + float4(g) + h;
}
"""

CROSSGL_FROM_METAL = [
    "vec4 a = texture(colorArray, s, vec3(in_.uvw.xy, uint(in_.uvw.z)));",
    "vec4 b = textureLod(colorArray, s, vec3(in_.uvw.xy, layerIndex), 1.0);",
    "textureGatherOffset(colorArray, s, vec3(in_.uvw.xy, uint(in_.uvw.z)), ivec2(1, 0), 1);",
    "float d = textureCompare(shadowArray, s, vec3(in_.uvw.xy, uint(in_.uvw.z)), 0.5);",
    "textureGatherCompare(shadowArray, s, vec3(in_.uvw.xy, uint(in_.uvw.z)), 0.5);",
    "vec4 f = texture(cubeArray, s, vec4(in_.dirLayer.xyz, uint(in_.dirLayer.w)));",
    "textureCompare(cubeShadowArray, s, vec4(in_.dirLayer.xyz, uint(in_.dirLayer.w)), 0.5);",
    "vec4 h = texture(colorArray, s, vec3(in_.uvw.xy, 2));",
]

LAYER_PROBE = """
shader LayerProbe {
    compute {
        layout(local_size_x = 10, local_size_y = 1, local_size_z = 1) in;
        uniform sampler2DArray layered;
        buffer float* layers;
        buffer float* results;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            float z = layers[i];
            results[i * 2u] = textureLod(layered, vec3(0.5, 0.5, z), 0.0).r;
            results[i * 2u + 1u] = textureGather(layered, vec3(0.5, 0.5, z), 0).x;
        }
    }
}
"""

# Constant layers are folded by the Metal compiler: uint(rint(-1.0)) used to select the last layer.
CONSTANT_LAYER_PROBE = """
shader ConstantLayerProbe {
    compute {
        layout(local_size_x = 10, local_size_y = 1, local_size_z = 1) in;
        uniform sampler2DArray layered;
        buffer float* layers;
        buffer float* results;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            float unused = layers[i] * 0.0;
            results[i * 2u] = textureLod(layered, vec3(0.5, 0.5, -1.0), 0.0).r + unused;
            results[i * 2u + 1u] = textureGather(layered, vec3(0.5, 0.5, 9.0), 0).x;
        }
    }
}
"""

PROBE_LAYERS = (0.4, 0.6, 1.5, 2.5, 1.49, -0.4, -0.6, -3.0, 2.6, 7.0)
# Layer l holds 10 * (l + 1); the layer is clamp(round(z), 0, 2) on GLSL, Vulkan and Direct3D.
PROBE_EXPECTED = (10.0, 20.0, 30.0, 30.0, 20.0, 10.0, 10.0, 10.0, 30.0, 30.0)


def _translate(tmp_path, source, backend, name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return " ".join(code.split())


def _metal_compiler_available():
    if not shutil.which("xcrun"):
        return False
    probe = subprocess.run(["xcrun", "--find", "metal"], capture_output=True, text=True)
    return probe.returncode == 0


def test_layer_helpers():
    assert (
        metal_sampled_array_layer("p.z") == "uint(rint(clamp(float(p.z), 0.0, 2047.0)))"
    )
    for metal_type in ("uint", "int", "ushort", "metal::uint", "constant uint&"):
        assert metal_integer_scalar_type(metal_type), metal_type
    for metal_type in (None, "", "float", "half", "uint2", "bool", "device uint*"):
        assert not metal_integer_scalar_type(metal_type), metal_type
    assert metal_sampled_layer_source_argument("z", "float") == "uint(z)"
    assert metal_sampled_layer_source_argument("z", None) == "uint(z)"
    assert metal_sampled_layer_source_argument("n", "int") == "n"


def test_metal_rounds_sampled_layers(tmp_path):
    code = _flat(_translate(tmp_path, SAMPLED_ARRAYS, "metal"))
    for expected in METAL_SAMPLED_LAYERS:
        assert expected in code, expected
    sampled_calls = re.findall(
        r"\.(?:sample|sample_compare|gather|gather_compare)\(", code
    )
    assert len(sampled_calls) == len(METAL_SAMPLED_LAYERS)
    assert code.count("uint(rint(") == len(METAL_SAMPLED_LAYERS)


def test_metal_keeps_fetched_layers_integral(tmp_path):
    code = _flat(_translate(tmp_path, SAMPLED_ARRAYS, "metal"))
    for expected in METAL_FETCHED_LAYERS:
        assert expected in code, expected


def test_metal_import_keeps_layer_truncation(tmp_path):
    code = _flat(_translate(tmp_path, METAL_FLOAT_LAYERS, "cgl", "s.metal"))
    for expected in CROSSGL_FROM_METAL:
        assert expected in code, expected


@pytest.mark.parametrize("backend", ("opengl", "vulkan"))
def test_metal_import_truncates_float_layers_for_rounding_targets(tmp_path, backend):
    code = _flat(_translate(tmp_path, METAL_FLOAT_LAYERS, backend, "s.metal"))
    if backend == "opengl":
        assert "texture(colorArray, vec3(uvw.xy, uint(uvw.z)))" in code
    else:
        assert "OpConvertFToU" in code


def test_metal_output_compiles_with_the_metal_compiler(tmp_path):
    if not _metal_compiler_available():
        pytest.skip("the Metal compiler is not installed")
    for name, source, file_name in (
        ("cgl", SAMPLED_ARRAYS, "shader.cgl"),
        ("roundtrip", METAL_FLOAT_LAYERS, "s.metal"),
        ("probe", LAYER_PROBE, "probe.cgl"),
        ("constant", CONSTANT_LAYER_PROBE, "constant.cgl"),
    ):
        path = tmp_path / f"{name}.metal"
        path.write_text(_translate(tmp_path, source, "metal", file_name))
        result = subprocess.run(
            ["xcrun", "metal", "-c", str(path), "-o", str(tmp_path / f"{name}.air")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


def test_glsl_output_from_metal_compiles(tmp_path):
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not installed")
    path = tmp_path / "out.frag"
    path.write_text(_translate(tmp_path, METAL_FLOAT_LAYERS, "opengl", "s.metal"))
    result = subprocess.run(
        ["glslangValidator", str(path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _run_layer_probe(metal_source):
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

    descriptor = Metal.MTLTextureDescriptor.new()
    descriptor.setTextureType_(Metal.MTLTextureType2DArray)
    descriptor.setPixelFormat_(Metal.MTLPixelFormatR32Float)
    descriptor.setWidth_(1)
    descriptor.setHeight_(1)
    descriptor.setArrayLength_(3)
    texture = device.newTextureWithDescriptor_(descriptor)
    for layer in range(3):
        texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage_(
            Metal.MTLRegionMake2D(0, 0, 1, 1),
            0,
            layer,
            struct.pack("<f", 10.0 * (layer + 1)),
            4,
            4,
        )

    layers = struct.pack(f"<{len(PROBE_LAYERS)}f", *PROBE_LAYERS)
    layer_buffer = device.newBufferWithBytes_length_options_(
        layers, len(layers), Metal.MTLResourceStorageModeShared
    )
    result_size = 8 * len(PROBE_LAYERS)
    result_buffer = device.newBufferWithLength_options_(
        result_size, Metal.MTLResourceStorageModeShared
    )
    command = device.newCommandQueue().commandBuffer()
    encoder = command.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setTexture_atIndex_(texture, 0)
    encoder.setBuffer_offset_atIndex_(layer_buffer, 0, 0)
    encoder.setBuffer_offset_atIndex_(result_buffer, 0, 1)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(len(PROBE_LAYERS), 1, 1),
        Metal.MTLSizeMake(len(PROBE_LAYERS), 1, 1),
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    assert command.error() is None, command.error()
    data = bytes(result_buffer.contents().as_buffer(result_size))
    return struct.unpack(f"<{2 * len(PROBE_LAYERS)}f", data)


def test_metal_layer_selection_matches_rounding_on_the_gpu(tmp_path):
    source = _translate(tmp_path, LAYER_PROBE, "metal", "probe.cgl")
    assert "kernel_main" in source
    results = _run_layer_probe(source)
    sampled = results[0::2]
    gathered = results[1::2]
    assert sampled == PROBE_EXPECTED
    assert gathered == PROBE_EXPECTED


def test_metal_constant_layers_clamp_like_the_other_apis_on_the_gpu(tmp_path):
    source = _translate(tmp_path, CONSTANT_LAYER_PROBE, "metal", "constant.cgl")
    results = _run_layer_probe(source)
    assert results[0::2] == (10.0,) * len(PROBE_LAYERS)
    assert results[1::2] == (30.0,) * len(PROBE_LAYERS)
