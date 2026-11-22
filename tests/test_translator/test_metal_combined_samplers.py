"""Metal binds a sampler for each combined sampler a program samples, instead of an inline sampler state."""

import re
import shutil
import struct
import subprocess
import sys

import pytest

import crosstl


def _translate(tmp_path, source, suffix=".frag"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _compile(tmp_path, code):
    if shutil.which("xcrun") is None:
        return
    source = tmp_path / "out.metal"
    source.write_text(code)
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "out.air"),
        ],
        capture_output=True,
        text=True,
    )
    if "unable to find utility" in result.stderr:
        return
    assert result.returncode == 0, result.stderr


SAMPLED = """#version 450
layout(set = 0, binding = 1) uniform sampler2D albedo;
layout(set = 0, binding = 2) uniform sampler2DShadow shadowMap;
layout(set = 0, binding = 3) uniform sampler2D layers[4];
layout(set = 0, binding = 7) uniform sampler2D fetchOnly;
layout(location = 0) in vec2 uv;
layout(location = 1) flat in int layer;
layout(location = 0) out vec4 color;
vec4 shade(sampler2D t, vec2 p) { return textureLod(t, p, 2.0); }
vec4 shadeTwice(sampler2D t, vec2 p) { return shade(t, p) + shade(t, p * 0.5); }
void main() {
    float lit = texture(shadowMap, vec3(uv, 0.5));
    color = shadeTwice(albedo, uv) * lit + texture(layers[layer], uv)
        + texelFetch(fetchOnly, ivec2(0), 0);
}
"""


def test_combined_samplers_take_bound_samplers(tmp_path):
    code = _translate(tmp_path, SAMPLED)
    # The host's sampler state replaces the inline linear sampler, which read level 0 with clamp addressing.
    assert "mag_filter::linear" not in code, code
    assert "sampler albedoSampler [[sampler(1)]]" in code, code
    assert "sampler shadowMapSampler [[sampler(2)]]" in code, code
    assert "array<sampler, 4> layersSampler [[sampler(3)]]" in code, code
    assert "shadowMap.sample_compare(shadowMapSampler, input.uv, 0.5)" in code, code
    assert (
        "layers[input.layer].sample(layersSampler[input.layer], input.uv)" in code
    ), code
    _compile(tmp_path, code)


def test_helpers_receive_the_sampler_with_the_texture(tmp_path):
    code = _translate(tmp_path, SAMPLED)
    assert re.search(
        r"float4 shade\(texture2d<float> t, sampler tSampler, float2 p\)", code
    ), code
    assert "t.sample(tSampler, p, level(2.0))" in code, code
    assert re.search(r"shade\(t, tSampler, p\)", code), code
    assert re.search(r"shadeTwice\(albedo, albedoSampler, input\.uv\)", code), code


def test_texel_fetches_need_no_sampler(tmp_path):
    code = _translate(tmp_path, SAMPLED)
    assert "fetchOnlySampler" not in code, code
    assert "fetchOnly.read(" in code, code


SEPARATE = """#version 450
layout(set = 0, binding = 0) uniform texture2D colorTexture;
layout(set = 0, binding = 1) uniform sampler pointSampler;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
void main() { color = texture(sampler2D(colorTexture, pointSampler), uv); }
"""


def test_separate_samplers_are_kept(tmp_path):
    code = _translate(tmp_path, SEPARATE)
    assert "colorTextureSampler" not in code, code
    assert "pointSampler" in code, code
    _compile(tmp_path, code)


SHARED_BINDING = """shader S {
    sampler2D colorMap @binding(0);
    sampler2D detailMap @binding(1);
    sampler linearSampler @binding(0);
    fragment {
        vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
            return texture(colorMap, linearSampler, uv) + texture(detailMap, uv);
        }
    }
}
"""


def test_a_declared_sampler_keeps_its_slot(tmp_path):
    code = _translate(tmp_path, SHARED_BINDING, suffix=".cgl")
    assert "sampler linearSampler [[sampler(0)]]" in code, code
    assert "colorMapSampler" not in code, code
    assert re.search(r"sampler detailMapSampler \[\[sampler\(1\)\]\]", code), code
    _compile(tmp_path, code)


# A combined sampler a stage block declares, as the coverage probes and most CrossGL programs write it. The texture is
# two texels wide and the host samples it with nearest filtering and repeat addressing, so each lane reads a texel
# value; an inline linear sampler with clamp addressing reads 1.5, 3, 3, 3 instead.
STAGE_BLOCK = """shader Probe {
    compute {
        layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        uniform sampler2D tex;
        vec4 fetch(sampler2D t, vec2 p) { return textureLod(t, p, 0.0); }
        void main() {
            uint i = gl_GlobalInvocationID.x;
            float u = 0.375 + 0.5 * float(i % 2u) + float(i / 2u);
            outp[i] = fetch(tex, vec2(u, 0.5)).x;
        }
    }
}
"""

STAGE_FRAGMENT = """shader S {
    fragment {
        uniform sampler2D albedo @binding(2);
        uniform samplerCube sky;
        vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
            return texture(albedo, uv) + texture(sky, vec3(uv, 1.0)) + texelFetch(albedo, ivec2(0), 0);
        }
    }
}
"""


def test_stage_block_combined_samplers_take_bound_samplers(tmp_path):
    code = _translate(tmp_path, STAGE_FRAGMENT, suffix=".cgl")
    assert "mag_filter::linear" not in code, code
    assert "sampler albedoSampler [[sampler(2)]]" in code, code
    assert re.search(r"sampler skySampler \[\[sampler\(\d+\)\]\]", code), code
    assert "albedo.sample(albedoSampler, " in code, code
    assert "sky.sample(skySampler, " in code, code
    _compile(tmp_path, code)


def test_stage_block_samplers_reach_helpers(tmp_path):
    code = _translate(tmp_path, STAGE_BLOCK, suffix=".cgl")
    assert "mag_filter::linear" not in code, code
    assert re.search(r"texture2d<float> tex \[\[texture\(0\)\]\]", code), code
    assert re.search(r"sampler texSampler \[\[sampler\(0\)\]\]", code), code
    assert re.search(r"fetch\(tex, texSampler, ", code), code
    assert "t.sample(tSampler, p, level(0.0))" in code, code


def _run_sampling_kernel(code):
    """Run ``code``'s kernel on four threads with a two-texel R32Float texture and a nearest, repeating sampler."""
    try:
        import Metal
    except ImportError:
        pytest.skip("PyObjC's Metal bindings are not installed")
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    library, error = device.newLibraryWithSource_options_error_(
        code, Metal.MTLCompileOptions.new(), None
    )
    assert library is not None, error
    function = library.newFunctionWithName_("kernel_main")
    pipeline, error = device.newComputePipelineStateWithFunction_error_(function, None)
    assert pipeline is not None, error
    descriptor = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
        Metal.MTLPixelFormatR32Float, 2, 1, False
    )
    texture = device.newTextureWithDescriptor_(descriptor)
    texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
        Metal.MTLRegionMake2D(0, 0, 2, 1), 0, struct.pack("<2f", 1.0, 3.0), 8
    )
    state = Metal.MTLSamplerDescriptor.new()
    state.setMinFilter_(Metal.MTLSamplerMinMagFilterNearest)
    state.setMagFilter_(Metal.MTLSamplerMinMagFilterNearest)
    state.setSAddressMode_(Metal.MTLSamplerAddressModeRepeat)
    state.setTAddressMode_(Metal.MTLSamplerAddressModeRepeat)
    sampler = device.newSamplerStateWithDescriptor_(state)
    output = device.newBufferWithLength_options_(16, Metal.MTLResourceStorageModeShared)
    command = device.newCommandQueue().commandBuffer()
    encoder = command.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(output, 0, 0)
    encoder.setTexture_atIndex_(texture, 0)
    encoder.setSamplerState_atIndex_(sampler, 0)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(4, 1, 1), Metal.MTLSizeMake(4, 1, 1)
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    assert command.error() is None, command.error()
    return struct.unpack("<4f", bytes(output.contents().as_buffer(16)))


@pytest.mark.skipif(sys.platform != "darwin", reason="needs a Metal device")
def test_stage_block_sampler_state_is_the_hosts_on_the_gpu(tmp_path):
    code = _translate(tmp_path, STAGE_BLOCK, suffix=".cgl")
    assert _run_sampling_kernel(code) == (1.0, 3.0, 1.0, 3.0)
