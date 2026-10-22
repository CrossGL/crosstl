"""Resource slots that name an enumerator or an integer constant bind at that integer in GLSL and SPIR-V."""

import re
import shutil
import subprocess

import crosstl

METAL_ENUM_SLOTS = """#include <metal_stdlib>
using namespace metal;
typedef enum BufferIndex {
    BufferIndexParams = 2,
    BufferIndexWeights,
} BufferIndex;
typedef enum TextureIndex { TextureIndexColor } TextureIndex;
struct Params { float4 tint; };
fragment float4 fmain(float4 pos [[position]],
                      constant Params &params [[buffer(BufferIndexParams)]],
                      device const float *weights [[buffer(BufferIndexWeights)]],
                      texture2d<float> color [[texture(TextureIndexColor)]]) {
    return params.tint * weights[0] * color.read(uint2(pos.xy));
}
"""

CROSSGL_CONSTANT_SLOT = """
shader main {
    const int PARAMS_SLOT = 5;
    cbuffer Params @binding(PARAMS_SLOT) { vec4 tint; };
    fragment {
        vec4 main() @ gl_FragColor { return tint; }
    }
}
"""


def _translate(tmp_path, source, name):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend="opengl", format_output=False)


def test_glsl_binds_resources_at_the_enumerators_they_name(tmp_path):
    glsl = _translate(tmp_path, METAL_ENUM_SLOTS, "shader.metal")

    # BufferIndexParams = 2, BufferIndexWeights counts on to 3, TextureIndexColor is 0.
    assert re.search(r"layout\(std140, binding = 2\) uniform \w+", glsl), glsl
    assert re.search(r"layout\(std430, binding = 3\) readonly buffer \w+", glsl), glsl
    assert re.search(r"layout\(binding = 0\) uniform sampler2D color;", glsl), glsl
    if shutil.which("glslangValidator"):
        result = subprocess.run(
            ["glslangValidator", "--stdin", "-S", "frag"],
            input=glsl,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_glsl_binds_a_block_at_the_constant_it_names(tmp_path):
    glsl = _translate(tmp_path, CROSSGL_CONSTANT_SLOT, "shader.cgl")

    assert re.search(r"layout\(std140, binding = 5\) uniform Params", glsl), glsl


def test_spirv_binds_resources_at_the_enumerators_they_name(tmp_path):
    path = tmp_path / "shader.metal"
    path.write_text(METAL_ENUM_SLOTS)
    spirv = crosstl.translate(str(path), backend="vulkan", format_output=False)

    bindings = {
        int(binding) for binding in re.findall(r"OpDecorate %\w+ Binding (\d+)", spirv)
    }
    assert {2, 3} <= bindings, spirv


METAL_NAMED_ATTACHMENTS = """#include <metal_stdlib>
using namespace metal;
typedef enum RenderTarget {
    RenderTargetLighting = 0,
    RenderTargetAlbedo,
    RenderTargetNormal,
    RenderTargetDepth,
} RenderTarget;
struct GBufferData {
    half4 lighting [[color(RenderTargetLighting)]];
    half4 albedo [[color(RenderTargetAlbedo)]];
    half4 normal [[color(RenderTargetNormal)]];
    float depth [[color(RenderTargetDepth)]];
};
struct VOut { float4 position [[position]]; };
fragment GBufferData gbuffer(VOut in [[stage_in]]) {
    GBufferData out;
    out.lighting = half4(0.0h);
    out.albedo = half4(1.0h);
    out.normal = half4(0.5h);
    out.depth = in.position.z;
    return out;
}
"""


def test_metal_colour_attachments_named_by_enumerators_keep_their_index(tmp_path):
    path = tmp_path / "shader.metal"
    path.write_text(METAL_NAMED_ATTACHMENTS)
    crossgl = crosstl.translate(str(path), backend="cgl", format_output=False)
    hlsl = crosstl.translate(str(path), backend="directx", format_output=False)

    # Every attachment used to collapse to colour 0.
    for member, attribute in (
        ("lighting", "@gl_FragColor"),
        ("albedo", "@gl_FragColor1"),
        ("normal", "@gl_FragColor2"),
        ("depth", "@gl_FragColor3"),
    ):
        assert re.search(rf"\b{member} {attribute}\b", crossgl), crossgl
    # A colour attachment may be a scalar, as MSL and HLSL allow.
    assert re.search(r"float depth\s*:\s*SV_TARGET3;", hlsl), hlsl
    assert re.search(r"albedo\s*:\s*SV_TARGET1;", hlsl), hlsl
