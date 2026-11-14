"""A Metal fragment function's value return is colour attachment 0 on HLSL whatever its numeric type.

The HLSL generator gave only ``float4`` returns the default ``SV_Target`` semantic, so ``fragment half4 shade(...)``
became ``float16_t4 PSMain(...)`` without one, which DXC rejects ("Semantic must be defined for all outputs"). Shared
Metal headers also include Apple's ``<TargetConditionals.h>``, which HLSL and GLSL kept and cannot resolve.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.hlsl_render_targets import is_render_target_value_type

HALF_FRAGMENT = """
#include <metal_stdlib>
#include <TargetConditionals.h>
using namespace metal;

struct Varyings {
    float4 position [[position]];
    float2 uv;
};

fragment half4 shade(Varyings in [[stage_in]],
                     texture2d<half> image [[texture(0)]],
                     sampler linear [[sampler(0)]]) {
    return image.sample(linear, in.uv);
}
"""


@pytest.mark.parametrize(
    "type_name",
    [
        "float4",
        "float",
        "float16_t4",
        "half3",
        "min16float2",
        "uint",
        "int4",
        "uint16_t2",
    ],
)
def test_numeric_scalars_and_vectors_are_render_target_values(type_name):
    assert is_render_target_value_type(type_name)


@pytest.mark.parametrize(
    "type_name", ["bool", "bool4", "float4x4", "float5", "Output", ""]
)
def test_other_types_are_not_render_target_values(type_name):
    assert not is_render_target_value_type(type_name)


def _translate(tmp_path, backend):
    path = tmp_path / "shade.metal"
    path.write_text(HALF_FRAGMENT, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def test_half_fragment_return_writes_the_first_render_target(tmp_path):
    hlsl = _translate(tmp_path, "directx")
    assert re.search(r"float16_t4 PSMain\(Varyings in_\)\s*:\s*SV_TARGET", hlsl), hlsl
    assert "#include" not in hlsl
    dxc = shutil.which("dxc")
    if dxc is None:
        return
    path = tmp_path / "shade.hlsl"
    path.write_text(hlsl, encoding="utf-8")
    completed = subprocess.run(
        [dxc, "-T", "ps_6_2", "-enable-16bit-types", "-E", "PSMain", str(path)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-2000:]


def test_glsl_drops_target_conditionals(tmp_path):
    glsl = _translate(tmp_path, "opengl")
    assert "TargetConditionals" not in glsl
    validator = shutil.which("glslangValidator")
    if validator:
        path = tmp_path / "shade.frag"
        path.write_text(glsl, encoding="utf-8")
        completed = subprocess.run(
            [validator, str(path)], capture_output=True, text=True
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr


HALF_TEXTURE_KERNEL = """
#include <metal_stdlib>
using namespace metal;

kernel void grayscale(texture2d<half, access::read> source [[texture(0)]],
                      texture2d<half, access::write> destination [[texture(1)]],
                      uint2 gid [[thread_position_in_grid]]) {
    half4 color = source.read(gid);
    half gray = dot(color.rgb, half3(0.2126h, 0.7152h, 0.0722h));
    destination.write(half4(gray, gray, gray, 1.0h), gid);
}
"""


def test_half_textures_are_single_hlsl_resources(tmp_path):
    # The importer marks MSL half textures for the Metal generator; HLSL read the marker as an array size.
    path = tmp_path / "grayscale.metal"
    path.write_text(HALF_TEXTURE_KERNEL, encoding="utf-8")
    hlsl = crosstl.translate(
        str(path), backend="directx", source_options={"threadgroup_size": (8, 8, 1)}
    )
    assert re.search(r"RWTexture2D<float4> source : register\(u0\);", hlsl), hlsl
    assert re.search(r"RWTexture2D<float4> destination : register\(u1\);", hlsl), hlsl
    assert "metal_texture_element" not in hlsl
