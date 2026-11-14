"""Fragment inputs that read colour attachments (MSL programmable blending) are refused on other targets.

HLSL emitted them as ``SV_Target`` inputs, which DXC rejects, and GLSL and SPIR-V as interpolated inputs, which read a
varying instead of the attachment. See :mod:`crosstl.translator.semantics.colour_attachment_inputs`.
"""

import pytest

import crosstl
from crosstl.translator.semantics.colour_attachment_inputs import (
    ColourAttachmentInputUnsupportedError,
)

SINGLE_PASS = """
#include <metal_stdlib>
using namespace metal;

struct Varyings {
    float4 position [[position]];
};

struct GBufferData {
    half4 albedo [[color(1)]];
    half4 normal [[color(2)]];
};

struct Lighting {
    half4 lighting [[color(0)]];
};

fragment Lighting light(Varyings in [[stage_in]], GBufferData gbuffer) {
    Lighting out;
    out.lighting = gbuffer.albedo * gbuffer.normal.w;
    return out;
}
"""

DIRECT_INPUT = """
#include <metal_stdlib>
using namespace metal;

struct Varyings {
    float4 position [[position]];
};

fragment half4 blend(Varyings in [[stage_in]], half4 destination [[color(0)]]) {
    return destination * 0.5h;
}
"""


@pytest.mark.parametrize(
    ("backend", "family", "reason"),
    [
        ("directx", "directx", "no framebuffer fetch"),
        ("vulkan", "vulkan", "input attachments"),
        ("opengl", "opengl", "no framebuffer fetch"),
    ],
)
@pytest.mark.parametrize(
    ("source", "name"), [(SINGLE_PASS, "gbuffer.albedo"), (DIRECT_INPUT, "destination")]
)
def test_colour_attachment_inputs_are_refused(
    tmp_path, backend, family, reason, source, name
):
    path = tmp_path / "light.metal"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ColourAttachmentInputUnsupportedError) as raised:
        crosstl.translate(str(path), backend=backend)
    assert (
        raised.value.project_diagnostic_code
        == f"project.translate.{family}-colour-attachment-input-unsupported"
    )
    assert f"through input '{name}'" in str(raised.value)
    assert reason in str(raised.value)


def test_colour_outputs_alone_still_translate(tmp_path):
    source = SINGLE_PASS.replace(", GBufferData gbuffer", "").replace(
        "gbuffer.albedo * gbuffer.normal.w", "half4(1.0h)"
    )
    path = tmp_path / "light.metal"
    path.write_text(source, encoding="utf-8")
    for backend in ("directx", "vulkan", "opengl"):
        assert crosstl.translate(str(path), backend=backend).strip()


def test_metal_keeps_programmable_blending(tmp_path):
    path = tmp_path / "light.metal"
    path.write_text(SINGLE_PASS, encoding="utf-8")
    metal = crosstl.translate(str(path), backend="metal")
    assert "[[color(1)]]" in metal
