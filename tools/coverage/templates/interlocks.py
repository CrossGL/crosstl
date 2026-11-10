"""Rasterizer-ordered probes: a fragment shader that reads and writes an image or a buffer at its own pixel, with the
accesses of overlapping fragments ordered as the primitives were submitted.

Each probe adds its texture coordinates to the value at the fragment's pixel and writes it back. GLSL brackets the
accesses with ``beginInvocationInterlockARB`` and ``endInvocationInterlockARB`` under
``layout(pixel_interlock_ordered) in`` (GL_ARB_fragment_shader_interlock). HLSL and Slang declare a
``RasterizerOrderedTexture2D`` or ``RasterizerOrderedStructuredBuffer``, which DXC and slangc lower to the same
interlocked region, Metal places the argument in a ``raster_order_group``, and CrossGL marks the resource
``@rasterizer_ordered``.

WGSL and GLSL ES 3.00 have no ordered fragment access, and the kernel languages have no fragment stage. rust-gpu sets
the ``pixel_interlock_ordered_ext`` execution mode on an entry point but cannot open the interlocked region: spirv-std
has no function for it, and rust-gpu's inline SPIR-V rejects ``OpBeginInvocationInterlockEXT`` as reserved for
SPV_EXT_fragment_shader_interlock.
"""

from __future__ import annotations

from .base import ABSENT, Template

NAMES = ("rasterizer-ordered-image", "rasterizer-ordered-buffer")

_GLSL_HEADER = (
    "#version 460\n#extension GL_ARB_fragment_shader_interlock : require\n"
    "layout(pixel_interlock_ordered) in;\n"
    "layout(location = 0) in vec2 uv;\nlayout(location = 0) out vec4 color;\n"
)
# Each probe as its resource and the statements that read and write it, in GLSL.
_GLSL = {
    "rasterizer-ordered-image": (
        "layout(rgba32f, binding = 0) uniform image2D accum;\n",
        "ivec2 p = ivec2(gl_FragCoord.xy);\n    beginInvocationInterlockARB();\n"
        "    imageStore(accum, p, imageLoad(accum, p) + vec4(uv, 0.0, 1.0));\n",
    ),
    "rasterizer-ordered-buffer": (
        "layout(std430, binding = 0) buffer Accum {\n    vec4 accum[];\n};\n",
        "uint k = uint(gl_FragCoord.y) * 64u + uint(gl_FragCoord.x);\n"
        "    beginInvocationInterlockARB();\n    accum[k] = accum[k] + vec4(uv, 0.0, 1.0);\n",
    ),
}
GLSL = {
    name: Template(
        program=f"{_GLSL_HEADER}{resource}void main() {{\n    {statements}"
        "    endInvocationInterlockARB();\n    color = vec4(uv, 0.0, 1.0);\n}\n"
    )
    for name, (resource, statements) in _GLSL.items()
}

# HLSL and Slang: the resource, the index type and how it is computed from the fragment position.
_HLSL = {
    "rasterizer-ordered-image": (
        "RasterizerOrderedTexture2D<float4> accum : register(u1);",
        "uint2 p = uint2(coord.xy);\n    accum[p] = accum[p] + float4(uv, 0.0, 1.0);",
    ),
    "rasterizer-ordered-buffer": (
        "RasterizerOrderedStructuredBuffer<float4> accum : register(u1);",
        "uint k = uint(coord.y) * 64u + uint(coord.x);\n"
        "    accum[k] = accum[k] + float4(uv, 0.0, 1.0);",
    ),
}


def _hlsl(attribute: str = "") -> dict[str, Template]:
    """The HLSL programs; Slang's name their stage with ``attribute`` on the entry point."""
    return {
        name: Template(
            program=f"{resource}\n\n{attribute}"
            "float4 main(float4 coord : SV_Position, float2 uv : TEXCOORD0) : SV_Target {\n"
            f"    {statements}\n    return float4(uv, 0.0, 1.0);\n}}\n",
            entry="main",
        )
        for name, (resource, statements) in _HLSL.items()
    }


HLSL = _hlsl()
SLANG = _hlsl('[shader("fragment")]\n')

_METAL = {
    "rasterizer-ordered-image": (
        "texture2d<float, access::read_write> accum [[texture(0), raster_order_group(0)]]",
        "uint2 p = uint2(in.coord.xy);\n"
        "    accum.write(accum.read(p) + float4(in.uv, 0.0, 1.0), p);",
    ),
    "rasterizer-ordered-buffer": (
        "device float4 *accum [[buffer(0), raster_order_group(0)]]",
        "uint k = uint(in.coord.y) * 64u + uint(in.coord.x);\n"
        "    accum[k] = accum[k] + float4(in.uv, 0.0, 1.0);",
    ),
}
METAL = {
    name: Template(
        program="#include <metal_stdlib>\nusing namespace metal;\n\n"
        "struct FragmentIn {\n    float4 coord [[position]];\n    float2 uv;\n};\n\n"
        f"fragment float4 probe(FragmentIn in [[stage_in]], {argument}) {{\n"
        f"    {statements}\n    return float4(in.uv, 0.0, 1.0);\n}}\n"
    )
    for name, (argument, statements) in _METAL.items()
}

WGSL = dict.fromkeys(NAMES, ABSENT)
WEBGL = dict.fromkeys(NAMES, ABSENT)
KERNELS = dict.fromkeys(NAMES, ABSENT)
RUST = dict.fromkeys(NAMES, ABSENT)

_CROSSGL = {
    "rasterizer-ordered-image": (
        "image2D accum @rasterizer_ordered @rgba32f;",
        "ivec2 p = ivec2(input.coord.xy);\n"
        "            imageStore(accum, p, imageLoad(accum, p) + vec4(input.uv, 0.0, 1.0));",
    ),
    "rasterizer-ordered-buffer": (
        "RWStructuredBuffer<vec4> accum @rasterizer_ordered;",
        "uint k = uint(input.coord.y) * 64u + uint(input.coord.x);\n"
        "            accum[k] = accum[k] + vec4(input.uv, 0.0, 1.0);",
    ),
}
CROSSGL = {
    name: Template(
        program=f"shader Probe {{\n    {resource}\n"
        "    struct FragmentInput {\n        vec4 coord @ gl_FragCoord;\n        vec2 uv;\n    }\n"
        "    struct FragmentOutput {\n        vec4 color;\n    }\n"
        "    fragment {\n        FragmentOutput main(FragmentInput input) {\n"
        f"            FragmentOutput output;\n            {statements}\n"
        "            output.color = vec4(input.uv, 0.0, 1.0);\n"
        "            return output;\n        }\n    }\n}\n"
    )
    for name, (resource, statements) in _CROSSGL.items()
}
