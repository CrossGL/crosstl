"""Fixed sampler state: decoded and validated, kept for Metal, refused where a target cannot declare it.

Metal's ``constexpr sampler`` used to lose its state on the way through CrossGL: program-scope samplers were dropped
and their sample calls fell back to a default sampler, GLSL sampled with whatever sampler object the host bound, and
HLSL received ``sampler(address::clamp_to_edge)`` verbatim. See :mod:`crosstl.translator.semantics.sampler_states`.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.semantics.sampler_states import (
    FixedSamplerStateUnsupportedError,
    SamplerState,
    SamplerStateArgumentError,
    decode_sampler_state,
    legalize_sampler_states,
)

SOURCE = """#include <metal_stdlib>
using namespace metal;
constexpr sampler linearRepeat(coord::normalized, s_address::repeat,
                               t_address::mirrored_repeat, filter::linear,
                               mip_filter::linear, max_anisotropy(8),
                               lod_clamp(0.0f, 4.0f));
struct VOut { float4 position [[position]]; float2 uv; };
fragment float4 shade(VOut in [[stage_in]], texture2d<float> tex [[texture(0)]],
                      depth2d<float> depth [[texture(1)]]) {
    constexpr sampler fallback;
    constexpr sampler compare(compare_func::less, address::clamp_to_border,
                              border_color::opaque_white);
    float4 color = tex.sample(linearRepeat, in.uv) + tex.sample(fallback, in.uv);
    float lit = depth.sample_compare(compare, in.uv, 0.5);
    return color * lit + tex.sample(sampler(metal::filter::nearest), in.uv);
}
"""


def call(*arguments):
    return A.FunctionCallNode(A.IdentifierNode("sampler"), list(arguments))


def ident(name):
    return A.IdentifierNode(name)


def number(value):
    return A.LiteralNode(value, A.PrimitiveType("float"))


def translate(tmp_path, backend):
    path = tmp_path / "shader.metal"
    path.write_text(SOURCE)
    return crosstl.translate(str(path), backend=backend)


def test_decoding_applies_metal_defaults_groups_and_later_overrides():
    assert decode_sampler_state(call()) == SamplerState()
    state = decode_sampler_state(
        call(
            ident("address::repeat"),
            ident("t_address::clamp_to_zero"),
            ident("filter::linear"),
            ident("metal::mip_filter::nearest"),
            ident("compare_func::greater_equal"),
            A.FunctionCallNode(ident("max_anisotropy"), [number(16)]),
            A.FunctionCallNode(ident("lod_clamp"), [number(1.0), number(2.5)]),
        )
    )

    assert (state.s_address, state.t_address, state.r_address) == (
        "repeat",
        "clamp_to_zero",
        "repeat",
    )
    assert (state.mag_filter, state.min_filter, state.mip_filter) == (
        "linear",
        "linear",
        "nearest",
    )
    assert state.compare_func == "greater_equal"
    assert state.max_anisotropy == 16
    assert state.lod_clamp == (1.0, 2.5)


@pytest.mark.parametrize(
    "argument",
    [
        ident("address::wrap"),
        ident("anisotropy::linear"),
        ident("linear"),
        A.FunctionCallNode(ident("max_anisotropy"), [number(17)]),
        A.FunctionCallNode(ident("lod_clamp"), [number(3.0), number(1.0)]),
        A.FunctionCallNode(ident("lod_bias"), [number(1.0)]),
    ],
)
def test_arguments_outside_the_vocabulary_are_refused_for_every_target(argument):
    with pytest.raises(SamplerStateArgumentError) as raised:
        decode_sampler_state(call(argument))

    assert (
        raised.value.project_diagnostic_code
        == "project.translate.sampler-state-argument-invalid"
    )


def test_metal_round_trip_keeps_every_fixed_sampler(tmp_path):
    crossgl = translate(tmp_path, "cgl")
    metal = translate(tmp_path, "metal")

    assert (
        "const sampler linearRepeat = sampler(coord::normalized, s_address::repeat"
        in crossgl
    )
    assert "max_anisotropy(8), lod_clamp(0.0f, 4.0f)" in crossgl
    assert "texture(tex, linearRepeat, in_.uv)" in crossgl
    assert "sampler(filter::nearest)" in crossgl
    assert "metal::" not in crossgl
    assert "linearRepeat = sampler(" in metal
    assert "tex.sample(linearRepeat, in_.uv)" in metal
    assert "tex.sample(fallback, in_.uv)" in metal
    assert "depth.sample_compare(compare, in_.uv, 0.5)" in metal
    assert "tex.sample(sampler(filter::nearest), in_.uv)" in metal


def test_metal_round_trip_compiles(tmp_path):
    if shutil.which("xcrun") is None:
        pytest.skip("the Metal toolchain is not installed")
    metal = tmp_path / "out.metal"
    metal.write_text(translate(tmp_path, "metal"))
    compiled = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(metal),
            "-o",
            str(tmp_path / "out.air"),
        ],
        capture_output=True,
        text=True,
    )
    if "unable to find utility" in compiled.stderr:
        pytest.skip("the Metal compiler is not installed")
    assert compiled.returncode == 0, compiled.stderr


@pytest.mark.parametrize(
    ("backend", "code", "cites"),
    [
        ("opengl", "opengl", "GLSL 4.60 section 4.1.7"),
        ("vulkan", "vulkan", "immutable samplers"),
        ("directx", "directx", "root-signature static samplers"),
    ],
)
def test_targets_without_shader_declared_sampler_state_refuse_it(
    tmp_path, backend, code, cites
):
    with pytest.raises(FixedSamplerStateUnsupportedError) as raised:
        translate(tmp_path, backend)

    assert (
        raised.value.project_diagnostic_code
        == f"project.translate.{code}-fixed-sampler-state-unsupported"
    )
    assert cites in str(raised.value)


def test_default_constructed_constant_samplers_are_fixed_state():
    program = crosstl.translator.parse("""
        shader main {
            sampler linearSampler;
            fragment {
                vec4 main(sampler2D tex @texture(0)) @gl_FragColor {
                    const sampler fallback;
                    return texture(tex, fallback, vec2(0.5));
                }
            }
        }
        """)
    bound_only = crosstl.translator.parse("""
        shader main {
            sampler linearSampler;
            fragment {
                vec4 main(sampler2D tex @texture(0)) @gl_FragColor {
                    return texture(tex, linearSampler, vec2(0.5));
                }
            }
        }
        """)

    assert legalize_sampler_states(program, "metal") == 1
    with pytest.raises(FixedSamplerStateUnsupportedError, match="'fallback'"):
        legalize_sampler_states(program, "opengl")
    # A sampler resource the host binds is not fixed state.
    assert legalize_sampler_states(bound_only, "opengl") == 0
