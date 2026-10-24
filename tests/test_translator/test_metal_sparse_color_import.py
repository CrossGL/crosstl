"""MSL sparse texture reads import as CrossGL texture residency queries.

``texture.sparse_sample(s, uv)`` and the other ``sparse_*`` members return ``sparse_color<T>``: the texel and whether
its tile is resident (MSL specification, section 6.12). A local initialised from one becomes a texel and a
``textureResidency`` code, ``value()`` the texel and ``resident()`` ``textureResident(code)``. Every other use of a
``sparse_color`` has no CrossGL form and is refused.
"""

from __future__ import annotations

import platform
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.import_refusals import MetalImportRefused
from crosstl.translator.codegen.texture_residency import (
    TextureResidencyUnsupportedError,
)

HEADER = """
#include <metal_stdlib>
using namespace metal;

struct Varyings {
    float4 position [[position]];
    float2 uv;
};
"""

FRAGMENT = HEADER + """
fragment float4 shade(Varyings in [[stage_in]],
                      texture2d<float> image [[texture(0)]],
                      depth2d<float> depth [[texture(1)]],
                      sampler linear [[sampler(0)]]) {
    sparse_color<float4> c = image.sparse_sample(linear, in.uv, level(1.0));
    auto r = image.sparse_read(uint2(in.position.xy), 0);
    sparse_color<float4> g = image.sparse_gather(linear, in.uv, int2(0), component::y);
    auto z = depth.sparse_sample_compare(linear, in.uv, 0.5);
    auto dg = depth.sparse_gather_compare(linear, in.uv, 0.5);
    float4 color = c.value() + r.value() + g.value() + float4(z.value()) + dg.value();
    color += image.sparse_sample(linear, in.uv).value();
    if (!c.resident() || !r.resident() || !g.resident() || !z.resident() || !dg.resident()) {
        color = float4(0.0);
    }
    c = image.sparse_sample(linear, in.uv + 0.5);
    if (c.resident()) {
        color += c.value();
    }
    return color;
}
"""

HALF_FRAGMENT = HEADER + """
fragment half4 shade(Varyings in [[stage_in]],
                     texture2d<half> image [[texture(0)]],
                     sampler linear [[sampler(0)]]) {
    sparse_color<half4> color = image.sparse_sample(linear, in.uv);
    half4 base = half4(0.0h);
    if (color.resident()) {
        base = color.value();
    }
    return base;
}
"""


def _write(tmp_path, source):
    path = tmp_path / "shade.metal"
    path.write_text(source, encoding="utf-8")
    return path


def _translate(tmp_path, source, backend="cgl"):
    return crosstl.translate(
        str(_write(tmp_path, source)), backend=backend, format_output=False
    )


def _compact(code):
    return " ".join(code.split())


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _has_metal_compiler():
    return (
        platform.system() == "Darwin"
        and shutil.which("xcrun") is not None
        and subprocess.run(
            ["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True
        ).returncode
        == 0
    )


def test_sparse_color_locals_import_as_residency_queries(tmp_path):
    # The importer renames the MSL parameter ``in``, a CrossGL keyword, to ``in_``.
    code = _compact(_translate(tmp_path, FRAGMENT))
    for line in (
        "vec4 crosstl_sparse_c_value; int crosstl_sparse_c_code = "
        "textureResidency(textureLod(image, linear, in_.uv, 1.0), crosstl_sparse_c_value);",
        "vec4 crosstl_sparse_r_value; int crosstl_sparse_r_code = "
        "textureResidency(texelFetch(image, uvec2(in_.position.xy), 0), crosstl_sparse_r_value);",
        "vec4 crosstl_sparse_g_value; int crosstl_sparse_g_code = "
        "textureResidency(textureGatherOffset(image, linear, in_.uv, ivec2(0), 1), crosstl_sparse_g_value);",
        "float crosstl_sparse_z_value; int crosstl_sparse_z_code = "
        "textureResidency(textureCompare(depth, linear, in_.uv, 0.5), crosstl_sparse_z_value);",
        "vec4 crosstl_sparse_dg_value; int crosstl_sparse_dg_code = "
        "textureResidency(textureGatherCompare(depth, linear, in_.uv, 0.5), crosstl_sparse_dg_value);",
        "vec4 color = crosstl_sparse_c_value + crosstl_sparse_r_value + crosstl_sparse_g_value + "
        "vec4(crosstl_sparse_z_value) + crosstl_sparse_dg_value;",
        # value() of a read with no local is the ordinary read.
        "color += texture(image, linear, in_.uv);",
        "(!textureResident(crosstl_sparse_c_code)) || (!textureResident(crosstl_sparse_r_code))",
        # Reassigning the local reads into the same texel and code.
        "crosstl_sparse_c_code = textureResidency(texture(image, linear, in_.uv + 0.5), crosstl_sparse_c_value);",
        "if (textureResident(crosstl_sparse_c_code)) {",
    ):
        assert line in code
    assert "sparse_color" not in code
    assert ".value()" not in code and ".resident()" not in code


def test_half_texel_type_is_kept(tmp_path):
    code = _compact(_translate(tmp_path, HALF_FRAGMENT))
    assert (
        "f16vec4 crosstl_sparse_color_value; int crosstl_sparse_color_code = "
        "textureResidency(texture(image, linear, in_.uv), crosstl_sparse_color_value);"
    ) in code
    assert "base = crosstl_sparse_color_value;" in code


@pytest.mark.parametrize("source", [FRAGMENT, HALF_FRAGMENT], ids=["float", "half"])
def test_sparse_reads_round_trip_to_msl(tmp_path, source):
    code = _translate(tmp_path, source, "metal")
    compact = _compact(code)
    assert "sparse_color" in compact
    assert "image.sparse_sample(linear, in_.uv" in compact
    if _has_metal_compiler():
        path = tmp_path / "out.metal"
        path.write_text(code)
        _run(
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metal",
                "-std=macos-metal2.3",
                "-c",
                str(path),
                "-o",
                str(tmp_path / "out.air"),
            ]
        )


def test_sparse_reads_translate_to_arb_sparse_texture2(tmp_path):
    code = _translate(tmp_path, FRAGMENT, "opengl")
    compact = _compact(code)
    assert "#extension GL_ARB_sparse_texture2 : require" in code
    for call in (
        "sparseTextureLodARB(",
        "sparseTexelFetchARB(",
        "sparseTextureGatherOffsetARB(",
        "sparseTexelsResidentARB(crosstl_sparse_c_code)",
    ):
        assert call in compact
    if shutil.which("glslangValidator"):
        path = tmp_path / "out.frag"
        path.write_text(code)
        _run(["glslangValidator", "-S", "frag", str(path)])


def test_sparse_reads_translate_to_hlsl_status_overloads(tmp_path):
    # HLSL compares through a SamplerComparisonState, so the depth reads take a sampler of their own.
    source = (
        FRAGMENT.replace(
            "sampler linear [[sampler(0)]]",
            "sampler linear [[sampler(0)]], sampler shadow [[sampler(1)]]",
        )
        .replace(
            "depth.sparse_sample_compare(linear", "depth.sparse_sample_compare(shadow"
        )
        .replace(
            "depth.sparse_gather_compare(linear", "depth.sparse_gather_compare(shadow"
        )
    )
    code = _translate(tmp_path, source, "directx")
    compact = _compact(code)
    for read in (
        "image.SampleLevel(linear_, in_.uv, 1.0, int2(0, 0), crosstl_residency_status_0)",
        "image.Load(int3(uint2(in_.position.xy), 0), int2(0, 0), crosstl_residency_status_1)",
        "depth.SampleCmp(shadow, in_.uv, 0.5, int2(0, 0), 0.0f, crosstl_residency_status_3)",
        "CheckAccessFullyMapped(crosstl_residency_status_0)) ? 0 : 1)",
    ):
        assert read in compact
    dxc = shutil.which("dxc")
    if dxc:
        path = tmp_path / "out.hlsl"
        path.write_text(code)
        _run([dxc, "-T", "ps_6_0", "-E", "PSMain", str(path)])


def test_vulkan_refuses_residency_queries(tmp_path):
    with pytest.raises(TextureResidencyUnsupportedError) as raised:
        _translate(tmp_path, HALF_FRAGMENT, "vulkan")
    assert raised.value.project_diagnostic_code == (
        "project.translate.vulkan-texture-residency-unsupported"
    )


def _fragment(body, parameters=""):
    return HEADER + f"""
{parameters}
fragment float4 shade(Varyings in [[stage_in]],
                      texture2d<float> image [[texture(0)]],
                      sampler linear [[sampler(0)]]) {{
{body}
}}
"""


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            _fragment(
                "return image.sparse_sample(linear, in.uv).resident() ? float4(1.0) : float4(0.0);"
            ),
            "needs a local to receive the texel",
        ),
        (
            _fragment("sparse_color<float4> c;\n    return c.value();"),
            "type 'sparse_color<float4>'",
        ),
        (
            _fragment(
                "sparse_color<float4> c = image.sparse_sample(linear, in.uv);\n"
                "    sparse_color<float4> d = c;\n    return d.value();"
            ),
            "only a single local assigned directly from a sparse texture read",
        ),
        (
            _fragment(
                "sparse_color<float4> c = image.sparse_sample(linear, in.uv);\n"
                "    return pick(c);",
                "float4 pick(sparse_color<float4> p) { return p.value(); }",
            ),
            "type 'sparse_color<float4>'",
        ),
        (
            _fragment(
                "sparse_color<float4> c = image.sparse_sample(linear, in.uv);\n"
                "    float4 v = c.value();\n    return v + pick(c);",
                "float4 pick(float4 p) { return p; }",
            ),
            "sparse_color value 'c'",
        ),
        (
            _fragment(
                "sparse_color<float4> c = image.sparse_sample(linear, in.uv);\n"
                "    return c.value(0);"
            ),
            "only value() and resident()",
        ),
        (
            _fragment(
                "sparse_color<float4> c[2] = {image.sparse_sample(linear, in.uv), "
                "image.sparse_sample(linear, in.uv)};\n    return c[0].value();"
            ),
            "sparse_color",
        ),
        (
            HEADER + """
struct Tile { sparse_color<float4> texel; };
fragment float4 shade(Varyings in [[stage_in]]) { return float4(0.0); }
""",
            "type 'sparse_color<float4>'",
        ),
    ],
)
def test_other_sparse_color_uses_are_refused(tmp_path, source, message):
    with pytest.raises(MetalImportRefused) as raised:
        _translate(tmp_path, source)
    assert message in str(raised.value)
    assert (
        raised.value.project_diagnostic_code
        == "project.translate.metal-import-unsupported"
    )


def test_a_local_is_read_any_number_of_times(tmp_path):
    source = _fragment(
        "sparse_color<float4> c = image.sparse_sample(linear, in.uv);\n"
        "    float4 v = c.value();\n    float4 w = c.value() * 2.0;\n"
        "    return v + w + float4(float(c.resident()));"
    )
    code = _compact(_translate(tmp_path, source))
    assert "sparse_color" not in code
    assert "vec4 v = crosstl_sparse_c_value;" in code
    assert "vec4 w = crosstl_sparse_c_value * 2.0;" in code
    assert "float(textureResident(crosstl_sparse_c_code))" in code
