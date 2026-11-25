"""Interpolation qualifiers cross Metal in both directions as one mode and one sampling location.

CrossGL states interpolation as GLSL and HLSL do: a mode (``@ flat``, ``@ smooth``, ``@ noperspective``) and a
sampling location (``@ centroid``, ``@ sample``). Metal spells the pair as one attribute (``[[flat]]``,
``[[center_no_perspective]]``, ``[[centroid_perspective]]``, ``[[sample_no_perspective]]`` ...).

Metal output lost the qualifier of a fragment entry parameter (``float4 main(nointerpolation float2 uv :
TEXCOORD0)``), whose generated ``[[stage_in]]`` member had only ``[[user(TEXCOORD0)]]``, and kept only the first
half of ``noperspective centroid``. Metal input imported ``[[centroid_perspective]]`` and its siblings verbatim, as
attributes no other generator reads, and did not map ``[[amplification_id]]`` to the view index.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.refusals import MetalRefusal

HLSL_QUALIFIERS = {
    "nointerpolation": "flat",
    "linear": None,
    "centroid": "centroid_perspective",
    "linear centroid": "centroid_perspective",
    "sample": "sample_perspective",
    "noperspective": "center_no_perspective",
    "noperspective centroid": "centroid_no_perspective",
    "noperspective sample": "sample_no_perspective",
}

METAL_QUALIFIERS = {
    "flat": ("@flat", "flat in vec2 uv", "nointerpolation float2 uv"),
    "center_no_perspective": (
        "@noperspective",
        "noperspective in vec2 uv",
        "noperspective float2 uv",
    ),
    "centroid_perspective": ("@centroid", "centroid in vec2 uv", "centroid float2 uv"),
    "centroid_no_perspective": (
        "@noperspective @centroid",
        "noperspective centroid in vec2 uv",
        "noperspective centroid float2 uv",
    ),
    "sample_perspective": ("@sample", "sample in vec2 uv", "sample float2 uv"),
    "sample_no_perspective": (
        "@noperspective @sample",
        "noperspective sample in vec2 uv",
        "noperspective sample float2 uv",
    ),
}


def _translate(tmp_path, source, name, backend):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _compile_metal(tmp_path, source):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return
    artifact = tmp_path / "translated.metal"
    artifact.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [xcrun, "--sdk", "macosx", "metal", "-Werror", "-c", str(artifact)]
        + ["-o", str(tmp_path / "translated.air")],
        capture_output=True,
        text=True,
        check=False,
    )
    if "unable to find utility" in result.stderr:
        return
    assert result.returncode == 0, result.stderr


def _metal_member(metal, name):
    match = re.search(rf"float2 {name}((?: \[\[[a-z_0-9()A-Z]+\]\])*);", metal)
    assert match is not None, metal
    return re.findall(r"\[\[([a-z_]+)", match.group(1))


@pytest.mark.parametrize("qualifier", sorted(HLSL_QUALIFIERS))
def test_hlsl_entry_parameter_keeps_its_interpolation_in_metal(tmp_path, qualifier):
    source = (
        f"float4 main({qualifier} float2 uv : TEXCOORD0) : SV_Target {{\n"
        "    return float4(uv, 0.0, 1.0);\n}\n"
    )
    metal = _translate(tmp_path, source, "probe.hlsl", "metal")

    attributes = _metal_member(metal, "uv")
    expected = HLSL_QUALIFIERS[qualifier]
    assert attributes == (["user", expected] if expected else ["user"])
    _compile_metal(tmp_path, metal)


@pytest.mark.parametrize("qualifier", sorted(HLSL_QUALIFIERS))
def test_hlsl_struct_member_keeps_both_halves_of_its_interpolation(tmp_path, qualifier):
    source = (
        "struct PSInput {\n"
        "    float4 position : SV_Position;\n"
        f"    {qualifier} float2 uv : TEXCOORD0;\n"
        "};\n"
        "float4 main(PSInput input) : SV_Target {\n"
        "    return float4(input.uv, 0.0, 1.0);\n}\n"
    )
    metal = _translate(tmp_path, source, "probe.hlsl", "metal")

    expected = HLSL_QUALIFIERS[qualifier]
    attributes = _metal_member(metal, "uv")
    assert [a for a in attributes if a != "user"] == ([expected] if expected else [])
    _compile_metal(tmp_path, metal)


@pytest.mark.parametrize(
    ("qualifier", "expected"),
    [
        ("nointerpolation", "flat"),
        ("linear_centroid", "centroid_perspective"),
        ("linear_noperspective_centroid", "centroid_no_perspective"),
        ("noperspective sample", "sample_no_perspective"),
    ],
)
def test_crossgl_parameter_qualifier_states_the_interpolation(
    tmp_path, qualifier, expected
):
    # The Slang importer writes ``nointerpolation vec2 uv``: a parameter qualifier, not an attribute.
    source = f"""shader main {{
    fragment {{
        vec4 main({qualifier} vec2 uv @ TexCoord0) @ gl_FragColor {{
            return vec4(uv, 0.0, 1.0);
        }}
    }}
}}
"""
    metal = _translate(tmp_path, source, "probe.cgl", "metal")

    assert _metal_member(metal, "uv") == ["user", expected]
    _compile_metal(tmp_path, metal)


def test_conflicting_interpolation_modes_fail_closed(tmp_path):
    source = """shader main {
    fragment {
        vec4 main(vec2 uv @ TexCoord0 @ centroid_perspective @ noperspective) @ gl_FragColor {
            return vec4(uv, 0.0, 1.0);
        }
    }
}
"""
    with pytest.raises(MetalRefusal, match="conflicting interpolation"):
        _translate(tmp_path, source, "probe.cgl", "metal")


@pytest.mark.parametrize("qualifier", sorted(METAL_QUALIFIERS))
def test_metal_interpolation_imports_as_mode_and_sampling(tmp_path, qualifier):
    source = (
        "#include <metal_stdlib>\nusing namespace metal;\n\n"
        f"struct FragmentIn {{\n    float2 uv [[{qualifier}]];\n}};\n\n"
        "fragment float4 probe(FragmentIn in [[stage_in]]) {\n"
        "    return float4(in.uv, 0.0, 1.0);\n}\n"
    )
    crossgl_attributes, glsl_declaration, hlsl_declaration = METAL_QUALIFIERS[qualifier]

    crossgl = _translate(tmp_path, source, "probe.metal", "cgl")
    assert f"vec2 uv {crossgl_attributes};" in crossgl
    assert f"@{qualifier}" not in crossgl or qualifier == "flat"
    assert glsl_declaration in _translate(tmp_path, source, "probe.metal", "opengl")
    assert hlsl_declaration in _translate(tmp_path, source, "probe.metal", "directx")

    metal = _translate(tmp_path, source, "probe.metal", "metal")
    assert _metal_member(metal, "uv") == [qualifier]
    _compile_metal(tmp_path, metal)


def test_metal_amplification_id_imports_as_the_view_index(tmp_path):
    source = """#include <metal_stdlib>
using namespace metal;

struct FragmentIn {
    float2 uv;
};

fragment float4 probe(FragmentIn in [[stage_in]], uint view [[amplification_id]]) {
    return float4(in.uv, float(view), 1.0);
}
"""
    crossgl = _translate(tmp_path, source, "probe.metal", "cgl")
    assert "uint view @gl_ViewIndex" in crossgl
    assert "@amplification_id" not in crossgl

    glsl = _translate(tmp_path, source, "probe.metal", "opengl")
    assert "GL_EXT_multiview" in glsl
    assert "gl_ViewIndex" in glsl
    assert "uint view : SV_ViewID" in _translate(
        tmp_path, source, "probe.metal", "directx"
    )

    metal = _translate(tmp_path, source, "probe.metal", "metal")
    assert "uint view [[amplification_id]]" in metal
    _compile_metal(tmp_path, metal)
