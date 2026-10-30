"""Barycentric and shading-rate fragment inputs read by name on HLSL and Metal, and the shading-rate flags."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import MetalBarycentricsUnsupportedError
from crosstl.translator.semantics.fragment_rate_barycentrics import SHADING_RATE_FLAGS


def _translate(tmp_path, source, backend, suffix=".frag"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _compile_metal(tmp_path, code):
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


def _barycentric(name):
    return f"""#version 450
#extension GL_EXT_fragment_shader_barycentric : require
layout(location = 0) in vec3 inColor;
layout(location = 0) out vec4 outColor;
float edge() {{ return min({name}.x, min({name}.y, {name}.z)); }}
void main() {{
    outColor = vec4(inColor * {name}, edge());
}}
"""


@pytest.mark.parametrize(
    ("name", "metal", "hlsl"),
    [
        (
            "gl_BaryCoordEXT",
            r"\[\[barycentric_coord\]\]",
            r"float3 \w+\s*:\s*SV_Barycentrics,|SV_Barycentrics\)",
        ),
        (
            "gl_BaryCoordNoPerspEXT",
            r"\[\[barycentric_coord\]\] \[\[center_no_perspective\]\]",
            r"noperspective float3 \w+\s*:\s*SV_Barycentrics1",
        ),
    ],
)
def test_barycentric_reads_become_entry_parameters(tmp_path, name, metal, hlsl):
    source = _barycentric(name)
    metal_code = _translate(tmp_path, source, "metal")
    assert not re.search(rf"\b{name}\b", metal_code), metal_code
    assert re.search(metal, metal_code), metal_code
    _compile_metal(tmp_path, metal_code)
    hlsl_code = _translate(tmp_path, source, "directx")
    assert not re.search(rf"\b{name}\b", hlsl_code), hlsl_code
    assert re.search(hlsl, hlsl_code), hlsl_code


BOTH = """#version 450
#extension GL_EXT_fragment_shader_barycentric : require
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(gl_BaryCoordEXT + gl_BaryCoordNoPerspEXT, 1.0); }
"""


def test_both_barycentric_interpolations_take_two_hlsl_indices(tmp_path):
    code = _translate(tmp_path, BOTH, "directx")
    assert re.search(r"float3 \w+\s*:\s*SV_Barycentrics,", code), code
    assert re.search(r"noperspective float3 \w+\s*:\s*SV_Barycentrics1\)", code), code


def test_both_barycentric_interpolations_are_refused_on_metal(tmp_path):
    with pytest.raises(MetalBarycentricsUnsupportedError) as caught:
        _translate(tmp_path, BOTH, "metal")
    assert caught.value.code == "project.translate.metal-barycentrics-unsupported"


SHADING_RATE = """#version 450
#extension GL_EXT_fragment_shading_rate : require
layout(location = 0) out vec4 outColor;
void main() {
    int h = (gl_ShadingRateEXT & gl_ShadingRateFlag4HorizontalPixelsEXT) != 0 ? 4 : 1;
    int v = (gl_ShadingRateEXT & gl_ShadingRateFlag2VerticalPixelsEXT) != 0 ? 2 : 1;
    outColor = vec4(float(h), float(v), 0.0, 1.0);
}
"""


def test_shading_rate_reads_and_flags_reach_hlsl(tmp_path):
    code = _translate(tmp_path, SHADING_RATE, "directx")
    assert re.search(r"uint \w+\s*:\s*SV_ShadingRate", code), code
    assert not re.search(r"\bgl_ShadingRate", code), code
    # D3D12_SHADING_RATE encodes log2(width) << 2 | log2(height), the bits of the GLSL flags.
    assert re.search(r"& 8\)", code) and re.search(r"& 1\)", code), code


def test_shading_rate_is_refused_on_metal(tmp_path):
    with pytest.raises(Exception, match="gl_ShadingRateEXT"):
        _translate(tmp_path, SHADING_RATE, "metal")


def test_shading_rate_flags_keep_their_glsl_names(tmp_path):
    code = _translate(tmp_path, SHADING_RATE, "opengl")
    assert "gl_ShadingRateFlag4HorizontalPixelsEXT" in code


def test_shading_rate_flag_values_match_the_extension():
    assert SHADING_RATE_FLAGS == {
        "gl_ShadingRateFlag2VerticalPixelsEXT": 1,
        "gl_ShadingRateFlag4VerticalPixelsEXT": 2,
        "gl_ShadingRateFlag2HorizontalPixelsEXT": 4,
        "gl_ShadingRateFlag4HorizontalPixelsEXT": 8,
    }
