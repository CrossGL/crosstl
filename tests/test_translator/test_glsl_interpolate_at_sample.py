"""GLSL output passes interpolateAtSample an ``int`` sample number.

GLSL declares ``interpolateAtSample(T interpolant, int sample)`` and converts no other integer type to ``int``
(GLSL 4.60 sections 8.13.2 and 4.1.10). HLSL's ``EvaluateAttributeAtSample`` and MSL's ``interpolate_at_sample`` take
a ``uint``, so a translated call converts the sample number.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.refusals import OpenGLRefusal

GLSLANG = shutil.which("glslangValidator")

HLSL = """
float4 PSMain(float4 pos : SV_Position, float2 uv : TEXCOORD0, uint s : SV_SampleIndex) : SV_Target {
    float2 a = EvaluateAttributeAtSample(uv, s);
    float2 b = EvaluateAttributeAtSample(uv, 2u);
    return float4(a, b);
}
"""

GLSL = """#version 450
in vec2 uv;
out vec4 color;
void main() {
    color = vec4(interpolateAtSample(uv, gl_SampleID), interpolateAtSample(uv, 1));
}
"""


def _translate(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return crosstl.translate(str(path), backend="opengl", format_output=False)


def _glslang(tmp_path, text):
    if GLSLANG is None:
        return
    path = tmp_path / "out.frag"
    path.write_text(text, encoding="utf-8")
    result = subprocess.run(
        [GLSLANG, str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_uint_sample_number_is_converted_to_int(tmp_path):
    translated = _translate(tmp_path, "eval.hlsl", HLSL)

    assert "interpolateAtSample(uv, int(s))" in translated
    assert "interpolateAtSample(uv, int(2u))" in translated
    _glslang(tmp_path, translated)


def test_an_int_sample_number_is_kept(tmp_path):
    translated = _translate(tmp_path, "eval.frag", GLSL)

    assert "interpolateAtSample(uv, gl_SampleID)" in translated
    assert "interpolateAtSample(uv, 1)" in translated
    assert "int(" not in translated
    _glslang(tmp_path, translated)


@pytest.mark.parametrize("sample", ["0.5", "uv"])
def test_a_non_integer_sample_number_is_refused(tmp_path, sample):
    source = GLSL.replace(
        "interpolateAtSample(uv, 1)", f"interpolateAtSample(uv, {sample})"
    )
    with pytest.raises(OpenGLRefusal, match="scalar integer"):
        _translate(tmp_path, "bad.frag", source)
