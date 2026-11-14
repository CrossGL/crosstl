"""``a.length()`` of a sized array becomes its element count for targets without that method.

HLSL and MSL printed the GLSL method call unchanged ("member reference base type 'const float [25]' is not a
structure or union") and SPIR-V refused it as a call through a callee expression.
See :mod:`crosstl.translator.semantics.array_length`.
"""

from __future__ import annotations

import pytest

import crosstl
from tests.conformance.corpus import checks

SOURCE = """
shader LengthShader {
    struct FragmentOutput { vec4 color; }
    const int TAPS = 4;
    const float table[3] = {1.0, 2.0, 3.0};
    fragment {
        float total(float values[TAPS]) {
            float sum = 0.0;
            for (int i = 0; i < values.length(); i++) { sum += values[i]; }
            return sum;
        }

        FragmentOutput main() {
            FragmentOutput output;
            float local[TAPS];
            float acc = 0.0;
            for (int i = 0; i < table.length(); i++) { acc += table[i]; }
            for (int j = 0; j < local.length(); j++) { local[j] = acc; }
            output.color = vec4(acc, float(local.length() >> 1), total(local), 1.0);
            return output;
        }
    }
}
"""

BACKEND = {"hlsl": "directx", "metal": "metal", "spirv": "vulkan", "glsl": "opengl"}
SUFFIX = {"hlsl": "hlsl", "metal": "metal", "spirv": "spvasm", "glsl": "glsl"}


def translate(tmp_path, target):
    path = tmp_path / "shader.cgl"
    path.write_text(SOURCE)
    return crosstl.translate(str(path), backend=BACKEND[target], format_output=False)


@pytest.mark.parametrize("target", ["hlsl", "metal", "spirv", "glsl"])
def test_length_of_sized_arrays_translates_to_checked_output(tmp_path, target):
    code = translate(tmp_path, target)
    artifact = tmp_path / f"out.{SUFFIX[target]}"
    artifact.write_text(code)
    verdict = checks.check(artifact, target, ["fragment"])
    if verdict["verdict"] == "unavailable":
        pytest.skip(f"no {target} checker installed")
    assert verdict["verdict"] == "accepted", verdict.get("log", "")


@pytest.mark.parametrize("target", ["hlsl", "metal"])
def test_length_becomes_the_element_count(tmp_path, target):
    code = translate(tmp_path, target)
    assert ".length()" not in code
    assert "< 3;" in code or "< 3)" in code
    assert "< 4;" in code or "< 4)" in code


def test_glsl_keeps_its_length_method(tmp_path):
    assert "table.length()" in translate(tmp_path, "glsl")
