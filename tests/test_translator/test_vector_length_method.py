"""``v.length()`` of a vector or a matrix becomes its constant count for targets without that method.

GLSL gives a vector's ``length()`` its component count and a matrix's its column count (GLSL 4.60 sections 5.5
and 5.6). HLSL and MSL printed the method call unchanged (``lights[i].length()``: "illegal vector component name
'l'") and SPIR-V refused it as a call through a callee expression.
See :mod:`crosstl.translator.semantics.array_length`.
"""

from __future__ import annotations

import pytest

import crosstl
from tests.conformance.corpus import checks

GLSL_SOURCE = """#version 450
layout (binding = 1) uniform Params {
    vec4 lights[4];
    mat3 basis;
    mat2x4 wide;
} params;
layout (location = 0) in vec2 inUV;
layout (location = 0) out vec4 outColor;

vec3 tint() { return vec3(0.5); }

void main() {
    vec3 c = vec3(0.0);
    for (int i = 0; i < params.lights[i].length(); i++) {
        c += params.lights[i].xyz;
    }
    int columns = params.basis.length() + params.wide.length();
    int rows = params.basis[0].length() + params.wide[1].length();
    int widths = inUV.length() + inUV.xyx.length() + tint().length() + (c * 2.0).length();
    outColor = vec4(c, float(columns * 100 + rows * 10 + widths));
}
"""

CGL_SOURCE = """
shader VectorLengthShader {
    struct FragmentOutput { vec4 color; }
    fragment {
        int width(vec3 v) { return v.length(); }

        FragmentOutput main() {
            FragmentOutput output;
            vec4 colour = vec4(0.25);
            mat4x3 frame = mat4x3(1.0);
            int count = colour.length() + colour.zw.length() + frame.length() + frame[2].length();
            output.color = vec4(float(count), float(width(colour.rgb)), 0.0, 1.0);
            return output;
        }
    }
}
"""

BACKEND = {"hlsl": "directx", "metal": "metal", "spirv": "vulkan", "glsl": "opengl"}
SUFFIX = {"hlsl": "hlsl", "metal": "metal", "spirv": "spvasm", "glsl": "glsl"}
SOURCES = {"glsl": ("shader.frag", GLSL_SOURCE), "cgl": ("shader.cgl", CGL_SOURCE)}


def translate(tmp_path, language, target):
    name, text = SOURCES[language]
    path = tmp_path / name
    path.write_text(text)
    return crosstl.translate(str(path), backend=BACKEND[target], format_output=False)


@pytest.mark.parametrize("language", ["glsl", "cgl"])
@pytest.mark.parametrize("target", ["hlsl", "metal", "spirv"])
def test_vector_and_matrix_length_translates_to_checked_output(
    tmp_path, language, target
):
    code = translate(tmp_path, language, target)
    artifact = tmp_path / f"out.{SUFFIX[target]}"
    artifact.write_text(code)
    verdict = checks.check(artifact, target, ["fragment"])
    if verdict["verdict"] == "unavailable":
        pytest.skip(f"no {target} checker installed")
    assert verdict["verdict"] == "accepted", verdict.get("log", "")


@pytest.mark.parametrize("target", ["hlsl", "metal"])
def test_glsl_vector_and_matrix_length_become_counts(tmp_path, target):
    code = translate(tmp_path, "glsl", target)
    assert ".length()" not in code
    # vec4 has four components; mat3 and mat2x4 have three and two columns of three and four rows.
    assert "< 4;" in code or "< 4)" in code
    assert "3 + 2" in code.replace("(", "").replace(")", "")
    assert "3 + 4" in code.replace("(", "").replace(")", "")
    assert "2 + 3 + 3 + 3" in code.replace("(", "").replace(")", "")


@pytest.mark.parametrize("target", ["hlsl", "metal"])
def test_crossgl_vector_and_matrix_length_become_counts(tmp_path, target):
    code = translate(tmp_path, "cgl", target)
    assert ".length()" not in code
    # vec4, vec2 swizzle, mat4x3 (four columns of three rows); a vec3 parameter has three components.
    assert "4 + 2 + 4 + 3" in code.replace("(", "").replace(")", "")
    assert "return 3;" in code


def test_glsl_keeps_its_length_method(tmp_path):
    code = translate(tmp_path, "glsl", "glsl")
    assert ".length()" in code
