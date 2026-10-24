"""A stage-input struct used as a whole value is rebuilt in GLSL.

GLSL flattens an entry point's struct parameter into one ``in`` variable per
member. A whole-value use such as ``sampleTexture(input)`` used to be emitted
as the bare parameter name, which GLSL never declares and, for ``input``,
reserves. The entry point now rebuilds the struct from its flattened members
first.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl

FRAGMENT = textwrap.dedent("""
    struct VSOutput {
        float4 Pos : SV_POSITION;
        float2 UV : TEXCOORD0;
        nointerpolation uint Index : TEXCOORD1;
    };

    float4 shade(VSOutput input) {
        return float4(input.UV, float(input.Index), input.Pos.z);
    }

    float4 main(VSOutput input) : SV_TARGET0 {
        float4 direct = float4(input.UV, 0.0, 1.0);
        return shade(input) + direct;
    }
    """)

VERTEX = textwrap.dedent("""
    struct VSInput {
        float3 Position : POSITION;
        float2 UV : TEXCOORD0;
    };

    struct VSOutput {
        float4 Pos : SV_POSITION;
        float2 UV : TEXCOORD0;
    };

    VSOutput transform(VSInput vin) {
        VSOutput result;
        result.Pos = float4(vin.Position, 1.0);
        result.UV = vin.UV;
        return result;
    }

    VSOutput main(VSInput input) {
        return transform(input);
    }
    """)


def _glsl(tmp_path, source, stage):
    path = tmp_path / f"shader.{stage}.hlsl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="opengl", source_backend="hlsl")


def _validate(tmp_path, code, stage):
    if shutil.which("glslangValidator") is None:
        pytest.skip("glslangValidator is not installed")
    (tmp_path / f"shader.{stage}").write_text(code, encoding="utf-8")
    result = subprocess.run(
        ["glslangValidator", "-S", stage, f"shader.{stage}"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout[-2000:]


def test_fragment_struct_input_is_rebuilt_for_a_whole_value_use(tmp_path):
    code = _glsl(tmp_path, FRAGMENT, "frag")
    assert (
        "VSOutput input_value = VSOutput(vec4(gl_FragCoord), vec2(UV), uint(Index));"
        in " ".join(code.split())
    )
    assert "shade(input_value)" in code
    # Member reads use the rebuilt copy, which holds the same values.
    assert "vec4(input_value.UV, 0.0, 1.0)" in code
    _validate(tmp_path, code, "frag")


def test_vertex_struct_input_is_rebuilt_for_a_whole_value_use(tmp_path):
    code = _glsl(tmp_path, VERTEX, "vert")
    assert "VSInput input_value = VSInput(" in " ".join(code.split())
    assert "transform(input_value)" in code
    _validate(tmp_path, code, "vert")


def test_member_only_uses_are_not_rebuilt(tmp_path):
    source = FRAGMENT.replace("return shade(input) + direct;", "return direct;")
    code = _glsl(tmp_path, source, "frag")
    assert "input_value" not in code
    assert "vec4(UV, 0.0, 1.0)" in code
    _validate(tmp_path, code, "frag")
