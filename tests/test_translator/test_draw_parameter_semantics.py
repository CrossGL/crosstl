"""The draw parameters ``SV_StartVertexLocation``, ``SV_StartInstanceLocation`` and ``SV_DrawIndex`` across targets.

The HLSL and Slang importers kept them as plain semantic names, so every generator bound them as user vertex
attributes: GLSL declared ``layout(location = 1) in int base``, SPIR-V gave them a ``Location`` and Metal an
``[[attribute(n)]]``. They now import as CrossGL's ``gl_BaseVertex``, ``gl_BaseInstance`` and ``gl_DrawID``.

HLSL output spells the first two with their Shader Model 6.8 system values; Slang output spells ``gl_DrawID`` as
``SV_DrawIndex`` (slangc rejects ``SV_DrawID`` as an unknown system value). Metal has no draw-index attribute, so a
``gl_DrawID`` vertex input is refused instead of being bound as a vertex-buffer attribute.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
import crosstl.translator
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.hlsl_refusals import DirectXLoweringRefused
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused

HLSL = """float4 main(float4 position : POSITION, int base : SV_StartVertexLocation,
            uint inst : SV_StartInstanceLocation) : SV_Position {
    return position * 2.0 + float4(float(base), float(inst), 0.0, 0.0);
}
"""

SLANG = """[shader("vertex")]
float4 main(float4 position : POSITION, int base : SV_StartVertexLocation,
            uint inst : SV_StartInstanceLocation, uint draw : SV_DrawIndex) : SV_Position {
    return position * 2.0 + float4(float(base), float(inst), float(draw), 0.0);
}
"""

MEMBERS = """shader Probe {
    struct VertexInput {
        vec4 position;
        int base @ gl_BaseVertex;
        int inst @ gl_BaseInstance;
    }
    struct VertexOutput {
        vec4 position @ gl_Position;
    }
    vertex {
        VertexOutput main(VertexInput input) {
            VertexOutput output;
            output.position = input.position * 2.0 + vec4(float(input.base), float(input.inst), 0.0, 0.0);
            return output;
        }
    }
}
"""

DRAW_MEMBER = """shader Probe {
    struct VertexInput {
        vec4 position;
        int draw @ gl_DrawID;
    }
    struct VertexOutput {
        vec4 position @ gl_Position;
    }
    vertex {
        VertexOutput main(VertexInput input) {
            VertexOutput output;
            output.position = input.position + vec4(float(input.draw), 0.0, 0.0, 0.0);
            return output;
        }
    }
}
"""

DRAW_PARAMETER = """shader Probe {
    vertex {
        vec4 main(vec4 position @ POSITION, uint draw @ gl_DrawID) @ gl_Position {
            return position + vec4(float(draw), 0.0, 0.0, 0.0);
        }
    }
}
"""


def _translate(tmp_path, source, backend, suffix):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _validate_glsl(tmp_path, glsl):
    if not shutil.which("glslangValidator"):
        return
    (tmp_path / "out.vert").write_text(glsl)
    result = subprocess.run(
        ["glslangValidator", "-S", "vert", "out.vert"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _decorations(spirv, builtin):
    return re.findall(rf"OpDecorate %\S+ BuiltIn {builtin}\b", spirv)


@pytest.mark.parametrize(
    ("source", "suffix", "expected"),
    [
        (HLSL, ".hlsl", ("int base @ gl_BaseVertex", "uint inst @ gl_BaseInstance")),
        (
            SLANG,
            ".slang",
            (
                "int base @ gl_BaseVertex",
                "uint inst @ gl_BaseInstance",
                "uint draw @ gl_DrawID",
            ),
        ),
    ],
    ids=["hlsl", "slang"],
)
def test_importers_map_draw_parameters_to_crossgl_built_ins(
    tmp_path, source, suffix, expected
):
    crossgl = _translate(tmp_path, source, "cgl", suffix)
    for parameter in expected:
        assert parameter in crossgl
    for semantic in ("SV_StartVertexLocation", "SV_StartInstanceLocation"):
        assert semantic not in crossgl
    assert "SV_DrawIndex" not in crossgl


@pytest.mark.parametrize(
    ("source", "suffix"), [(HLSL, ".hlsl"), (SLANG, ".slang")], ids=["hlsl", "slang"]
)
def test_glsl_output_reads_the_draw_parameters_instead_of_attributes(
    tmp_path, source, suffix
):
    glsl = _translate(tmp_path, source, "opengl", suffix)
    assert "#extension GL_ARB_shader_draw_parameters : require" in glsl
    assert "float(gl_BaseVertexARB)" in glsl
    assert "uint(gl_BaseInstanceARB)" in glsl
    assert "layout(location = 0) in vec4 position;" in glsl
    assert "layout(location = 1)" not in glsl
    if suffix == ".slang":
        assert "uint(gl_DrawIDARB)" in glsl
    _validate_glsl(tmp_path, glsl)


@pytest.mark.parametrize(
    ("source", "suffix", "builtins"),
    [
        (HLSL, ".hlsl", ("BaseVertex", "BaseInstance")),
        (SLANG, ".slang", ("BaseVertex", "BaseInstance", "DrawIndex")),
    ],
    ids=["hlsl", "slang"],
)
def test_spirv_output_decorates_the_draw_parameters_as_built_ins(
    tmp_path, source, suffix, builtins
):
    spirv = _translate(tmp_path, source, "vulkan", suffix)
    assert "OpCapability DrawParameters" in spirv
    for builtin in builtins:
        assert len(_decorations(spirv, builtin)) == 1, builtin
    assert len(re.findall(r"OpDecorate %\S+ Location \d+", spirv)) == 1


def test_hlsl_round_trip_keeps_the_shader_model_6_8_system_values(tmp_path):
    hlsl = _translate(tmp_path, HLSL, "directx", ".hlsl")
    assert "int base : SV_StartVertexLocation" in hlsl
    assert "uint inst : SV_StartInstanceLocation" in hlsl


def test_hlsl_output_spells_crossgl_draw_parameter_members(tmp_path):
    hlsl = _translate(tmp_path, MEMBERS, "directx", ".cgl")
    assert re.search(r"int base\s*:\s*SV_StartVertexLocation;", hlsl)
    assert re.search(r"int inst\s*:\s*SV_StartInstanceLocation;", hlsl)
    assert "gl_BaseVertex" not in hlsl and "gl_BaseInstance" not in hlsl


@pytest.mark.parametrize("profile", ["dx10", "dx11"])
def test_hlsl_profiles_before_shader_model_6_8_refuse_the_draw_parameters(profile):
    with pytest.raises(DirectXLoweringRefused) as refused:
        HLSLCodeGen(target_profile=profile).generate(crosstl.translator.parse(MEMBERS))
    message = str(refused.value)
    assert "SV_StartVertexLocation requires Shader Model 6.8" in message
    assert "SV_StartInstanceLocation requires Shader Model 6.8" in message
    assert f"DirectX profile {profile} does not provide" in message


def test_dx12_profile_keeps_the_draw_parameters():
    hlsl = HLSLCodeGen(target_profile="dx12").generate(
        crosstl.translator.parse(MEMBERS)
    )
    assert re.search(r"int base\s*:\s*SV_StartVertexLocation;", hlsl)


def test_hlsl_output_still_refuses_the_draw_index(tmp_path):
    with pytest.raises(DirectXLoweringRefused, match="@gl_DrawID"):
        _translate(tmp_path, SLANG, "directx", ".slang")


def test_slang_output_spells_the_draw_index_sv_draw_index(tmp_path):
    slang = _translate(tmp_path, SLANG, "slang", ".slang")
    assert "uint draw : SV_DrawIndex" in slang
    assert "SV_DrawID" not in slang
    assert "int base : SV_StartVertexLocation" in slang
    assert "uint inst : SV_StartInstanceLocation" in slang


def test_metal_output_reads_base_vertex_and_base_instance_attributes(tmp_path):
    metal = _translate(tmp_path, HLSL, "metal", ".hlsl")
    assert "[[base_vertex]]" in metal and "[[base_instance]]" in metal
    assert "[[attribute(0)]]" in metal
    assert "[[attribute(1)]]" not in metal


@pytest.mark.parametrize(
    ("source", "suffix", "name"),
    [
        (DRAW_MEMBER, ".cgl", "VertexInput.draw"),
        (DRAW_PARAMETER, ".cgl", "draw"),
        (SLANG, ".slang", "draw"),
    ],
    ids=["member", "parameter", "slang"],
)
def test_metal_output_refuses_a_draw_index_vertex_input(tmp_path, source, suffix, name):
    with pytest.raises(MetalLoweringRefused) as refused:
        _translate(tmp_path, source, "metal", suffix)
    message = str(refused.value)
    assert (
        f"vertex input '{name}': Metal has no vertex-function attribute for the built-in @gl_DrawID"
        in message
    )
    assert "gl_BaseVertex" not in message and "gl_BaseInstance" not in message
