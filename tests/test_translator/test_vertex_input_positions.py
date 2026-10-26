"""An HLSL vertex input with the SV_Position semantic is an ordinary vertex attribute.

Direct3D binds it to the input-assembler element named SV_Position (DXC records system value NONE for it).
GLSL used to declare it without a location, Metal put ``[[position]]`` in the ``[[stage_in]]`` structure,
WGSL put ``@builtin(position)`` on a vertex input, and SPIR-V read the ``gl_Position`` output instead.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import VertexInputPositionUnsupportedError

STRUCT_INPUT = """
struct VSIn { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
VSOut VSMain(VSIn i) {
    VSOut o;
    o.pos = i.pos * 2.0;
    o.uv = i.uv;
    return o;
}
float4 PSMain(VSOut i) : SV_Target { return float4(i.uv, i.pos.x, 1.0); }
"""

PARAMETER_INPUT = """
float4 VSMain(float4 p : SV_Position, float2 uv : TEXCOORD0) : SV_Position {
    return p * 2.0 + float4(uv, 0.0, 0.0);
}
"""

SHARED_STRUCT = """
struct V { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
V VSMain(V i) { i.pos *= 2.0; return i; }
float4 PSMain(V i) : SV_Target { return float4(i.uv, 0.0, 1.0); }
"""

TWO_POSITIONS = """
struct VSIn { float4 a : POSITION; float4 b : SV_Position; };
float4 VSMain(VSIn i) : SV_Position { return i.a + i.b; }
"""

INOUT_PARAMETER = """
void VSMain(inout float4 p : SV_Position, float2 uv : TEXCOORD0) {
    p = p * 2.0 + float4(uv, 0.0, 0.0);
}
"""

INOUT_STRUCT = """
struct V { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
void VSMain(inout V v) { v.pos *= 2.0; }
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.hlsl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _flat(code):
    return " ".join(code.split())


def _spirv_id(code, name):
    """The result id of ``OpName <id> "name"``; symbolic with spirv-tools installed, numeric otherwise."""
    match = re.search(rf'OpName (%\S+) "{re.escape(name)}"$', code, re.MULTILINE)
    assert match is not None, code
    return re.escape(match.group(1))


def _run(command, cwd):
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=cwd, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_glsl_declares_the_input_at_the_position_attribute_location(tmp_path):
    flat = _flat(_translate(tmp_path, STRUCT_INPUT, "opengl"))
    assert "layout(location = 0) in vec4 pos;" in flat, flat
    assert "gl_Position = (pos * 2.0);" in flat, flat


def test_metal_reads_the_input_from_a_vertex_attribute(tmp_path):
    code = _translate(tmp_path, STRUCT_INPUT, "metal")
    flat = _flat(code)
    assert "struct VSIn { float4 pos [[attribute(0)]];" in flat, code
    assert "struct VSOut { float4 pos [[position]];" in flat, code
    if shutil.which("xcrun") is None:
        return
    (tmp_path / "out.metal").write_text(code)
    _run(["xcrun", "metal", "-c", "out.metal", "-o", "out.air"], tmp_path)


@pytest.mark.parametrize(
    "source", [STRUCT_INPUT, PARAMETER_INPUT], ids=["struct", "parameter"]
)
def test_wgsl_vertex_input_is_a_location(tmp_path, source):
    code = _translate(tmp_path, source, "wgsl")
    flat = _flat(code)
    if source is STRUCT_INPUT:
        assert "struct VSIn { @location(0) pos: vec4<f32>," in flat, code
        assert "struct VSOut { @builtin(position) pos: vec4<f32>," in flat, code
    else:
        assert "fn vertex_main(@location(0) p: vec4<f32>," in flat, code
    naga = shutil.which("naga")
    if naga is None:
        return
    (tmp_path / "out.wgsl").write_text(code)
    _run([naga, "out.wgsl"], tmp_path)


def test_spirv_reads_an_input_variable_not_the_position_output(tmp_path):
    code = _translate(tmp_path, STRUCT_INPUT, "vulkan")
    position_input = _spirv_id(code, "CrossGL_vertex_input_i_pos")
    assert re.search(
        rf"^\s*OpDecorate {position_input} Location 0$", code, re.MULTILINE
    ), code
    position_output = _spirv_id(code, "gl_Position")
    assert not re.search(rf"= OpLoad \S+ {position_output}$", code, re.MULTILINE), code
    assembler = shutil.which("spirv-as")
    validator = shutil.which("spirv-val")
    if assembler is None or validator is None:
        return
    (tmp_path / "out.spvasm").write_text(code)
    _run(
        [assembler, "out.spvasm", "-o", "out.spv", "--target-env", "vulkan1.1"],
        tmp_path,
    )
    _run([validator, "out.spv"], tmp_path)


@pytest.mark.parametrize(
    "source", [STRUCT_INPUT, PARAMETER_INPUT], ids=["struct", "parameter"]
)
def test_hlsl_keeps_the_sv_position_input_element(tmp_path, source):
    # Without clang-format the HLSL generator writes ``name: SEMANTIC``; normalise the spacing.
    flat = re.sub(r"\s*:\s*", " : ", _flat(_translate(tmp_path, source, "directx")))
    if source is STRUCT_INPUT:
        assert "struct VSIn { float4 pos : SV_POSITION;" in flat, flat
    else:
        assert "float4 VSMain(float4 p : SV_POSITION," in flat, flat


@pytest.mark.parametrize("backend", ["opengl", "metal", "wgsl", "vulkan"])
def test_structure_that_is_also_an_output_is_refused(tmp_path, backend):
    with pytest.raises(VertexInputPositionUnsupportedError, match="Structure 'V'"):
        _translate(tmp_path, SHARED_STRUCT, backend)


@pytest.mark.parametrize("backend", ["opengl", "metal", "wgsl", "vulkan"])
@pytest.mark.parametrize(
    "source, name",
    [(INOUT_PARAMETER, "p"), (INOUT_STRUCT, "v_pos")],
    ids=["parameter", "struct"],
)
def test_inout_position_parameter_is_refused(tmp_path, source, name, backend):
    # The input reads the vertex attribute and the output writes the position built-in.
    with pytest.raises(
        VertexInputPositionUnsupportedError, match=f"inout parameter '{name}'"
    ):
        _translate(tmp_path, source, backend)


def test_hlsl_keeps_an_inout_sv_position_parameter(tmp_path):
    code = _translate(tmp_path, INOUT_PARAMETER, "directx")
    flat = re.sub(r"\s*:\s*", " : ", _flat(code))
    assert "void VSMain(inout float4 p : SV_POSITION," in flat, flat


@pytest.mark.parametrize("backend", ["opengl", "metal", "wgsl", "vulkan"])
def test_second_position_attribute_is_refused(tmp_path, backend):
    with pytest.raises(
        VertexInputPositionUnsupportedError, match="would all be the Position"
    ):
        _translate(tmp_path, TWO_POSITIONS, backend)
