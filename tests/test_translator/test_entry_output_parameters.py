"""HLSL-style entry points that write stage outputs through ``out`` parameters.

An HLSL vertex or pixel shader may return its outputs through ``out`` parameters (``out float2 uv : TEXCOORD0``),
alone or beside a return value, and ``SV_Target<N>`` places an output at colour attachment ``N``.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.interface_conflicts import (
    InterfaceLocationConflictError,
)

OUT_PARAMETERS = """
void VSMain(float3 p : POSITION, float2 t : TEXCOORD0,
            out float4 opos : SV_Position, out float2 ouv : TEXCOORD0) {
    opos = float4(p, 1.0);
    ouv = t * 2.0;
}
void PSMain(float4 pos : SV_Position, float2 uv : TEXCOORD0, out float4 color : SV_Target0) {
    color = float4(uv, 0.0, 1.0);
}
"""

RETURN_AND_OUT_PARAMETERS = """
cbuffer Params : register(b0) { float scale; };
float4 VSMain(float3 p : POSITION, float2 t : TEXCOORD0,
              out float2 ouv : TEXCOORD0, out float3 onrm : NORMAL) : SV_Position {
    ouv = t * scale;
    onrm = normalize(p);
    return float4(p, 1.0);
}
float4 PSMain(float4 pos : SV_Position, float2 uv : TEXCOORD0, float3 n : NORMAL,
              out float4 extra : SV_Target1) : SV_Target0 {
    extra = float4(n, 1.0);
    return float4(uv, pos.z, 1.0);
}
"""

MISSING_OUTPUT = """
void VSMain(float3 p : POSITION, out float4 opos : SV_Position, out float2 ouv : TEXCOORD0) {
    opos = float4(p, 1.0);
    ouv = p.xy;
}
float4 PSMain(float4 pos : SV_Position, float2 uv : TEXCOORD1) : SV_Target {
    return float4(uv, 0.0, 1.0);
}
"""

DUPLICATE_OUTPUT = """
void VSMain(float3 p : POSITION, out float4 opos : SV_Position,
            out float2 a : TEXCOORD0, out float2 b : TEXCOORD0) {
    opos = float4(p, 1.0);
    a = p.xy;
    b = p.yz;
}
"""

ATTACHMENTS_OUT_OF_ORDER = """
struct PSOut { float4 c2 : SV_Target2; float4 c0 : SV_Target0; };
PSOut PSMain(float4 pos : SV_Position) {
    PSOut o;
    o.c2 = pos;
    o.c0 = pos * 2.0;
    return o;
}
"""

ATTACHMENT_PARAMETERS = """
void PSMain(float4 pos : SV_Position, out float4 c0 : SV_Target, out float4 c1 : SV_Target1) {
    c0 = pos;
    c1 = pos * 2.0;
}
"""

EARLY_RETURN = """
float4 tint;
void PSMain(float4 pos : SV_Position, out float4 color : SV_Target0) {
    color = float4(1.0, 0.0, 0.0, 1.0);
    if (pos.x < tint.x) {
        color = float4(0.0, 1.0, 0.0, 1.0);
        return;
    }
    color = float4(0.0, 0.0, 1.0, 1.0);
}
"""

INOUT_PARAMETER = """
void VSMain(float3 p : POSITION, inout float2 uv : TEXCOORD0, out float4 pos : SV_Position) {
    pos = float4(p, 1.0);
    uv = uv * 2.0;
}
float4 PSMain(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
    return float4(uv, 0.0, 1.0);
}
"""

COMPUTE_OUT_PARAMETER = """
RWStructuredBuffer<float> results : register(u0);
[numthreads(8, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID, out float value : SV_Target0) {
    value = 1.0;
    results[id.x] = 2.0;
}
"""

STRUCT_RETURN_AND_OUT_PARAMETER = """
struct VSOut { float4 pos : SV_Position; };
VSOut VSMain(float3 p : POSITION, out float2 ouv : TEXCOORD0) {
    VSOut o;
    o.pos = float4(p, 1.0);
    ouv = p.xy;
    return o;
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.hlsl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return " ".join(code.split())


def _run(command, tmp_path):
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=tmp_path, check=False
    )
    if "unable to find utility" in result.stderr:
        pytest.skip("xcrun has no Metal compiler")
    assert result.returncode == 0, result.stdout + result.stderr


def _metal_compiles(tmp_path, code):
    if shutil.which("xcrun"):
        (tmp_path / "out.metal").write_text(code)
        _run(
            ["xcrun", "-sdk", "macosx", "metal", "-c", "out.metal", "-o", "out.air"],
            tmp_path,
        )


def _glsl_links(tmp_path, code):
    validator = shutil.which("glslangValidator")
    if validator is None:
        return
    # glslang defines GL_VERTEX_SHADER or GL_FRAGMENT_SHADER from the file's stage.
    (tmp_path / "out.vert").write_text(code)
    (tmp_path / "out.frag").write_text(code)
    _run([validator, "-l", "out.vert", "out.frag"], tmp_path)


def _spirv_validates(tmp_path, code):
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if assembler is None or validator is None:
        return
    (tmp_path / "out.spvasm").write_text(code)
    _run(
        [assembler, "--target-env", "vulkan1.1", "out.spvasm", "-o", "out.spv"],
        tmp_path,
    )
    _run([validator, "--target-env", "vulkan1.1", "out.spv"], tmp_path)


def _wgsl_validates(tmp_path, code):
    validator = shutil.which("naga")
    if validator is None:
        return
    (tmp_path / "out.wgsl").write_text(code)
    _run([validator, "out.wgsl"], tmp_path)


@pytest.mark.parametrize(
    "source", [OUT_PARAMETERS, RETURN_AND_OUT_PARAMETERS, INOUT_PARAMETER]
)
def test_out_parameters_feed_the_next_stage(tmp_path, source):
    _metal_compiles(tmp_path, _translate(tmp_path, source, "metal"))
    _glsl_links(tmp_path, _translate(tmp_path, source, "opengl"))
    _spirv_validates(tmp_path, _translate(tmp_path, source, "vulkan"))
    _wgsl_validates(tmp_path, _translate(tmp_path, source, "wgsl"))


def test_vertex_out_parameters_link_by_semantic(tmp_path):
    glsl = _translate(tmp_path, OUT_PARAMETERS, "opengl")
    output = re.search(r"layout\(location = (\d+)\) out vec2 ouv;", glsl)
    varying = re.search(r"layout\(location = (\d+)\) in vec2 uv;", glsl)
    assert output and varying, glsl
    assert output.group(1) == varying.group(1)
    assert "gl_Position = vec4(p, 1.0);" in glsl

    metal = _flat(_translate(tmp_path, OUT_PARAMETERS, "metal"))
    assert "float2 ouv [[user(TEXCOORD0)]];" in metal
    assert "float2 uv [[user(TEXCOORD0)]];" in metal


def test_a_missing_vertex_output_is_still_refused(tmp_path):
    with pytest.raises(ValueError, match="Missing cross-stage interface output"):
        _translate(tmp_path, MISSING_OUTPUT, "opengl")


def test_hlsl_input_and_output_signatures_are_separate(tmp_path):
    hlsl = _flat(_translate(tmp_path, OUT_PARAMETERS, "directx"))
    assert "float2 t : TexCoord0, out float4 opos : SV_POSITION" in hlsl
    assert "out float2 ouv : TexCoord0" in hlsl
    with pytest.raises(InterfaceLocationConflictError, match="'b'"):
        _translate(tmp_path, DUPLICATE_OUTPUT, "directx")


@pytest.mark.parametrize(
    "backend, expected",
    [
        (
            "opengl",
            ["layout(location = 2) out vec4 c2;", "layout(location = 0) out vec4 c0;"],
        ),
        ("wgsl", ["@location(2) c2: vec4<f32>", "@location(0) c0: vec4<f32>"]),
        ("slang", ["float4 c2 : SV_Target2;", "float4 c0 : SV_Target0;"]),
        ("metal", ["float4 c2 [[color(2)]];", "float4 c0 [[color(0)]];"]),
        ("directx", ["float4 c2: SV_TARGET2;", "float4 c0: SV_TARGET0;"]),
    ],
)
def test_colour_attachment_indices_survive_member_order(tmp_path, backend, expected):
    code = _flat(_translate(tmp_path, ATTACHMENTS_OUT_OF_ORDER, backend))
    for text in expected:
        assert text in code, code


@pytest.mark.parametrize(
    "backend, expected",
    [
        ("opengl", ["layout(location = 1) out vec4 c1;"]),
        ("slang", ["out float4 c1 : SV_Target1"]),
        ("wgsl", ["@location(0) c0: vec4<f32>", "@location(1) c1: vec4<f32>"]),
        ("metal", ["float4 c0 [[color(0)]];", "float4 c1 [[color(1)]];"]),
    ],
)
def test_colour_attachment_out_parameters(tmp_path, backend, expected):
    code = _flat(_translate(tmp_path, ATTACHMENT_PARAMETERS, backend))
    for text in expected:
        assert text in code, code
    if backend == "metal":
        assert "gl_FragData" not in code
        _metal_compiles(tmp_path, code)


def test_metal_return_inside_the_body_returns_the_outputs(tmp_path):
    metal = _translate(tmp_path, EARLY_RETURN, "metal")
    branch = _flat(
        metal[metal.index("if (") : metal.index("color = float4(0.0, 0.0, 1.0")]
    )
    assert (
        "color = float4(0.0, 1.0, 0.0, 1.0); { PSMain_Return _crossglOutput; "
        "_crossglOutput.color = color; return _crossglOutput; }" in branch
    ), metal
    _metal_compiles(tmp_path, metal)


def test_metal_return_value_beside_out_parameters(tmp_path):
    metal = _translate(tmp_path, RETURN_AND_OUT_PARAMETERS, "metal")
    flat = _flat(metal)
    assert "float4 _crossglReturn [[position]];" in flat
    assert "float4 extra [[color(1)]]; float4 _crossglReturn [[color(0)]];" in flat
    assert (
        "_crossglOutput._crossglReturn = float4(uv, pos.z, 1.0); "
        "_crossglOutput.extra = extra; return _crossglOutput;" in flat
    )
    _metal_compiles(tmp_path, metal)


def test_metal_refuses_a_structure_return_beside_out_parameters(tmp_path):
    with pytest.raises(ValueError, match="structure return type"):
        _translate(tmp_path, STRUCT_RETURN_AND_OUT_PARAMETER, "metal")


def test_wgsl_out_parameters_become_an_output_structure(tmp_path):
    wgsl = _flat(_translate(tmp_path, OUT_PARAMETERS, "wgsl"))
    output = re.search(r"@location\((\d+)\) ouv: vec2<f32>", wgsl)
    varying = re.search(r"@location\((\d+)\) uv: vec2<f32>\)", wgsl)
    assert output and varying, wgsl
    assert output.group(1) == varying.group(1)
    assert (
        "struct VSMainOutput { @builtin(position) opos: vec4<f32>, "
        "@location(" + output.group(1) + ") ouv: vec2<f32>, };" in wgsl
    )
    assert "-> VSMainOutput { var opos: vec4<f32>; var ouv: vec2<f32>;" in wgsl
    assert "return VSMainOutput(opos, ouv); }" in wgsl
    assert "-> PSMainOutput { var color: vec4<f32>;" in wgsl


def test_wgsl_return_value_beside_out_parameters(tmp_path):
    wgsl = _flat(_translate(tmp_path, RETURN_AND_OUT_PARAMETERS, "wgsl"))
    assert "@builtin(position) _crossglReturn: vec4<f32>," in wgsl
    assert (
        "struct PSMainOutput { @location(0) _crossglReturn: vec4<f32>, "
        "@location(1) extra: vec4<f32>, };" in wgsl
    )
    assert "return PSMainOutput(vec4<f32>(uv, pos.z, 1.0), extra);" in wgsl


def test_wgsl_inout_parameter_starts_from_its_input(tmp_path):
    wgsl = _flat(_translate(tmp_path, INOUT_PARAMETER, "wgsl"))
    assert re.search(r"@location\(\d+\) uv_in: vec2<f32>\) -> VSMainOutput", wgsl)
    assert "var uv: vec2<f32> = uv_in;" in wgsl
    assert "return VSMainOutput(uv, pos);" in wgsl


def test_wgsl_return_inside_the_body_returns_the_outputs(tmp_path):
    wgsl = _translate(tmp_path, EARLY_RETURN, "wgsl")
    branch = _flat(
        wgsl[wgsl.index("if (") : wgsl.index("color = vec4<f32>(0.0, 0.0, 1.0")]
    )
    assert "color = vec4<f32>(0.0, 1.0, 0.0, 1.0); return PSMainOutput(color); }" in (
        branch
    ), wgsl
    assert _flat(wgsl).endswith("return PSMainOutput(color); }")
    _wgsl_validates(tmp_path, wgsl)


def test_wgsl_refuses_a_structure_return_beside_out_parameters(tmp_path):
    with pytest.raises(ValueError, match="also returns a structure"):
        _translate(tmp_path, STRUCT_RETURN_AND_OUT_PARAMETER, "wgsl")


def test_wgsl_refuses_a_compute_output_parameter(tmp_path):
    with pytest.raises(ValueError, match="compute shader has no stage outputs"):
        _translate(tmp_path, COMPUTE_OUT_PARAMETER, "wgsl")
