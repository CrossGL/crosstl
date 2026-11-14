"""Clip and cull distance values keep their place in the target's distance array.

HLSL declares each clip or cull distance as its own float or float vector, ``SV_ClipDistanceN``; DXC packs them
into the one SPIR-V ``ClipDistance`` array in ascending semantic index. CrossGL keeps the index
(``gl_ClipDistanceN``), and GLSL and SPIR-V write every value to the elements DXC gives it. Metal, which has a
single ``[[clip_distance]]`` float or float array and no cull distances, refuses what it cannot express.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.metal_varyings import MetalVaryingLinkError
from crosstl.translator.codegen.stage_distance_layout import (
    distance_layout,
    distance_semantic,
)

VERTEX_HLSL = """
struct VSInput { float4 Pos : POSITION0; };
struct VSOutput
{
    float4 Pos : SV_POSITION;
    float Upper : SV_ClipDistance1;
    float2 Lower : SV_ClipDistance0;
    float Cull : SV_CullDistance0;
    float3 Normal : NORMAL0;
};
VSOutput main(VSInput input)
{
    VSOutput output = (VSOutput)0;
    output.Pos = input.Pos;
    output.Lower = float2(input.Pos.x, input.Pos.y);
    output.Upper = input.Pos.z;
    output.Cull = input.Pos.w;
    output.Normal = input.Pos.xyz;
    return output;
}
"""

FRAGMENT_HLSL = """
struct PSInput
{
    float4 Pos : SV_POSITION;
    float Upper : SV_ClipDistance1;
    float2 Lower : SV_ClipDistance0;
    float3 Normal : NORMAL0;
};
float4 main(PSInput input) : SV_TARGET
{
    return float4(input.Normal * input.Lower.y, input.Upper);
}
"""


def _translate(tmp_path, source, name, backend):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend="directx"
    )


def _glslang(tmp_path, code, stage):
    if shutil.which("glslangValidator") is None:
        return
    path = tmp_path / f"out.{stage}"
    path.write_text(code)
    result = subprocess.run(
        ["glslangValidator", str(path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _spirv_val(tmp_path, code):
    if shutil.which("spirv-as") is None or shutil.which("spirv-val") is None:
        return
    asm = tmp_path / "out.spvasm"
    asm.write_text(code)
    binary = tmp_path / "out.spv"
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


def _spirv_element_stores(code, builtin_name):
    """``{element: stored value id}`` for stores through access chains into a distance built-in."""
    variable = re.search(rf"OpDecorate %(\w+) BuiltIn {builtin_name}\b", code).group(1)
    constants = {
        name: int(value)
        for name, value in re.findall(r"%(\w+) = OpConstant %\w+ (\d+)\b", code)
    }
    chains = {
        result: constants[index]
        for result, index in re.findall(
            rf"%(\w+) = OpAccessChain %\w+ %{variable} %(\w+)", code
        )
    }
    return {
        chains[pointer]: value
        for pointer, value in re.findall(r"OpStore %(\w+) %(\w+)", code)
        if pointer in chains
    }


def test_layout_orders_values_by_semantic_index_not_declaration():
    layout = distance_layout(
        [
            ("upper", "gl_ClipDistance1", "float", None),
            ("lower", "gl_ClipDistance", "vec2", None),
            ("cull", "gl_CullDistance", "float", None),
            ("normal", "Normal", "vec3", None),
        ]
    )

    assert [(s.name, s.offset, s.count) for s in layout.slots] == [
        ("lower", 0, 2),
        ("upper", 2, 1),
        ("cull", 0, 1),
    ]
    assert layout.size("gl_ClipDistance") == 3
    assert layout.size("gl_CullDistance") == 1
    assert distance_semantic("gl_CullDistance3") == ("gl_CullDistance", 3)
    assert distance_semantic("gl_Position") is None


def test_layout_refuses_shared_index_and_non_float_values():
    with pytest.raises(ValueError, match="share semantic index 0"):
        distance_layout(
            [("a", "gl_ClipDistance", "float", None)]
            + [("b", "gl_ClipDistance0", "float", None)]
        )
    with pytest.raises(ValueError, match="float, a float vector or a float array"):
        distance_layout([("a", "gl_ClipDistance", "ivec2", None)])


def test_hlsl_import_keeps_the_semantic_index(tmp_path):
    crossgl = _translate(tmp_path, VERTEX_HLSL, "clip.vert", "cgl")

    assert "float Upper @ gl_ClipDistance1;" in crossgl
    assert "vec2 Lower @ gl_ClipDistance;" in crossgl
    assert "float Cull @ gl_CullDistance;" in crossgl


def test_hlsl_round_trip_spells_the_index(tmp_path):
    hlsl = _translate(tmp_path, VERTEX_HLSL, "clip.vert", "directx")

    assert re.search(r"float Upper\s*:\s*SV_ClipDistance1;", hlsl)
    assert re.search(r"float2 Lower\s*:\s*SV_ClipDistance;", hlsl)


def test_glsl_vertex_outputs_fill_the_elements_dxc_assigns(tmp_path):
    code = _translate(tmp_path, VERTEX_HLSL, "clip.vert", "opengl")

    # ``Lower`` (index 0) takes elements 0 and 1, ``Upper`` (index 1) element 2.
    assert "gl_ClipDistance[2] = Pos.z;" in code
    assert re.search(r"gl_ClipDistance\[0\] = (\w+)\[0\];", code)
    assert re.search(r"gl_ClipDistance\[1\] = (\w+)\[1\];", code)
    assert "gl_CullDistance[0] = Pos.w;" in code
    assert not re.search(r"\bout\b[^;]*\b(Lower|Upper|Cull)\b", code)
    _glslang(tmp_path, code, "vert")


def test_glsl_fragment_inputs_read_the_elements_dxc_assigns(tmp_path):
    code = _translate(tmp_path, FRAGMENT_HLSL, "clip.frag", "opengl")

    assert "vec2(gl_ClipDistance[0], gl_ClipDistance[1]).y" in code
    assert "gl_ClipDistance[2]" in code
    assert not re.search(r"\bin\b[^;]*\b(Lower|Upper)\b", code)
    _glslang(tmp_path, code, "frag")


def test_spirv_vertex_outputs_fill_one_sized_builtin_array(tmp_path):
    code = _translate(tmp_path, VERTEX_HLSL, "clip.vert", "vulkan")

    assert "OpCapability ClipDistance" in code
    assert "OpCapability CullDistance" in code
    clip = _spirv_element_stores(code, "ClipDistance")
    cull = _spirv_element_stores(code, "CullDistance")
    assert sorted(clip) == [0, 1, 2]
    assert sorted(cull) == [0]
    # Element 2 holds ``Upper``, a member extracted whole; elements 0 and 1 are components of ``Lower``.
    extracts = dict(re.findall(r"%(\w+) = OpCompositeExtract %\w+ %\w+ (\d+)", code))
    assert extracts[clip[0]] == "0" and extracts[clip[1]] == "1"
    clip_array = re.search(r"OpDecorate %(\w+) BuiltIn ClipDistance", code).group(1)
    pointer_type = re.search(rf"%{clip_array} = OpVariable %(\w+) Output", code).group(
        1
    )
    array_type = re.search(
        rf"%{pointer_type} = OpTypePointer Output %(\w+)", code
    ).group(1)
    length = re.search(rf"%{array_type} = OpTypeArray %\w+ %(\w+)", code).group(1)
    assert re.search(rf"%{length} = OpConstant %\w+ 3\b", code)
    assert "Location" not in "".join(
        line for line in code.splitlines() if "Lower" in line or "Upper" in line
    )
    _spirv_val(tmp_path, code)


def test_spirv_fragment_inputs_load_their_elements(tmp_path):
    code = _translate(tmp_path, FRAGMENT_HLSL, "clip.frag", "vulkan")

    assert "OpDecorate" in code and re.search(r"BuiltIn ClipDistance\b", code)
    variable = re.search(r"OpDecorate %(\w+) BuiltIn ClipDistance", code).group(1)
    assert re.search(rf"%\w+ = OpVariable %\w+ Input", code)
    assert len(re.findall(rf"OpAccessChain %\w+ %{variable} ", code)) == 3
    _spirv_val(tmp_path, code)


def test_metal_keeps_a_single_scalar_clip_distance(tmp_path):
    source = """
struct VSOutput { float4 Pos : SV_POSITION; float Clip : SV_ClipDistance0; };
VSOutput main(float4 pos : POSITION0)
{
    VSOutput output;
    output.Pos = pos;
    output.Clip = pos.x;
    return output;
}
"""
    code = _translate(tmp_path, source, "clip.vert", "metal")

    assert "float Clip [[clip_distance]];" in code


@pytest.mark.parametrize(
    "source, reason",
    [
        (
            VERTEX_HLSL.replace("float Upper : SV_ClipDistance1;", "").replace(
                "output.Upper = input.Pos.z;", ""
            ),
            "is a float or a float array, not 'float2'",
        ),
        (
            VERTEX_HLSL.replace("float2 Lower", "float Lower").replace(
                "float2(input.Pos.x, input.Pos.y)", "input.Pos.x"
            ),
            "semantic index 1",
        ),
        (
            VERTEX_HLSL.replace("float2 Lower", "float Lower")
            .replace("float2(input.Pos.x, input.Pos.y)", "input.Pos.x")
            .replace("float Upper : SV_ClipDistance1;", "")
            .replace("output.Upper = input.Pos.z;", ""),
            "no cull distance attribute",
        ),
    ],
)
def test_metal_refuses_distances_it_cannot_express(tmp_path, source, reason):
    with pytest.raises(MetalVaryingLinkError, match=re.escape(reason)):
        _translate(tmp_path, source, "clip.vert", "metal")


CALL_RETURN_HLSL = """
struct VSOutput { float4 Pos : SV_POSITION; float2 Clip : SV_ClipDistance0; float3 N : NORMAL0; };
VSOutput make(float4 p) { VSOutput o; o.Pos = p; o.Clip = p.xy; o.N = p.xyz; return o; }
VSOutput main(float4 p : POSITION0) { return make(p); }
"""


def test_glsl_entry_returning_a_call_copies_every_output(tmp_path):
    # ``void main()`` cannot return the structure, so each member is copied to its output.
    code = _translate(tmp_path, CALL_RETURN_HLSL, "call.vert", "opengl")
    body = code[code.index("void main()") :]

    local = re.search(r"VSOutput (\w+) = make\(p\);", body).group(1)
    assert f"gl_Position = {local}.Pos;" in body
    assert re.search(rf"\bN = {local}\.N;", body)
    assert re.search(r"gl_ClipDistance\[1\] = \w+\[1\];", body)
    assert body.index(f"= {local}.Pos;") < body.index("gl_ClipDistance[0]")
    _glslang(tmp_path, code, "vert")
