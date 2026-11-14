"""A vertex or fragment entry's scalar or vector return value reaches the Metal output its semantic names.

A Metal vertex function returning float4 returns the position and a fragment function returning a vector returns
colour attachment 0 (MSL 3.2 sections 5.2.3.3 and 5.2.3.4). A return semantic naming another output used to be
dropped, so ``SV_Target1`` wrote attachment 0 and ``TEXCOORD0`` became the vertex position. A GLSL
``layout(location = N) out`` fragment output (``@ location(N)``) wrote attachment 0 the same way, and the Slang
import spelled ``SV_Target<N>`` as ``Out_Color<N>``, which the GLSL and SPIR-V generators placed at location 0.
"""

import pytest

import crosstl
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused


def _metal(tmp_path, source, name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _entry(stage, return_type, semantic):
    return f"""
shader S {{
    {stage} {{
        {return_type} main() @ {semantic} {{
            return {return_type}(1.0);
        }}
    }}
}}
"""


def test_hlsl_sv_target1_return_writes_colour_attachment_1(tmp_path):
    source = "float4 main(float4 p : SV_Position) : SV_Target1 { return float4(1, 0, 0, 1); }\n"
    output = _metal(tmp_path, source, "shader.hlsl")
    assert "[[color(1)]]" in output
    assert "fragment float4 " not in output


def test_frag_data_return_index_selects_the_attachment(tmp_path):
    output = _metal(tmp_path, _entry("fragment", "vec4", "gl_FragData[2]"))
    assert "float4 color [[color(2)]];" in output


def test_glsl_location_output_writes_its_colour_attachment(tmp_path):
    source = "#version 450\nlayout(location = 2) out vec4 fragColor;\nvoid main() { fragColor = vec4(1.0); }\n"
    output = _metal(tmp_path, source, "shader.frag")
    assert "float4 color [[color(2)]];" in output
    assert "fragment float4 " not in output


def test_location_zero_return_stays_a_plain_return(tmp_path):
    output = _metal(tmp_path, _entry("fragment", "vec4", "location(0)"))
    assert "fragment float4 fragment_main()" in output


def test_location_return_of_a_vertex_entry_is_refused(tmp_path):
    with pytest.raises(MetalLoweringRefused, match="no Metal vertex output"):
        _metal(tmp_path, _entry("vertex", "vec4", "location(1)"))


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("metal", "float4 color [[color(1)]];"),
        ("opengl", "layout(location = 1) out vec4"),
        ("vulkan", "Location 1"),
        ("directx", ": SV_TARGET1"),
    ],
)
def test_slang_indexed_target_reaches_its_attachment(tmp_path, backend, expected):
    path = tmp_path / "shader.slang"
    path.write_text(
        "float4 main(float4 pos : SV_Position) : SV_Target1 { return pos; }\n"
    )
    output = crosstl.translate(str(path), backend=backend, format_output=False)
    assert expected in output
    if backend == "vulkan":
        assert "Location 0" not in output


def test_frag_data_return_zero_stays_a_plain_return(tmp_path):
    output = _metal(tmp_path, _entry("fragment", "vec4", "gl_FragData[0]"))
    assert "fragment float4 fragment_main()" in output


@pytest.mark.parametrize("semantic", ["TEXCOORD0", "POSITION", "COLOR1"])
def test_vertex_return_with_a_user_semantic_is_refused(tmp_path, semantic):
    with pytest.raises(
        MetalLoweringRefused, match=f"semantic '{semantic}'.*no Metal vertex output"
    ):
        _metal(tmp_path, _entry("vertex", "vec4", semantic))


@pytest.mark.parametrize("semantic", ["TEXCOORD0", "POSITION"])
def test_fragment_return_with_a_non_output_semantic_is_refused(tmp_path, semantic):
    with pytest.raises(
        MetalLoweringRefused, match=f"semantic '{semantic}'.*no Metal fragment output"
    ):
        _metal(tmp_path, _entry("fragment", "vec4", semantic))


def test_vertex_return_of_a_non_position_type_is_refused(tmp_path):
    source = """
shader S {
    vertex {
        float main() {
            return 1.0;
        }
    }
}
"""
    with pytest.raises(
        MetalLoweringRefused, match="returns float: a Metal vertex function returns"
    ):
        _metal(tmp_path, source)


def test_fragment_return_of_a_matrix_without_semantic_is_refused(tmp_path):
    source = """
shader S {
    fragment {
        mat2 main() {
            return mat2(1.0);
        }
    }
}
"""
    with pytest.raises(MetalLoweringRefused, match="colour attachment 0 cannot hold"):
        _metal(tmp_path, source)


@pytest.mark.parametrize(
    "stage, return_type, semantic, expected",
    [
        ("vertex", "vec4", "gl_Position", "vertex float4 vertex_main()"),
        ("vertex", "vec4", "SV_Position", "vertex float4 vertex_main()"),
        ("fragment", "vec4", "gl_FragColor", "fragment float4 fragment_main()"),
        ("fragment", "vec4", "SV_Target", "fragment float4 fragment_main()"),
        ("fragment", "float", "SV_Depth", "float depth [[depth(any)]];"),
    ],
)
def test_returns_naming_a_metal_output_are_kept(
    tmp_path, stage, return_type, semantic, expected
):
    assert expected in _metal(tmp_path, _entry(stage, return_type, semantic))


def test_vertex_return_with_user_semantic_beside_an_out_position_is_a_user_member(
    tmp_path,
):
    source = (
        "float4 main(out float4 pos : SV_Position) : TEXCOORD0 {\n"
        "    pos = float4(0, 0, 0, 1);\n"
        "    return float4(1, 0, 0, 1);\n"
        "}\n"
    )
    output = _metal(tmp_path, source, "shader.vert.hlsl")
    assert "[[user(TEXCOORD0)]]" in output
    assert "[[position]]" in output
