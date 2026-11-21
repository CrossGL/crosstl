"""A Metal vertex or fragment entry with several stage inputs reads them through one ``[[stage_in]]`` structure.

Apple's compiler rejects a second ``[[stage_in]]`` parameter. The emitter used to mark every structure parameter of a
vertex or fragment entry ``[[stage_in]]`` and to wrap its loose inputs in one more, so an HLSL vertex shader with a
per-vertex and a per-instance structure did not compile (see :mod:`crosstl.translator.codegen.metal_stage_input_merge`).
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.metal_stage_input_merge import (
    MetalStageInputMergeError,
)

TWO_STRUCTS = """
shader TwoStructs {
    struct PosIn {
        vec3 position @POSITION;
    };
    struct ColorIn {
        vec4 color @COLOR;
    };
    struct VSOut {
        vec4 position @gl_Position;
        vec4 color @COLOR;
    };
    vertex {
        VSOut main(PosIn p, ColorIn c) {
            VSOut o;
            o.position = vec4(p.position, 1.0);
            o.color = c.color;
            return o;
        }
    }
}
"""

STRUCT_AND_LOOSE = """
shader StructAndLoose {
    struct Vertex {
        vec3 position @location(0);
        vec3 normal @location(1);
    };
    struct VSOut {
        vec4 position @gl_Position;
        vec4 color @COLOR;
        vec2 uv @TEXCOORD0;
    };
    struct Colour {
        vec4 color @COLOR;
    };
    struct Coordinates {
        vec2 uv @TEXCOORD0;
    };
    vertex {
        VSOut main(Vertex v, vec4 weight @BLENDWEIGHTX, vec2 uv @TEXCOORD0) {
            VSOut o;
            o.position = vec4(v.position + v.normal, 1.0) * weight;
            o.color = weight;
            o.uv = uv;
            return o;
        }
    }
    fragment {
        vec4 main(Colour a, Coordinates b) @gl_FragColor {
            return a.color * b.uv.x;
        }
    }
}
"""

HLSL_INSTANCED = """
struct VertexIn { float3 position : POSITION; float2 uv : TEXCOORD0; };
struct InstanceIn { float4 offset : TEXCOORD1; float4 tint : COLOR0; };
struct VSOut { float4 position : SV_Position; float2 uv : TEXCOORD0; float4 tint : COLOR0; };
VSOut main(VertexIn v, InstanceIn inst, uint iid : SV_InstanceID) {
    VSOut o;
    o.position = float4(v.position + inst.offset.xyz * float(iid), 1.0);
    o.uv = v.uv;
    o.tint = inst.tint;
    return o;
}
"""


def _metal(tmp_path, source, suffix=".cgl"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _compile(tmp_path, metal):
    if shutil.which("xcrun") is None:
        return
    shader = tmp_path / "out.metal"
    shader.write_text(metal, encoding="utf-8")
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(shader)]
        + ["-o", str(tmp_path / "out.air")],
        capture_output=True,
        text=True,
        check=False,
    )
    if "unable to find utility" in result.stderr:
        pytest.skip("the Metal toolchain is not installed")
    assert result.returncode == 0, result.stderr


def _signature(metal, stage):
    match = re.search(
        rf"^{stage} \w+ {stage}_main\((.*?)\) {{", metal, re.MULTILINE | re.DOTALL
    )
    assert match is not None, metal
    return match.group(1)


def test_two_structure_inputs_share_one_stage_in(tmp_path):
    metal = _metal(tmp_path, TWO_STRUCTS)

    assert _signature(metal, "vertex") == (
        "vertex_main_Input _crossglInput [[stage_in]]"
    )
    assert "struct vertex_main_Input {\n  PosIn p;\n  ColorIn c;\n};" in metal
    assert "PosIn p = _crossglInput.p;" in metal
    assert "ColorIn c = _crossglInput.c;" in metal
    # The nested structures keep the attributes the host binds.
    assert "float3 position [[attribute(0)]];" in metal
    assert "float4 color [[attribute(13)]];" in metal
    _compile(tmp_path, metal)


def test_loose_inputs_join_the_structure_input_and_skip_its_attributes(tmp_path):
    metal = _metal(tmp_path, STRUCT_AND_LOOSE)

    assert _signature(metal, "vertex") == (
        "vertex_main_Input _crossglInput [[stage_in]]"
    )
    assert "  Vertex v;\n" in metal
    # Vertex declares attributes 0 and 1, so the loose input with a user semantic takes 2.
    assert "float4 weight [[attribute(2)]];" in metal
    assert "float2 uv [[attribute(5)]];" in metal
    assert metal.count("[[stage_in]]") == 2
    _compile(tmp_path, metal)


def test_two_fragment_structure_inputs_share_one_stage_in(tmp_path):
    metal = _metal(tmp_path, STRUCT_AND_LOOSE)

    assert _signature(metal, "fragment") == (
        "fragment_main_Input _crossglInput [[stage_in]]"
    )
    assert "Colour a = _crossglInput.a;" in metal
    assert "Coordinates b = _crossglInput.b;" in metal
    assert "float4 color [[user(COLOR0)]];" in metal


def test_hlsl_per_vertex_and_per_instance_structures_compile(tmp_path):
    metal = _metal(tmp_path, HLSL_INSTANCED, ".hlsl")

    signature = _signature(metal, "vertex")
    assert signature.count("[[stage_in]]") == 1
    assert "[[instance_id]]" in signature
    assert "float4 offset [[attribute(6)]];" in metal
    _compile(tmp_path, metal)


def test_single_structure_input_keeps_its_parameter(tmp_path):
    source = TWO_STRUCTS.replace(
        "VSOut main(PosIn p, ColorIn c)", "VSOut main(PosIn p)"
    )
    source = source.replace("o.color = c.color;", "o.color = vec4(1.0);")

    metal = _metal(tmp_path, source)

    assert _signature(metal, "vertex") == "PosIn p [[stage_in]]"
    assert "_crossglInput" not in metal


def test_inputs_sharing_a_vertex_attribute_are_refused(tmp_path):
    source = TWO_STRUCTS.replace("@POSITION", "@location(0)").replace(
        "@COLOR;\n    };\n    struct VSOut", "@location(0);\n    };\n    struct VSOut"
    )

    with pytest.raises(MetalStageInputMergeError, match="both use vertex attribute 0"):
        _metal(tmp_path, source)
