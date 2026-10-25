"""Parameter-only built-in members of Metal stage-input structures.

Metal takes ``[[vertex_id]]``, ``[[instance_id]]`` and ``[[amplification_id]]`` only on entry-point parameters, and a
vertex ``[[stage_in]]`` structure holds only ``[[attribute(n)]]`` members. A stage-input structure with such a member
(``uint id : SV_VertexID;``) must reach Metal as a ``[[stage_in]]`` structure of the other members plus one parameter
per built-in, which :mod:`crosstl.translator.codegen.metal_stage_input_builtins` produces. The Metal compiler checks
each translation where it is installed.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import parse
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.metal_stage_input_builtins import (
    split_metal_stage_input_builtins,
)

_VERTEX_IDS = """
struct VSIn {
    float3 pos : POSITION;
    uint id : SV_VertexID;
    uint inst : SV_InstanceID;
};
struct VSOut {
    float4 pos : SV_Position;
    float4 col : COLOR0;
};
VSOut main(VSIn input) {
    VSOut o;
    o.pos = float4(input.pos + float3(input.id, input.inst, 0), 1.0);
    o.col = float4(1, 0, 0, 1);
    return o;
}
"""

_ONLY_BUILTINS = """
struct VSIn { uint id : SV_VertexID; uint inst : SV_InstanceID; };
float4 main(VSIn input) : SV_Position {
    return float4(float(input.id), float(input.inst), 0, 1);
}
"""

# A matrix and an array member are split into per-column and per-element attributes, a helper takes the structure
# by value, the vertex index is declared ``int`` and the generated parameter names are already taken.
_SPLIT_MEMBERS = """
struct VSIn {
    float4x4 world : WORLD;
    float3 pos : POSITION;
    float2 uvs[2] : TEXCOORD0;
    int id : SV_VertexID;
};
struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
float4 shade(VSIn v) { return mul(v.world, float4(v.pos, 1.0)) + float(v.id); }
VSOut main(VSIn input, uint input_id : SV_InstanceID) {
    VSOut o;
    int input_id2 = 3;
    o.pos = shade(input) + float(input_id + input_id2);
    o.uv = input.uvs[1];
    return o;
}
"""

_AMPLIFIED = """
shader S {
    struct VIn { vec3 position @ Position; uint vid @ gl_VertexID; }
    struct VOut { vec4 position @ gl_Position; vec2 uv @ TexCoord0; }
    struct FIn { vec2 uv @ TexCoord0; uint view @ gl_ViewIndex; bool front @ gl_FrontFacing; }
    struct FOut { vec4 color @ gl_FragColor; }
    vertex {
        VOut main(VIn input) {
            VOut o;
            o.position = vec4(input.position, 1.0);
            o.uv = vec2(float(input.vid), 0.0);
            return o;
        }
    }
    fragment {
        FOut main(FIn input) {
            FOut o;
            o.color = vec4(input.uv, float(input.view), input.front ? 1.0 : 0.0);
            return o;
        }
    }
}
"""

_PLAIN = """
shader S {
    struct VIn { vec3 position @ Position; }
    struct VOut { vec4 position @ gl_Position; }
    vertex {
        VOut main(VIn input) {
            VOut o;
            o.position = vec4(input.position, 1.0);
            return o;
        }
    }
}
"""


def _translate(tmp_path, source, suffix):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal")


def _compile(tmp_path, code):
    if not shutil.which("xcrun"):
        return
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    if found.returncode != 0:
        return
    source = tmp_path / "out.metal"
    source.write_text(code)
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.0", "-c", str(source)]
        + ["-o", str(tmp_path / "out.air")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + code
    assert "warning" not in result.stderr, result.stderr + code


def _struct_body(code, name):
    match = re.search(rf"struct {name} \{{(.*?)\}};", code, flags=re.S)
    assert match is not None, code
    return match.group(1)


def _entry_signature(code, stage):
    match = re.search(rf"^{stage} \w+ \w+\((.*?)\) \{{", code, flags=re.S | re.M)
    assert match is not None, code
    return " ".join(match.group(1).split())


def test_vertex_and_instance_ids_become_entry_parameters(tmp_path):
    code = _translate(tmp_path, _VERTEX_IDS, ".hlsl")

    # The structure used to carry `uint id [[vertex_id]];`, which Metal rejects outside a parameter list.
    for name in ("VSIn", "VSIn_StageIn"):
        assert "vertex_id" not in _struct_body(code, name)
        assert "instance_id" not in _struct_body(code, name)
    assert "float3 pos [[attribute(0)]];" in _struct_body(code, "VSIn_StageIn")
    assert _entry_signature(code, "vertex") == (
        "VSIn_StageIn input_stage_in [[stage_in]], uint input_id [[vertex_id]], "
        "uint input_inst [[instance_id]]"
    )
    for line in (
        "VSIn input;",
        "input.pos = input_stage_in.pos;",
        "input.id = input_id;",
        "input.inst = input_inst;",
    ):
        assert line in code
    _compile(tmp_path, code)


def test_structure_of_only_builtins_takes_no_stage_input(tmp_path):
    code = _translate(tmp_path, _ONLY_BUILTINS, ".hlsl")

    assert "[[stage_in]]" not in code
    assert "VSIn_StageIn" not in code
    assert _entry_signature(code, "vertex") == (
        "uint input_id [[vertex_id]], uint input_inst [[instance_id]]"
    )
    _compile(tmp_path, code)


def test_split_members_helpers_and_taken_names(tmp_path):
    code = _translate(tmp_path, _SPLIT_MEMBERS, ".hlsl")

    # The structure the body and `shade` use is a plain value: no member keeps a stage attribute.
    assert "[[" not in _struct_body(code, "VSIn")
    stage_in = _struct_body(code, "VSIn_StageIn")
    assert "float4 world_0 [[attribute(" in stage_in
    assert "float2 uvs_1 [[attribute(" in stage_in
    assert " id" not in stage_in
    signature = _entry_signature(code, "vertex")
    assert "[[vertex_id]]" in signature
    assert "uint input_id [[instance_id]]" in signature
    # The `int` vertex index arrives as `uint` and is converted once, under a name no local takes.
    assert re.search(r"int input_id3 = int\(\w+\);", code)
    assert "input.id = input_id3;" in code
    assert "shade(input)" in code
    _compile(tmp_path, code)


def test_fragment_amplification_id_becomes_a_parameter(tmp_path):
    code = _translate(tmp_path, _AMPLIFIED, ".cgl")

    assert "amplification_id" not in _struct_body(code, "FIn_StageIn")
    assert "bool front [[front_facing]];" in _struct_body(code, "FIn_StageIn")
    assert _entry_signature(code, "fragment") == (
        "FIn_StageIn input_stage_in [[stage_in]], uint input_view [[amplification_id]]"
    )
    assert _entry_signature(code, "vertex") == (
        "VIn_StageIn input_stage_in [[stage_in]], uint input_vid [[vertex_id]]"
    )
    # The vertex output and the fragment input still link by the same user key.
    assert "float2 uv [[user(TEXCOORD0)]];" in _struct_body(code, "VOut")
    assert "float2 uv [[user(TEXCOORD0)]];" in _struct_body(code, "FIn_StageIn")
    _compile(tmp_path, code)


@pytest.mark.parametrize("backend", ["opengl", "directx", "wgsl"])
def test_other_targets_keep_the_input_structure(tmp_path, backend):
    path = tmp_path / "shader.hlsl"
    path.write_text(_VERTEX_IDS)
    code = crosstl.translate(str(path), backend=backend)

    assert "_StageIn" not in code
    assert "input_stage_in" not in code


def test_programs_without_parameter_only_members_are_returned_unchanged():
    ast = parse(_PLAIN)

    assert split_metal_stage_input_builtins(ast, MetalCodeGen()) is ast


def test_rewrite_leaves_the_input_ast_unchanged():
    ast = parse(_AMPLIFIED)
    vertex = next(
        stage.entry_point
        for stage_type, stage in ast.stages.items()
        if "VERTEX" in str(stage_type).upper()
    )
    before = [parameter.name for parameter in vertex.parameters]

    rewritten = split_metal_stage_input_builtins(ast, MetalCodeGen())

    assert rewritten is not ast
    assert [parameter.name for parameter in vertex.parameters] == before
    assert [struct.name for struct in ast.structs] == ["VIn", "VOut", "FIn", "FOut"]
    assert [struct.name for struct in rewritten.structs] == [
        "VIn",
        "VIn_StageIn",
        "VOut",
        "FIn",
        "FIn_StageIn",
        "FOut",
    ]
