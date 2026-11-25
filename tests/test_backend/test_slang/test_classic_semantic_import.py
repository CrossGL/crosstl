"""Slang input writes the classic vertex semantics as the HLSL importer does.

The Slang importer wrote ``POSITION``, ``NORMAL``, ``TANGENT`` and ``BINORMAL`` as ``in_Position``, ``in_Normal``,
``in_Tangent`` and ``in_Binormal``, names with no fixed location, so a Slang vertex input took its location from the
declaration order while the same HLSL input took the semantic's fixed location (``POSITION`` 0, ``NORMAL`` 1,
``TANGENT`` 2, ``BINORMAL`` 3; ``stage_io_semantic_locations``). The GLSL generator refused a geometry input array
with such a semantic, because it places those by semantic only. The importer now writes ``Position``, ``Normal``,
``Tangent`` and ``Binormal``, the spellings the HLSL importer writes.
"""

import re

import pytest

import crosstl

VERTEX = """struct VertexIn {
    float3 normal : NORMAL;
    float3 tangent : TANGENT0;
    float3 binormal : BINORMAL;
    float3 position : POSITION;
};

[shader("vertex")]
float4 main(VertexIn v) : SV_Position {
    return float4(v.position + v.normal + v.tangent + v.binormal, 1.0);
}
"""

GEOMETRY = """struct GeometryOut {
    float4 position : SV_Position;
};

[shader("geometry")]
[maxvertexcount(2)]
void main(line float4 position[2] : POSITION, inout PointStream<GeometryOut> stream) {
    GeometryOut output;
    output.position = position[0];
    stream.Append(output);
    output.position = position[1];
    stream.Append(output);
}
"""


def _translate(tmp_path, source, backend, name="shader.slang"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def test_classic_semantics_import_under_the_hlsl_importer_names(tmp_path):
    crossgl = _translate(tmp_path, VERTEX, "cgl")
    for member, semantic in (
        ("normal", "Normal"),
        ("tangent", "Tangent"),
        ("binormal", "Binormal"),
        ("position", "Position"),
    ):
        assert f"vec3 {member} @ {semantic};" in crossgl
    assert "in_" not in crossgl


@pytest.mark.parametrize(
    "hlsl_semantic", ["POSITION", "POSITION0"], ids=["unindexed", "index-0"]
)
def test_position_index_0_is_the_unindexed_semantic(tmp_path, hlsl_semantic):
    source = VERTEX.replace(": POSITION;", f": {hlsl_semantic};")
    assert "vec3 position @ Position;" in _translate(tmp_path, source, "cgl")


def test_vertex_inputs_take_the_fixed_locations_of_their_semantics(tmp_path):
    glsl = _translate(tmp_path, VERTEX, "opengl")
    locations = {
        name: int(location)
        for location, name in re.findall(
            r"layout\(location = (\d+)\) in vec3 (\w+);", glsl
        )
    }
    assert locations == {"position": 0, "normal": 1, "tangent": 2, "binormal": 3}
    hlsl_glsl = _translate(
        tmp_path,
        VERTEX.replace('[shader("vertex")]\n', ""),
        "opengl",
        name="shader.hlsl",
    )
    assert re.findall(r"layout\(location = \d+\) in vec3 \w+;", hlsl_glsl) == (
        re.findall(r"layout\(location = \d+\) in vec3 \w+;", glsl)
    )


def test_geometry_input_array_imports_its_position_semantic(tmp_path):
    crossgl = _translate(tmp_path, GEOMETRY, "cgl")
    assert "line vec4 position[2] @ Position," in crossgl
