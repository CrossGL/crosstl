"""Slang input keeps the configuration of geometry, hull and domain entry points.

The Slang importer dropped a geometry stage's input primitive (``triangle float4 p[3]``), its ``[maxvertexcount]``
and ``[instance]``, every hull attribute (``[domain]``, ``[partitioning]``, ``[outputtopology]``,
``[outputcontrolpoints]``, ``[patchconstantfunc]``, ``[maxtessfactor]``) and the domain stage's ``[domain]``, kept
the tessellation system values under their HLSL names, and wrote the patch constant function's ``InputPatch`` as a
plain array. The Slang generator then refused every one of these stages. The importer now writes them as the HLSL
importer does, so they translate back to Slang and on to HLSL and SPIR-V.
"""

import re
import shutil
import subprocess

import pytest

import crosstl

GEOMETRY = """struct GeometryOut {
    float4 position : SV_Position;
};

[shader("geometry")]
[maxvertexcount(1)]
[instance(2)]
void main(triangle float4 position[3] : POSITION, uint instance : SV_GSInstanceID,
          inout PointStream<GeometryOut> stream) {
    GeometryOut output;
    output.position = position[0] + position[1] + position[2] + float(instance);
    stream.Append(output);
    stream.RestartStrip();
}
"""

HULL = """struct ControlPoint {
    float4 position : POSITION;
};

struct PatchConstants {
    float edges[3] : SV_TessFactor;
    float inside : SV_InsideTessFactor;
};

PatchConstants constants(InputPatch<ControlPoint, 3> patch) {
    PatchConstants output;
    output.edges[0] = patch[0].position.x;
    output.edges[1] = 1.0;
    output.edges[2] = 1.0;
    output.inside = 1.0;
    return output;
}

[shader("hull")]
[domain("tri")]
[partitioning("integer")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("constants")]
[maxtessfactor(16.0)]
ControlPoint main(InputPatch<ControlPoint, 3> patch, uint id : SV_OutputControlPointID) {
    ControlPoint output;
    output.position = patch[id].position * 2.0;
    return output;
}
"""

DOMAIN = """struct ControlPoint {
    float4 position : POSITION;
};

struct PatchConstants {
    float edges[4] : SV_TessFactor;
    float inside[2] : SV_InsideTessFactor;
};

[shader("domain")]
[domain("quad")]
float4 main(PatchConstants constants, float2 coord : SV_DomainLocation,
            const OutputPatch<ControlPoint, 4> patch) : SV_Position {
    float4 a = lerp(patch[0].position, patch[1].position, coord.x);
    float4 b = lerp(patch[3].position, patch[2].position, coord.x);
    return lerp(a, b, coord.y) * constants.edges[0];
}
"""


def _translate(tmp_path, source, backend, name="shader.slang"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _entry(crossgl):
    return " ".join(
        re.search(r"\bmain\((.*?)\)\s*(@ gl_Position\s*)?\{", crossgl, flags=re.S)
        .group(1)
        .split()
    )


def _slangc(tmp_path, source, stage):
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    path = tmp_path / f"{stage}.slang"
    path.write_text(source)
    result = subprocess.run(
        [
            slangc,
            str(path),
            "-target",
            "spirv",
            "-entry",
            "main",
            "-stage",
            stage,
            "-o",
            str(tmp_path / f"{stage}.spv"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr + source


def test_geometry_entry_keeps_its_primitive_vertex_count_and_instances(tmp_path):
    crossgl = _translate(tmp_path, GEOMETRY, "cgl")
    assert re.search(r"@ maxvertexcount\(1\)\s*@ instance\(2\)", crossgl), crossgl
    assert _entry(crossgl) == (
        "triangle vec4 position[3] @ Position, uint instance @ gl_InvocationID, "
        "inout PointStream<GeometryOut> stream"
    )
    slang = _translate(tmp_path, crossgl, "slang", name="shader.cgl")
    assert re.search(r"\[maxvertexcount\(1\)\]", slang), slang
    assert re.search(r"\[instance\(2\)\]", slang), slang
    assert re.search(r"\btriangle\s+float4\s+position\[3\]", slang), slang
    assert "SV_GSInstanceID" in slang, slang
    _slangc(tmp_path, slang, "geometry")


def test_hull_entry_keeps_its_attributes_and_patch_constant_function(tmp_path):
    crossgl = _translate(tmp_path, HULL, "cgl")
    attributes = re.findall(r"^\s*(@ ?\w+\(.*?\))\s*$", crossgl, flags=re.M)
    assert attributes == [
        '@domain("tri")',
        '@ partitioning("integer")',
        '@ outputtopology("triangle_cw")',
        "@ outputcontrolpoints(3)",
        '@ patchconstantfunc("constants")',
        "@ maxtessfactor(16.0)",
    ], crossgl
    assert "PatchConstants constants(InputPatch<ControlPoint, 3> patch)" in crossgl
    assert "float edges[3] @ gl_TessLevelOuter;" in crossgl
    assert "float inside @ gl_TessLevelInner;" in crossgl
    assert _entry(crossgl) == (
        "InputPatch<ControlPoint, 3> patch, uint id @ gl_InvocationID"
    )
    slang = _translate(tmp_path, crossgl, "slang", name="shader.cgl")
    for attribute in (
        r'\[domain\("tri"\)\]',
        r"\[outputcontrolpoints\(3\)\]",
        r'\[patchconstantfunc\("constants"\)\]',
        r"\[maxtessfactor\(16\.0\)\]",
    ):
        assert re.search(attribute, slang), slang
    assert "constants(InputPatch<ControlPoint, 3> patch)" in slang, slang
    _slangc(tmp_path, slang, "hull")
    hlsl = _translate(tmp_path, HULL, "directx")
    assert re.search(r"\[outputcontrolpoints\(3\)\]", hlsl), hlsl


def test_domain_entry_keeps_its_domain_and_tessellation_coordinate(tmp_path):
    crossgl = _translate(tmp_path, DOMAIN, "cgl")
    assert re.search(r'@domain\("quad"\)', crossgl), crossgl
    assert "vec2 coord @ gl_TessCoord" in crossgl, crossgl
    assert "float edges[4] @ gl_TessLevelOuter;" in crossgl, crossgl
    slang = _translate(tmp_path, crossgl, "slang", name="shader.cgl")
    assert re.search(r'\[domain\("quad"\)\]', slang), slang
    assert re.search(r"float2\s+coord\s*:\s*SV_DomainLocation", slang), slang
    _slangc(tmp_path, slang, "domain")
    spirv = _translate(tmp_path, DOMAIN, "vulkan")
    assert re.search(r"OpExecutionMode %\w+ Quads", spirv), spirv
    assert re.search(r"OpDecorate %\w+ BuiltIn TessCoord", spirv), spirv


def test_ordinary_function_keeps_patch_parameters_as_arrays(tmp_path):
    source = """struct ControlPoint { float4 position : POSITION; };
float4 first(InputPatch<ControlPoint, 3> patch) { return patch[0].position; }
"""
    crossgl = _translate(tmp_path, source, "cgl")
    assert "vec4 first(ControlPoint patch[3])" in crossgl, crossgl
