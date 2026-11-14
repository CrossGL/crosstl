"""HLSL-style hull and domain stages lower to GLSL's tessellation interface for the GLSL and SPIR-V targets.

An HLSL hull shader returns one control point per invocation and names a patch-constant function the pipeline runs
once per patch; a domain shader receives the patch constants and the output patch as parameters. GLSL and SPIR-V
exchange control points through per-vertex arrays, ``patch`` variables and ``gl_TessLevel*``, with the patch work
done by invocation 0 after a barrier. The control-point and patch-constant varyings of both stages must take the
same locations, and the hull's inputs and the domain's outputs the locations their vertex and fragment neighbours
use.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import TessellationPatchUnsupportedError

COMMON_HLSL = """
struct VSOut { float4 Pos : SV_Position; float3 Normal : NORMAL0; float2 UV : TEXCOORD0; };
struct HSOut { float4 Pos : SV_Position; float3 Normal : NORMAL0; float2 UV : TEXCOORD0; };
struct PatchOut
{
    float Edges[3] : SV_TessFactor;
    float Inside : SV_InsideTessFactor;
    float3 Bias : BIAS;
    float Weight : WEIGHT;
    float3 Control[2] : CONTROL;
};
struct DSOut { float4 Pos : SV_Position; float3 Normal : NORMAL0; float2 UV : TEXCOORD0; };
cbuffer Params : register(b0) { float TessLevel; };
"""

HULL_HLSL = """
float EdgeLength(InputPatch<VSOut, 3> patch, int a, int b)
{
    return distance(patch[a].Pos.xyz, patch[b].Pos.xyz);
}

PatchOut PatchFunc(InputPatch<VSOut, 3> patch, const OutputPatch<HSOut, 3> points, uint PrimitiveID : SV_PrimitiveID)
{
    PatchOut o;
    o.Edges[0] = TessLevel * EdgeLength(patch, 1, 2);
    o.Edges[1] = TessLevel * EdgeLength(patch, 2, 0);
    o.Edges[2] = TessLevel * EdgeLength(patch, 0, 1);
    o.Inside = TessLevel;
    o.Bias = points[0].Normal + points[1].Normal + points[2].Normal;
    o.Weight = float(PrimitiveID);
    o.Control[0] = points[0].Pos.xyz;
    o.Control[1] = points[1].Pos.xyz;
    return o;
}

[domain("tri")]
[partitioning("fractional_odd")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("PatchFunc")]
[maxtessfactor(16.0)]
HSOut HSMain(InputPatch<VSOut, 3> patch, uint id : SV_OutputControlPointID)
{
    HSOut o;
    o.Pos = patch[id].Pos;
    o.Normal = patch[id].Normal;
    o.UV = patch[id].UV;
    return o;
}
"""

DOMAIN_HLSL = """
[domain("tri")]
DSOut DSMain(PatchOut pc, float3 uvw : SV_DomainLocation, const OutputPatch<HSOut, 3> patch)
{
    DSOut o;
    o.Pos = patch[0].Pos * uvw.x + patch[1].Pos * uvw.y + patch[2].Pos * uvw.z;
    o.Pos.xyz += pc.Bias * pc.Weight * pc.Edges[0] * pc.Inside + pc.Control[1];
    o.Normal = normalize(patch[0].Normal * uvw.x + patch[1].Normal * uvw.y + patch[2].Normal * uvw.z);
    o.UV = patch[0].UV * uvw.x + patch[1].UV * uvw.y + patch[2].UV * uvw.z;
    return o;
}
"""

VERTEX_HLSL = """
struct VSIn { float3 Pos : POSITION0; float3 Normal : NORMAL0; float2 UV : TEXCOORD0; };
struct VSOut { float4 Pos : SV_Position; float3 Normal : NORMAL0; float2 UV : TEXCOORD0; };
VSOut main(VSIn input)
{
    VSOut o;
    o.Pos = float4(input.Pos, 1.0);
    o.Normal = input.Normal;
    o.UV = input.UV;
    return o;
}
"""

FRAGMENT_HLSL = """
struct PSIn { float4 Pos : SV_Position; float3 Normal : NORMAL0; float2 UV : TEXCOORD0; };
float4 main(PSIn input) : SV_Target { return float4(input.Normal, input.UV.x); }
"""

QUAD_HLSL = """
struct CP { float4 Pos : SV_Position; };
struct QuadPatch { float Edges[4] : SV_TessFactor; float Inside[2] : SV_InsideTessFactor; };
QuadPatch QuadConstants(InputPatch<CP, 4> patch)
{
    QuadPatch o;
    for (int i = 0; i < 4; i++) { o.Edges[i] = 4.0; }
    o.Inside[0] = 4.0;
    o.Inside[1] = 4.0;
    return o;
}
[domain("quad")]
[partitioning("integer")]
[outputtopology("point")]
[outputcontrolpoints(4)]
[patchconstantfunc("QuadConstants")]
CP QuadHull(InputPatch<CP, 4> patch, uint id : SV_OutputControlPointID)
{
    CP o;
    o.Pos = patch[id].Pos;
    return o;
}
[domain("quad")]
CP QuadDomain(QuadPatch pc, float2 uv : SV_DomainLocation, const OutputPatch<CP, 4> patch)
{
    CP o;
    float4 top = lerp(patch[0].Pos, patch[1].Pos, uv.x);
    float4 bottom = lerp(patch[3].Pos, patch[2].Pos, uv.x);
    o.Pos = lerp(top, bottom, uv.y) * (pc.Inside[1] / pc.Edges[3]);
    return o;
}
"""


def _translate(tmp_path, source, name, backend, entry_point=None):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(
        str(path),
        backend=backend,
        format_output=False,
        source_backend="directx",
        entry_point=entry_point,
    )


def _glslang(tmp_path, code, stage):
    if shutil.which("glslangValidator") is None:
        return
    path = tmp_path / f"out.{stage}"
    path.write_text(code)
    result = subprocess.run(
        ["glslangValidator", "-V", str(path), "-o", str(tmp_path / f"{stage}.spv")],
        capture_output=True,
        text=True,
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


def _glsl_locations(code, direction):
    """Sorted ``(location, patch)`` pairs of the user varyings a GLSL shader declares ``in`` or ``out``."""
    pattern = rf"layout\(location = (\d+)\)\s+((?:\w+\s+)*?){direction}\s+\w+\s+\w+"
    return sorted(
        (int(location), "patch" in qualifiers.split())
        for location, qualifiers in re.findall(pattern, code)
    )


def _spirv_entry_modes(code, model):
    entry = re.search(rf'OpEntryPoint {model} %(\w+) "', code).group(1)
    return re.findall(rf"OpExecutionMode %{entry} (\w+(?: \d+)?)", code)


def _spirv_locations(code, storage_class, model):
    """Sorted ``(location, patch)`` of the user variables of ``storage_class`` an entry point of ``model`` lists."""
    interface = re.search(rf'OpEntryPoint {model} %\w+ "\w+"([^\n]*)', code).group(1)
    names = set(re.findall(r"%(\w+)", interface))
    pointers = set(re.findall(rf"%(\w+) = OpTypePointer {storage_class} ", code))
    variables = {
        variable
        for variable, pointer in re.findall(r"%(\w+) = OpVariable %(\w+) \w+", code)
        if pointer in pointers and variable in names
    }
    patch = set(re.findall(r"OpDecorate %(\w+) Patch", code))
    return sorted(
        (int(location), variable in patch)
        for variable, location in re.findall(r"OpDecorate %(\w+) Location (\d+)", code)
        if variable in variables
    )


def test_hlsl_hull_and_domain_lower_to_glsl_tessellation_stages(tmp_path):
    source = COMMON_HLSL + HULL_HLSL + DOMAIN_HLSL
    hull = _translate(tmp_path, source, "t.hlsl", "opengl", entry_point="HSMain")
    domain = _translate(tmp_path, source, "t.hlsl", "opengl", entry_point="DSMain")

    assert "layout(vertices = 3) out;" in hull
    assert re.search(
        r"HSOut hs_point = HSMain_control_point\(uint\(gl_InvocationID\)\);", hull
    )
    assert "gl_out[gl_InvocationID].gl_Position = hs_point.Pos;" in hull
    barrier = hull.index("barrier();")
    first = hull.index("if ((gl_InvocationID == 0))")
    assert barrier < first
    assert "PatchOut hs_patch = PatchFunc(uint(gl_PrimitiveID));" in hull[first:]
    assert "gl_TessLevelOuter[2] = hs_patch.Edges[2];" in hull
    assert "gl_TessLevelInner[0] = hs_patch.Inside;" in hull
    # A patch used as a whole is copied into a local array, and the helper takes that array.
    assert re.search(r"float EdgeLength\(VSOut patch_?\[3\], int a, int b\)", hull)
    for removed in ("InputPatch", "OutputPatch", "patchconstantfunc", "maxtessfactor"):
        assert removed not in hull

    assert "layout(triangles, fractional_odd_spacing, cw) in;" in domain
    assert "ds_patch.Edges[0] = gl_TessLevelOuter[0];" in domain
    assert "ds_patch.Inside = gl_TessLevelInner[0];" in domain
    assert "DSOut ds_vertex = DSMain_domain_point(ds_patch, gl_TessCoord);" in domain
    assert "gl_Position = ds_vertex.Pos;" in domain

    # Control points at 0 (NORMAL0) and 1 (TEXCOORD0), then BIAS 2, WEIGHT 3 and CONTROL 4-5 as patch varyings.
    expected = [(0, False), (1, False), (2, True), (3, True), (4, True)]
    assert _glsl_locations(hull, "out") == expected
    assert _glsl_locations(domain, "in") == expected
    _glslang(tmp_path, hull, "tesc")
    _glslang(tmp_path, domain, "tese")


def test_tessellation_stages_link_with_vertex_and_fragment_stages_by_glsl_location(
    tmp_path,
):
    source = COMMON_HLSL + HULL_HLSL + DOMAIN_HLSL
    vertex = _translate(tmp_path, VERTEX_HLSL, "v.vert.hlsl", "opengl")
    hull = _translate(tmp_path, source, "t.hlsl", "opengl", entry_point="HSMain")
    domain = _translate(tmp_path, source, "t.hlsl", "opengl", entry_point="DSMain")
    fragment = _translate(tmp_path, FRAGMENT_HLSL, "f.frag.hlsl", "opengl")

    # NORMAL0 1 and TEXCOORD0 5 on both sides of the vertex-hull and domain-fragment interfaces.
    assert _glsl_locations(vertex, "out") == _glsl_locations(hull, "in")
    assert _glsl_locations(hull, "in") == [(1, False), (5, False)]
    assert _glsl_locations(domain, "out") == _glsl_locations(fragment, "in")
    assert _glsl_locations(domain, "out") == [(1, False), (5, False)]


def test_hlsl_hull_and_domain_lower_to_spirv_with_the_execution_modes_dxc_states(
    tmp_path,
):
    source = COMMON_HLSL + HULL_HLSL + DOMAIN_HLSL
    spirv = _translate(tmp_path, source, "t.hlsl", "vulkan")

    # DXC states the domain, spacing and winding on the control stage as well.
    assert _spirv_entry_modes(spirv, "TessellationControl") == [
        "OutputVertices 3",
        "Triangles",
        "SpacingFractionalOdd",
        "VertexOrderCw",
    ]
    assert _spirv_entry_modes(spirv, "TessellationEvaluation") == [
        "Triangles",
        "SpacingFractionalOdd",
        "VertexOrderCw",
    ]
    outputs = _spirv_locations(spirv, "Output", "TessellationControl")
    inputs = _spirv_locations(spirv, "Input", "TessellationEvaluation")
    assert (
        outputs == inputs == [(0, False), (1, False), (2, True), (3, True), (4, True)]
    )
    assert "OpControlBarrier" in spirv
    _spirv_val(tmp_path, spirv)


def test_separately_translated_hull_and_domain_link_for_spirv(tmp_path):
    hull = _translate(tmp_path, COMMON_HLSL + HULL_HLSL, "h.hlsl", "vulkan")
    domain = _translate(tmp_path, COMMON_HLSL + DOMAIN_HLSL, "d.hlsl", "vulkan")

    assert _spirv_locations(hull, "Output", "TessellationControl") == _spirv_locations(
        domain, "Input", "TessellationEvaluation"
    )
    # Without its hull the domain stage states only its domain, as DXC does; the hull states the rest.
    assert _spirv_entry_modes(domain, "TessellationEvaluation") == ["Triangles"]
    _spirv_val(tmp_path, hull)
    _spirv_val(tmp_path, domain)


def test_domain_stage_without_its_hull_is_refused_for_glsl(tmp_path):
    with pytest.raises(TessellationPatchUnsupportedError) as error:
        _translate(tmp_path, COMMON_HLSL + DOMAIN_HLSL, "d.hlsl", "opengl")
    assert error.value.reason_code == "domain-without-hull"
    assert error.value.code == "project.translate.opengl-tessellation-patch-unsupported"


def test_domain_stage_links_to_the_hull_by_semantic(tmp_path):
    # The domain stage declares the control point with its members in another order, as HLSL allows.
    domain = DOMAIN_HLSL.replace("OutputPatch<HSOut, 3>", "OutputPatch<DSIn, 3>")
    source = (
        COMMON_HLSL
        + "struct DSIn { float2 UV : TEXCOORD0; float4 Pos : SV_Position; float3 Normal : NORMAL0; };\n"
        + HULL_HLSL
        + domain
    )
    hull_code = _translate(tmp_path, source, "t.hlsl", "opengl", entry_point="HSMain")
    domain_code = _translate(tmp_path, source, "t.hlsl", "opengl", entry_point="DSMain")

    hull_uv = re.search(r"layout\(location = (\d+)\) out vec2 tc_out_UV", hull_code)
    domain_uv = re.search(r"layout\(location = (\d+)\) in vec2 te_in_UV", domain_code)
    assert hull_uv.group(1) == domain_uv.group(1) == "1"
    _glslang(tmp_path, domain_code, "tese")

    unlinked = source.replace(
        "float3 Normal : NORMAL0; };\n" + HULL_HLSL,
        "float3 Normal : NORMAL1; };\n" + HULL_HLSL,
    )
    with pytest.raises(TessellationPatchUnsupportedError) as error:
        _translate(tmp_path, unlinked, "u.hlsl", "opengl", entry_point="DSMain")
    assert error.value.reason_code == "unlinked-domain-input"


def test_quad_domain_with_point_output(tmp_path):
    hull = _translate(tmp_path, QUAD_HLSL, "q.hlsl", "opengl", entry_point="QuadHull")
    domain = _translate(
        tmp_path, QUAD_HLSL, "q.hlsl", "opengl", entry_point="QuadDomain"
    )
    assert "gl_TessLevelOuter[3] = hs_patch.Edges[3];" in hull
    assert "gl_TessLevelInner[1] = hs_patch.Inside[1];" in hull
    assert "layout(quads, equal_spacing, point_mode) in;" in domain
    assert "QuadDomain_domain_point(ds_patch, gl_TessCoord.xy)" in domain
    _glslang(tmp_path, hull, "tesc")
    _glslang(tmp_path, domain, "tese")

    spirv = _translate(tmp_path, QUAD_HLSL, "q.hlsl", "vulkan")
    assert _spirv_entry_modes(spirv, "TessellationEvaluation") == [
        "Quads",
        "SpacingEqual",
        "PointMode",
    ]
    _spirv_val(tmp_path, spirv)


CLIP_HULL_HLSL = """
struct Clip { float4 Pos : SV_Position; float c : SV_ClipDistance0; };
PatchOut ClipConstants(InputPatch<VSOut, 3> patch)
{
    PatchOut o = (PatchOut)0;
    o.Edges[0] = 1.0; o.Edges[1] = 1.0; o.Edges[2] = 1.0; o.Inside = 1.0;
    return o;
}
[domain("tri")]
[partitioning("integer")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("ClipConstants")]
Clip ClipHull(InputPatch<VSOut, 3> patch, uint id : SV_OutputControlPointID)
{
    Clip o;
    o.Pos = patch[id].Pos;
    o.c = 0.0;
    return o;
}
"""


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        (
            HULL_HLSL.replace('[patchconstantfunc("PatchFunc")]\n', ""),
            "patchconstantfunc",
        ),
        (CLIP_HULL_HLSL, "system-value"),
    ],
)
def test_unlowered_tessellation_constructs_are_refused(tmp_path, source, reason):
    with pytest.raises(TessellationPatchUnsupportedError) as error:
        _translate(tmp_path, COMMON_HLSL + source, "r.hlsl", "opengl")
    assert error.value.reason_code == reason
