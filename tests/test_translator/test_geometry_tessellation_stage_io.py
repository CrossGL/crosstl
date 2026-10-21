"""Geometry and tessellation stage interfaces keep their meaning through CrossGL.

HLSL hull, domain and geometry shaders import with their ``InputPatch<T, N>``, ``OutputPatch<T, N>`` and stream
element types and write their system values back as ``SV_OutputControlPointID``, ``SV_GSInstanceID``,
``SV_TessFactor``, ``SV_InsideTessFactor`` and ``SV_DomainLocation``. GLSL's per-vertex built-in arrays
``gl_in``/``gl_out``, unsized per-vertex varyings, ``patch`` varyings, the tessellation control ``barrier()`` and the
geometry primitive built-ins lower to the SPIR-V that glslang writes for them.
"""

import re
import shutil
import subprocess

import pytest

import crosstl

HULL_HLSL = """
struct VSOutput { float4 Pos : SV_POSITION; float3 Normal : NORMAL0; };
struct HSOutput { float4 Pos : SV_POSITION; float3 Normal : NORMAL0; };
struct ConstantsHSOutput
{
    float TessLevelOuter[3] : SV_TessFactor;
    float TessLevelInner : SV_InsideTessFactor;
};
ConstantsHSOutput ConstantsHS(InputPatch<VSOutput, 3> patch, uint PrimitiveID : SV_PrimitiveID)
{
    ConstantsHSOutput output = (ConstantsHSOutput)0;
    output.TessLevelInner = 1;
    output.TessLevelOuter[0] = 1;
    output.TessLevelOuter[1] = 1;
    output.TessLevelOuter[2] = 1;
    return output;
}
[domain("tri")]
[partitioning("integer")]
[outputtopology("triangle_ccw")]
[outputcontrolpoints(3)]
[patchconstantfunc("ConstantsHS")]
HSOutput main(InputPatch<VSOutput, 3> patch, uint InvocationID : SV_OutputControlPointID)
{
    HSOutput output = (HSOutput)0;
    output.Pos = patch[InvocationID].Pos;
    output.Normal = patch[InvocationID].Normal;
    return output;
}
"""

DOMAIN_HLSL = """
struct ConstantsHSOutput
{
    float TessLevelOuter[3] : SV_TessFactor;
    float TessLevelInner : SV_InsideTessFactor;
};
struct HSOutput { float4 Pos : SV_POSITION; float3 Normal : NORMAL0; };
struct DSOutput { float4 Pos : SV_POSITION; float3 Normal : NORMAL0; };
[domain("tri")]
DSOutput main(ConstantsHSOutput input, float3 TessCoord : SV_DomainLocation,
              const OutputPatch<HSOutput, 3> patch)
{
    DSOutput output = (DSOutput)0;
    output.Pos = TessCoord.x * patch[0].Pos + TessCoord.y * patch[1].Pos + TessCoord.z * patch[2].Pos;
    output.Normal = patch[0].Normal;
    return output;
}
"""

GEOMETRY_HLSL = """
struct VSOutput { float4 Pos : SV_POSITION; float3 Normal : NORMAL0; };
struct GSOutput { float4 Pos : SV_POSITION; float3 Normal : NORMAL0; };
[instance(2)]
[maxvertexcount(3)]
void main(triangle VSOutput input[3], uint InvocationID : SV_GSInstanceID,
          inout TriangleStream<GSOutput> outStream)
{
    for (int i = 0; i < 3; i++)
    {
        GSOutput o;
        o.Pos = input[i].Pos;
        o.Normal = input[i].Normal * float(InvocationID);
        outStream.Append(o);
    }
    outStream.RestartStrip();
}
"""

CONTROL_GLSL = """#version 450
layout (vertices = 3) out;
layout (location = 0) in vec3 inNormal[];
layout (location = 0) out vec3 outNormal[3];
layout (location = 1) patch out vec4 patchColor;
void main()
{
    if (gl_InvocationID == 0)
    {
        gl_TessLevelInner[0] = 1.0;
        gl_TessLevelOuter[0] = 1.0;
        gl_TessLevelOuter[1] = 1.0;
        gl_TessLevelOuter[2] = float(gl_PatchVerticesIn);
        patchColor = vec4(1.0);
    }
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    outNormal[gl_InvocationID] = inNormal[gl_InvocationID];
    barrier();
}
"""

EVALUATION_GLSL = """#version 450
layout (triangles, fractional_odd_spacing, cw) in;
layout (location = 0) in vec3 inNormal[];
layout (location = 1) patch in vec4 patchColor;
layout (location = 0) out vec3 outNormal;
void main()
{
    outNormal = gl_TessCoord.x * inNormal[0] + gl_TessCoord.y * inNormal[1]
        + gl_TessCoord.z * inNormal[2] + patchColor.xyz;
    gl_Position = gl_TessCoord.x * gl_in[0].gl_Position
        + gl_TessCoord.y * gl_in[1].gl_Position + gl_TessCoord.z * gl_in[2].gl_Position;
}
"""

GEOMETRY_GLSL = """#version 450
layout (triangles, invocations = 2) in;
layout (triangle_strip, max_vertices = 3) out;
layout (location = 0) in vec3 inNormal[];
layout (location = 0) out vec3 outNormal;
void main()
{
    for (int i = 0; i < gl_in.length(); i++)
    {
        outNormal = inNormal[i];
        gl_Position = gl_in[i].gl_Position;
        gl_Layer = gl_InvocationID;
        gl_ViewportIndex = gl_InvocationID;
        gl_PrimitiveID = gl_PrimitiveIDIn;
        EmitVertex();
    }
    EndPrimitive();
}
"""


def _translate(tmp_path, source, name, backend, source_backend):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend=source_backend
    )


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


def _dxc(tmp_path, code, profile, entry):
    if shutil.which("dxc") is None:
        return
    path = tmp_path / "out.hlsl"
    path.write_text(code)
    result = subprocess.run(
        ["dxc", "-T", profile, "-E", entry, str(path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _decorated(code, decoration):
    """Ids carrying ``OpDecorate %id <decoration>``."""
    return set(re.findall(rf"OpDecorate %(\w+) {decoration}\n", code + "\n"))


def _variable_type(code, variable):
    """Pointee type text of an ``OpVariable``: ``vec4[32]``-like spelling from the type declarations."""
    pointer = re.search(rf"%{variable} = OpVariable %(\w+) (\w+)", code).group(1)
    pointee = re.search(rf"%{pointer} = OpTypePointer \w+ %(\w+)", code).group(1)
    array = re.search(rf"%{pointee} = OpTypeArray %(\w+) %(\w+)", code)
    if array is None:
        return pointee, None
    length = re.search(rf"%{array.group(2)} = OpConstant %\w+ (\d+)", code).group(1)
    return array.group(1), int(length)


def test_hlsl_patch_and_stream_parameters_import_with_their_element_types(tmp_path):
    hull = _translate(tmp_path, HULL_HLSL, "hull.tesc.hlsl", "cgl", "directx")
    domain = _translate(tmp_path, DOMAIN_HLSL, "domain.tese.hlsl", "cgl", "directx")
    geometry = _translate(tmp_path, GEOMETRY_HLSL, "geo.geom.hlsl", "cgl", "directx")

    assert "InputPatch<VSOutput, 3> patch" in hull
    assert "const OutputPatch<HSOutput, 3> patch" in domain
    assert "inout TriangleStream<GSOutput> outStream" in geometry
    assert "outStream.Append(o);" in geometry
    assert "buffer_append" not in geometry
    for code in (hull, domain, geometry):
        assert "inputPatch" not in code and "outputPatch" not in code


def test_hlsl_hull_domain_and_geometry_round_trip_their_system_values(tmp_path):
    hull = _translate(tmp_path, HULL_HLSL, "hull.tesc.hlsl", "directx", "directx")
    domain = _translate(tmp_path, DOMAIN_HLSL, "domain.tese.hlsl", "directx", "directx")
    geometry = _translate(
        tmp_path, GEOMETRY_HLSL, "geo.geom.hlsl", "directx", "directx"
    )

    assert (
        "HSMain(InputPatch<VSOutput, 3> patch, uint InvocationID : SV_OutputControlPointID)"
        in hull
    )
    assert (
        "ConstantsHS(InputPatch<VSOutput, 3> patch, uint PrimitiveID : SV_PrimitiveID)"
        in hull
    )
    assert "float TessLevelOuter[3] : SV_TessFactor;" in hull
    assert "float TessLevelInner : SV_InsideTessFactor;" in hull
    assert "float3 TessCoord : SV_DomainLocation" in domain
    assert "const OutputPatch<HSOutput, 3> patch" in domain
    assert "[instance(2)]" in geometry
    assert "uint InvocationID : SV_GSInstanceID" in geometry
    assert "outStream.Append(o);" in geometry
    _dxc(tmp_path, hull, "hs_6_0", "HSMain")
    _dxc(tmp_path, domain, "ds_6_0", "DSMain")
    _dxc(tmp_path, geometry, "gs_6_0", "GSMain")


def test_glsl_tessellation_control_lowers_per_vertex_arrays_to_spirv(tmp_path):
    code = _translate(tmp_path, CONTROL_GLSL, "a.tesc", "vulkan", "opengl")

    positions = _decorated(code, "BuiltIn Position")
    assert len(positions) == 2
    sizes = sorted(_variable_type(code, variable)[1] for variable in positions)
    # gl_in has gl_MaxPatchVertices elements, gl_out one per output control point.
    assert sizes == [3, 32]
    normal_in = re.search(r'OpName %(\w+) "inNormal"', code).group(1)
    assert _variable_type(code, normal_in)[1] == 32
    patch_color = re.search(r'OpName %(\w+) "patchColor"', code).group(1)
    assert patch_color in _decorated(code, "Patch")
    for builtin in ("TessLevelOuter", "TessLevelInner"):
        assert _decorated(code, f"BuiltIn {builtin}") <= _decorated(code, "Patch")
    assert _decorated(code, "BuiltIn PatchVertices")
    barrier = re.search(r"OpControlBarrier %(\w+) %(\w+) %(\w+)", code)
    constants = dict(re.findall(r"%(\w+) = OpConstant %\w+ (\d+)\b", code))
    # Workgroup execution scope, Invocation memory scope, no memory semantics, as glslang writes it.
    assert [constants[barrier.group(index)] for index in (1, 2, 3)] == ["2", "4", "0"]
    _spirv_val(tmp_path, code)


def test_glsl_tessellation_evaluation_reads_gl_in_and_patch_inputs(tmp_path):
    code = _translate(tmp_path, EVALUATION_GLSL, "a.tese", "vulkan", "opengl")

    positions = _decorated(code, "BuiltIn Position")
    assert len(positions) == 2
    sizes = sorted((_variable_type(code, variable)[1] or 0) for variable in positions)
    assert sizes == [0, 32]
    patch_color = re.search(r'OpName %(\w+) "patchColor"', code).group(1)
    assert patch_color in _decorated(code, "Patch")
    for mode in ("Triangles", "SpacingFractionalOdd", "VertexOrderCw"):
        assert re.search(rf"OpExecutionMode %\w+ {mode}\n", code)
    _spirv_val(tmp_path, code)


def test_glsl_geometry_lowers_gl_in_length_and_primitive_builtins(tmp_path):
    code = _translate(tmp_path, GEOMETRY_GLSL, "a.geom", "vulkan", "opengl")

    position_in = [
        variable
        for variable in _decorated(code, "BuiltIn Position")
        if _variable_type(code, variable)[1] == 3
    ]
    assert len(position_in) == 1
    assert "OpCapability MultiViewport" in code
    assert _decorated(code, "BuiltIn Layer")
    assert _decorated(code, "BuiltIn ViewportIndex")
    primitive_ids = _decorated(code, "BuiltIn PrimitiveId")
    storage = {
        re.search(rf"%{variable} = OpVariable %\w+ (\w+)", code).group(1)
        for variable in primitive_ids
    }
    # gl_PrimitiveIDIn is read, gl_PrimitiveID written for the emitted primitive.
    assert storage == {"Input", "Output"}
    assert "OpEmitVertex" in code and "OpEndPrimitive" in code
    assert "Invocations 2" in code
    _spirv_val(tmp_path, code)


def test_glsl_patch_input_member_reads_keep_their_name(tmp_path):
    code = _translate(tmp_path, EVALUATION_GLSL, "a.tese", "cgl", "opengl")

    assert "patchColor.xyz" in code
    assert "input.patchColor" not in code


@pytest.mark.parametrize("member", ["gl_ClipDistance", "gl_CullDistance"])
def test_unlowered_per_vertex_members_are_refused(tmp_path, member):
    source = CONTROL_GLSL.replace(
        "barrier();", f"float d = gl_in[0].{member}[0];\n    barrier();"
    )
    with pytest.raises(Exception, match="gl_in"):
        _translate(tmp_path, source, "a.tesc", "vulkan", "opengl")
