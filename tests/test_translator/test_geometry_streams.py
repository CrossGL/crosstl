"""HLSL-style geometry stages lower to GLSL's geometry interface for the GLSL and SPIR-V targets.

An HLSL geometry shader reads a topology-qualified input array and appends vertices to a stream object; GLSL and
SPIR-V read per-vertex input arrays and ``gl_in[]``, write stage outputs and call ``EmitVertex()``. Each user varying
must take the location the neighbouring vertex or fragment stage gives the same structure member, or the stages do
not link.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import GeometryStreamUnsupportedError

VERTEX_HLSL = """
struct VSInput { float3 Pos : POSITION0; float3 Normal : NORMAL0; float2 UV : TEXCOORD0; };
struct VSOutput { float4 Pos : SV_POSITION; float3 Normal : NORMAL0; float2 UV : TEXCOORD0; float3 Color : COLOR0; };
VSOutput main(VSInput input)
{
    VSOutput output = (VSOutput)0;
    output.Pos = float4(input.Pos, 1.0);
    output.Normal = input.Normal;
    output.UV = input.UV;
    output.Color = input.Normal;
    return output;
}
"""

GEOMETRY_HLSL = """
struct VSOutput { float4 Pos : SV_POSITION; float3 Normal : NORMAL0; float2 UV : TEXCOORD0; float3 Color : COLOR0; };
struct GSOutput
{
    float4 Pos : SV_POSITION;
    float3 Normal : NORMAL0;
    float2 UV : TEXCOORD0;
    float3 Color : COLOR0;
    uint Layer : SV_RenderTargetArrayIndex;
};
[instance(2)]
[maxvertexcount(3)]
void main(triangle VSOutput input[3], uint InvocationID : SV_GSInstanceID, uint PrimitiveID : SV_PrimitiveID,
          inout TriangleStream<GSOutput> outStream)
{
    for (int i = 0; i < 3; i++)
    {
        GSOutput output = (GSOutput)0;
        output.Pos = input[i].Pos;
        output.Normal = input[i].Normal;
        output.UV = input[i].UV * float(PrimitiveID);
        output.Color = input[i].Color;
        output.Layer = InvocationID;
        outStream.Append(output);
    }
    outStream.RestartStrip();
}
"""

FRAGMENT_HLSL = """
struct PSInput { float4 Pos : SV_POSITION; float3 Normal : NORMAL0; float2 UV : TEXCOORD0; float3 Color : COLOR0; };
float4 main(PSInput input) : SV_TARGET
{
    return float4(input.Normal * input.Color, input.UV.x);
}
"""

HELPER_GEOMETRY_HLSL = """
struct VSOutput { float4 Pos : SV_POSITION; float3 Normal : NORMAL0; };
struct GSOutput { float4 Pos : SV_POSITION; float3 Normal : NORMAL0; };
GSOutput Scaled(VSOutput v, float scale)
{
    GSOutput o;
    o.Pos = v.Pos * scale;
    o.Normal = v.Normal;
    return o;
}
void EmitTriangle(VSOutput tri[3], inout TriangleStream<GSOutput> stream, float scale)
{
    for (int i = 0; i < 3; i++)
    {
        stream.Append(Scaled(tri[i], scale));
    }
    stream.RestartStrip();
}
[maxvertexcount(6)]
void main(triangle VSOutput input[3], inout TriangleStream<GSOutput> outStream)
{
    EmitTriangle(input, outStream, 1.0);
    VSOutput first = input[0];
    if (first.Pos.w > 0.0)
        EmitTriangle(input, outStream, 0.5);
}
"""

POINT_GEOMETRY_HLSL = """
struct VSOut
{
    float4 Pos : SV_POSITION;
    nointerpolation uint Id : TEXCOORD1;
    noperspective float2 UV : TEXCOORD0;
    float Size : PSIZE;
};
struct GSOut
{
    float4 Pos : SV_POSITION;
    nointerpolation uint Id : TEXCOORD1;
    centroid float2 UV : TEXCOORD0;
    uint Viewport : SV_ViewportArrayIndex;
    uint Primitive : SV_PrimitiveID;
};
[maxvertexcount(2)]
void main(point VSOut v[1], uint pid : SV_PrimitiveID, inout LineStream<GSOut> s)
{
    GSOut o = (GSOut)0;
    o.Pos = v[0].Pos;
    o.Id = v[0].Id;
    o.UV = v[0].UV * v[0].Size;
    o.Viewport = 0;
    o.Primitive = pid;
    s.Append(o);
    o.Pos.x += 1.0;
    s.Append(o);
}
"""

MULTI_STAGE_CGL = """
shader main {
    struct VSOutput {
        vec4 Pos @ gl_Position;
        vec3 Normal @ Normal;
    }
    struct GSOutput {
        vec4 Pos @ gl_Position;
        vec3 Color @ Color0;
    }
    struct FSOutput {
        vec4 color @ gl_FragColor;
    }
    vertex {
        VSOutput main(vec3 position @ Position) {
            VSOutput o;
            o.Pos = vec4(position, 1.0);
            o.Normal = position;
            return o;
        }
    }
    geometry {
        @maxvertexcount(3)
        void main(triangle VSOutput input[3], inout TriangleStream<GSOutput> stream) {
            for (int i = 0; i < 3; i++) {
                GSOutput o;
                o.Pos = input[i].Pos;
                o.Color = input[i].Normal;
                stream.Append(o);
            }
        }
    }
    fragment {
        FSOutput main(GSOutput input) {
            FSOutput o;
            o.color = vec4(input.Color, 1.0);
            return o;
        }
    }
}
"""


def _translate(tmp_path, source, name, backend, source_backend="directx"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend=source_backend
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


def _glsl_locations(code, direction):
    """``name -> location`` of the user varyings a GLSL shader declares ``in`` or ``out``."""
    pattern = rf"layout\(location = (\d+)\)\s+(?:\w+\s+)*?{direction}\s+\w+\s+(\w+)"
    return {name: int(location) for location, name in re.findall(pattern, code)}


def _spirv_locations(code, storage_class):
    """Sorted locations of the user ``Input`` or ``Output`` variables of a SPIR-V module."""
    pointers = set(re.findall(rf"%(\w+) = OpTypePointer {storage_class} ", code))
    variables = {
        variable
        for variable, pointer in re.findall(r"%(\w+) = OpVariable %(\w+) \w+", code)
        if pointer in pointers
    }
    return sorted(
        int(location)
        for variable, location in re.findall(r"OpDecorate %(\w+) Location (\d+)", code)
        if variable in variables
    )


def test_hlsl_geometry_shader_lowers_to_glsl_stage_interface(tmp_path):
    code = _translate(tmp_path, GEOMETRY_HLSL, "g.geom.hlsl", "opengl")

    assert "layout(triangles, invocations = 2) in;" in code
    assert "layout(triangle_strip, max_vertices = 3) out;" in code
    assert "uint InvocationID = uint(gl_InvocationID);" in code
    assert "uint PrimitiveID = uint(gl_PrimitiveIDIn);" in code
    assert "gl_in[i].gl_Position" in code
    assert (
        "gl_Layer = int(output_.Layer);" in code
        or "gl_Layer = int(output.Layer);" in code
    )
    assert code.count("EmitVertex();") == 1
    assert "EndPrimitive();" in code
    assert "Append" not in code and "TriangleStream" not in code
    _glslang(tmp_path, code, "geom")


def test_hlsl_vertex_geometry_fragment_chain_links_by_glsl_location(tmp_path):
    vertex = _translate(tmp_path, VERTEX_HLSL, "v.vert.hlsl", "opengl")
    geometry = _translate(tmp_path, GEOMETRY_HLSL, "g.geom.hlsl", "opengl")
    fragment = _translate(tmp_path, FRAGMENT_HLSL, "f.frag.hlsl", "opengl")

    vertex_out = sorted(_glsl_locations(vertex, "out").values())
    geometry_in = sorted(_glsl_locations(geometry, "in").values())
    geometry_out = sorted(_glsl_locations(geometry, "out").values())
    fragment_in = sorted(_glsl_locations(fragment, "in").values())
    # NORMAL0 1, TEXCOORD0 5, COLOR0 13 on every side of both interfaces.
    assert vertex_out == geometry_in == [1, 5, 13]
    assert geometry_out == fragment_in == [1, 5, 13]
    _glslang(tmp_path, vertex, "vert")
    _glslang(tmp_path, fragment, "frag")


def test_hlsl_geometry_shader_lowers_to_spirv_and_links_with_its_neighbours(tmp_path):
    vertex = _translate(tmp_path, VERTEX_HLSL, "v.vert.hlsl", "vulkan")
    geometry = _translate(tmp_path, GEOMETRY_HLSL, "g.geom.hlsl", "vulkan")
    fragment = _translate(tmp_path, FRAGMENT_HLSL, "f.frag.hlsl", "vulkan")

    assert "OpEntryPoint Geometry" in geometry
    for mode in (
        "Triangles",
        "OutputTriangleStrip",
        "OutputVertices 3",
        "Invocations 2",
    ):
        assert re.search(rf"OpExecutionMode %\w+ {mode}\n", geometry), mode
    assert "OpEmitVertex" in geometry and "OpEndPrimitive" in geometry
    assert "BuiltIn Layer" in geometry and "BuiltIn PrimitiveId" in geometry
    assert _spirv_locations(vertex, "Output") == _spirv_locations(geometry, "Input")
    assert _spirv_locations(geometry, "Output") == _spirv_locations(fragment, "Input")
    _spirv_val(tmp_path, geometry)


def test_geometry_interpolation_qualifiers_and_integer_builtins(tmp_path):
    code = _translate(tmp_path, POINT_GEOMETRY_HLSL, "p.geom.hlsl", "opengl")

    assert "layout(points) in;" in code
    assert "layout(line_strip, max_vertices = 2) out;" in code
    assert re.search(r"layout\(location = 6\) flat in uint \w+\[1\];", code)
    assert re.search(r"layout\(location = 5\) noperspective in vec2 \w+\[1\];", code)
    assert re.search(r"layout\(location = 5\) centroid out vec2 \w+;", code)
    assert "gl_in[0].gl_PointSize" in code
    assert "gl_ViewportIndex = int(o.Viewport);" in code
    assert "gl_PrimitiveID = int(o.Primitive);" in code
    assert code.count("EmitVertex();") == 2
    _glslang(tmp_path, code, "geom")
    _spirv_val(
        tmp_path, _translate(tmp_path, POINT_GEOMETRY_HLSL, "p.geom.hlsl", "vulkan")
    )


def test_geometry_stream_helpers_and_whole_input_values(tmp_path):
    code = _translate(tmp_path, HELPER_GEOMETRY_HLSL, "h.geom.hlsl", "opengl")

    # The helper loses its stream parameter and emits through the stage outputs.
    assert re.search(r"void EmitTriangle\(VSOutput tri\[3\], float scale\) \{", code)
    assert "GSOutput gs_vertex = Scaled(tri[i], scale);" in code
    # The input array is passed whole, so the entry point reads a local copy of it.
    assert re.search(r"VSOutput input_?\[3\];", code)
    assert re.search(r"input_?\[2\]\.Normal = \w+\[2\];", code)
    assert re.search(r"EmitTriangle\(input_?, 0\.5\);", code)
    assert "TriangleStream" not in code
    _glslang(tmp_path, code, "geom")
    _spirv_val(
        tmp_path, _translate(tmp_path, HELPER_GEOMETRY_HLSL, "h.geom.hlsl", "vulkan")
    )


def test_hlsl_stream_helper_imports_as_a_helper_not_a_geometry_entry(tmp_path):
    code = _translate(tmp_path, HELPER_GEOMETRY_HLSL, "h.geom.hlsl", "cgl")

    assert code.count("geometry {") == 1
    assert "stage_entry" not in code
    assert re.search(
        r"void EmitTriangle\(in VSOutput tri\[3\], inout TriangleStream<GSOutput> stream",
        code,
    )


def test_multi_stage_crossgl_geometry_program_for_glsl_and_spirv(tmp_path):
    code = _translate(
        tmp_path, MULTI_STAGE_CGL, "m.cgl", "opengl", source_backend="cgl"
    )

    geometry_section = code.split("#ifdef GL_GEOMETRY_SHADER", 1)[1]
    assert "struct GSOutput" in code
    assert re.search(
        r"#ifdef GL_GEOMETRY_SHADER\nlayout\(location = 13\) out vec3 \w+;\n#endif",
        code,
    )
    assert "layout(triangles) in;" in geometry_section
    for stage in ("vert", "geom", "frag"):
        _glslang(tmp_path, code, stage)

    spirv = _translate(
        tmp_path, MULTI_STAGE_CGL, "m.cgl", "vulkan", source_backend="cgl"
    )
    assert "OpEntryPoint Geometry" in spirv
    _spirv_val(tmp_path, spirv)


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        (
            """
struct V { float4 Pos : SV_POSITION; float3 W : WORLDPOS; };
[maxvertexcount(1)]
void main(point V input[1], inout PointStream<V> s) { s.Append(input[0]); }
""",
            "unlocated-varying",
        ),
        (
            """
struct V { float4 Pos : SV_POSITION; float c : SV_ClipDistance0; };
[maxvertexcount(1)]
void main(point V input[1], inout PointStream<V> s) { s.Append(input[0]); }
""",
            "input-system-value",
        ),
        (
            """
struct V { float4 Pos : SV_POSITION; };
[maxvertexcount(1)]
void main(point V input[1], inout PointStream<V> a, inout PointStream<V> b) { a.Append(input[0]); b.Append(input[0]); }
""",
            "multiple-streams",
        ),
    ],
)
def test_unlowered_geometry_constructs_are_refused(tmp_path, source, reason):
    with pytest.raises(GeometryStreamUnsupportedError) as error:
        _translate(tmp_path, source, "r.geom.hlsl", "opengl")
    assert error.value.reason_code == reason
    assert error.value.code == "project.translate.opengl-geometry-stream-unsupported"


def test_color_semantics_have_glsl_locations_and_are_not_spirv_colour_targets(tmp_path):
    glsl = _translate(tmp_path, VERTEX_HLSL, "v.vert.hlsl", "opengl")
    assert re.search(r"layout\(location = 13\) out vec3 \w*Color;", glsl)

    source = """
struct VSOutput { float3 A : TEXCOORD3; float4 Pos : SV_POSITION; float2 UV : COLOR0; };
VSOutput main(float3 p : POSITION0) { VSOutput o = (VSOutput)0; o.Pos = float4(p, 1.0); o.A = p; return o; }
"""
    spirv = _translate(tmp_path, source, "c.vert.hlsl", "vulkan")
    # COLOR0 on a vertex output is a user varying, not colour attachment 0 (which A already uses).
    assert _spirv_locations(spirv, "Output") == [0, 2]
    _spirv_val(tmp_path, spirv)
