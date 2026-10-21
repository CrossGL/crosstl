"""GLSL-style geometry stages become HLSL's stream form for the DirectX target.

GLSL geometry shaders declare their primitives as stage layouts, read per-vertex input arrays and ``gl_in[]``, and
emit a vertex by writing stage outputs and calling ``EmitVertex()``. HLSL takes the primitive as a topology-qualified
array parameter and appends vertices to a stream object. The DirectX generator printed the GLSL form unchanged, with
undeclared ``gl_Position`` and ``EmitVertex``, which DXC rejects.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.diagnostics import CODE_PREFIX
from crosstl.translator.semantics import HlslGeometryStreamUnsupportedError

# The coverage probe for the geometry stage (tools/coverage/crossgl.py).
PROBE_CGL = """
shader Probe {
    in vec4 inPosition[1] @location(0);
    geometry {
        layout(points) in;
        layout(points, max_vertices = 1) out;
        void main() {
            gl_Position = inPosition[0] * 2.0;
            EmitVertex();
            EndPrimitive();
        }
    }
}
"""

TRIANGLE_STRIP_GLSL = """#version 450
layout(points) in;
layout(triangle_strip, max_vertices = 3) out;
layout(location = 0) in vec2 vuv[];
layout(location = 0) out vec2 guv;
void main() {
    for (int k = 0; k < 3; k++) {
        gl_Position = gl_in[0].gl_Position + vec4(float(k), 0.0, 0.0, 0.0);
        guv = vuv[0];
        EmitVertex();
    }
    EndPrimitive();
}
"""

HELPER_GLSL = """#version 450
layout(triangles, invocations = 2) in;
layout(line_strip, max_vertices = 4) out;
layout(location = 0) in vec3 vnormal[];
layout(location = 1) flat in int vid[];
layout(location = 0) flat out int gid;
layout(location = 1) out vec3 gnormal;
void emit(int k) {
    gl_Position = gl_in[k].gl_Position;
    gnormal = vnormal[k];
    gid = vid[k] + gl_PrimitiveIDIn;
    gl_Layer = gl_InvocationID;
    EmitVertex();
}
void main() {
    for (int k = 0; k < 3; k++) {
        emit(k);
    }
    EndPrimitive();
}
"""

WHOLE_ARRAY_GLSL = """#version 450
layout(lines) in;
layout(points, max_vertices = 2) out;
layout(location = 0) in vec3 vcolor[];
layout(location = 2) out vec3 gcolor;
void main() {
    vec3 copies[2] = vcolor;
    for (int k = 0; k < vcolor.length(); k++) {
        gl_Position = gl_in[k].gl_Position;
        gcolor = copies[k];
        gl_PrimitiveID = gl_PrimitiveIDIn;
        EmitVertex();
    }
}
"""

VERTEX_GLSL = """#version 450
layout(location = 0) in vec4 position;
layout(location = 1) in vec3 normal;
layout(location = 0) out vec3 vnormal;
layout(location = 1) flat out int vid;
void main() {
    vnormal = normal;
    vid = gl_VertexIndex;
    gl_Position = position;
}
"""

HLSL_GEOMETRY = """
struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
struct GSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; uint layer : SV_RenderTargetArrayIndex; };
[instance(2)]
[maxvertexcount(3)]
void main(triangle VSOut input[3], uint pid : SV_PrimitiveID, uint gid : SV_GSInstanceID,
          inout TriangleStream<GSOut> stream) {
    for (int k = 0; k < 3; k++) {
        GSOut o;
        o.pos = input[k].pos;
        o.uv = input[k].uv * float(pid);
        o.layer = gid;
        stream.Append(o);
    }
    stream.RestartStrip();
}
"""


# GLSL built-ins and stream functions, none of which HLSL declares.
_GLSL_NAMES = re.compile(r"\b(gl_\w+|EmitVertex|EndPrimitive)\b")


def _translate(tmp_path, source, name, backend="directx"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def _dxc(tmp_path, code, profile="gs_6_0"):
    dxc = shutil.which("dxc")
    if dxc is None:
        return
    path = tmp_path / f"out_{profile}.hlsl"
    path.write_text(code, encoding="utf-8")
    entry = re.search(r"\]\s*void\s+(\w+)\s*\(", code) or re.search(
        r"\b(\w+Main)\s*\(", code
    )
    result = subprocess.run(
        [
            dxc,
            "-T",
            profile,
            "-E",
            entry.group(1),
            str(path),
            "-Fo",
            str(tmp_path / "o"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _struct_semantics(code, struct_name):
    body = re.search(rf"struct {struct_name}\s*\{{(.*?)\}};", code, re.S).group(1)
    return {
        name: semantic for name, semantic in re.findall(r"(\w+)\s*:\s*(\w+)\s*;", body)
    }


def test_the_crossgl_probe_becomes_a_point_stream_entry(tmp_path):
    code = _translate(tmp_path, PROBE_CGL, "probe.cgl")
    flat = _flat(code)
    assert "[maxvertexcount(1)]" in flat
    assert re.search(
        r"void GSMain\(point GeometryInput input\[1\], inout PointStream<GeometryOutput> stream\)",
        flat,
    )
    assert "output.position = (input[0].inPosition * 2.0);" in flat
    assert "stream.Append(output); stream.RestartStrip();" in flat
    assert _struct_semantics(code, "GeometryInput") == {"inPosition": "TEXCOORD0"}
    assert _struct_semantics(code, "GeometryOutput") == {"position": "SV_POSITION"}
    assert not _GLSL_NAMES.search(code)
    _dxc(tmp_path, code)


def test_glsl_geometry_shader_reads_gl_in_and_appends_its_outputs(tmp_path):
    code = _translate(tmp_path, TRIANGLE_STRIP_GLSL, "shader.geom")
    flat = _flat(code)
    assert "[maxvertexcount(3)]" in flat
    assert "inout TriangleStream<GeometryOutput> stream" in flat
    assert (
        "output.position = (input[0].position + float4(float(k), 0.0, 0.0, 0.0));"
        in flat
    )
    assert "output.guv = input[0].vuv; stream.Append(output);" in flat
    assert _struct_semantics(code, "GeometryInput") == {
        "vuv": "TEXCOORD0",
        "position": "SV_POSITION",
    }
    assert _struct_semantics(code, "GeometryOutput") == {
        "guv": "TEXCOORD0",
        "position": "SV_POSITION",
    }
    # The stage's varyings are entry structures now, not global (constant-buffer) variables.
    assert not re.search(r"^float2 (vuv|guv)", code, re.M)
    assert not _GLSL_NAMES.search(code)
    _dxc(tmp_path, code)


def test_helpers_take_the_interface_and_system_values_arrive_as_parameters(tmp_path):
    code = _translate(tmp_path, HELPER_GLSL, "shader.geom")
    flat = _flat(code)
    assert "[instance(2)][maxvertexcount(4)]" in flat
    assert re.search(
        r"void GSMain\(triangle GeometryInput input\[3\], uint invocationID : SV_GSInstanceID, "
        r"uint primitiveID : SV_PrimitiveID, inout LineStream<GeometryOutput> stream\)",
        flat,
    )
    assert (
        "void emit(int k, GeometryInput input[3], inout GeometryOutput output, "
        "inout LineStream<GeometryOutput> stream, uint invocationID, uint primitiveID)"
    ) in flat
    assert "emit(k, input, output, stream, invocationID, primitiveID);" in flat
    # HLSL declares SV_RenderTargetArrayIndex uint and GLSL gl_Layer int.
    assert "output.layer = uint(int(invocationID));" in flat
    assert "nointerpolation int gid : TEXCOORD0;" in flat
    assert _struct_semantics(code, "GeometryOutput")["layer"] == (
        "SV_RenderTargetArrayIndex"
    )
    assert not _GLSL_NAMES.search(code)
    _dxc(tmp_path, code)


def test_a_whole_input_array_reads_a_local_copy(tmp_path):
    code = _translate(tmp_path, WHOLE_ARRAY_GLSL, "shader.geom")
    flat = _flat(code)
    assert (
        "float3 vcolor[2]; vcolor[0] = input[0].vcolor; vcolor[1] = input[1].vcolor;"
    ) in flat
    assert "(k < 2)" in flat
    assert "output.primitiveID = uint(int(primitiveID));" in flat
    assert "uint primitiveID : SV_PrimitiveID;" in flat
    assert _struct_semantics(code, "GeometryOutput")["gcolor"] == "TEXCOORD2"
    assert not _GLSL_NAMES.search(code)
    _dxc(tmp_path, code)


def test_geometry_inputs_link_with_the_vertex_stage_outputs(tmp_path):
    vertex = _translate(tmp_path, VERTEX_GLSL, "shader.vert")
    geometry = _translate(tmp_path, HELPER_GLSL, "shader.geom")
    outputs = _struct_semantics(vertex, "VertexOutput")
    inputs = _struct_semantics(geometry, "GeometryInput")
    assert {name: outputs[name] for name in ("vnormal", "vid")} == {
        name: inputs[name] for name in ("vnormal", "vid")
    }
    assert "SV_POSITION" in outputs.values() and inputs["position"] == "SV_POSITION"
    _dxc(tmp_path, vertex, "vs_6_0")


def test_hlsl_geometry_round_trips_through_glsl(tmp_path):
    glsl = _translate(tmp_path, HLSL_GEOMETRY, "shader.hlsl", "opengl")
    assert "EmitVertex();" in glsl
    code = _translate(tmp_path, glsl, "round.geom")
    flat = _flat(code)
    assert "[instance(2)][maxvertexcount(3)]" in flat
    assert "inout TriangleStream<GeometryOutput> stream" in flat
    assert "SV_RenderTargetArrayIndex" in flat and "SV_GSInstanceID" in flat
    assert not _GLSL_NAMES.search(code)
    _dxc(tmp_path, code)


def test_other_targets_keep_the_glsl_form(tmp_path):
    glsl = _translate(tmp_path, TRIANGLE_STRIP_GLSL, "shader.geom", "opengl")
    assert "EmitVertex();" in glsl and "gl_in[0].gl_Position" in glsl
    assert "GeometryOutput" not in glsl


def _geometry(
    body, layouts="layout(points) in;\nlayout(points, max_vertices = 1) out;"
):
    return (
        "shader P {\n    in vec4 v[1] @location(0);\n    geometry {\n"
        f"        {layouts}\n        void main() {{\n            {body}\n        }}\n    }}\n}}\n"
    )


@pytest.mark.parametrize(
    "source, reason",
    [
        (_geometry("gl_Position = v[0]; EmitStreamVertex(1);"), "multiple-streams"),
        (_geometry("gl_PointSize = 2.0; EmitVertex();"), "point-size"),
        (
            _geometry("gl_Position = vec4(gl_in[0].gl_ClipDistance[0]); EmitVertex();"),
            "clip-distance",
        ),
        (
            _geometry(
                "gl_Position = v[0]; EmitVertex();",
                "layout(points, max_vertices = 1) out;",
            ),
            "input-primitive",
        ),
        (
            _geometry(
                "gl_Position = v[0]; EmitVertex();",
                "layout(points) in;\nlayout(points, max_vertices = 1, stream = 1) out;",
            ),
            "multiple-streams",
        ),
    ],
)
def test_constructs_without_an_hlsl_stream_form_are_refused(tmp_path, source, reason):
    with pytest.raises(HlslGeometryStreamUnsupportedError) as raised:
        _translate(tmp_path, source, "shader.cgl")
    assert raised.value.reason_code == reason
    assert raised.value.code == f"{CODE_PREFIX}hlsl-geometry-stream-unsupported"
