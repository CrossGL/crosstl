"""Slang receives GLSL-style geometry stages in the HLSL stream form.

The GLSL import declares a geometry stage's interface as per-vertex input arrays and stage outputs and emits vertices
with ``EmitVertex()``. Slang, like HLSL, takes a topology-qualified input array and appends to a stream object, and
the Slang generator printed the GLSL form, which slangc rejected (undefined ``gl_in``, ``EmitVertex``, outputs that
are uniforms). The HLSL stream raising (hlsl_geometry_streams.py) now also runs for Slang, and ``[instance(N)]``
carries GLSL's ``invocations = N``. The raising rewrote integer built-in outputs through ``AssignmentNode.target`` only,
which left the ``left`` alias the Slang generator reads on ``gl_Layer``.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

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

INSTANCED_GLSL = """#version 450
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

TWO_STREAMS_GLSL = """#version 450
layout(points) in;
layout(points, max_vertices = 2) out;
layout(stream = 0, location = 0) out vec4 a;
layout(stream = 1, location = 1) out vec4 b;
void main() {
    a = gl_in[0].gl_Position;
    EmitStreamVertex(0);
    b = gl_in[0].gl_Position;
    EmitStreamVertex(1);
}
"""

_GLSL_NAMES = re.compile(r"\b(gl_\w+|EmitVertex|EndPrimitive)\b")


def _slang(tmp_path, source):
    path = tmp_path / "shader.geom"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="slang", format_output=False)


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _slangc_spirv(tmp_path, code):
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    source = tmp_path / "geometry.slang"
    source.write_text(code, encoding="utf-8")
    output = tmp_path / "geometry.spvasm"
    completed = subprocess.run(
        [slangc, str(source), "-entry", "main", "-stage", "geometry"]
        + ["-target", "spirv-asm", "-o", str(output)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return output.read_text(encoding="utf-8")


def test_glsl_geometry_stage_becomes_a_stream_entry(tmp_path):
    code = _slang(tmp_path, TRIANGLE_STRIP_GLSL)
    flat = _flat(code)
    assert "[maxvertexcount(3)]" in flat, code
    assert re.search(
        r"void main\(point GeometryInput input\[1\], inout TriangleStream<GeometryOutput> stream\)",
        flat,
    ), code
    assert "stream.Append(output);" in flat and "stream.RestartStrip();" in flat, code
    assert not _GLSL_NAMES.search(code), code
    spirv = _slangc_spirv(tmp_path, code)
    assert "OpExecutionMode %main OutputTriangleStrip" in spirv
    assert "OpExecutionMode %main OutputVertices 3" in spirv


def test_invocations_layer_and_primitive_id_reach_their_system_values(tmp_path):
    code = _slang(tmp_path, INSTANCED_GLSL)
    flat = _flat(code)
    assert "[instance(2)]" in flat, code
    assert "output.layer = uint(int(invocationID));" in flat, code
    assert "uint layer : SV_RenderTargetArrayIndex;" in flat, code
    assert not _GLSL_NAMES.search(code), code
    spirv = _slangc_spirv(tmp_path, code)
    assert "OpExecutionMode %main Invocations 2" in spirv
    for builtin in ("Layer", "InvocationId", "PrimitiveId", "Position"):
        assert re.search(rf"OpDecorate %\S+ BuiltIn {builtin}\b", spirv), builtin


def test_several_vertex_streams_are_refused(tmp_path):
    # slangc 2026.20 aborts on a second stream object (internal error in spirv-emit), so the generator refuses.
    with pytest.raises(ValueError, match="at most one stream output parameter"):
        _slang(tmp_path, TWO_STREAMS_GLSL)


def test_directx_keeps_the_same_stream_form(tmp_path):
    path = tmp_path / "shader.geom"
    path.write_text(INSTANCED_GLSL, encoding="utf-8")
    hlsl = _flat(crosstl.translate(str(path), backend="directx", format_output=False))
    assert "output.layer = uint(int(invocationID));" in hlsl, hlsl
