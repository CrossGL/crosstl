"""Stage interface arrays GLSL sizes implicitly reach every target with the count their stage gives them.

Geometry inputs, tessellation control outputs, ``pervertex`` inputs and ``gl_ClipDistance`` may omit their size in
GLSL; the hub had refused them as unsized arrays of values. Tessellation inputs, which the pipeline sizes, stay
unsized. The per-vertex dimension of an arrayed interface takes no locations, in GLSL and in SPIR-V.
See :mod:`crosstl.translator.semantics.stage_io_arrays`.
"""

from __future__ import annotations

import re

import pytest

import crosstl
from crosstl.translator.semantics import (
    PerVertexInputUnsupportedError,
    UnsizedArrayUnsupportedError,
)
from tests.conformance.corpus import checks

GEOMETRY = """#version 450
layout(triangles) in;
layout(line_strip, max_vertices = 6) out;
layout(location = 0) in vec3 inNormal[];
layout(location = 1) in vec3 inColor[];
layout(location = 0) out vec3 outColor;
void main() {
    for (int i = 0; i < gl_in.length(); i++) {
        gl_Position = gl_in[i].gl_Position;
        outColor = inColor[i];
        EmitVertex();
        gl_Position = gl_in[i].gl_Position + vec4(inNormal[i], 0.0);
        EmitVertex();
        EndPrimitive();
    }
}
"""

TESS_CONTROL = """#version 450
layout(vertices = 3) out;
layout(location = 0) in vec3 inNormal[];
layout(location = 1) in vec2 inUV[];
layout(location = 0) out vec3 outNormal[];
layout(location = 1) out vec2 outUV[];
void main() {
    gl_TessLevelInner[0] = 1.0;
    gl_TessLevelOuter[0] = 1.0;
    gl_TessLevelOuter[1] = 1.0;
    gl_TessLevelOuter[2] = 1.0;
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    outNormal[gl_InvocationID] = inNormal[gl_InvocationID];
    outUV[gl_InvocationID] = inUV[gl_InvocationID];
}
"""

TESS_EVALUATION = """#version 450
layout(triangles, equal_spacing, cw) in;
layout(location = 0) in vec3 inNormal[];
layout(location = 0) out vec3 outNormal;
void main() {
    gl_Position = gl_TessCoord.x * gl_in[0].gl_Position + gl_TessCoord.y * gl_in[1].gl_Position
        + gl_TessCoord.z * gl_in[2].gl_Position;
    outNormal = gl_TessCoord.x * inNormal[0] + gl_TessCoord.y * inNormal[1] + gl_TessCoord.z * inNormal[2];
}
"""

CLIP_DISTANCE = """#version 450
layout(location = 0) in vec3 inPos;
layout(binding = 0) uniform UBO { mat4 mvp; vec4 plane; } ubo;
out gl_PerVertex { vec4 gl_Position; float gl_ClipDistance[]; };
void main() {
    gl_Position = ubo.mvp * vec4(inPos, 1.0);
    gl_ClipDistance[0] = dot(vec4(inPos, 1.0), ubo.plane);
    gl_ClipDistance[1] = -gl_ClipDistance[0];
}
"""

PER_VERTEX = """#version 450
#extension GL_EXT_fragment_shader_barycentric : require
layout(location = 0) in pervertexEXT vec3 inColor[];
layout(location = 0) out vec4 outColor;
void main() {
    outColor = vec4(inColor[0] * gl_BaryCoordEXT.x + inColor[1] * gl_BaryCoordEXT.y
        + inColor[2] * gl_BaryCoordEXT.z, 1.0);
}
"""

NO_PRIMITIVE = """
shader NoPrimitive {
    in vec3 inNormal @location(0)[];
    geometry {
        layout(line_strip, max_vertices = 2) out;
        void main() {
            gl_Position = vec4(inNormal[0], 1.0);
            EmitVertex();
        }
    }
}
"""

STAGE_OF = {
    ".geom": "geometry",
    ".tesc": "tess_control",
    ".tese": "tess_eval",
    ".vert": "vertex",
    ".frag": "fragment",
}
TARGET_SUFFIX = {"glsl": "glsl", "spirv": "spvasm", "hlsl": "hlsl", "metal": "metal"}
BACKEND = {"glsl": "opengl", "spirv": "vulkan", "hlsl": "directx", "metal": "metal"}


def translate(tmp_path, source, suffix, target):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(
        str(path),
        backend=BACKEND[target],
        source_backend="opengl" if suffix != ".cgl" else None,
        format_output=False,
    )


def assert_accepted(tmp_path, code, target, stage):
    artifact = tmp_path / f"out.{TARGET_SUFFIX[target]}"
    artifact.write_text(code)
    verdict = checks.check(artifact, target, [stage])
    if verdict["verdict"] == "unavailable":
        pytest.skip(f"no {target} checker installed")
    assert verdict["verdict"] == "accepted", verdict.get("log", "")


# The SPIR-V emitter does not lower gl_in yet, so the geometry and tessellation cases are checked as GLSL.
CHECKED = [
    (GEOMETRY, ".geom", "glsl"),
    (TESS_CONTROL, ".tesc", "glsl"),
    (TESS_EVALUATION, ".tese", "glsl"),
    (CLIP_DISTANCE, ".vert", "glsl"),
    (CLIP_DISTANCE, ".vert", "spirv"),
    (PER_VERTEX, ".frag", "glsl"),
    (PER_VERTEX, ".frag", "spirv"),
]


@pytest.mark.parametrize(
    "source, suffix, target",
    CHECKED,
    ids=[f"{STAGE_OF[suffix]}-{target}" for _, suffix, target in CHECKED],
)
def test_implicitly_sized_stage_arrays_translate_to_checked_output(
    tmp_path, source, suffix, target
):
    code = translate(tmp_path, source, suffix, target)
    assert_accepted(tmp_path, code, target, STAGE_OF[suffix])


def test_geometry_inputs_take_the_primitive_vertex_count(tmp_path):
    glsl = translate(tmp_path, GEOMETRY, ".geom", "glsl")
    assert re.search(r"layout\(location = 0\) in vec3 inNormal\[3\];", glsl)
    # The per-vertex dimension takes no locations: inColor at location 1 does not overlap inNormal.
    assert re.search(r"layout\(location = 1\) in vec3 inColor\[3\];", glsl)


def test_tessellation_control_outputs_take_the_patch_size(tmp_path):
    glsl = translate(tmp_path, TESS_CONTROL, ".tesc", "glsl")
    assert re.search(r"out vec3 outNormal\[3\];", glsl)
    # Control shader inputs are sized by the pipeline (gl_MaxPatchVertices) and stay unsized.
    assert re.search(r"in vec3 inNormal\[\];", glsl)


def test_tessellation_evaluation_inputs_stay_pipeline_sized(tmp_path):
    glsl = translate(tmp_path, TESS_EVALUATION, ".tese", "glsl")
    assert re.search(r"in vec3 inNormal\[\];", glsl)


def test_clip_distance_takes_the_largest_constant_index(tmp_path):
    glsl = translate(tmp_path, CLIP_DISTANCE, ".vert", "glsl")
    spirv = translate(tmp_path, CLIP_DISTANCE, ".vert", "spirv")
    assert "gl_ClipDistance[]" not in glsl
    assert re.search(r"OpTypeArray %\w+ %\w*2\b|OpTypeArray %float %uint_2", spirv)


def test_per_vertex_inputs_take_three_elements(tmp_path):
    glsl = translate(tmp_path, PER_VERTEX, ".frag", "glsl")
    assert re.search(r"pervertexEXT in vec3 inColor\[3\];", glsl)
    spirv = translate(tmp_path, PER_VERTEX, ".frag", "spirv")
    assert re.search(r"OpDecorate %\w+ PerVertexKHR", spirv)


@pytest.mark.parametrize("target", ["hlsl", "metal"])
def test_per_vertex_inputs_are_refused_where_no_emitter_lowers_them(tmp_path, target):
    with pytest.raises(PerVertexInputUnsupportedError) as raised:
        translate(tmp_path, PER_VERTEX, ".frag", target)
    backend = BACKEND[target]
    assert (
        raised.value.code == f"project.translate.{backend}-per-vertex-input-unsupported"
    )


@pytest.mark.parametrize("target", ["glsl", "hlsl"])
def test_geometry_inputs_without_a_primitive_are_still_refused(tmp_path, target):
    with pytest.raises(UnsizedArrayUnsupportedError):
        translate(tmp_path, NO_PRIMITIVE, ".cgl", target)
