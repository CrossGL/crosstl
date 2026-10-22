"""Fragment built-in inputs read by name reach HLSL and MSL as entry-point parameters threaded into helpers."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import BuiltinValueUnsupportedError

BUILTIN_READS = """#version 450
layout(location = 0) out vec4 o_color;
vec4 shade(vec2 uv) {
    return vec4(gl_FragCoord.xy * uv, float(gl_SampleID), gl_FrontFacing ? 1.0 : 0.0);
}
vec4 outer() { return shade(vec2(0.5)) + vec4(float(gl_PrimitiveID)); }
void main() { o_color = outer() + vec4(float(gl_SampleID)) + gl_FragCoord; }
"""

POINT_COORD = """#version 450
layout(location = 0) out vec4 o_color;
vec2 sprite() { return gl_PointCoord; }
void main() { o_color = vec4(sprite(), 0.0, 1.0); }
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.frag"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr


def _compile_metal(tmp_path, metal):
    if shutil.which("xcrun") is None:
        return
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    if found.returncode != 0:
        return
    source = tmp_path / "out.metal"
    source.write_text(metal)
    _run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(source)]
        + ["-o", str(tmp_path / "out.air")]
    )


def test_metal_binds_fragment_builtins_and_threads_them_into_helpers(tmp_path):
    metal = _translate(tmp_path, BUILTIN_READS, "metal")

    assert not re.search(r"\bgl_\w+", metal), metal
    for attribute, type_name in (
        ("position", "float4"),
        ("front_facing", "bool"),
        ("sample_id", "uint"),
        ("primitive_id", "uint"),
    ):
        assert re.search(rf"{type_name} \w+ \[\[{attribute}\]\]", metal), metal
    # shade reads three built-ins; outer reads one and passes on shade's three.
    assert re.search(r"float4 shade\(float2 uv, float4 \w+, bool \w+, int \w+\)", metal)
    assert re.search(r"float4 outer\(float4 \w+, bool \w+, int \w+, int \w+\)", metal)
    assert re.search(r"int (\w+) = int\(\w+_bits\);", metal)
    _compile_metal(tmp_path, metal)


def test_hlsl_binds_fragment_builtins_and_threads_them_into_helpers(tmp_path):
    hlsl = _translate(tmp_path, BUILTIN_READS, "directx")

    assert not re.search(r"\bgl_\w+", hlsl), hlsl
    for semantic, type_name in (
        ("SV_Position", "float4"),
        ("SV_IsFrontFace", "bool"),
        ("SV_SampleIndex", "uint"),
        ("SV_PrimitiveID", "uint"),
    ):
        assert re.search(rf"{type_name} \w+ : {semantic}", hlsl), hlsl
    assert re.search(r"float4 shade\(float2 uv, bool \w+, int \w+, float4 \w+\)", hlsl)


def test_metal_passes_the_point_coordinate_to_a_helper(tmp_path):
    metal = _translate(tmp_path, POINT_COORD, "metal")

    assert re.search(r"float2 (\w+) \[\[point_coord\]\]", metal), metal
    assert re.search(r"float2 sprite\(float2 \w+\)", metal), metal
    _compile_metal(tmp_path, metal)


def test_hlsl_refuses_the_point_coordinate(tmp_path):
    with pytest.raises(BuiltinValueUnsupportedError, match="gl_PointCoord"):
        _translate(tmp_path, POINT_COORD, "directx")


@pytest.mark.parametrize("backend", ["opengl", "vulkan"])
def test_targets_with_builtin_globals_keep_reading_them(tmp_path, backend):
    code = _translate(tmp_path, BUILTIN_READS, backend)

    assert "gl_SampleID" in code
    assert "cgl_SampleID" not in code


INSTANCE_INDEX = """#version 450
layout(location = 0) in vec4 pos;
layout(location = 0) out flat int o_id;
vec4 offset() { return vec4(float(gl_InstanceIndex), float(gl_VertexIndex), 0.0, 0.0); }
void main() { o_id = gl_InstanceIndex; gl_Position = pos + offset(); }
"""


def _translate_vertex(tmp_path, backend):
    path = tmp_path / "shader.vert"
    path.write_text(INSTANCE_INDEX)
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize(
    "backend, bindings",
    [
        ("directx", [r"uint \w+ : SV_InstanceID", r"uint \w+ : SV_VertexID"]),
        ("metal", [r"uint \w+ \[\[instance_id\]\]", r"uint \w+ \[\[vertex_id\]\]"]),
    ],
)
def test_vertex_indices_reach_the_entry_point_and_its_helpers(
    tmp_path, backend, bindings
):
    code = _translate_vertex(tmp_path, backend)

    assert not re.search(r"\bgl_(Instance|Vertex)Index\b", code), code
    for binding in bindings:
        assert re.search(binding, code), code
    assert re.search(r"float4 offset\(int \w+, int \w+\)", code), code
    if backend == "metal":
        _compile_metal(tmp_path, code)


def test_glsl_adds_the_base_instance_to_the_instance_id(tmp_path):
    glsl = _translate_vertex(tmp_path, "opengl")

    # Vulkan's gl_InstanceIndex counts from the draw's first instance; OpenGL's gl_InstanceID from 0.
    assert "(gl_InstanceID + gl_BaseInstanceARB)" in glsl
    assert "#extension GL_ARB_shader_draw_parameters : require" in glsl
    assert "gl_InstanceIndex" not in glsl
    if shutil.which("glslangValidator"):
        _run(["glslangValidator", "--stdin", "-S", "vert"], input=glsl)


def test_glsl_es_reads_the_instance_index_as_the_instance_id(tmp_path):
    essl = _translate_vertex(tmp_path, "webgl")

    # GLSL ES 3.00 draws have no base instance.
    assert "gl_InstanceID" in essl
    assert "gl_InstanceIndex" not in essl
    assert "gl_BaseInstance" not in essl
    if shutil.which("glslangValidator"):
        _run(["glslangValidator", "--stdin", "-S", "vert"], input=essl)


def test_spirv_declares_the_instance_index_builtin(tmp_path):
    spirv = _translate_vertex(tmp_path, "vulkan")

    assert re.search(r"OpDecorate %\w+ BuiltIn InstanceIndex", spirv), spirv
    assert re.search(r"OpDecorate %\w+ BuiltIn VertexIndex", spirv), spirv


def test_instance_index_parameters_bind_the_instance_builtin(tmp_path):
    source = """
    shader main {
        vertex {
            vec4 main(int ii @ gl_InstanceIndex, int vi @ gl_VertexIndex) @ gl_Position {
                return vec4(float(ii), float(vi), 0.0, 1.0);
            }
        }
    }
    """
    path = tmp_path / "shader.cgl"
    path.write_text(source)

    hlsl = crosstl.translate(str(path), backend="directx", format_output=False)
    metal = crosstl.translate(str(path), backend="metal", format_output=False)
    spirv = crosstl.translate(str(path), backend="vulkan", format_output=False)
    glsl = crosstl.translate(str(path), backend="opengl", format_output=False)

    assert re.search(r"int ii : SV_InstanceID", hlsl), hlsl
    assert re.search(r"\[\[instance_id\]\]", metal), metal
    assert "BuiltIn InstanceIndex" in spirv
    assert "(gl_InstanceID + gl_BaseInstanceARB)" in glsl
