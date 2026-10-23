"""GLSL ``in``/``out`` interface blocks keep their locations through CrossGL to GLSL and HLSL."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.GLSL.OpenglParser import UnsupportedGLSLFeatureError
from crosstl.translator.codegen.hlsl_refusals import DirectXLoweringRefused

INSTANCE_VERTEX = """#version 460
layout (location = 0) in vec3 inPos;
layout (location = 1) out VertexData {
    vec2 uv;
    mat2 basis;
    flat uint texture_index;
} vertex_out;
layout (location = 6) out vec3 after;
void main() {
    vertex_out.uv = inPos.xy;
    vertex_out.basis = mat2(1.0);
    vertex_out.texture_index = 3u;
    after = inPos;
    gl_Position = vec4(inPos, 1.0);
}
"""

INSTANCE_FRAGMENT = """#version 460
layout (location = 1) in VertexData {
    vec2 uv;
    mat2 basis;
    flat uint texture_index;
} vertex_in;
layout (location = 6) in vec3 after;
layout (location = 0) out vec4 outColor;
void main() {
    outColor = vec4(vertex_in.uv + vertex_in.basis[1], float(vertex_in.texture_index), after.x);
}
"""

ANONYMOUS_FRAGMENT = """#version 460
layout (location = 2) in VertexData {
    vec2 uv;
    mat3 basis;
    vec4 arr[2];
    layout(location = 9) flat uint texture_index;
    vec3 tail;
};
layout (location = 0) out vec4 outColor;
void main() {
    outColor = vec4(uv + basis[2].xy + arr[1].zw, float(texture_index), tail.x);
}
"""

LOCATED_FIELD_VERTEX = """#version 460
layout (location = 0) in vec3 inPos;
layout (location = 1) out VertexData {
    vec2 uv;
    layout(location = 5) flat uint texture_index;
} vertex_out;
void main() {
    vertex_out.uv = inPos.xy;
    vertex_out.texture_index = 3u;
    gl_Position = vec4(inPos, 1.0);
}
"""

UNCOUNTABLE_FRAGMENT = """#version 460
struct Inner { vec2 a; };
layout (location = 0) in VertexData {
    Inner inner;
    vec2 uv;
};
layout (location = 0) out vec4 outColor;
void main() {
    outColor = vec4(inner.a, uv);
}
"""


def _translate(tmp_path, source, name, backend):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _validate(command):
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _glslang(tmp_path, code, stage):
    if shutil.which("glslangValidator") is None:
        return
    path = tmp_path / f"out.{stage}"
    path.write_text(code)
    _validate(["glslangValidator", "-S", stage, str(path)])


def _dxc(tmp_path, code, profile, entry):
    if shutil.which("dxc") is None:
        return
    path = tmp_path / "out.hlsl"
    path.write_text(code)
    _validate(["dxc", "-T", profile, "-E", entry, str(path)])


def _locations(crossgl, struct_name):
    body = crossgl.split(f"struct {struct_name} {{", 1)[1].split("}", 1)[0]
    return {
        name: int(location)
        for name, location in re.findall(r"(\w+)(?:\[\d+\])? @location\((\d+)\)", body)
    }


def test_anonymous_block_members_take_consecutive_locations(tmp_path):
    crossgl = _translate(tmp_path, ANONYMOUS_FRAGMENT, "anon.frag", "cgl")

    # GLSL 4.60 section 4.4.1: mat3 spans 3 locations, vec4[2] spans 2, and an explicit location restarts the count.
    assert _locations(crossgl, "FragmentInput") == {
        "uv": 2,
        "basis": 3,
        "arr": 6,
        "texture_index": 9,
        "tail": 10,
    }


def test_anonymous_block_member_after_uncountable_type_is_refused(tmp_path):
    with pytest.raises(UnsupportedGLSLFeatureError, match="layout\\(location\\)"):
        _translate(tmp_path, UNCOUNTABLE_FRAGMENT, "uncountable.frag", "cgl")


def test_glsl_declares_anonymous_input_block_once(tmp_path):
    glsl = _translate(tmp_path, ANONYMOUS_FRAGMENT, "anon.frag", "opengl")

    assert glsl.count("vec2 uv;") == 1
    assert re.search(r"layout\(location = 2\) in VertexData \{", glsl)
    assert "layout(location = 9) flat uint texture_index;" in glsl
    assert not re.search(r"\bin vec3 tail\b", glsl)
    _glslang(tmp_path, glsl, "frag")


@pytest.mark.parametrize(
    "source, name, stage, instance",
    [
        (INSTANCE_VERTEX, "inst.vert", "vert", "vertex_out"),
        (INSTANCE_FRAGMENT, "inst.frag", "frag", "vertex_in"),
    ],
)
def test_glsl_declares_instance_block_once(tmp_path, source, name, stage, instance):
    glsl = _translate(tmp_path, source, name, "opengl")

    assert len(re.findall(r"\bVertexData \{", glsl)) == 1
    assert re.search(rf"\}}\s*{instance};", glsl)
    assert not re.search(rf"\b(?:in|out) VertexData {instance};", glsl)
    # The block spans locations 1-4, so the plain varying keeps location 6.
    assert re.search(r"layout\(location = 6\) (?:in|out) vec3 after;", glsl)
    _glslang(tmp_path, glsl, stage)


def test_hlsl_struct_varyings_number_their_fields_from_the_block_location(tmp_path):
    vertex = _translate(tmp_path, INSTANCE_VERTEX, "inst.vert", "directx")
    fragment = _translate(tmp_path, INSTANCE_FRAGMENT, "inst.frag", "directx")

    for code, member in ((vertex, "vertex_out"), (fragment, "vertex_in")):
        assert re.search(rf"VertexData {member}\s*:\s*TEXCOORD1;", code)
        assert re.search(r"float3 after\s*:\s*TEXCOORD6;", code)
    _dxc(tmp_path, vertex, "vs_6_0", "VSMain")
    _dxc(tmp_path, fragment, "ps_6_0", "PSMain")


def test_hlsl_refuses_struct_varying_whose_fields_have_locations(tmp_path):
    with pytest.raises(DirectXLoweringRefused, match="declare their own locations"):
        _translate(tmp_path, LOCATED_FIELD_VERTEX, "located.vert", "directx")
