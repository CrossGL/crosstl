"""GLSL output keeps a resource's Vulkan descriptor set next to its OpenGL binding point.

OpenGL GLSL has no ``set`` qualifier, so GLSL output wrote only the OpenGL binding point it plans for a resource of
another set (:mod:`crosstl.translator.codegen.glsl_descriptor_sets`), and a Vulkan compilation of the output bound
the resource in set 0. A resource whose set is not 0 now takes ``CROSSGL_BINDING(set, vulkan binding, OpenGL
binding)``, which the output defines under ``#ifdef VULKAN``. See
:mod:`crosstl.translator.codegen.glsl_vulkan_descriptor_sets`.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

CBUFFER = """shader Probe {
    cbuffer Globals @binding(0) {
        float bias;
    };
    cbuffer Params @set(1) @binding(3) {
        float scale;
    };
    cbuffer Moved @set(2) @binding(0) {
        float offset;
    };
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = fin[i] * scale + bias + offset;
        }
    }
}
"""

HLSL_SPACES = """Texture2D albedo : register(t0, space1);
SamplerState linearSampler : register(s0, space1);
cbuffer Material : register(b2, space3) { float4 tint; };
float4 main(float2 uv : TEXCOORD0) : SV_Target {
    return albedo.Sample(linearSampler, uv) * tint;
}
"""

GLSL_VULKAN = """#version 450
layout(local_size_x = 1) in;
layout(set = 2, binding = 0) buffer Data { float v[]; } data;
layout(set = 1, binding = 0) uniform sampler2D tex;
layout(set = 0, binding = 0) buffer Other { float w[]; } other;
void main() {
    data.v[gl_GlobalInvocationID.x] = textureLod(tex, vec2(0.5), 0.0).x + other.w[0];
}
"""

SET_ZERO = """shader Probe {
    cbuffer Params @set(0) @binding(3) { float scale; };
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        void main() { outp[0] = scale; }
    }
}
"""


def translate(tmp_path, source, name):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend="opengl", format_output=False)


def _tools():
    glslang, spirv_dis = shutil.which("glslangValidator"), shutil.which("spirv-dis")
    if not (glslang and spirv_dis):
        pytest.skip("glslangValidator and spirv-dis are required")
    return glslang, spirv_dis


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def vulkan_descriptors(tmp_path, glsl, stage):
    """``{variable id: (set, binding)}`` of a Vulkan compilation of ``glsl``."""
    glslang, spirv_dis = _tools()
    source = tmp_path / f"out.{stage}"
    source.write_text(glsl)
    _run([glslang, "-V", str(source), "-o", str(tmp_path / "out.spv")])
    assembly = _run([spirv_dis, str(tmp_path / "out.spv")])
    found = {}
    for target, kind, value in re.findall(
        r"OpDecorate (%\w+) (DescriptorSet|Binding) (\d+)", assembly
    ):
        entry = found.setdefault(target, [None, None])
        entry[0 if kind == "DescriptorSet" else 1] = int(value)
    return sorted(tuple(value) for value in found.values())


def opengl_bindings(tmp_path, glsl, stage):
    """``{name: binding}`` of the uniforms and blocks an OpenGL compilation of ``glsl`` reflects."""
    glslang, _ = _tools()
    source = tmp_path / f"out.{stage}"
    source.write_text(glsl)
    reflection = _run([glslang, "-l", "-q", str(source)])
    return {
        name: int(binding)
        for name, binding in re.findall(
            r"^(\w+): .*?binding (-?\d+)", reflection, flags=re.M
        )
        if int(binding) >= 0
    }


def test_constant_buffers_keep_their_set_for_vulkan(tmp_path):
    glsl = translate(tmp_path, CBUFFER, "shader.cgl")
    assert (
        "#ifdef VULKAN\n#define CROSSGL_BINDING(s, vulkan, opengl) set = s, binding = vulkan\n"
        in glsl
    )
    assert (
        "#else\n#define CROSSGL_BINDING(s, vulkan, opengl) binding = opengl\n#endif\n"
        in glsl
    )
    assert "layout(std140, binding = 0) uniform Globals" in glsl
    assert "layout(std140, CROSSGL_BINDING(1, 3, 3)) uniform Params" in glsl
    # Binding 0 of the uniform blocks is taken by Globals, so OpenGL moves Moved to the lowest free one.
    assert "layout(std140, CROSSGL_BINDING(2, 0, 1)) uniform Moved" in glsl
    assert (2, 0) in vulkan_descriptors(tmp_path, glsl, "comp")
    assert (1, 3) in vulkan_descriptors(tmp_path, glsl, "comp")
    found = opengl_bindings(tmp_path, glsl, "comp")
    assert (found["Globals"], found["Params"], found["Moved"]) == (0, 3, 1)


def test_hlsl_register_spaces_become_vulkan_sets(tmp_path):
    glsl = translate(tmp_path, HLSL_SPACES, "shader.hlsl")
    assert "CROSSGL_BINDING(1, 0, " in glsl
    assert "CROSSGL_BINDING(3, 2, " in glsl
    descriptors = vulkan_descriptors(tmp_path, glsl, "frag")
    assert (1, 0) in descriptors and (3, 2) in descriptors
    assert all(
        binding < 80 for binding in opengl_bindings(tmp_path, glsl, "frag").values()
    )


def test_glsl_round_trip_keeps_the_sets(tmp_path):
    glsl = translate(tmp_path, GLSL_VULKAN, "shader.comp")
    assert sorted(vulkan_descriptors(tmp_path, glsl, "comp")) == [
        (0, 0),
        (1, 0),
        (2, 0),
    ]
    found = opengl_bindings(tmp_path, glsl, "comp")
    assert found["otherBuffer"] == 0 and found["dataBuffer"] == 1 and found["tex"] == 0


def test_set_zero_resources_need_no_macro(tmp_path):
    glsl = translate(tmp_path, SET_ZERO, "shader.cgl")
    assert "CROSSGL_BINDING" not in glsl
    assert "layout(std140, binding = 3) uniform Params" in glsl


def test_opengl_host_reflection_reads_the_opengl_binding(tmp_path):
    from crosstl.project.host_reflection import reflect_target_host_interface

    artifact = tmp_path / "shader.glsl"
    artifact.write_text(translate(tmp_path, CBUFFER, "shader.cgl"))
    reflected = reflect_target_host_interface(
        artifact, target="opengl", stage="compute"
    )
    blocks = {
        resource["name"]: (resource["set"], resource["binding"])
        for resource in reflected["resources"]
        if resource["kind"] == "constant-buffer"
    }
    assert blocks == {"Globals": (0, 0), "Params": (0, 3), "Moved": (0, 1)}
