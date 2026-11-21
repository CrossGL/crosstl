"""GLSL output gives each Vulkan descriptor type its own set 0 binding.

OpenGL numbers uniform blocks, storage blocks, textures and images separately, so GLSL output put several of them at
``binding = 0``, and a Vulkan compilation (glslang ``-V`` defines ``VULKAN``) bound two descriptor types at one set 0
binding, which no pipeline layout can provide. See :mod:`crosstl.translator.codegen.glsl_vulkan_binding_classes`.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.glsl_vulkan_binding_classes import (
    vulkan_descriptor_type,
    with_distinct_vulkan_bindings,
)

UBO_AND_BUFFERS = """shader Probe {
    cbuffer Params {
        float scale;
    };
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = fin[i] * scale;
        }
    }
}
"""

SEPARATE_TEXTURE = """shader Probe {
    uniform texture2D albedo;
    uniform sampler point;
    cbuffer Params { float scale; };
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer float* outp;
        layout(rgba32f) uniform image2D img;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            vec4 c = textureLod(albedo, point, vec2(fin[i], 0.5), 0.0);
            imageStore(img, ivec2(0, 0), c);
            outp[i] = c.x * scale;
        }
    }
}
"""

EXPLICIT_BLOCK = """shader Probe {
    cbuffer Globals @binding(0) {
        float bias;
    };
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = fin[i] + bias;
        }
    }
}
"""

HLSL_REGISTERS = """cbuffer Params : register(b0) { float scale; };
StructuredBuffer<float> fin : register(t0);
RWStructuredBuffer<float> outp : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    outp[id.x] = fin[id.x] * scale;
}
"""

VULKAN_GLSL = """#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) uniform Params { float scale; } params;
layout(set = 0, binding = 1) buffer In { float v[]; } fin;
layout(set = 0, binding = 2) buffer Out { float v[]; } outp;
void main() {
    outp.v[gl_GlobalInvocationID.x] = fin.v[gl_GlobalInvocationID.x] * params.scale;
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


def vulkan_bindings(tmp_path, glsl):
    """``{(set, binding): [storage class of each variable]}`` of a Vulkan compilation of ``glsl``."""
    glslang, spirv_dis = _tools()
    source = tmp_path / "out.comp"
    source.write_text(glsl)
    # Vulkan 1.1 (SPIR-V 1.3) gives storage blocks the StorageBuffer storage class.
    _run(
        [glslang, "-V", "--target-env", "vulkan1.1", str(source)]
        + ["-o", str(tmp_path / "out.spv")]
    )
    if shutil.which("spirv-val"):
        _run(["spirv-val", "--target-env", "vulkan1.1", str(tmp_path / "out.spv")])
    assembly = _run([spirv_dis, "--raw-id", str(tmp_path / "out.spv")])
    decorations = {}
    for target, kind, value in re.findall(
        r"OpDecorate (%\w+) (DescriptorSet|Binding) (\d+)", assembly
    ):
        decorations.setdefault(target, {})[kind] = int(value)
    pointers = dict(
        re.findall(r"(%\w+) = OpTypePointer (\w+) %\w+", assembly, flags=re.MULTILINE)
    )
    variables = dict(
        re.findall(r"(%\w+) = OpVariable (%\w+) \w+", assembly, flags=re.MULTILINE)
    )
    found = {}
    for target, decoration in decorations.items():
        key = (decoration.get("DescriptorSet", 0), decoration["Binding"])
        found.setdefault(key, []).append(pointers[variables[target]])
    return found


def opengl_bindings(tmp_path, glsl):
    """``{name: binding}`` of the blocks and uniforms an OpenGL compilation of ``glsl`` reflects."""
    glslang, _ = _tools()
    source = tmp_path / "out.comp"
    source.write_text(glsl)
    reflection = _run([glslang, "-l", "-q", str(source)])
    return {
        name: int(binding)
        for name, binding in re.findall(
            r"^(\w+): .*?binding (-?\d+)", reflection, flags=re.MULTILINE
        )
        if int(binding) >= 0
    }


def test_a_uniform_block_leaves_the_storage_buffers_binding(tmp_path):
    glsl = translate(tmp_path, UBO_AND_BUFFERS, "probe.cgl")
    assert "layout(std430, binding = 0) buffer finBuffer" in glsl
    assert "layout(std430, binding = 1) buffer outpBuffer" in glsl
    assert "layout(std140, CROSSGL_BINDING(0, 2, 0)) uniform Params" in glsl
    assert (
        "#ifdef VULKAN\n#define CROSSGL_BINDING(s, vulkan, opengl) set = s, binding = vulkan\n"
        "#else\n#define CROSSGL_BINDING(s, vulkan, opengl) binding = opengl\n#endif\n"
    ) in glsl
    assert vulkan_bindings(tmp_path, glsl) == {
        (0, 0): ["StorageBuffer"],
        (0, 1): ["StorageBuffer"],
        (0, 2): ["Uniform"],
    }
    found = opengl_bindings(tmp_path, glsl)
    assert (found["Params"], found["finBuffer"], found["outpBuffer"]) == (0, 0, 1)


def test_textures_and_images_leave_the_buffers_bindings(tmp_path):
    glsl = translate(tmp_path, SEPARATE_TEXTURE, "probe.cgl")
    # The Vulkan-only texture declaration takes its binding directly; the OpenGL one keeps texture unit 0.
    assert (
        "#ifdef VULKAN\nlayout(binding = 3) uniform texture2D albedo;\n#else\n"
        "layout(binding = 0) uniform sampler2D albedo;\n#endif\n"
    ) in glsl
    assert "layout(rgba32f, CROSSGL_BINDING(0, 4, 0)) uniform image2D img;" in glsl
    assert "layout(std140, CROSSGL_BINDING(0, 5, 0)) uniform Params" in glsl
    found = vulkan_bindings(tmp_path, glsl)
    assert sorted(found) == [(0, binding) for binding in range(6)]
    assert all(len(classes) == 1 for classes in found.values())


def test_a_binding_the_source_wrote_keeps_its_number(tmp_path):
    glsl = translate(tmp_path, EXPLICIT_BLOCK, "probe.cgl")
    assert "layout(std140, binding = 0) uniform Globals" in glsl
    assert "layout(std430, CROSSGL_BINDING(0, 2, 0)) buffer finBuffer" in glsl
    assert vulkan_bindings(tmp_path, glsl) == {
        (0, 0): ["Uniform"],
        (0, 1): ["StorageBuffer"],
        (0, 2): ["StorageBuffer"],
    }


def test_hlsl_registers_of_two_classes_get_two_vulkan_bindings(tmp_path):
    glsl = translate(tmp_path, HLSL_REGISTERS, "probe.hlsl")
    found = vulkan_bindings(tmp_path, glsl)
    assert all(len(classes) == 1 for classes in found.values())
    storage = sorted(
        key for key, classes in found.items() if classes == ["StorageBuffer"]
    )
    assert storage == [(0, 0), (0, 1)]
    assert opengl_bindings(tmp_path, glsl)["Params"] == 0


def test_distinct_vulkan_bindings_are_unchanged(tmp_path):
    glsl = translate(tmp_path, VULKAN_GLSL, "probe.comp")
    assert "CROSSGL_BINDING" not in glsl
    assert vulkan_bindings(tmp_path, glsl) == {
        (0, 0): ["Uniform"],
        (0, 1): ["StorageBuffer"],
        (0, 2): ["StorageBuffer"],
    }


def test_resource_binding_records_keep_their_fields(tmp_path):
    path = tmp_path / "probe.cgl"
    path.write_text(EXPLICIT_BLOCK)
    codegen = GLSLCodeGen()
    codegen.generate(crosstl.translator.parse(EXPLICIT_BLOCK))
    records = codegen.resource_binding_records()
    assert {record["glslName"]: record["binding"] for record in records} == {
        "fin": 0,
        "outp": 1,
        "Globals": 0,
    }
    assert all(not key.startswith("_") for record in records for key in record)


def test_views_of_one_resource_move_together():
    code = (
        "#version 450\n"
        "layout(std430, binding = 0) buffer A { float a[]; };\n"
        "layout(r32f, binding = 0) uniform image2D view;\n"
        "layout(r32ui, binding = 0) uniform uimage2D bits;\n"
        "layout(binding = 0) uniform sampler2D tex;\n"
        "void main() {}\n"
    )
    moved = with_distinct_vulkan_bindings(code)
    assert "layout(r32f, CROSSGL_BINDING(0, 1, 0)) uniform image2D view;" in moved
    assert "layout(r32ui, CROSSGL_BINDING(0, 1, 0)) uniform uimage2D bits;" in moved
    assert "layout(CROSSGL_BINDING(0, 2, 0)) uniform sampler2D tex;" in moved
    # The definitions precede the first declaration, after the version line.
    assert moved.startswith("#version 450\n#ifdef VULKAN\n#define CROSSGL_BINDING(")


def test_the_definition_precedes_a_conditional_first_declaration():
    code = (
        "#version 450\n"
        "#extension GL_EXT_nonuniform_qualifier : require\n"
        "#ifdef VULKAN\n"
        "layout(binding = 0) uniform texture2D t;\n"
        "#else\n"
        "layout(binding = 0) uniform sampler2D t;\n"
        "#endif\n"
        "layout(std430, binding = 0) buffer A { float a[]; };\n"
        "layout(std140, binding = 0) uniform P { float p; };\n"
    )
    moved = with_distinct_vulkan_bindings(code)
    lines = moved.splitlines()
    definition = lines.index(
        "#define CROSSGL_BINDING(s, vulkan, opengl) set = s, binding = vulkan"
    )
    assert lines[definition - 1] == "#ifdef VULKAN"
    assert lines[definition - 2] == "#extension GL_EXT_nonuniform_qualifier : require"
    assert "#ifdef VULKAN\nlayout(binding = 1) uniform texture2D t;\n#else\n" in moved
    assert "#else\nlayout(binding = 0) uniform sampler2D t;\n#endif\n" in moved
    assert "layout(std140, CROSSGL_BINDING(0, 2, 0)) uniform P" in moved


def test_opengl_only_declarations_and_other_sets_are_ignored():
    code = (
        "#version 450\n"
        "#ifndef VULKAN\n"
        "layout(binding = 0) uniform sampler2D combined;\n"
        "#endif\n"
        "layout(std430, binding = 0) buffer A { float a[]; };\n"
        "layout(std140, CROSSGL_BINDING(1, 0, 1)) uniform P { float p; };\n"
        "layout(binding = 0, offset = 0) uniform atomic_uint counter;\n"
    )
    assert with_distinct_vulkan_bindings(code) == code


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        ("buffer Data { float v[]; };", "storage-buffer"),
        ("readonly buffer Data", "storage-buffer"),
        ("uniform Params { float s; };", "uniform-buffer"),
        ("uniform Params", "uniform-buffer"),
        ("uniform highp sampler2D tex;", "combined-image-sampler"),
        ("uniform usamplerBuffer texels;", "uniform-texel-buffer"),
        ("uniform texture2D tex[];", "sampled-image"),
        ("uniform sampler point;", "sampler"),
        ("uniform samplerShadow cmp;", "sampler"),
        ("uniform coherent uimage2D img;", "storage-image"),
        ("uniform imageBuffer texels;", "storage-texel-buffer"),
        ("uniform accelerationStructureEXT scene;", "acceleration-structure"),
        ("uniform subpassInput color;", "input-attachment"),
        ("uniform atomic_uint counter;", None),
        ("in vec2 uv;", None),
    ],
)
def test_vulkan_descriptor_type(declaration, expected):
    assert vulkan_descriptor_type(declaration) == expected
