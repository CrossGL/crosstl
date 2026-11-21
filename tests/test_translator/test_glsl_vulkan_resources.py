"""GLSL output keeps separate samplers and runtime texture arrays for a Vulkan compilation.

OpenGL GLSL has only combined image samplers and sized texture arrays, so GLSL output combined each separate
texture with its samplers and sized each unsized texture array, and a Vulkan compilation of the output (glslang
``-V`` defines ``VULKAN``) lost the source's ``OpTypeSampler`` and ``RuntimeDescriptorArray``. Such resources are
now declared once per API, and sampling calls pick their operand through ``CROSSGL_SAMPLED``. See
:mod:`crosstl.translator.codegen.glsl_vulkan_resources`.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.ast import IdentifierNode
from crosstl.translator.codegen.glsl_vulkan_resources import VulkanResourceForms

SHARED = """shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer float* outp;
        uniform texture2D texa;
        uniform texture2D texb;
        uniform sampler smp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = textureLod(texa, smp, vec2(fin[i], 0.5), 0.0).x
                + textureLod(texb, smp, vec2(fin[i], 0.5), 0.0).y;
        }
    }
}
"""

# One texture read through two samplers: OpenGL needs a second combined uniform, Vulkan a second sampler.
SPLIT = """shader Probe {
    uniform texture2D albedo;
    uniform sampler point;
    uniform sampler linear;
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = texture(albedo, point, vec2(fin[i], 0.5)).x
                + textureLod(albedo, linear, vec2(fin[i], 0.5), 0.0).y;
        }
    }
}
"""

PLACED = """shader Probe {
    uniform texture2D texa @binding(3);
    uniform sampler smp @set(1) @binding(5);
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        void main() {
            outp[0] = textureLod(texa, smp, vec2(0.5, 0.5), 0.0).x;
        }
    }
}
"""

# Vulkan GLSL needs GL_EXT_samplerless_texture_functions for textureSize of a texture2D, so both compilations keep
# the combined form.
SIZE_QUERY = """shader Probe {
    uniform texture2D texa;
    uniform sampler smp;
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        void main() {
            outp[0] = textureLod(texa, smp, vec2(0.5, 0.5), 0.0).x + float(textureSize(texa, 0).x);
        }
    }
}
"""

HELPER = """shader Probe {
    uniform texture2D texa;
    uniform sampler smp;
    float fetch(texture2D t, sampler s) {
        return textureLod(t, s, vec2(0.5, 0.5), 0.0).x;
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        void main() {
            outp[0] = fetch(texa, smp);
        }
    }
}
"""

UNSIZED = """shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer float* outp;
        uniform sampler2D tex[];
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int k = int(fin[0]) & 1;
            outp[i] = textureLod(tex[k], vec2(fin[i], 0.5), 0.0).x;
        }
    }
}
"""

FRAGMENT = """shader Web {
    uniform texture2D texa;
    uniform sampler smp;
    fragment {
        vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
            return texture(texa, smp, uv);
        }
    }
}
"""

VULKAN_GLSL = """#version 460
#extension GL_EXT_nonuniform_qualifier : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(std430, binding = 0) buffer B0 { float fin[]; };
layout(std430, binding = 1) buffer B1 { float outp[]; };
layout(binding = 3) uniform texture2D texa;
layout(binding = 4) uniform texture2D texb;
layout(binding = 5) uniform sampler smp;
layout(binding = 6) uniform sampler2D tex[];
void main() {
    uint i = gl_GlobalInvocationID.x;
    int k = int(fin[0]) & 1;
    outp[i] = textureLod(sampler2D(texa, smp), vec2(fin[i], 0.5), 0.0).x
        + textureLod(sampler2D(texb, smp), vec2(fin[i], 0.5), 0.0).y
        + textureLod(tex[k], vec2(fin[i], 0.5), 0.0).z;
}
"""


def translate(tmp_path, source, name, backend="opengl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _glslang():
    glslang, spirv_dis = shutil.which("glslangValidator"), shutil.which("spirv-dis")
    if not (glslang and spirv_dis):
        pytest.skip("glslangValidator and spirv-dis are required")
    return glslang, spirv_dis


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def compile_both(tmp_path, glsl):
    """The SPIR-V assembly of a Vulkan compilation of ``glsl``, after checking that OpenGL compiles it as well."""
    glslang, spirv_dis = _glslang()
    source = tmp_path / "out.comp"
    source.write_text(glsl)
    _run([glslang, "-S", "comp", str(source)])
    _run([glslang, "-G", "-S", "comp", str(source), "-o", str(tmp_path / "gl.spv")])
    _run([glslang, "-V", "-S", "comp", str(source), "-o", str(tmp_path / "vk.spv")])
    return _run([spirv_dis, str(tmp_path / "vk.spv")])


def opengl_text(glsl):
    """``glsl`` as an OpenGL compilation preprocesses it: the ``#else`` branch of each ``#ifdef VULKAN``."""
    kept, stack = [], []
    for line in glsl.splitlines():
        stripped = line.strip()
        if stripped in ("#ifdef VULKAN", "#ifndef VULKAN"):
            stack.append(stripped == "#ifndef VULKAN")
            continue
        if stack and stripped == "#else":
            stack[-1] = not stack[-1]
            continue
        if stack and stripped == "#endif":
            stack.pop()
            continue
        if all(stack):
            kept.append(line)
    return re.sub(r"CROSSGL_SAMPLED\(\w+\(\w+, \w+\), (\w+)\)", r"\1", "\n".join(kept))


def test_shared_sampler_stays_separate_for_vulkan(tmp_path):
    glsl = translate(tmp_path, SHARED, "shared.cgl")
    assert "#ifdef VULKAN\n#define CROSSGL_SAMPLED(vulkan, opengl) vulkan\n" in glsl
    assert "#else\n#define CROSSGL_SAMPLED(vulkan, opengl) opengl\n#endif\n" in glsl
    assert (
        "#ifdef VULKAN\nlayout(binding = 0) uniform texture2D texa;\n#else\n"
        "layout(binding = 0) uniform sampler2D texa;\n#endif\n"
    ) in glsl
    # The buffers take bindings 0 and 1 and the textures 0 and 1, so the sampler takes the lowest free one.
    assert "#ifdef VULKAN\nlayout(binding = 2) uniform sampler smp;\n#endif\n" in glsl
    assert "textureLod(CROSSGL_SAMPLED(sampler2D(texa, smp), texa)," in glsl
    assert "textureLod(CROSSGL_SAMPLED(sampler2D(texb, smp), texb)," in glsl
    assembly = compile_both(tmp_path, glsl)
    assert re.search(r"= OpTypeSampler$", assembly, re.M)
    assert assembly.count("OpSampledImage") == 2


def test_the_opengl_branch_is_the_combined_lowering(tmp_path):
    glsl = opengl_text(translate(tmp_path, SPLIT, "split.cgl"))
    assert "layout(binding = 0) uniform sampler2D albedo;" in glsl
    assert "layout(binding = 1) uniform sampler2D albedo_linear;" in glsl
    assert "uniform sampler " not in glsl
    assert "texture(albedo, vec2(fin[i], 0.5))" in glsl
    assert "textureLod(albedo_linear, vec2(fin[i], 0.5), 0.0)" in glsl


def test_each_sampler_pair_reads_the_one_texture_for_vulkan(tmp_path):
    glsl = translate(tmp_path, SPLIT, "split.cgl")
    assert (
        "#ifndef VULKAN\nlayout(binding = 1) uniform sampler2D albedo_linear;\n#endif\n"
        in glsl
    )
    assert (
        "#ifdef VULKAN\nlayout(binding = 2) uniform sampler point;\n"
        "layout(binding = 3) uniform sampler linear;\n#endif\n"
    ) in glsl
    assert "texture(CROSSGL_SAMPLED(sampler2D(albedo, point), albedo)," in glsl
    assert (
        "textureLod(CROSSGL_SAMPLED(sampler2D(albedo, linear), albedo_linear)," in glsl
    )
    assembly = compile_both(tmp_path, glsl)
    assert len(re.findall(r"OpTypePointer UniformConstant %\w+", assembly)) >= 2
    assert len(re.findall(r"OpDecorate %\w+ Binding", assembly)) == 5


def test_a_sampler_keeps_its_set_and_binding(tmp_path):
    glsl = translate(tmp_path, PLACED, "placed.cgl")
    assert (
        "#ifdef VULKAN\nlayout(set = 1, binding = 5) uniform sampler smp;\n#endif\n"
        in glsl
    )
    assert "#ifdef VULKAN\nlayout(binding = 3) uniform texture2D texa;\n" in glsl
    assembly = compile_both(tmp_path, glsl)
    sampler = re.search(r"^\s*(%\w+) = OpTypeSampler$", assembly, re.M).group(1)
    variable = re.search(
        rf"^\s*%\w+ = OpTypePointer UniformConstant {sampler}$", assembly, re.M
    )
    assert variable is not None
    assert "DescriptorSet 1" in assembly


@pytest.mark.parametrize("source", [SIZE_QUERY, HELPER], ids=["size-query", "helper"])
def test_other_uses_keep_the_combined_form(tmp_path, source):
    glsl = translate(tmp_path, source, "other.cgl")
    assert "VULKAN" not in glsl
    assert "uniform sampler2D texa;" in glsl
    compile_both(tmp_path, glsl)


def test_unsized_texture_array_is_a_runtime_array_for_vulkan(tmp_path):
    glsl = translate(tmp_path, UNSIZED, "unsized.cgl")
    assert (
        "#ifdef VULKAN\n#extension GL_EXT_nonuniform_qualifier : require\n#endif\n"
        in glsl
    )
    assert (
        "#ifdef VULKAN\nlayout(binding = 0) uniform sampler2D tex[];\n#else\n"
        "layout(binding = 0) uniform sampler2D tex[2];\n#endif\n"
    ) in glsl
    assembly = compile_both(tmp_path, glsl)
    assert "OpCapability RuntimeDescriptorArray" in assembly
    assert re.search(r"= OpTypeRuntimeArray %\w+$", assembly, re.M)


def test_an_array_used_without_an_index_stays_sized(tmp_path):
    source = UNSIZED.replace(
        "outp[i] = textureLod(tex[k], vec2(fin[i], 0.5), 0.0).x;",
        "outp[i] = textureLod(tex[k], vec2(fin[i], 0.5), 0.0).x + float(tex.length());",
    )
    glsl = translate(tmp_path, source, "length.cgl")
    assert "VULKAN" not in glsl
    assert "uniform sampler2D tex[2];" in glsl


def test_vulkan_glsl_round_trips_its_separate_resources(tmp_path):
    glsl = translate(tmp_path, VULKAN_GLSL, "probe.comp")
    assert "#ifdef VULKAN\nlayout(binding = 3) uniform texture2D texa;\n" in glsl
    assert "#ifdef VULKAN\nlayout(binding = 5) uniform sampler smp;\n#endif\n" in glsl
    assert "#ifdef VULKAN\nlayout(binding = 6) uniform sampler2D tex[];\n" in glsl
    assembly = compile_both(tmp_path, glsl)
    assert re.search(r"= OpTypeSampler$", assembly, re.M)
    assert "OpCapability RuntimeDescriptorArray" in assembly
    bindings = sorted(
        int(value) for value in re.findall(r"OpDecorate %\w+ Binding (\d+)", assembly)
    )
    assert bindings == [0, 1, 3, 4, 5, 6]


def test_webgl_and_versions_without_bindings_keep_one_form(tmp_path):
    assert "VULKAN" not in translate(tmp_path, FRAGMENT, "web.cgl", backend="webgl")
    # GLSL before 4.20 has no layout(binding), so a Vulkan compilation could not place the descriptors.
    glsl = translate(tmp_path, "#version 330 core\n" + FRAGMENT, "old.cgl")
    assert "layout(binding" not in glsl
    assert "VULKAN" not in glsl
    assert "uniform sampler2D texa;" in glsl


def test_an_unwrapped_use_is_reported():
    forms = VulkanResourceForms(
        textures={
            "texa": {
                "type": "texture2D",
                "sampled": "sampler2D",
                "uniforms": {"texa": "smp"},
            }
        },
        arrays={"tex"},
    )
    forms.active = True
    code = forms.declaration(
        "texa", "texa", "layout(binding = 0) uniform", "sampler2D texa", "sampler2D", ""
    )
    code += forms.declaration(
        "tex",
        "tex",
        "layout(binding = 1) uniform",
        "sampler2D tex[2]",
        "sampler2D",
        "[2]",
    )
    operand = forms.sampled(
        [IdentifierNode("texa"), IdentifierNode("smp")], "texa", "smp"
    )
    assert operand == "CROSSGL_SAMPLED(sampler2D(texa, smp), texa)"
    code += f"textureLod({operand}, uv, 0.0) + texture(tex[1], uv)"
    assert forms.unwrapped(code) == set()
    assert forms.unwrapped(code + " + textureSize(texa, 0).x") == {"texa"}
    assert forms.unwrapped(code + " + tex.length()") == {"tex"}
