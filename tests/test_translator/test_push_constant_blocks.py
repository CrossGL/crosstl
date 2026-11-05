"""Push-constant blocks in GLSL and Slang output.

A CrossGL ``cbuffer`` or ``ConstantBuffer<T>`` marked ``@push_constant`` is read from the constants a command buffer
pushes, not from a descriptor. GLSL and Slang output declared it as an ordinary uniform buffer at binding 0, which
the host never binds for push constants. GLSL output now declares a ``layout(push_constant)`` block when compiled
for Vulkan and keeps the OpenGL uniform block otherwise; Slang output declares ``[[vk::push_constant]]``.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.push_constant_blocks import PushConstantBlockError

BODY = """
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = fin[i] * %s;
        }
    }
}
"""

CBUFFER = """shader Probe {
    cbuffer Constants @push_constant {
        float scale;
        float bias;
    };
""" + BODY % "scale + bias"

OFFSET_CBUFFER = """shader Probe {
    cbuffer Constants @push_constant @buffer_layout(std430) {
        float scale @offset(16);
        float bias;
    };
""" + BODY % "scale + bias"

LAYOUT_CBUFFER = """shader Probe {
    cbuffer Constants @push_constant @buffer_layout(%s) {
        vec3 direction;
        float scale;
    };
""" + BODY % "scale + direction.x"

CONSTANT_BUFFER = """shader Probe {
    struct Constants {
        float scale;
        float bias;
    };
    ConstantBuffer<Constants> pc @push_constant;
""" + BODY % "pc.scale + pc.bias"

UNIFORM_STRUCT = CONSTANT_BUFFER.replace(
    "ConstantBuffer<Constants> pc @push_constant;",
    "uniform Constants pc @push_constant;",
)

TWO_BLOCKS = """shader Probe {
    cbuffer First @push_constant {
        float scale;
    };
    cbuffer Second @push_constant {
        float bias;
    };
""" + BODY % "scale + bias"

STRUCT_OFFSET = CONSTANT_BUFFER.replace("float scale;", "float scale @offset(16);")
ARRAY_OF_BLOCKS = CONSTANT_BUFFER.replace(
    "ConstantBuffer<Constants> pc @push_constant;",
    "ConstantBuffer<Constants> pc[2] @push_constant;",
).replace("pc.scale + pc.bias", "pc[0].scale + pc[1].bias")


def _translate(tmp_path, source, backend, name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _vulkan_spirv(tmp_path, glsl):
    """The SPIR-V assembly glslang writes for ``glsl`` compiled for Vulkan, or ``None`` without the tools."""
    glslang, disassembler = shutil.which("glslangValidator"), shutil.which("spirv-dis")
    if glslang is None or disassembler is None:
        return None
    source, binary = tmp_path / "out.comp", tmp_path / "out.spv"
    source.write_text(glsl)
    result = subprocess.run(
        [glslang, "-V", "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return subprocess.run(
        [disassembler, str(binary)], capture_output=True, text=True, check=True
    ).stdout


def _check_opengl(tmp_path, glsl):
    """glslang accepts ``glsl`` as OpenGL GLSL, where ``VULKAN`` is not defined."""
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        return
    source = tmp_path / "opengl.comp"
    source.write_text(glsl)
    result = subprocess.run([glslang, str(source)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _slang_spirv(tmp_path, code):
    slangc = shutil.which("slangc")
    if slangc is None:
        return None
    source, output = tmp_path / "out.slang", tmp_path / "out.spvasm"
    source.write_text(code)
    result = subprocess.run(
        [slangc, str(source), "-entry", "main", "-stage", "compute"]
        + ["-target", "spirv-asm", "-o", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return output.read_text()


def _member_offsets(spirv, block):
    """The ``Offset`` decorations of the members of the SPIR-V type named ``block``."""
    return [
        int(offset)
        for offset in re.findall(
            rf"OpMemberDecorate %{block} \d+ Offset (\d+)", spirv or ""
        )
    ]


def _assert_push_constant(spirv):
    if spirv is not None:
        assert re.search(r"OpVariable %\w+ PushConstant", spirv), spirv


# GLSL ---------------------------------------------------------------------------------------------------------------


def test_glsl_declares_a_cbuffer_as_a_vulkan_push_constant_block(tmp_path):
    glsl = _translate(tmp_path, CBUFFER, "opengl")
    vulkan, opengl = glsl.split("#else\n", 1)
    assert "#ifdef VULKAN\nlayout(push_constant) uniform Constants {" in vulkan
    assert "binding" not in vulkan.split("#ifdef VULKAN", 1)[1]
    assert "layout(std140, binding = 0) uniform Constants {" in opengl
    spirv = _vulkan_spirv(tmp_path, glsl)
    _assert_push_constant(spirv)
    if spirv is not None:
        assert _member_offsets(spirv, "Constants") == [0, 4]
        variable = re.search(r"(%\w+) = OpVariable %\w+ PushConstant", spirv).group(1)
        assert not re.search(rf"OpDecorate {variable} (DescriptorSet|Binding)", spirv)
    _check_opengl(tmp_path, glsl)


def test_glsl_push_constant_member_keeps_its_offset(tmp_path):
    glsl = _translate(tmp_path, OFFSET_CBUFFER, "opengl")
    vulkan, opengl = glsl.split("#else\n", 1)
    assert "layout(offset = 16) float scale;" in vulkan
    # OpenGL keeps the std430 offsets through its register loader, which does nothing for Vulkan.
    assert "void crossgl_load_Constants() {}" in vulkan
    assert "crossgl_cbuffer_registers" in opengl
    assert "crossgl_load_Constants();" in glsl.rsplit("#endif", 1)[1]
    spirv = _vulkan_spirv(tmp_path, glsl)
    _assert_push_constant(spirv)
    if spirv is not None:
        assert _member_offsets(spirv, "Constants") == [16, 20]
    _check_opengl(tmp_path, glsl)


@pytest.mark.parametrize(
    "layout, qualifier, offsets",
    [
        ("std430", "layout(push_constant)", [0, 12]),
        ("std140", "layout(push_constant, std140)", [0, 12]),
        ("scalar", "layout(push_constant, scalar)", [0, 12]),
    ],
)
def test_glsl_push_constant_block_keeps_its_layout(
    tmp_path, layout, qualifier, offsets
):
    glsl = _translate(tmp_path, LAYOUT_CBUFFER % layout, "opengl")
    vulkan = glsl.split("#else\n", 1)[0]
    assert f"{qualifier} uniform Constants {{" in vulkan
    if layout == "scalar":
        assert "#extension GL_EXT_scalar_block_layout : require" in vulkan
    spirv = _vulkan_spirv(tmp_path, glsl)
    _assert_push_constant(spirv)
    if spirv is not None:
        assert _member_offsets(spirv, "Constants") == offsets
    _check_opengl(tmp_path, glsl)


@pytest.mark.parametrize("source", [CONSTANT_BUFFER, UNIFORM_STRUCT])
def test_glsl_declares_a_push_constant_variable_as_a_named_block(tmp_path, source):
    glsl = _translate(tmp_path, source, "opengl")
    vulkan, opengl = glsl.split("#else\n", 1)
    assert re.search(
        r"layout\(push_constant\) uniform Constants \{[^}]*\}\s*pc;", vulkan
    ), vulkan
    assert re.search(
        r"layout\(std140, binding = 0\) uniform Constants \{[^}]*\}\s*pc;", opengl
    ), opengl
    assert "pc.scale" in glsl
    _assert_push_constant(_vulkan_spirv(tmp_path, glsl))
    _check_opengl(tmp_path, glsl)


def test_glsl_push_constant_block_round_trips_through_glsl(tmp_path):
    source = """#version 450
layout(local_size_x = 1) in;
layout(push_constant) uniform Constants {
    layout(offset = 16) float scale;
    float bias;
} pc;
layout(std430, binding = 0) buffer Data { float values[]; };
void main() { values[gl_GlobalInvocationID.x] *= pc.scale + pc.bias; }
"""
    glsl = _translate(tmp_path, source, "opengl", "shader.comp")
    assert "layout(push_constant) uniform Constants {" in glsl
    assert "layout(offset = 16) float scale;" in glsl
    spirv = _vulkan_spirv(tmp_path, glsl)
    _assert_push_constant(spirv)
    if spirv is not None:
        assert _member_offsets(spirv, "Constants") == [16, 20]


@pytest.mark.parametrize(
    "source, message",
    [
        (TWO_BLOCKS, "already declares push-constant block 'First'"),
        (STRUCT_OFFSET, "member 'scale' of struct 'Constants' has an explicit offset"),
        (ARRAY_OF_BLOCKS, "it is an array"),
        (
            LAYOUT_CBUFFER % "hlsl_cbuffer",
            "push constants are laid out with the std430, std140 or scalar rules",
        ),
    ],
)
def test_glsl_refuses_push_constants_it_cannot_declare(tmp_path, source, message):
    with pytest.raises(PushConstantBlockError) as raised:
        _translate(tmp_path, source, "opengl")
    assert (
        raised.value.code == "project.translate.opengl-push-constant-block-unsupported"
    )
    assert message in str(raised.value)


# Slang --------------------------------------------------------------------------------------------------------------


def test_slang_declares_a_cbuffer_as_a_push_constant_block(tmp_path):
    slang = _translate(tmp_path, OFFSET_CBUFFER, "slang")
    assert "[[vk::push_constant]] cbuffer Constants" in slang
    assert "[[vk::offset(16)]] float scale;" in slang
    declaration = slang.split("cbuffer Constants", 1)[0].rsplit("\n", 1)[-1]
    assert "vk::binding" not in declaration
    spirv = _slang_spirv(tmp_path, slang)
    _assert_push_constant(spirv)
    if spirv is not None:
        assert _member_offsets(spirv, "SLANG_ParameterGroup_Constants_std430") == [
            16,
            20,
        ]


@pytest.mark.parametrize("source", [CONSTANT_BUFFER, UNIFORM_STRUCT])
def test_slang_declares_a_push_constant_variable(tmp_path, source):
    slang = _translate(tmp_path, source, "slang")
    assert re.search(r"\[\[vk::push_constant\]\] \w[\w<> ]* pc;", slang), slang
    assert "pc : register" not in slang
    _assert_push_constant(_slang_spirv(tmp_path, slang))


def test_slang_push_constant_block_round_trips_through_slang(tmp_path):
    source = """[[vk::push_constant]] cbuffer Constants
{
    float scale;
    [[vk::offset(16)]] float bias;
};
RWStructuredBuffer<float> outp;
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) { outp[id.x] = outp[id.x] * scale + bias; }
"""
    crossgl = _translate(tmp_path, source, "cgl", "shader.slang")
    assert re.search(r"cbuffer Constants @push_constant \{", crossgl), crossgl
    assert "float bias @offset(16);" in crossgl
    slang = _translate(tmp_path, source, "slang", "shader.slang")
    assert "[[vk::offset(16)]] float bias;" in slang
    spirv = _slang_spirv(tmp_path, slang)
    _assert_push_constant(spirv)
    if spirv is not None:
        assert _member_offsets(spirv, "SLANG_ParameterGroup_Constants_std430") == [
            0,
            16,
        ]


@pytest.mark.parametrize("layout", ["std140", "scalar"])
def test_slang_refuses_a_push_constant_block_with_another_layout(tmp_path, layout):
    with pytest.raises(PushConstantBlockError) as raised:
        _translate(tmp_path, LAYOUT_CBUFFER % layout, "slang")
    assert (
        raised.value.code == "project.translate.slang-push-constant-block-unsupported"
    )
    assert "std430 rules only" in str(raised.value)
