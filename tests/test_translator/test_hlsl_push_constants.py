"""Push-constant blocks in HLSL output.

A CrossGL ``cbuffer`` or ``ConstantBuffer<T>`` marked ``@push_constant`` holds constants a command buffer pushes,
laid out by the std430 rules unless it declares another layout. HLSL output declared it as an ordinary ``cbuffer``
at ``register(b0)``, laid out by HLSL packing, which DXC compiles for Vulkan to a uniform buffer the host never binds.
HLSL output now declares ``[[vk::push_constant]] ConstantBuffer<...>`` with every member's ``[[vk::offset(N)]]`` when
DXC compiles for SPIR-V, and the Direct3D constant buffer, at the same offsets, otherwise.
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
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = %s;
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

# Members std430 places at other bytes than HLSL packing and DXC's relaxed rules: a vec3 after a float, matrices,
# arrays (of structs as well) and a nested struct whose vec3 follows a float.
MIXED = (
    """shader Probe {
    struct Inner {
        float a;
        vec3 b;
    };
    struct Light {
        vec2 position;
        float intensity;
    };
    cbuffer Constants @push_constant {
        float s;
        vec3 v;
        mat2 m2;
        mat3 m3;
        float f3[3];
        vec3 v3[2];
        Inner inner;
        Light lights[2];
        float t @offset(240);
    };
"""
    + BODY
    % "s + v.y + (m2 * vec2(s, t)).y + (m3 * v).z + f3[1] + v3[1].z + inner.b.y + lights[1].intensity + m3[1][2]"
)

STRUCT_BLOCK = """shader Probe {
    struct Constants {
        float scale;
        vec3 dir;
        float bias;
    };
    %s
""" + BODY % "pc.scale + pc.dir.y + pc.bias"

LAYOUT_CBUFFER = """shader Probe {
    cbuffer Constants @push_constant @buffer_layout(%s) {
        %s
    };
""" + BODY % "%s"


def _translate(tmp_path, source, backend="directx", name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _dxc(tmp_path, hlsl, *flags, output="out.spv"):
    """Compile ``hlsl`` with DXC; ``None`` without it."""
    dxc = shutil.which("dxc")
    if dxc is None:
        return None
    source, binary = tmp_path / "out.hlsl", tmp_path / output
    source.write_text(hlsl)
    command = [dxc, *flags, "-T", "cs_6_0", "-E", "CSMain", "-HV", "2021"]
    result = subprocess.run(
        command + [str(source), "-Fo", str(binary)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


def _dxc_spirv(tmp_path, hlsl):
    """The SPIR-V assembly DXC writes for ``hlsl``, validated; ``None`` without the tools."""
    if shutil.which("spirv-dis") is None or shutil.which("spirv-val") is None:
        return None
    binary = _dxc(tmp_path, hlsl, "-spirv", "-fspv-target-env=vulkan1.1")
    if binary is None:
        return None
    validation = subprocess.run(
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
        capture_output=True,
        text=True,
    )
    assert validation.returncode == 0, validation.stdout + validation.stderr
    return subprocess.run(
        ["spirv-dis", str(binary)], capture_output=True, text=True, check=True
    ).stdout


def _glslang_spirv(tmp_path, glsl):
    glslang, disassembler = shutil.which("glslangValidator"), shutil.which("spirv-dis")
    if glslang is None or disassembler is None:
        return None
    source, binary = tmp_path / "out.comp", tmp_path / "glsl.spv"
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


def _member_layout(spirv, type_name):
    """``(member, decoration, value)`` of the ``Offset`` and ``MatrixStride`` decorations of a SPIR-V type."""
    return sorted(
        (int(member), decoration, int(value))
        for member, decoration, value in re.findall(
            rf"OpMemberDecorate %{type_name} (\d+) (Offset|MatrixStride) (\d+)", spirv
        )
    )


def _array_strides(spirv):
    return sorted(int(value) for value in re.findall(r"ArrayStride (\d+)", spirv))


def _push_constant_variable(spirv):
    match = re.search(r"(%\w+) = OpVariable %\w+ PushConstant", spirv)
    assert match, spirv
    assert not re.search(rf"OpDecorate {match.group(1)} (DescriptorSet|Binding)", spirv)
    return match.group(1)


def _branches(hlsl):
    spirv, rest = hlsl.split("#ifdef __spirv__\n", 1)[1].split("#else\n", 1)
    direct3d, entry = rest.split("#endif\n", 1)
    return spirv, direct3d, entry


def test_hlsl_declares_a_cbuffer_as_a_spirv_push_constant_block(tmp_path):
    hlsl = _translate(tmp_path, CBUFFER)
    spirv, direct3d, entry = _branches(hlsl)
    assert "struct __crossgl_Constants_push_constants\n{\n" in spirv
    assert (
        "    [[vk::offset(0)]] float scale;\n    [[vk::offset(4)]] float bias;\n"
        in spirv
    )
    assert (
        "[[vk::push_constant]] ConstantBuffer<__crossgl_Constants_push_constants> Constants;\n"
        in spirv
    )
    assert "static float scale;\nstatic float bias;\n" in spirv
    assert "    scale = Constants.scale;\n    bias = Constants.bias;\n" in spirv
    assert "register" not in spirv
    assert "cbuffer Constants : register(b0)" in direct3d
    assert re.search(r"void __crossgl_load_Constants\(\)\s*\{\s*\}", direct3d)
    assert "__crossgl_load_Constants();" in entry
    assembly = _dxc_spirv(tmp_path, hlsl)
    if assembly is not None:
        _push_constant_variable(assembly)
        block = "type_ConstantBuffer___crossgl_Constants_push_constants"
        assert _member_layout(assembly, block) == [(0, "Offset", 0), (1, "Offset", 4)]
    _dxc(tmp_path, hlsl, output="out.dxil")


def test_hlsl_push_constant_block_keeps_the_std430_offsets_glsl_gives(tmp_path):
    hlsl = _translate(tmp_path, MIXED)
    spirv, direct3d, _entry = _branches(hlsl)
    for declaration in (
        "[[vk::offset(16)]] float3 v;",
        "[[vk::offset(32)]] float2x2 m2;",
        "[[vk::offset(48)]] float3x3 m3;",
        "[[vk::offset(96)]] float f3[3];",
        "[[vk::offset(112)]] float3 v3[2];",
        "[[vk::offset(144)]] __crossgl_push_Inner inner;",
        "[[vk::offset(176)]] __crossgl_push_Light lights[2];",
        "[[vk::offset(240)]] float t;",
    ):
        assert declaration in spirv
    assert "struct __crossgl_push_Inner\n{\n    [[vk::offset(0)]] float a;\n" in spirv
    assert "    [[vk::offset(16)]] float3 b;\n" in spirv
    assert "    inner.b = Constants.inner.b;\n" in spirv
    assert (
        "for (int __crossgl_push_i0 = 0; __crossgl_push_i0 < 2; ++__crossgl_push_i0)"
        in spirv
    )
    assert (
        "lights[__crossgl_push_i0].intensity = Constants.lights[__crossgl_push_i0].intensity;"
        in spirv
    )
    # Direct3D reads the same std430 bytes through the block's registers.
    assert "__crossgl_Constants_registers" in direct3d
    assembly = _dxc_spirv(tmp_path, hlsl)
    reference = _glslang_spirv(
        tmp_path, _translate(tmp_path, MIXED, "opengl", "glsl.cgl")
    )
    if assembly is not None:
        _push_constant_variable(assembly)
        block = "type_ConstantBuffer___crossgl_Constants_push_constants"
        layout = _member_layout(assembly, block)
        assert [value for _member, kind, value in layout if kind == "Offset"] == [
            0,
            16,
            32,
            48,
            96,
            112,
            144,
            176,
            240,
        ]
        if reference is not None:
            assert layout == _member_layout(reference, "Constants")
            for struct in ("Inner", "Light"):
                assert _member_layout(assembly, f"__crossgl_push_{struct}") == (
                    _member_layout(reference, struct)
                )
            assert _array_strides(assembly) == _array_strides(reference)
    _dxc(tmp_path, hlsl, output="out.dxil")


@pytest.mark.parametrize(
    "declaration",
    [
        "ConstantBuffer<Constants> pc @push_constant;",
        "uniform Constants pc @push_constant;",
    ],
)
def test_hlsl_declares_a_constant_buffer_variable_as_a_push_constant_block(
    tmp_path, declaration
):
    hlsl = _translate(tmp_path, STRUCT_BLOCK % declaration)
    spirv, direct3d, _entry = _branches(hlsl)
    assert (
        "struct __crossgl_push_Constants\n{\n    [[vk::offset(0)]] float scale;\n"
        "    [[vk::offset(16)]] float3 dir;\n    [[vk::offset(28)]] float bias;\n};\n"
    ) in spirv
    assert "[[vk::offset(0)]] __crossgl_push_Constants pc;" in spirv
    assert "static Constants pc;\n" in spirv
    assert "    pc.dir = pc_block.pc.dir;\n" in spirv
    # HLSL packing would read bias from byte 28 as well, but dir must not start at byte 4.
    assert "cbuffer pc_block : register(b0)" in direct3d
    if declaration.startswith("ConstantBuffer"):
        assembly = _dxc_spirv(tmp_path, hlsl)
        if assembly is not None:
            _push_constant_variable(assembly)
            assert _member_layout(assembly, "__crossgl_push_Constants") == [
                (0, "Offset", 0),
                (1, "Offset", 16),
                (2, "Offset", 28),
            ]


@pytest.mark.parametrize(
    "layout, members, offsets",
    [
        ("std140", "float s; vec4 v;", [0, 16]),
        ("std140", "float s; vec3 v[2];", [0, 16]),
        ("scalar", "float s; vec3 v;", [0, 4]),
        ("scalar", "float s; float f[2];", [0, 4]),
    ],
)
def test_hlsl_push_constant_block_keeps_its_declared_layout(
    tmp_path, layout, members, offsets
):
    hlsl = _translate(tmp_path, LAYOUT_CBUFFER % (layout, members, "s"))
    spirv, _direct3d, _entry = _branches(hlsl)
    found = [
        int(offset) for offset in re.findall(r"\[\[vk::offset\((\d+)\)\]\]", spirv)
    ]
    assert found == offsets


@pytest.mark.parametrize(
    "layout, members, value, words, registers",
    [
        # A std140 float[] has a 16-byte stride and a std140 mat2 16-byte columns; DXC gives them 4 and 8.
        ("std140", "float s; float f[2];", "s + f[1]", 12, 3),
        ("std140", "mat2 m;", "m[1][0]", 8, 2),
        # A scalar vec3[] has a 12-byte stride where DXC uses 16, and a vec3 at byte 8 crosses 16 bytes.
        ("scalar", "vec3 v[2];", "v[1].y", 6, 2),
        ("scalar", "float a; float b; vec3 c;", "a + b + c.z", 5, 2),
    ],
)
def test_hlsl_reads_a_push_constant_block_dxc_cannot_place_as_words(
    tmp_path, layout, members, value, words, registers
):
    hlsl = _translate(tmp_path, LAYOUT_CBUFFER % (layout, members, value))
    spirv, direct3d, _entry = _branches(hlsl)
    assert f"    [[vk::offset(0)]] uint words[{words}];\n" in spirv
    assert f"static uint4 __crossgl_Constants_registers[{registers}];\n" in spirv
    assert re.search(
        r"__crossgl_Constants_registers\[\(__crossgl_push_word\) >> 2\]"
        r"\[\(__crossgl_push_word\) & 3\] =\s*Constants\.words\[__crossgl_push_word\];",
        spirv,
    )
    loader = direct3d.split("void __crossgl_load_Constants()", 1)[1]
    loader = loader.split("{", 1)[1].rsplit("}", 1)[0].strip()
    if layout == "scalar":
        # Both branches assign the members with the same register loader.
        assert loader and loader in spirv
    else:
        # A Direct3D constant-buffer array or matrix already has std140's 16-byte stride.
        assert not loader
        assert "__crossgl_Constants_registers" not in direct3d
    assembly = _dxc_spirv(tmp_path, hlsl)
    if assembly is not None:
        _push_constant_variable(assembly)
        block = "type_ConstantBuffer___crossgl_Constants_push_constants"
        assert _member_layout(assembly, block) == [(0, "Offset", 0)]
    _dxc(tmp_path, hlsl, output="out.dxil")


def test_hlsl_push_constant_words_start_at_the_first_member(tmp_path):
    hlsl = _translate(
        tmp_path,
        LAYOUT_CBUFFER % ("std140", "float f[2] @offset(32); float g;", "f[1] + g"),
    )
    spirv, _direct3d, _entry = _branches(hlsl)
    # f takes bytes 32 to 64 (a 16-byte stride) and g bytes 64 to 68: words 8 to 17.
    assert "    [[vk::offset(32)]] uint words[9];\n" in spirv
    assert "[(__crossgl_push_word + 8) >> 2][(__crossgl_push_word + 8) & 3]" in spirv
    assembly = _dxc_spirv(tmp_path, hlsl)
    if assembly is not None:
        block = "type_ConstantBuffer___crossgl_Constants_push_constants"
        assert _member_layout(assembly, block) == [(0, "Offset", 32)]


def test_hlsl_refuses_a_push_constant_layout_push_constants_do_not_have(tmp_path):
    with pytest.raises(PushConstantBlockError, match="declares the metal layout"):
        _translate(tmp_path, LAYOUT_CBUFFER % ("metal", "float a;", "a"))


@pytest.mark.parametrize(
    "source, reason",
    [
        (
            CBUFFER.replace(
                "    compute {",
                "    cbuffer Second @push_constant {\n        float other;\n    };\n    compute {",
            ),
            "already declares push-constant block 'Constants'",
        ),
        (
            STRUCT_BLOCK % "ConstantBuffer<Constants> pc[2] @push_constant;",
            "it is an array",
        ),
        (
            STRUCT_BLOCK % "uniform float pc @push_constant;",
            "only a constant block",
        ),
    ],
)
def test_hlsl_refuses_push_constants_one_block_cannot_hold(tmp_path, source, reason):
    with pytest.raises(PushConstantBlockError, match=re.escape(reason)):
        _translate(tmp_path, source)


def test_hlsl_cbuffer_without_push_constant_is_unchanged(tmp_path):
    hlsl = _translate(tmp_path, CBUFFER.replace(" @push_constant", ""))
    assert "__spirv__" not in hlsl
    assert "vk::push_constant" not in hlsl
    assert "cbuffer Constants : register(b0)" in hlsl
