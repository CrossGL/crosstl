"""Loose uniforms reach SPIR-V as members of one std140 ``Uniform`` block.

The emitter gave ``uniform mat4 m;`` and every other non-opaque uniform outside a block ``Private`` storage, so the
module validated but each invocation read its own uninitialised copy instead of the host's value. They are now
members of ``CrossGLDefaultUniforms``, the module's default uniform block (DXC's ``$Globals``, glslang's
``gl_DefaultUniformBlock``), laid out with std140 rules. Initializers and conflicting redeclarations have no Vulkan
form and are refused.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

import crosstl
from crosstl.translator.codegen.SPIRV_codegen import UnsupportedSPIRVFeatureError
from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused

ROOT = Path(__file__).resolve().parents[2]

LOOSE = """
shader Loose {
    uniform float gain;
    struct VSIn { vec3 position; }
    struct VSOut { vec4 position; }
    struct FSOut { vec4 color; }
    vertex {
        uniform mat4 mvp;
        uniform vec3 offsets[2];
        VSOut main(VSIn input) {
            VSOut o;
            o.position = mvp * vec4(input.position + offsets[1], gain);
            return o;
        }
    }
    fragment {
        uniform vec4 tint;
        uniform sampler2D tex;
        uniform int count;
        uniform float gain;
        FSOut main(VSOut input) {
            FSOut o;
            o.color = tint * float(count) * gain * texture(tex, vec2(0.5));
            return o;
        }
    }
}
"""

LAYOUT = """
shader Layout {
    uniform bool enabled;
    uniform float weights[3];
    uniform mat3 basis;
    uniform bvec2 mask;
    uniform ivec3 counts[2];
    float scaled(float x) { return enabled ? x * weights[2] : x; }
    compute {
        buffer float out_data[];
        layout(local_size_x = 1) in;
        void main() {
            vec3 v = basis * vec3(weights[0], weights[1], scaled(1.0));
            if (mask.y && enabled) { v.x += float(counts[1].z); }
            out_data[0] = v.x + v.y + v.z;
        }
    }
}
"""


def translate(tmp_path, source, name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend="vulkan", format_output=False)


def validate(tmp_path, spirv):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    asm, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
    asm.write_text(spirv)
    env = ["--target-env", "vulkan1.1"]
    for command in (
        ["spirv-as", *env, str(asm), "-o", str(binary)],
        ["spirv-val", *env, str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def names(spirv):
    return {
        ident: name
        for ident, name in re.findall(r'^OpName %(\w+) "(\w+)"$', spirv, re.M)
    }


def block_layout(spirv, block_name="CrossGLDefaultUniforms"):
    block = next(i for i, n in names(spirv).items() if n == block_name)
    members = dict(re.findall(rf'^OpMemberName %{block} (\d+) "(\w+)"$', spirv, re.M))
    offsets = re.findall(
        rf"^OpMemberDecorate %{block} (\d+) Offset (\d+)$", spirv, re.M
    )
    return block, {members[index]: int(offset) for index, offset in offsets}


def test_loose_uniforms_share_one_block_and_no_private_copy(tmp_path):
    spirv = translate(tmp_path, LOOSE)
    validate(tmp_path, spirv)
    block, layout = block_layout(spirv)
    # Shader scope first, then each stage; the fragment stage's `gain` is the same uniform.
    assert layout == {"gain": 0, "mvp": 16, "offsets": 80, "tint": 112, "count": 128}
    assert f"OpDecorate %{block} Block" in spirv
    assert " Private" not in spirv
    variable = next(
        i for i, n in names(spirv).items() if n == "crossgl_default_uniforms"
    )
    texture = next(i for i, n in names(spirv).items() if n == "tex")
    # Declared resources keep their bindings; the block takes the next free one.
    assert f"OpDecorate %{texture} Binding 0" in spirv
    assert f"OpDecorate %{variable} Binding 1" in spirv
    assert f"OpDecorate %{variable} DescriptorSet 0" in spirv
    assert re.search(rf"OpAccessChain %\w+ %{variable} %\w+", spirv)


def test_std140_offsets_strides_and_boolean_stand_ins(tmp_path):
    spirv = translate(tmp_path, LAYOUT)
    validate(tmp_path, spirv)
    block, layout = block_layout(spirv)
    # std140 (GLSL 4.60 section 7.6.2.2): scalar arrays and mat3 columns round up to 16 bytes, bvec2 is a uvec2.
    assert layout == {
        "enabled": 0,
        "weights": 16,
        "basis": 64,
        "mask": 112,
        "counts": 128,
    }
    assert f"OpMemberDecorate %{block} 2 MatrixStride 16" in spirv
    member_types = re.search(rf"^%{block} = OpTypeStruct (.*)$", spirv, re.M).group(1)
    uint = re.search(r"^%(\w+) = OpTypeInt 32 0$", spirv, re.M).group(1)
    uvec2 = re.search(rf"^%(\w+) = OpTypeVector %{uint} 2$", spirv, re.M).group(1)
    assert member_types.split()[0] == f"%{uint}"
    assert member_types.split()[3] == f"%{uvec2}"
    assert " Private" not in spirv


def test_boolean_members_of_uniform_structs_use_uint_vectors(tmp_path):
    source = """
    shader StructBools {
        struct Params { bvec3 on; float k; }
        uniform Params p;
        compute {
            buffer float out_data[];
            void main() { out_data[0] = p.on.z ? p.k : 0.0; }
        }
    }
    """
    spirv = translate(tmp_path, source)
    validate(tmp_path, spirv)
    layout_struct = next(
        i for i, n in names(spirv).items() if n.startswith("Params_std140")
    )
    first_member = re.search(rf"^%{layout_struct} = OpTypeStruct %(\w+) ", spirv, re.M)
    uint = re.search(r"^%(\w+) = OpTypeInt 32 0$", spirv, re.M).group(1)
    assert re.search(
        rf"^%{first_member.group(1)} = OpTypeVector %{uint} 3$", spirv, re.M
    )


def test_uniform_initializers_are_refused(tmp_path):
    source = """
    shader Initialized {
        uniform float gain = 2.0;
        compute {
            buffer float out_data[];
            void main() { out_data[0] = gain; }
        }
    }
    """
    with pytest.raises(
        UnsupportedSPIRVFeatureError, match="cannot take an initializer"
    ):
        translate(tmp_path, source)


def test_conflicting_uniform_types_are_refused(tmp_path):
    source = """
    shader Conflict {
        struct FSOut { vec4 color; }
        vertex {
            uniform vec4 tint;
            vec4 main() { return tint; }
        }
        fragment {
            uniform vec3 tint;
            FSOut main() { FSOut o; o.color = vec4(tint, 1.0); return o; }
        }
    }
    """
    with pytest.raises(UnsupportedSPIRVFeatureError, match="4.3.5"):
        translate(tmp_path, source)


def test_assigning_a_uniform_is_refused(tmp_path):
    source = """
    shader Store {
        uniform float gain;
        compute {
            buffer float out_data[];
            void main() { gain = 2.0; out_data[0] = gain; }
        }
    }
    """
    with pytest.raises(SpirvLoweringRefused, match="read-only uniform gain"):
        translate(tmp_path, source)


def test_readme_showcase_uniforms_reach_the_host(tmp_path):
    source = (ROOT / "demos/readme/showcase_pbr.cgl").read_text()
    spirv = translate(tmp_path, source)
    validate(tmp_path, spirv)
    _, layout = block_layout(spirv)
    assert list(layout)[:3] == ["modelMatrix", "viewProjectionMatrix", "normalMatrix"]
    assert layout["lightPositions"] % 16 == 0
    assert " Private" not in spirv
