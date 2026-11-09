"""SPIR-V output places the members of a ``ConstantBuffer<T>`` block where ``T`` declares them.

A struct member written ``float scale @offset(16);`` (GLSL ``layout(offset = 16)``, HLSL ``[[vk::offset(16)]]``)
starts at byte 16 of the block; the following members continue from there under the block's rules: std430 for a
push-constant block unless the variable declares another ``@buffer_layout``, std140 for a uniform block (Vulkan 1.3
section 15.6.4). The SPIR-V emitter used to lay ``T`` out from byte 0 and ignore the offsets, so the shader read
``scale`` from the bytes the host wrote for another member. A placement the block's rules refuse (misaligned or
overlapping) raises :class:`UnsupportedSPIRVFeatureError`.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from crosstl.translator.codegen.SPIRV_codegen import (
    UnsupportedSPIRVFeatureError,
    VulkanSPIRVCodeGen,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from tests.spirv_friendly_names import friendly_spirv


def program(structs, declaration, body):
    return f"""shader Probe {{
    {structs}
    {declaration}
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            {body}
        }}
    }}
}}"""


def generate(source):
    ast = Parser(Lexer(source).get_tokens()).parse()
    return friendly_spirv(VulkanSPIRVCodeGen().generate(ast))


def spirv_val(tmp_path, code):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools is not installed")
    (tmp_path / "out.spvasm").write_text(code)
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", "out.spvasm", "-o", "out.spv"],
        ["spirv-val", "--target-env", "vulkan1.1", "out.spv"],
    ):
        result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def block_offsets(code, variable, storage_class):
    """The member offsets of the block type ``variable`` points to."""
    pointer = re.search(rf"%{variable} = OpVariable %(\w+) {storage_class}\b", code)
    assert pointer is not None, code
    block = re.search(
        rf"%{pointer.group(1)} = OpTypePointer {storage_class} %(\w+)", code
    ).group(1)
    assert re.search(rf"OpDecorate %{block} Block\b", code)
    return {
        int(index): int(offset)
        for index, offset in re.findall(
            rf"OpMemberDecorate %{block} (\d+) Offset (\d+)", code
        )
    }


def test_push_constant_struct_member_keeps_explicit_offset(tmp_path):
    code = generate(
        program(
            "struct Constants { float scale @offset(16); float bias; };",
            "ConstantBuffer<Constants> pc @push_constant;",
            "outp[i] = pc.scale * float(i) + pc.bias;",
        )
    )

    assert block_offsets(code, "pc", "PushConstant") == {0: 16, 1: 20}
    assert not re.search(r"%pc\w* (DescriptorSet|Binding)", code)
    spirv_val(tmp_path, code)


def test_push_constant_struct_without_offsets_keeps_std430(tmp_path):
    code = generate(
        program(
            "struct Constants { float scale; float pad[2]; vec3 v; };",
            "ConstantBuffer<Constants> pc @push_constant;",
            "outp[i] = pc.scale + pc.pad[1] + pc.v.y;",
        )
    )

    assert block_offsets(code, "pc", "PushConstant") == {0: 0, 1: 4, 2: 16}
    spirv_val(tmp_path, code)


def test_push_constant_struct_follows_declared_std140_layout(tmp_path):
    code = generate(
        program(
            "struct Constants { float scale; float pad[2]; vec3 v; };",
            "ConstantBuffer<Constants> pc @push_constant @buffer_layout(std140);",
            "outp[i] = pc.scale + pc.pad[1] + pc.v.y;",
        )
    )

    assert block_offsets(code, "pc", "PushConstant") == {0: 0, 1: 16, 2: 48}
    assert re.search(r"OpDecorate %\w+ ArrayStride 16", code)
    spirv_val(tmp_path, code)


def test_placed_push_constant_struct_with_nested_matrix_and_array_members(tmp_path):
    code = generate(
        program(
            "struct Inner { vec2 a; float b; }; "
            "struct Constants { float s; Inner inner @offset(32); mat4 m @offset(64); float arr[3] @offset(128); };",
            "ConstantBuffer<Constants> pc @push_constant;",
            "Constants copy = pc; "
            "outp[i] = pc.s + pc.inner.b + pc.m[1][2] + pc.arr[2] + copy.inner.a.x + copy.m[0][0];",
        )
    )

    assert block_offsets(code, "pc", "PushConstant") == {0: 0, 1: 32, 2: 64, 3: 128}
    # The function-scope copy keeps the logical type, which carries no explicit layout.
    spirv_val(tmp_path, code)


@pytest.mark.parametrize(
    "members, body, message",
    [
        (
            "float scale; vec3 v @offset(4);",
            "outp[i] = pc.scale + pc.v.y;",
            "not a multiple of its std430 alignment 16",
        ),
        (
            "vec4 s @offset(16); float b @offset(20);",
            "outp[i] = pc.s.y + pc.b;",
            "lies within or before",
        ),
    ],
)
def test_push_constant_struct_refuses_offsets_std430_cannot_place(
    members, body, message
):
    source = program(
        f"struct Constants {{ {members} }};",
        "ConstantBuffer<Constants> pc @push_constant;",
        body,
    )
    with pytest.raises(UnsupportedSPIRVFeatureError, match=message):
        generate(source)


def test_uniform_struct_member_keeps_explicit_offset(tmp_path):
    code = generate(
        program(
            "struct Constants { float scale @offset(16); vec3 v @offset(32); };",
            "ConstantBuffer<Constants> cb @binding(0);",
            "outp[i] = cb.scale + cb.v.y;",
        )
    )

    assert block_offsets(code, "cb", "Uniform") == {0: 16, 1: 32}
    assert re.search(r"OpDecorate %cb Binding 0", code)
    assert re.search(r"OpDecorate %cb DescriptorSet 0", code)
    spirv_val(tmp_path, code)


def test_uniform_struct_without_offsets_keeps_std140(tmp_path):
    code = generate(
        program(
            "struct Constants { float scale; vec3 v; };",
            "ConstantBuffer<Constants> cb @binding(0);",
            "outp[i] = cb.scale + cb.v.y;",
        )
    )

    assert block_offsets(code, "cb", "Uniform") == {0: 0, 1: 16}
    spirv_val(tmp_path, code)


def test_placed_uniform_struct_leaves_struct_used_elsewhere_undecorated(tmp_path):
    code = generate(
        program(
            "struct Inner { vec2 a; float b; }; "
            "struct Constants { float s; Inner inner @offset(32); mat4 m @offset(64); float arr[3] @offset(128); };",
            "ConstantBuffer<Constants> cb @binding(0); ConstantBuffer<Constants> other @binding(1);",
            "Constants copy = cb; "
            "outp[i] = cb.s + cb.inner.b + cb.m[1][2] + cb.arr[2] + copy.inner.a.x + other.m[0][0];",
        )
    )

    assert block_offsets(code, "cb", "Uniform") == {0: 0, 1: 32, 2: 64, 3: 128}
    assert block_offsets(code, "other", "Uniform") == {0: 0, 1: 32, 2: 64, 3: 128}
    spirv_val(tmp_path, code)


def test_uniform_struct_refuses_offset_std140_cannot_place():
    with pytest.raises(
        UnsupportedSPIRVFeatureError, match="not a multiple of its std140 alignment 16"
    ):
        generate(
            program(
                "struct Constants { float scale; vec3 v @offset(4); };",
                "ConstantBuffer<Constants> cb @binding(0);",
                "outp[i] = cb.scale + cb.v.y;",
            )
        )
