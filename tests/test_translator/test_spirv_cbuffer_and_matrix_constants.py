"""Constant buffers and module-scope matrix constants on the SPIR-V emitter.

``cbuffer ubo { UBO ubo; }`` is the usual HLSL spelling of a uniform block whose only member is a struct of the
same name: the constant-buffer name is not in expression scope, so ``ubo.model`` reads the member. A module-scope
``const mat4`` must become ``OpConstantComposite`` columns, because ``OpCompositeConstruct`` may only appear inside
a block (SPIR-V 1.6 section 2.16.1) and an unfolded initializer must never become the zero matrix.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def translate(tmp_path, name, source, backend="vulkan"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def validate(tmp_path, spirv):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    asm, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
    asm.write_text(spirv)
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def module_scope(spirv):
    return spirv.split(" = OpFunction ", 1)[0]


def constants_by_id(spirv):
    scalars = dict(re.findall(r"%(\w+) = OpConstant %\w+ (\S+)", spirv))
    composites = {
        name: parts.split()
        for name, parts in re.findall(
            r"%(\w+) = OpConstantComposite %\w+ ([^\n]+)", spirv
        )
    }
    return scalars, composites


def test_constant_buffer_member_named_like_the_buffer_is_read_through_the_block(
    tmp_path,
):
    spirv = translate(
        tmp_path,
        "ubo.hlsl",
        """
        struct UBO { float4x4 projection; float4 lightPos; };
        cbuffer ubo : register(b0) { UBO ubo; }
        float4 main(float3 pos : POSITION0) : SV_POSITION
        {
            return mul(ubo.projection, float4(pos, 1.0)) + ubo.lightPos;
        }
        """,
    )
    assert "OpDecorate" in spirv and " Block" in spirv
    # ubo (block member 0, the UBO struct) -> projection / lightPos: every read is a two-step access chain.
    chains = dict(re.findall(r"%(\w+) = OpAccessChain %\w+ %(\w+) %\w+\n", spirv))
    assert sum(1 for base in chains.values() if base in chains) == 2
    assert re.search(r"OpMemberDecorate %\w+ 1 Offset 64", spirv)
    validate(tmp_path, spirv)


def test_module_scope_matrix_constants_are_constant_composites(tmp_path):
    source = """
    shader main {
        const mat4 bias = mat4(0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 1.0);
        const mat2 pair = mat2(vec2(1.0, 2.0), vec2(3.0, 4.0));
        const mat3 diagonal = mat3(2.0);
        vertex {
            vec4 main(vec4 pos @ Position) @ gl_Position {
                return bias * pos + vec4(pair[1], diagonal[2][2], 0.0);
            }
        }
    }
    """
    Parser(Lexer(source).get_tokens()).parse()
    spirv = translate(tmp_path, "constants.cgl", source)
    head = module_scope(spirv)
    assert "OpCompositeConstruct" not in head
    scalars, composites = constants_by_id(head)

    def column_values(column_id):
        return [float(scalars[part.lstrip("%")]) for part in composites[column_id]]

    matrices = {
        name: [column_values(column.lstrip("%")) for column in columns]
        for name, columns in composites.items()
        if all(column.lstrip("%") in composites for column in columns)
    }
    assert [
        [0.5, 0, 0, 0],
        [0, 0.5, 0, 0],
        [0, 0, 1, 0],
        [0.5, 0.5, 0, 1],
    ] in matrices.values()
    assert [[1, 2], [3, 4]] in matrices.values()
    assert [[2, 0, 0], [0, 2, 0], [0, 0, 2]] in matrices.values()
    validate(tmp_path, spirv)


def test_hlsl_static_const_matrix_reaches_spirv_as_constant(tmp_path):
    spirv = translate(
        tmp_path,
        "bias.hlsl",
        """
        static const float4x4 biasMat = float4x4(
            0.5, 0.0, 0.0, 0.5,
            0.0, 0.5, 0.0, 0.5,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0);
        float4 main(float4 pos : POSITION0) : SV_POSITION
        {
            return mul(biasMat, pos);
        }
        """,
    )
    assert "OpCompositeConstruct" not in module_scope(spirv)
    validate(tmp_path, spirv)
