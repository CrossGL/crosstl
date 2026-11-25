"""HLSL input reads DXC's ``[[vk::constant_id(N)]]`` as a CrossGL specialization constant.

DXC's SPIR-V output makes ``[[vk::constant_id(0)]] const int N = 3;`` an ``OpSpecConstant %int 3`` with
``SpecId 0``. Once a non-static global became a ``$Globals`` member, the importer wrote
``@ vk_constant_id(0) @ hlsl_program_constant int N;``, which no generator reads as a specialization constant: GLSL
got a loose ``uniform int N;`` that Vulkan GLSL rejects, SPIR-V ``OpSpecConstant %int 0`` and the default 3 was lost.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.backend.import_refusals import DirectXImportRefused

KERNEL = """
    {declaration}
    RWStructuredBuffer<int> v : register(u0);
    [numthreads(1, 1, 1)]
    void main() {{ v[0] = {read}; }}
    """


def _translate(tmp_path, declaration, backend, read="N"):
    path = tmp_path / "shader.hlsl"
    path.write_text(
        textwrap.dedent(KERNEL.format(declaration=declaration, read=read)),
        encoding="utf-8",
    )
    return crosstl.translate(str(path), backend=backend)


@pytest.mark.parametrize(
    "declaration",
    [
        "[[vk::constant_id(0)]] const int N = 3;",
        # DXC's SPIR-V back end specializes the global without ``const`` too.
        "[[vk::constant_id(0)]] int N = 3;",
    ],
)
def test_import_writes_a_specialization_constant_with_its_default(
    tmp_path, declaration
):
    crossgl = _translate(tmp_path, declaration, "cgl")

    assert "const int N @constant_id(0) = 3;" in crossgl
    assert "hlsl_program_constant" not in crossgl
    assert "vk_constant_id" not in crossgl


def test_glsl_output_declares_a_vulkan_specialization_constant(tmp_path):
    glsl = _translate(tmp_path, "[[vk::constant_id(0)]] const int N = 3;", "opengl")

    assert "layout(constant_id = 0) const int N = 3;" in glsl
    assert not re.search(r"\buniform\s+int\s+N\b", glsl)
    if shutil.which("glslangValidator"):
        path = tmp_path / "shader.comp"
        path.write_text(glsl, encoding="utf-8")
        result = subprocess.run(
            [
                "glslangValidator",
                "--target-env",
                "vulkan1.1",
                "-o",
                str(tmp_path / "shader.spv"),
                str(path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout


@pytest.mark.parametrize(
    "declaration, read, constant",
    [
        ("[[vk::constant_id(0)]] const int N = 3;", "N", r"OpSpecConstant %\w+ 3\b"),
        (
            "[[vk::constant_id(2)]] const bool B = true;",
            "B ? 1 : 0",
            r"OpSpecConstantTrue %\w+",
        ),
        (
            "[[vk::constant_id(4)]] const float G = 1.5;",
            "int(G * 2.0)",
            r"OpSpecConstant %\w+ 1\.5\b",
        ),
    ],
)
def test_spirv_output_keeps_the_spec_id_and_the_default(
    tmp_path, declaration, read, constant
):
    spirv = _translate(tmp_path, declaration, "vulkan", read=read)
    spec_id = re.search(r"\[\[vk::constant_id\((\d+)\)\]\]", declaration).group(1)

    assert re.search(rf"OpDecorate %\w+ SpecId {spec_id}\b", spirv)
    assert re.search(constant, spirv)
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        assembly = tmp_path / "shader.spvasm"
        assembly.write_text(spirv, encoding="utf-8")
        binary = str(tmp_path / "shader.spv")
        subprocess.run(
            ["spirv-as", "--target-env", "vulkan1.1", str(assembly), "-o", binary],
            check=True,
        )
        subprocess.run(["spirv-val", "--target-env", "vulkan1.1", binary], check=True)


def test_metal_output_defaults_the_function_constant(tmp_path):
    metal = _translate(tmp_path, "[[vk::constant_id(0)]] const int N = 3;", "metal")

    assert "[[function_constant(0)]]" in metal
    assert re.search(
        r"is_function_constant_defined\(\w+\)\s*\?\s*\w+\s*:\s*int\(3\)", metal
    )


def test_a_constant_without_the_attribute_stays_a_globals_member(tmp_path):
    crossgl = _translate(tmp_path, "const int N = 3;", "cgl")

    assert "@ hlsl_program_constant" in crossgl
    assert "constant_id" not in crossgl


@pytest.mark.parametrize(
    "declaration, reason",
    [
        # Each is a DXC -spirv error: "missing default value for specialization constant",
        # "specialization constant must be externally visible", "'constant_id' attribute only applies to global
        # variables of scalar type".
        ("[[vk::constant_id(0)]] int N;", "needs a default value"),
        ("[[vk::constant_id(0)]] static const int N = 3;", "externally visible"),
        ("[[vk::constant_id(0)]] const int2 N = int2(1, 2);", "scalar type"),
    ],
)
def test_constants_dxc_rejects_are_refused(tmp_path, declaration, reason):
    read = "N.x" if "int2" in declaration else "N"
    with pytest.raises(DirectXImportRefused, match=reason):
        _translate(tmp_path, declaration, "cgl", read=read)
