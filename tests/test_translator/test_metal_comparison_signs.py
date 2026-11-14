"""Comparisons of ``int`` with ``uint`` reach Metal with GLSL's ``int`` to ``uint`` conversion written out.

GLSL converts the ``int`` operand of a mixed comparison to ``uint``; MSL computes the same, but warns
(``-Wsign-compare``), which fails a warnings-as-errors build (sascha-vulkan computecloth, computenbody and
computeparticles compute shaders).
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.metal_comparison_signs import (
    metal_sign_matched_comparison_operands,
)

COMPUTE = """#version 450
layout (local_size_x = 16) in;
layout (binding = 0) buffer Values { float values[]; };
layout (binding = 1) uniform UBO { int count; ivec2 size; } ubo;
void main()
{
    uint index = gl_GlobalInvocationID.x;
    if (index >= ubo.count) {
        return;
    }
    for (int j = 0; j < gl_WorkGroupSize.x; j++) {
        values[index] += float(j);
    }
    if (gl_GlobalInvocationID.y < ubo.size.y - 1 && ubo.count != index) {
        values[index] = 0.0;
    }
}
"""


@pytest.mark.parametrize(
    "left_type, right_type, expected",
    [
        ("uint", "int", ("a", "uint(b)")),
        ("int", "uint", ("uint(a)", "b")),
        ("uint3", "int3", ("a", "uint3(b)")),
        ("int2", "uint2", ("uint2(a)", "b")),
        ("uint", "uint", ("a", "b")),
        ("int", "int", ("a", "b")),
        ("uint", "float", ("a", "b")),
        ("uint2", "int3", ("a", "b")),
        ("ushort", "int", ("a", "b")),
    ],
)
def test_signed_operands_take_the_unsigned_type(left_type, right_type, expected):
    assert (
        metal_sign_matched_comparison_operands(left_type, right_type, "a", "b")
        == expected
    )


def test_non_negative_literals_are_kept():
    assert metal_sign_matched_comparison_operands("uint", "int", "a", "0") == ("a", "0")
    assert metal_sign_matched_comparison_operands("int", "uint", "16", "b") == (
        "16",
        "b",
    )
    assert metal_sign_matched_comparison_operands("uint", "int", "a", "-1") == (
        "a",
        "uint(-1)",
    )


def test_mixed_sign_comparisons_compile_without_warnings(tmp_path):
    path = tmp_path / "shader.comp"
    path.write_text(COMPUTE)
    code = crosstl.translate(str(path), backend="metal", format_output=False)
    assert "uint(" in code
    if shutil.which("xcrun") is None:
        pytest.skip("the Metal toolchain is not installed")
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    if found.returncode != 0:
        pytest.skip("the Metal toolchain is not installed")
    artifact = tmp_path / "out.metal"
    artifact.write_text(code)
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-Werror", "-c", str(artifact)]
        + ["-o", str(tmp_path / "out.air")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
