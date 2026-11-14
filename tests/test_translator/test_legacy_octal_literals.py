"""C-family octal literals reach CrossGL as octal (issue #3282).

C11 6.4.4.1, C++17 [lex.icon], GLSL 4.60 section 4.1.3, HLSL and Slang read ``017`` as octal 15, and CrossGL reads
it as decimal 17. The HLSL and Metal importers already converted such literals; the GLSL, Slang, CUDA, HIP and
OpenCL C importers copied them, so ``017 + 010`` became 27 instead of 23 in every target.
"""

import re

import pytest

import crosstl
from crosstl.translator.integer_literals import (
    IntegerLiteralError,
    crossgl_legacy_octal,
)

EXPRESSION = "017 + 010 + 0 + 00 + 0x1F + 7"

SOURCES = {
    "comp": (
        "#version 450\nlayout(local_size_x = 1) in;\n"
        "layout(std430, binding = 0) buffer Out { int outp[]; };\n"
        f"void main() {{ outp[0] = {EXPRESSION}; }}\n"
    ),
    "slang": (
        "RWStructuredBuffer<int> outp;\n"
        '[shader("compute")][numthreads(1,1,1)] '
        f"void main() {{ outp[0] = {EXPRESSION}; }}\n"
    ),
    "hlsl": (
        "RWStructuredBuffer<int> outp : register(u0);\n"
        f"[numthreads(1,1,1)] void main() {{ outp[0] = {EXPRESSION}; }}\n"
    ),
    "cu": f"__global__ void k(int* outp) {{ outp[0] = {EXPRESSION}; }}\n",
    "hip": f"__global__ void k(int* outp) {{ outp[0] = {EXPRESSION}; }}\n",
    "cl": f"kernel void k(global int* outp) {{ outp[0] = {EXPRESSION}; }}\n",
    "metal": (
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void k(device int* outp [[buffer(0)]]) "
        f"{{ outp[0] = {EXPRESSION}; }}\n"
    ),
}


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("017", "0o17"),
        ("017u", "0o17u"),
        ("017UL", "0o17UL"),
        ("00", "0o0"),
        ("0", "0"),
        ("17", "17"),
        ("0x1F", "0x1F"),
        ("0.5", "0.5"),
    ],
)
def test_legacy_octal_spelling(spelling, expected):
    assert crossgl_legacy_octal(spelling) == expected


def test_a_leading_zero_with_an_eight_is_not_an_integer():
    with pytest.raises(IntegerLiteralError):
        crossgl_legacy_octal("018")


@pytest.mark.parametrize("extension", sorted(SOURCES))
def test_importers_keep_octal_values(tmp_path, extension):
    source = tmp_path / f"oct.{extension}"
    source.write_text(SOURCES[extension])
    crossgl = crosstl.translate(str(source), backend="cgl")
    assert not re.search(r"(?<![\w.])0[0-7]+\b", crossgl.replace("0o", "@@"))
    path = tmp_path / "oct.cgl"
    path.write_text(crossgl)
    glsl = crosstl.translate(str(path), backend="opengl", format_output=False)
    # 017 + 010 is 15 + 8, whatever spelling CrossGL keeps for it.
    assert "15 + 8" in glsl
