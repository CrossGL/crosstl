"""Mixed-signedness vector arithmetic declared with an integer vector type (#3089).

HLSL evaluates ``int4 + uint4`` as ``uint4`` and converts the result to the
declared type. Metal rejects mixed vector operands, so the generator converts
the ``int4`` operand to ``uint4``; it must not then convert the other operand
back to the declared ``int4``, which mixes the operands again.
"""

import shutil
import subprocess

import pytest

import crosstl

SHADER = """RWStructuredBuffer<uint> iin;
RWStructuredBuffer<int> outp;
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    int4 a = int4(iin[0], 1, 2, 3);
    uint4 b = uint4(iin[1], 1u, 2u, 3u);
    float4 g = float4(1.0, 2.0, 3.0, 4.0);
    int4 r = a + b;
    int4 s = a * b;
    int4 q = a + g;
    uint4 u = a - b;
    float4 f = a + b;
    int4 h = a + a;
    outp[0] = r.y + s.x + q.z + int(u.w) + int(f.x) + h.y;
}
"""


def translate(tmp_path):
    path = tmp_path / "mixed.hlsl"
    path.write_text(SHADER)
    return crosstl.translate(str(path), backend="metal", format_output=False)


def test_integer_declared_type_keeps_the_common_operand_type(tmp_path):
    code = translate(tmp_path)

    assert "int4 r = int4(uint4(a) + b);" in code
    assert "int4 s = int4(uint4(a) * b);" in code
    assert "int4 q = int4(float4(a) + g);" in code
    assert "uint4 u = uint4(a) - b;" in code
    assert "int4(b)" not in code
    assert "int4(g)" not in code


def test_float_declared_type_and_matching_operands_are_unchanged(tmp_path):
    code = translate(tmp_path)

    assert "float4 f = float4(float4(uint4(a)) + float4(b));" in code
    assert "int4 h = as_type<int4>(as_type<uint4>(a) + as_type<uint4>(a));" in code


def test_mixed_vector_declarations_compile(tmp_path):
    code = translate(tmp_path)
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
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(artifact)]
        + ["-o", str(tmp_path / "out.air")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
