"""Metal output converts the result of a vector expression, not its operands (issue #3166).

``float4 x = a / int4(2, 2, 2, 2)`` with ``int4 a`` is an integer division converted to ``float4`` in HLSL, GLSL and
CrossGL. The Metal generator converted each operand to the declared type instead, so the division became a floating
one, and ``int4 x = b / 2u`` a signed one. MSL also converts a scalar operand to the vector's component type
(MSL 2.12), so ``int4 * 0.5`` needs the vector converted to ``float4`` first, as HLSL and GLSL do.
"""

import shutil
import subprocess

import pytest

import crosstl

SHADER = """RWStructuredBuffer<int4> ia : register(u0);
RWStructuredBuffer<uint4> ub : register(u1);
RWStructuredBuffer<float4> fo : register(u2);
RWStructuredBuffer<int4> io : register(u3);
RWStructuredBuffer<uint4> uo : register(u4);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    int4 a = ia[0];
    uint4 b = ub[0];
    float4 x1 = a / int4(2, 2, 2, 2);
    int4 x2 = b / 2u;
    uint4 x3 = a / int4(2, 2, 2, 2);
    float4 x4 = a + b;
    float4 x5 = a * 0.5;
    fo[0] = x1 + x4 + x5;
    io[0] = x2;
    uo[0] = x3;
}
"""


@pytest.fixture
def metal(tmp_path):
    path = tmp_path / "vectors.hlsl"
    path.write_text(SHADER)
    return crosstl.translate(str(path), backend="metal", format_output=False)


@pytest.mark.parametrize(
    "declaration",
    [
        # Integer division, then the conversion.
        "float4 x1 = float4(a / int4(2, 2, 2, 2));",
        # Unsigned division of the uint4.
        "int4 x2 = int4(b / 2u);",
        # Signed division of the int4.
        "uint4 x3 = uint4(a / int4(2, 2, 2, 2));",
        # int4 + uint4 is a wrapping uint4 sum.
        "float4 x4 = float4(uint4(a) + b);",
        # int4 * 0.5 multiplies in float.
        "float4 x5 = float4(float4(a) * 0.5);",
    ],
)
def test_declarations_convert_the_result(metal, declaration):
    assert declaration in metal


def test_metal_compiles(metal, tmp_path):
    if shutil.which("xcrun") is None:
        return
    source = tmp_path / "vectors.metal"
    source.write_text(metal)
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "v.air"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
