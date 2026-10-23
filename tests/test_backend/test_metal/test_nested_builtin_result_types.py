"""Metal front-end typing of matrix arithmetic and of saturate, cross and faceforward inside other expressions.

``-normalize(m * v)`` and ``1.0 - smoothstep(0.0, 1.0, saturate(x))`` were refused with
``MetalBuiltinResultTypeResolutionError``: once a call sat inside another expression its argument types were needed,
and the front end typed neither ``half3x3 * half3`` nor ``saturate``, ``cross`` or ``faceforward``.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.matrix_arithmetic_types import matrix_binary_result_type
from crosstl.translator.semantics.half import HalfFloatUnsupportedError

SOURCE = """
#include <metal_stdlib>
using namespace metal;

struct Params { float elevation; float slope; };

struct VertexOut {
    float4 position [[position]];
    half3 tangent;
    half3 bitangent;
    half3 normal;
};

inline float fade(thread Params &params)
{
    return 1.0 - smoothstep(0.0, 1.0, saturate((params.elevation - 0.009) * (1.0 / 0.05)))
        + length(normalize(cross(float3(params.slope), float3(1.0, 0.0, 0.0))))
        + length(faceforward(float3(1.0), float3(params.elevation), float3(0.0, 1.0, 0.0)));
}

vertex VertexOut vs(constant float3x3 &normalMatrix [[buffer(0)]],
                    constant Params &params [[buffer(1)]],
                    uint vid [[vertex_id]])
{
    VertexOut out;
    half3x3 m = half3x3(normalMatrix);
    half3 t = half3(1.0h, 0.0h, 0.0h);
    Params local = params;
    out.position = float4(fade(local));
    out.tangent = -normalize(m * t);
    out.bitangent = half3(0.0h) - normalize(t * m);
    out.normal = -normalize((m * m) * (m * 2.0h) * t);
    return out;
}
"""


@pytest.mark.parametrize(
    "operator, left, right, expected",
    [
        ("*", ("matrix", "half", 3, 3), ("vector", "half", 3), "half3"),
        ("*", ("matrix", "float", 4, 3), ("vector", "float", 4), "float3"),
        ("*", ("vector", "float", 3), ("matrix", "float", 4, 3), "float4"),
        ("*", ("matrix", "float", 2, 3), ("matrix", "float", 4, 2), "float4x3"),
        ("*", ("matrix", "float", 2, 3), ("matrix", "float", 3, 2), "float3x3"),
        ("*", ("matrix", "float", 3, 3), ("scalar", "float"), "float3x3"),
        ("*", ("scalar", "half"), ("matrix", "half", 2, 2), "half2x2"),
        ("+", ("matrix", "float", 3, 2), ("matrix", "float", 3, 2), "float3x2"),
        ("-", ("matrix", "float", 3, 2), ("matrix", "float", 3, 2), "float3x2"),
    ],
)
def test_matrix_rules_give_the_msl_result_type(operator, left, right, expected):
    assert matrix_binary_result_type(operator, left, right) == expected


@pytest.mark.parametrize(
    "operator, left, right",
    [
        ("*", ("matrix", "float", 4, 3), ("vector", "float", 3)),
        ("*", ("vector", "float", 4), ("matrix", "float", 4, 3)),
        ("*", ("matrix", "float", 2, 3), ("matrix", "float", 2, 3)),
        ("*", ("matrix", "half", 3, 3), ("vector", "float", 3)),
        ("+", ("matrix", "float", 3, 3), ("matrix", "float", 3, 2)),
        ("+", ("matrix", "float", 3, 3), ("scalar", "float")),
        ("/", ("matrix", "float", 3, 3), ("matrix", "float", 3, 3)),
        ("*", ("vector", "float", 3), ("scalar", "float")),
    ],
)
def test_mismatched_shapes_have_no_matrix_result(operator, left, right):
    assert matrix_binary_result_type(operator, left, right) is None


def _translate(tmp_path, backend):
    path = tmp_path / "nested.metal"
    path.write_text(SOURCE)
    return crosstl.translate(str(path), backend=backend)


def _tool(*names):
    return all(shutil.which(name) for name in names)


def test_metal_round_trip_compiles(tmp_path):
    code = _translate(tmp_path, "metal")
    assert "normalize(m * t)" in code
    assert "saturate(" in code
    if not _tool("xcrun"):
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


def test_glsl_spells_saturate_as_clamp(tmp_path):
    code = _translate(tmp_path, "opengl")
    assert "clamp(" in code
    assert "saturate(" not in code


def test_directx_translates(tmp_path):
    assert "saturate(" in _translate(tmp_path, "directx")


def test_spirv_refuses_only_the_half_matrix(tmp_path):
    # SPIR-V output has no 16-bit float type; the program now reaches that typed refusal.
    with pytest.raises(HalfFloatUnsupportedError, match="half matrix"):
        _translate(tmp_path, "vulkan")
