"""Metal expression types for matrix arithmetic and the saturate family of builtins.

The importer types expressions to resolve builtin overloads. It had no rule for matrix arithmetic, so the argument of
``-normalize(normalMatrix * n)`` was untyped and the program was refused. ``saturate``, ``cross`` and ``faceforward``
were not known builtins either, so ``smoothstep(0.0, 1.0, saturate(x))`` was
refused the same way. See :mod:`crosstl.backend.Metal.matrix_arithmetic`.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.matrix_arithmetic import NOT_MATRIX, matrix_arithmetic_shape

SOURCE = """
#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float3 position [[attribute(0)]];
    half3 tangent [[attribute(1)]];
};

struct VertexOut {
    float4 position [[position]];
    half3 tangent;
    float fade;
};

vertex VertexOut shade(VertexIn in [[stage_in]],
                       constant float4x4 &transform [[buffer(1)]],
                       constant float3x3 &normals [[buffer(2)]]) {
    VertexOut out;
    half3x3 normal_matrix = half3x3(normals);
    float4 world = transform * float4(in.position, 1.0);
    float3 row = float3(1.0, 0.0, 0.0) * normals;
    float4x4 twice = (transform * transform) / 2.0;
    out.position = twice * world + float4(row, 0.0);
    out.tangent = -normalize(normal_matrix * in.tangent);
    float3 up = cross(normalize(row), float3(0.0, 1.0, 0.0));
    float3 facing = faceforward(up, row, float3(0.0, 0.0, 1.0));
    out.fade = 1.0 - smoothstep(0.0, 1.0, saturate(world.z * 0.5)) + facing.y;
    return out;
}
"""


@pytest.mark.parametrize(
    ("op", "left", "right", "left_lanes", "right_lanes", "shape"),
    [
        ("*", (4, 3), None, None, 4, ("vector", 3)),
        ("*", None, (4, 3), 3, None, ("vector", 4)),
        ("*", (2, 3), (4, 2), None, None, ("matrix", 4, 3)),
        ("*", (3, 3), None, None, 0, ("matrix", 3, 3)),
        ("*", None, (3, 3), 0, None, ("matrix", 3, 3)),
        ("/", (4, 4), None, None, 0, ("matrix", 4, 4)),
        ("+", (2, 2), (2, 2), None, None, ("matrix", 2, 2)),
        ("*", (4, 3), None, None, 3, None),
        ("*", (2, 3), (2, 3), None, None, None),
        ("-", (2, 2), (3, 3), None, None, None),
        ("/", None, (2, 2), 0, None, None),
    ],
)
def test_matrix_arithmetic_shapes(op, left, right, left_lanes, right_lanes, shape):
    assert matrix_arithmetic_shape(op, left, right, left_lanes, right_lanes) == shape


def test_operations_without_a_matrix_are_not_matrix_arithmetic():
    assert matrix_arithmetic_shape("*", None, None, 4, 4) is NOT_MATRIX


def _translate(tmp_path, backend):
    path = tmp_path / "shade.metal"
    path.write_text(SOURCE, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def test_typed_matrix_arithmetic_and_builtins_import(tmp_path):
    crossgl = _translate(tmp_path, "cgl")
    assert "out_.tangent = (-normalize(normal_matrix * in_.tangent));" in crossgl
    assert "smoothstep(0.0, 1.0, saturate(world.z * 0.5))" in crossgl
    assert "vec3 up = cross(normalize(row), vec3(0.0, 1.0, 0.0));" in crossgl
    assert "faceforward(up, row, vec3(0.0, 0.0, 1.0))" in crossgl


def test_metal_round_trip_compiles(tmp_path):
    metal = _translate(tmp_path, "metal")
    xcrun = shutil.which("xcrun")
    if (
        xcrun is None
        or subprocess.run(
            [xcrun, "-sdk", "macosx", "-f", "metal"], capture_output=True
        ).returncode
    ):
        return
    shader = tmp_path / "out.metal"
    shader.write_text(metal, encoding="utf-8")
    completed = subprocess.run(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader),
            "-o",
            str(tmp_path / "out.air"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]


@pytest.mark.parametrize("backend", ["directx", "opengl"])
def test_other_targets_translate(tmp_path, backend):
    output = _translate(tmp_path, backend)
    assert "normalize(" in output
    assert "cross(" in output
