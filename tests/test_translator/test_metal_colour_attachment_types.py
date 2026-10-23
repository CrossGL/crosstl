"""MSL colour attachment outputs may be scalars or narrower vectors.

``[[color(n)]]`` takes any scalar or vector of up to four float, half or integer components (MSL 3.2 section
5.2.3.4), as in a G-buffer's ``float depth [[color(3)]]``. The Metal generator accepted only ``float4`` and
``half4`` members and refused the rest.
"""

import shutil
import subprocess

import pytest

import crosstl

GBUFFER = """
#include <metal_stdlib>
using namespace metal;

struct Varyings {
    float4 position [[position]];
    float2 uv;
};

struct GBufferData {
    half4 lighting [[color(0)]];
    half2 normal [[color(1)]];
    float depth [[color(2)]];
    uint material [[color(3)]];
};

fragment GBufferData gbuffer(Varyings in [[stage_in]]) {
    GBufferData out;
    out.lighting = half4(0.0h);
    out.normal = half2(in.uv);
    out.depth = in.position.z;
    out.material = 7u;
    return out;
}
"""


def _translate(tmp_path, source):
    path = tmp_path / "gbuffer.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="metal")


def test_scalar_and_narrow_colour_outputs_keep_their_types(tmp_path):
    metal = _translate(tmp_path, GBUFFER)
    for declaration in (
        "half4 lighting [[color(0)]];",
        "half2 normal [[color(1)]];",
        "float depth [[color(2)]];",
        "uint material [[color(3)]];",
    ):
        assert declaration in metal
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


def test_matrix_colour_outputs_are_still_refused(tmp_path):
    source = GBUFFER.replace(
        "float depth [[color(2)]];", "float2x2 depth [[color(2)]];"
    )
    source = source.replace("out.depth = in.position.z;", "out.depth = float2x2(1.0);")
    with pytest.raises(ValueError, match="requires member 'depth'"):
        _translate(tmp_path, source)
