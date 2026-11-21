"""Metal import keeps the direction a fragment depth output promises.

``[[depth(greater)]]`` and ``[[depth(less)]]`` on a fragment output promise that the shader only moves the depth
in that direction (MSL 3.2 section 5.2.3.5). CrossGL states the promise as a conservative depth layout next to
``gl_FragDepth``. The importer wrote every depth argument as plain ``@gl_FragDepth``, so a round trip through
CrossGL wrote ``[[depth(any)]]`` and other targets lost ``SV_DepthGreaterEqual`` or ``DepthGreater``.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl

SOURCE = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    struct FragmentIn {
        float2 uv;
    };

    struct FragmentOut {
        float4 color [[color(0)]];
        float depth [[depth(ARGUMENT)]];
    };

    fragment FragmentOut probe(FragmentIn in [[stage_in]]) {
        FragmentOut out;
        out.color = float4(in.uv, 0.0, 1.0);
        out.depth = in.uv.x * 0.5 + 0.25;
        return out;
    }
    """)

ARGUMENTS = [
    ("greater", "@gl_FragDepth @depth_greater;", "SV_DepthGreaterEqual"),
    ("less", "@gl_FragDepth @depth_less;", "SV_DepthLessEqual"),
    ("any", "@gl_FragDepth;", "SV_DEPTH;"),
]


def _translate(tmp_path, source, backend):
    path = tmp_path / "probe.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend="metal"
    )


@pytest.mark.parametrize(
    "argument, crossgl_semantic, hlsl_semantic",
    ARGUMENTS,
    ids=[argument for argument, _, _ in ARGUMENTS],
)
def test_depth_argument_survives_import(
    tmp_path, argument, crossgl_semantic, hlsl_semantic
):
    source = SOURCE.replace("ARGUMENT", argument)

    crossgl = _translate(tmp_path, source, "cgl")
    assert f"float depth {crossgl_semantic}" in crossgl

    metal = _translate(tmp_path, source, "metal")
    assert f"float depth [[depth({argument})]];" in metal

    hlsl = _translate(tmp_path, source, "directx")
    assert f"float depth: {hlsl_semantic}" in hlsl

    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return
    shader = tmp_path / "round_trip.metal"
    shader.write_text(metal, encoding="utf-8")
    result = subprocess.run(
        [xcrun, "-sdk", "macosx", "metal", "-std=metal3.0", "-c", str(shader)]
        + ["-o", str(tmp_path / "round_trip.air")],
        capture_output=True,
        text=True,
        check=False,
    )
    if "unable to find utility" not in result.stderr:
        assert result.returncode == 0, result.stderr
