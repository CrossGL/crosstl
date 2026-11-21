"""Metal import reads a raster order group argument as a CrossGL rasterizer-ordered resource.

``[[raster_order_group(N)]]`` on a fragment function's buffer or texture argument orders the accesses of
overlapping fragments to it. The importer wrote the attribute as ``@raster_order_group(N)``, a name CrossGL does
not define, so Metal output refused the round trip and HLSL output declared an unordered view. CrossGL has one
order, ``@rasterizer_ordered``, which orders at least the accesses any Metal group orders.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl

BUFFER = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    struct FragmentIn {
        float4 coord [[position]];
        float2 uv;
    };

    fragment float4 probe(FragmentIn in [[stage_in]],
                          device float4 *accum [[buffer(0), raster_order_group(GROUP)]]) {
        uint k = uint(in.coord.y) * 64u + uint(in.coord.x);
        accum[k] = accum[k] + float4(in.uv, 0.0, 1.0);
        return float4(in.uv, 0.0, 1.0);
    }
    """)

TEXTURE = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    struct FragmentIn {
        float4 coord [[position]];
        float2 uv;
    };

    fragment float4 probe(FragmentIn in [[stage_in]],
                          texture2d<float, access::read_write> accum [[texture(0), raster_order_group(GROUP)]]) {
        uint2 p = uint2(in.coord.xy);
        accum.write(accum.read(p) + float4(in.uv, 0.0, 1.0), p);
        return float4(in.uv, 0.0, 1.0);
    }
    """)

CASES = {
    "buffer": (BUFFER, "RasterizerOrderedStructuredBuffer<float4> accum"),
    "texture": (TEXTURE, "RasterizerOrderedTexture2D<float4> accum"),
}


def _import(tmp_path, source):
    path = tmp_path / "probe.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="cgl", format_output=False)


def _translate(tmp_path, crossgl, backend):
    path = tmp_path / "probe.cgl"
    path.write_text(crossgl, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _assert_compiles(tmp_path, metal):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return
    shader = tmp_path / "out.metal"
    shader.write_text(metal, encoding="utf-8")
    result = subprocess.run(
        [xcrun, "-sdk", "macosx", "metal", "-std=metal3.0", "-c", str(shader)]
        + ["-o", str(tmp_path / "out.air")],
        capture_output=True,
        text=True,
        check=False,
    )
    if "unable to find utility" not in result.stderr:
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("group", ["0", "2"])
@pytest.mark.parametrize("kind", sorted(CASES))
def test_raster_order_group_imports_as_rasterizer_ordered(tmp_path, kind, group):
    source, _ = CASES[kind]

    crossgl = _import(tmp_path, source.replace("GROUP", group))

    accum = next(line for line in crossgl.splitlines() if "accum @" in line)
    assert "@rasterizer_ordered" in accum
    assert "raster_order_group" not in crossgl


@pytest.mark.parametrize("kind", sorted(CASES))
def test_raster_order_group_round_trips_to_metal(tmp_path, kind):
    source, _ = CASES[kind]
    crossgl = _import(tmp_path, source.replace("GROUP", "1"))

    metal = _translate(tmp_path, crossgl, "metal")

    assert "[[raster_order_group(0)]]" in metal
    _assert_compiles(tmp_path, metal)


@pytest.mark.parametrize("kind", sorted(CASES))
def test_raster_order_group_reaches_hlsl_as_an_ordered_view(tmp_path, kind):
    source, hlsl_declaration = CASES[kind]
    crossgl = _import(tmp_path, source.replace("GROUP", "0"))

    hlsl = _translate(tmp_path, crossgl, "directx")

    assert hlsl_declaration in hlsl
