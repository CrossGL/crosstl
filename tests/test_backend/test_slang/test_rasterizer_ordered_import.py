"""Slang ``RasterizerOrdered*`` views import as the HLSL importer writes them.

A rasterizer-ordered view reads and writes like its ``RW*`` view, so CrossGL declares the read-write resource with
``@rasterizer_ordered``: ``RasterizerOrderedTexture2D<float4>`` becomes ``image2D ... @rasterizer_ordered`` and its
subscripts become ``imageLoad`` and ``imageStore``. A function parameter takes the read-write type alone. Each
generator then binds the view in its own raster-ordered form, and the result compiles.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

ORDERED_TEXTURE = """RasterizerOrderedTexture2D<float4> accum : register(u1);

void blend(RasterizerOrderedTexture2D<float4> target, uint2 p, float4 c) {
    target[p] = target[p] * 0.5 + c;
}

[shader("fragment")]
float4 main(float4 coord : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
    uint2 p = uint2(coord.xy);
    float4 c = float4(uv, 0.0, 1.0);
    accum[p] = accum[p] + c;
    blend(accum, p, c);
    return c;
}
"""

ORDERED_VIEWS = """RasterizerOrderedTexture1D<float4> t1 : register(u1);
RasterizerOrderedTexture2DArray<uint> t2a : register(u2);
RasterizerOrderedTexture3D<int2> t3 : register(u3);
RasterizerOrderedStructuredBuffer<float> sb : register(u5);
RasterizerOrderedByteAddressBuffer bab : register(u6);

[shader("fragment")]
float4 main(float4 coord : SV_Position) : SV_Target {
    uint2 p = uint2(coord.xy);
    t1[p.x] = t1[p.x] + 1.0;
    t2a[uint3(p, 0)] = t2a[uint3(p, 0)] + 1u;
    t3[uint3(p, 1)] = int2(1, 2);
    sb[p.x] = sb[p.x] + 1.0;
    bab.Store(0, bab.Load(4) + 1u);
    return float4(1.0, 0.0, 0.0, 1.0);
}
"""


def _translate(tmp_path, source, backend, suffix=".slang"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(tmp_path, command):
    result = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, check=False, timeout=180
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]


def _statements(crossgl):
    """The body statements of ``crossgl``, which both importers write alike."""
    return [
        line.strip()
        for line in crossgl.splitlines()
        if line.strip().startswith(("imageStore(", "sb[", "buffer_store(", "blend("))
    ]


def _declaration(crossgl, name):
    """The type and attributes CrossGL declares global ``name`` with, in one canonical order."""
    compact = re.sub(r"\s+", " ", crossgl)
    match = re.search(
        r"((?:@ ?\w+(?:\([^)]*\))? )*)([\w<>]+) "
        + name
        + r"((?: @ ?\w+(?:\([^)]*\))?)*);",
        compact,
    )
    assert match, compact
    attributes = re.findall(r"@ ?(\w+)", match.group(1) + match.group(3))
    return match.group(2), sorted(attributes)


def test_rasterizer_ordered_texture_imports_as_an_ordered_storage_image(tmp_path):
    crossgl = _translate(tmp_path, ORDERED_TEXTURE, "cgl")

    assert _declaration(crossgl, "accum") == (
        "image2D",
        ["format", "rasterizer_ordered", "register"],
    )
    assert "void blend(image2D target, uvec2 p, vec4 c)" in crossgl
    assert _statements(crossgl) == [
        "imageStore(target, p, imageLoad(target, p) * 0.5 + c);",
        "imageStore(accum, p, imageLoad(accum, p) + c);",
        "blend(accum, p, c);",
    ]
    assert "accum[" not in crossgl and "target[" not in crossgl


@pytest.mark.parametrize(
    "source", [ORDERED_TEXTURE, ORDERED_VIEWS], ids=["texture", "views"]
)
def test_slang_and_hlsl_import_the_same_accesses(tmp_path, source):
    from_slang = _translate(tmp_path, source, "cgl")
    from_hlsl = _translate(tmp_path, source, "cgl", suffix=".hlsl")

    # The two importers parenthesize and cast differently; the operations they write are the same.
    def operations(crossgl):
        return [
            re.findall(
                r"\b(imageLoad|imageStore|buffer_load|buffer_store|blend)\(\s*(\w+)",
                line,
            )
            for line in _statements(crossgl)
        ]

    assert operations(from_slang) == operations(from_hlsl)
    for name in ("t1", "t2a", "t3", "sb", "bab", "accum"):
        if f" {name} " in source:
            assert "rasterizer_ordered" in _declaration(from_slang, name)[1]
            assert _declaration(from_slang, name)[0] == _declaration(from_hlsl, name)[0]


def test_every_rasterizer_ordered_view_keeps_its_ordering(tmp_path):
    crossgl = _translate(tmp_path, ORDERED_VIEWS, "cgl")

    assert [
        _declaration(crossgl, name)[0] for name in ("t1", "t2a", "t3", "sb", "bab")
    ] == [
        "image1D",
        "uimage2DArray",
        "iimage3D",
        "RWStructuredBuffer<float>",
        "RWByteAddressBuffer",
    ]
    assert "RasterizerOrdered" not in crossgl


def _compile_metal(tmp_path, code):
    if shutil.which("xcrun") is None:
        pytest.skip("the Metal toolchain is not installed")
    (tmp_path / "out.metal").write_text(code, encoding="utf-8")
    _run(
        tmp_path,
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-Werror",
            "-c",
            "out.metal",
            "-o",
            "out.air",
        ],
    )


def _compile_slang(tmp_path, code):
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    (tmp_path / "out.slang").write_text(code, encoding="utf-8")
    _run(
        tmp_path,
        [
            slangc,
            "out.slang",
            "-target",
            "spirv",
            "-entry",
            "main",
            "-stage",
            "fragment",
            "-o",
            "out.spv",
        ],
    )


def _compile_glsl(tmp_path, code):
    tool = shutil.which("glslangValidator")
    if tool is None:
        pytest.skip("glslangValidator is not installed")
    (tmp_path / "out.frag").write_text(code, encoding="utf-8")
    _run(tmp_path, [tool, "out.frag"])


def _compile_spirv(tmp_path, code):
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if assembler is None or validator is None:
        pytest.skip("SPIRV-Tools is not installed")
    (tmp_path / "out.spvasm").write_text(code, encoding="utf-8")
    _run(tmp_path, [assembler, "out.spvasm", "-o", "out.spv"])
    _run(tmp_path, [validator, "out.spv"])


def test_metal_binds_the_imported_views_in_a_raster_order_group(tmp_path):
    code = _translate(tmp_path, ORDERED_TEXTURE, "metal")

    assert re.search(
        r"accum\s*\[\[texture\(1\)\]\]\s*\[\[raster_order_group\(0\)\]\]", code
    )
    assert "accum.write(" in code and "accum.read(" in code
    _compile_metal(tmp_path, code)


def test_slang_output_declares_the_rasterizer_ordered_views(tmp_path):
    code = _translate(tmp_path, ORDERED_TEXTURE, "slang")

    assert "RasterizerOrderedTexture2D<float4> accum" in code
    assert "void blend(RasterizerOrderedTexture2D<float4> target" in code
    _compile_slang(tmp_path, code)


def test_glsl_output_orders_the_imported_view_with_fragment_interlock(tmp_path):
    code = _translate(tmp_path, ORDERED_TEXTURE, "opengl")

    assert "beginInvocationInterlockARB()" in code
    _compile_glsl(tmp_path, code)


def test_spirv_output_orders_the_imported_view_with_pixel_interlock(tmp_path):
    code = _translate(tmp_path, ORDERED_TEXTURE, "vulkan")

    assert "PixelInterlockOrderedEXT" in code
    _compile_spirv(tmp_path, code)


def test_hlsl_output_declares_the_rasterizer_ordered_view(tmp_path):
    code = _translate(tmp_path, ORDERED_TEXTURE, "directx")

    assert "RasterizerOrderedTexture2D<float4> accum" in code
