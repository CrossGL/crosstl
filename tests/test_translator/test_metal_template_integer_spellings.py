"""Metal imports keep the meaning of multi-word integer spellings and reference arguments.

MSL (section 2.1) spells ``uint`` also as ``unsigned int`` and inherits C++'s ``signed``/``unsigned`` forms.
The type parser folded them at the top level of a declaration, but not inside template arguments, so
``texture2d<unsigned int>`` reached the CrossGL parser unchanged and the translation failed with a syntax error.

A scalar fragment argument passed by reference, ``constant float &bias [[buffer(n)]]``, names a ``float`` value
([dcl.ref]); Metal's LOD and bias checks read its declared ``float&`` type and refused it as a non-scalar argument.
"""

import shutil
import subprocess

import pytest

import crosstl

UNSIGNED_TEXTURE = """
#include <metal_stdlib>
using namespace metal;

kernel void copy_texels(texture2d<unsigned int> source [[texture(0)]],
                        texture2d<signed int, access::read> offsets [[texture(1)]],
                        device uint *destination [[buffer(2)]],
                        uint2 gid [[thread_position_in_grid]]) {
    vec<unsigned short, 2> narrow = vec<unsigned short, 2>(gid);
    uint value = source.read(gid).x + uint(offsets.read(gid).x);
    destination[gid.y * 16 + gid.x] = value + uint(narrow.x);
}
"""

REFERENCE_LOD = """
#include <metal_stdlib>
using namespace metal;

struct Varyings {
    float4 position [[position]];
    float2 uv;
};

fragment half4 shade(Varyings in [[stage_in]],
                     texture2d<half> image [[texture(0)]],
                     sampler linear [[sampler(0)]],
                     constant float &mipBias [[buffer(0)]]) {
    return image.sample(linear, in.uv, level(mipBias)) + image.sample(linear, in.uv, bias(mipBias));
}
"""


def _metal(tmp_path, source, name):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return str(path)


def _compile_metal(tmp_path, source):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return
    probe = subprocess.run(
        [xcrun, "-sdk", "macosx", "-f", "metal"], capture_output=True
    )
    if probe.returncode != 0:
        return
    shader = tmp_path / "output.metal"
    shader.write_text(source, encoding="utf-8")
    completed = subprocess.run(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-Werror",
            "-c",
            str(shader),
            "-o",
            str(tmp_path / "output.air"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "spelling, canonical",
    [
        ("unsigned int", "uint"),
        ("unsigned", "uint"),
        ("unsigned short", "ushort"),
        ("unsigned char", "uchar"),
        ("unsigned long", "ulong"),
        ("signed int", "int"),
        ("signed char", "char"),
        ("signed", "int"),
    ],
)
def test_template_arguments_fold_signed_spellings(spelling, canonical):
    from crosstl.backend.Metal.signed_type_spellings import fold_signed_type_spellings

    assert fold_signed_type_spellings(spelling.split()) == [canonical]
    assert fold_signed_type_spellings(
        ["vec", "<", *spelling.split(), ",", "2", ">"]
    ) == [
        "vec",
        "<",
        canonical,
        ",",
        "2",
        ">",
    ]


def test_unsigned_texture_imports_as_an_unsigned_texture(tmp_path):
    crossgl = crosstl.translate(
        _metal(tmp_path, UNSIGNED_TEXTURE, "copy.metal"), backend="cgl"
    )
    assert "utexture2D source @texture(0)" in crossgl
    assert "iimage2D offsets @texture(1) @readonly" in crossgl
    assert "u16vec2 narrow = u16vec2(gid);" in crossgl
    assert "unsigned" not in crossgl
    assert "signed int" not in crossgl


@pytest.mark.parametrize("backend", ["metal", "spirv", "directx", "opengl"])
def test_unsigned_texture_translates(tmp_path, backend):
    output = crosstl.translate(
        _metal(tmp_path, UNSIGNED_TEXTURE, "copy.metal"), backend=backend
    )
    assert output.strip()
    if backend == "metal":
        assert "texture2d<uint" in output
        assert "texture2d<int" in output
        _compile_metal(tmp_path, output)


def test_reference_argument_is_a_scalar_lod_and_bias(tmp_path):
    output = crosstl.translate(
        _metal(tmp_path, REFERENCE_LOD, "shade.metal"), backend="metal"
    )
    assert "level(mipBias)" in output
    assert "bias(mipBias)" in output
    _compile_metal(tmp_path, output)
