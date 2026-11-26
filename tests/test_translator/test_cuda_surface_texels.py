"""Storage images in CUDA and HIP output read and write their texels as memory holds them.

A CUDA or HIP surface holds raw bytes: ``surf2Dread<T>`` reads ``sizeof(T)`` bytes at a byte offset and
``surf2Dwrite`` writes the bytes of its value (CUDA C++ Programming Guide, "Surface Memory"). The generators wrote
every float image as ``float4``, so an ``r32f`` image was read 16 bytes at a time at offsets 16 bytes apart, an
``rgba8`` image written with 16-byte texels, and ``rgba16f`` read as 32-bit floats. The surface element now comes
from the image's format or ``@texel`` type; a load widens it to the four components ``imageLoad`` returns, filling
green and blue with 0 and alpha with 1, and normalizes unorm, snorm and half channels, and a store converts back.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.cuda_surface_texels import (
    SurfaceTexel,
    UnsupportedSurfaceTexel,
    cuda_value_width,
    format_surface_texel,
    texel_type_surface_texel,
)
from crosstl.translator.codegen.lowering_refusals import LoweringRefused

BACKENDS = ["cuda", "hip"]
READ = {"cuda": "surf2Dread", "hip": "cgl_surf2Dread"}

PROGRAM = """
shader Probe {{
    compute {{
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer float* outp;
        {declaration}
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            ivec2 p = ivec2(int(i), 0);
            {body}
            outp[i] = 0.0;
        }}
    }}
}}
"""

COPY = "{texel} v = imageLoad(img, p);\nimageStore(img, p + ivec2(1, 0), v);"


def _translate(tmp_path, backend, declaration, body):
    path = tmp_path / "images.cgl"
    path.write_text(PROGRAM.format(declaration=declaration, body=body))
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _compile(tmp_path, code, backend):
    """Compile ``code`` with the target's own compiler when it is installed."""
    if backend == "cuda" and shutil.which("nvcc"):
        (tmp_path / "out.cu").write_text(code)
        command = [shutil.which("nvcc"), "-std=c++17", "--ptx", "out.cu"]
    elif backend == "hip" and shutil.which("hipcc"):
        (tmp_path / "out.hip").write_text(code)
        command = [
            shutil.which("hipcc"),
            "-std=c++17",
            "--offload-device-only",
            "-c",
            "out.hip",
        ]
    else:
        return
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=tmp_path, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("declaration", "texel", "element", "helper"),
    [
        ("layout(r32f) uniform image2D img;", "vec4", "float", "cgl_r32f_texel"),
        ("layout(rg32f) uniform image2D img;", "vec4", "float2", "cgl_rg32f_texel"),
        ("layout(rgba32f) uniform image2D img;", "vec4", "float4", None),
        ("layout(r32i) uniform iimage2D img;", "ivec4", "int", "cgl_r32i_texel"),
        ("layout(rgba32ui) uniform uimage2D img;", "uvec4", "uint4", None),
        ("layout(rgba8) uniform image2D img;", "vec4", "uchar4", "cgl_rgba8_texel"),
        (
            "layout(rg8_snorm) uniform image2D img;",
            "vec4",
            "char2",
            "cgl_rg8_snorm_texel",
        ),
        ("layout(r16) uniform image2D img;", "vec4", "unsigned short", "cgl_r16_texel"),
        (
            "layout(rgba16f) uniform image2D img;",
            "vec4",
            "ushort4",
            "cgl_rgba16f_texel",
        ),
        ("@format(r32f) image2D img;", "vec4", "float", "cgl_r32f_texel"),
        (
            "@format(unknown) @texel(ivec2) iimage2D img;",
            "ivec4",
            "int2",
            "cgl_texel_ivec2_texel",
        ),
    ],
)
def test_surface_element_is_the_format_texel(
    tmp_path, backend, declaration, texel, element, helper
):
    code = _translate(tmp_path, backend, declaration, COPY.format(texel=texel))
    read = f"{READ[backend]}<{element}>(img, p.x * sizeof({element}), p.y)"
    if helper is None:
        assert f"= {read};" in code
    else:
        assert f"= {helper}({read});" in code
    write = re.search(r"surf2Dwrite\(([^;]*)\);", code)
    assert write is not None, code
    value = "v" if helper is None else helper.removesuffix("_texel") + "_element(v)"
    assert write.group(1).startswith(f"{value}, img, "), write.group(1)
    assert f".x * sizeof({element})" in write.group(1)
    _compile(tmp_path, code, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_image_without_format_keeps_four_float_components(tmp_path, backend):
    code = _translate(
        tmp_path, backend, "uniform image2D img;", COPY.format(texel="vec4")
    )
    assert f"{READ[backend]}<float4>(img, p.x * sizeof(float4), p.y)" in code


def test_conversions_follow_the_format():
    rgba8 = format_surface_texel("rgba8")
    _, load = rgba8.load_helper()
    _, store = rgba8.store_helper()
    assert "c.x / 255.0f" in load and "c.w / 255.0f" in load
    assert "(unsigned char)__float2uint_rn(__saturatef(v.x) * 255.0f)" in store
    snorm = format_surface_texel("r16_snorm")
    _, load = snorm.load_helper()
    _, store = snorm.store_helper()
    assert "fmaxf(c / 32767.0f, -1.0f)" in load
    assert "(short)__float2int_rn(fminf(fmaxf(v.x, -1.0f), 1.0f) * 32767.0f)" in store
    half = format_surface_texel("rg16f")
    _, load = half.load_helper()
    _, store = half.store_helper()
    assert "__half2float(__ushort_as_half(c.y))" in load
    assert "__half_as_ushort(__float2half_rn(v.x))" in store
    # Green and blue a format lacks read as 0, alpha as 1.
    _, load = format_surface_texel("rg32i").load_helper()
    assert "make_int4(c.x, c.y, 0, 1)" in load
    _, load = format_surface_texel("r32ui").load_helper()
    assert "make_uint4(c, 0u, 0u, 1u)" in load


def test_narrow_integer_channels_extend_and_keep_low_bits():
    # A load zero- or sign-extends each channel to the 32-bit integer of its signedness; a store keeps the low
    # bits, one of the values Vulkan allows for an integer the format cannot represent ("Texel Output Format
    # Conversion").
    r8ui = format_surface_texel("r8ui")
    assert (r8ui.kind, r8ui.storage, r8ui.count) == ("uint", "unsigned char", 1)
    _, load = r8ui.load_helper()
    _, store = r8ui.store_helper()
    assert "make_uint4((unsigned int)c, 0u, 0u, 1u)" in load
    assert "(unsigned char)v.x" in store
    rgba16i = format_surface_texel("rgba16i")
    assert (rgba16i.kind, rgba16i.storage, rgba16i.count) == ("int", "short", 4)
    _, load = rgba16i.load_helper()
    _, store = rgba16i.store_helper()
    assert "(int)c.x" in load and "(int)c.w" in load
    assert "(short)v.x" in store and "(short)v.w" in store


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("declaration", "texel", "element", "helper"),
    [
        (
            "layout(r8ui) uniform uimage2D img;",
            "uvec4",
            "unsigned char",
            "cgl_r8ui_texel",
        ),
        ("layout(rg8i) uniform iimage2D img;", "ivec4", "char2", "cgl_rg8i_texel"),
        (
            "layout(rgba16i) uniform iimage2D img;",
            "ivec4",
            "short4",
            "cgl_rgba16i_texel",
        ),
        (
            "layout(r16ui) uniform uimage2D img;",
            "uvec4",
            "unsigned short",
            "cgl_r16ui_texel",
        ),
    ],
)
def test_narrow_integer_formats_read_and_write_their_element(
    tmp_path, backend, declaration, texel, element, helper
):
    code = _translate(tmp_path, backend, declaration, COPY.format(texel=texel))
    read = f"{READ[backend]}<{element}>(img, p.x * sizeof({element}), p.y)"
    assert f"= {helper}({read});" in code
    write = re.search(r"surf2Dwrite\(([^;]*)\);", code)
    assert write is not None, code
    value = helper.removesuffix("_texel") + "_element(v)"
    assert write.group(1).startswith(f"{value}, img, "), write.group(1)
    _compile(tmp_path, code, backend)


def test_texel_types_and_widths():
    assert texel_type_surface_texel(["uvec4"]) == SurfaceTexel(
        "texel_uvec4", "uint", 4, "uint"
    )
    assert isinstance(texel_type_surface_texel(["vec3"]), UnsupportedSurfaceTexel)
    assert isinstance(
        texel_type_surface_texel(["unorm", "vec4"]), UnsupportedSurfaceTexel
    )
    assert cuda_value_width("uint") == 1
    assert cuda_value_width("unsigned int") == 1
    assert cuda_value_width("float2") == 2
    assert cuda_value_width("int4") == 4
    assert cuda_value_width("MyStruct") is None


@pytest.mark.parametrize("backend", BACKENDS)
def test_narrow_contexts_load_leading_components(tmp_path, backend):
    body = (
        "uint count = imageLoad(img, p);\n"
        "imageStore(img, p, count + 1u);\n"
        "uint first = imageLoad(img, p).x;\n"
        "uint again;\n"
        "again = imageLoad(img, p);\n"
        "outp[i] = float(count + first + again);"
    )
    code = _translate(tmp_path, backend, "layout(r32ui) uniform uimage2D img;", body)
    read = f"{READ[backend]}<uint>(img, p.x * sizeof(uint), p.y)"
    # A scalar context loads the one channel itself; a stored scalar is the element.
    assert f"count = {read};" in code
    assert f"again = {read};" in code
    assert re.search(r"surf2Dwrite\(\(?count \+ 1u\)?, img,", code), code
    # The four-component value of a swizzled load widens the element first.
    assert f"first = cgl_r32ui_texel({read}).x;" in code
    _compile(tmp_path, code, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_stored_scalar_is_splatted_into_a_converted_element(tmp_path, backend):
    body = "imageStore(img, p, fin[i]);"
    code = _translate(tmp_path, backend, "layout(rgba8) uniform image2D img;", body)
    assert "cgl_rgba8_element1(fin[i])" in code
    helper = re.search(
        r"uchar4 cgl_rgba8_element1\(float v\)\s*\{\s*return (.*?);", code, re.S
    )
    assert helper is not None, code
    assert helper.group(1).count("__saturatef(v)") == 4
    _compile(tmp_path, code, backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("declaration", "reason"),
    [
        ("layout(rgb10_a2) uniform image2D img;", "bit fields"),
        ("@format(unknown) @texel(vec3) image2D img;", "three components"),
    ],
)
def test_formats_without_a_surface_element_are_refused(
    tmp_path, backend, declaration, reason
):
    texel = (
        "uvec4"
        if "uimage" in declaration
        else "ivec4" if "iimage" in declaration else "vec4"
    )
    with pytest.raises(LoweringRefused, match=reason):
        _translate(tmp_path, backend, declaration, COPY.format(texel=texel))
