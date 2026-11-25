"""CUDA and HIP surfaces import as CrossGL storage images whose format holds the texels they are accessed as.

A surface holds raw bytes: ``surf2Dread<T>(s, x, y)`` and ``surf2Dread(&v, s, x, y)`` read ``sizeof(T)`` bytes at
byte offset ``x``, and ``surf2Dwrite(v, s, x, y)`` writes the bytes of ``v`` (CUDA C++ Programming Guide, "Surface
Object API"). A CrossGL storage image holds texels of one format and takes a texel coordinate, so the importers
give each surface the format of the one element type its accesses name (``float`` -> ``r32f``, ``uchar4`` ->
``rgba8ui``), divide ``x`` by ``sizeof(T)``, and refuse a surface accessed as two element types or as a type no
format holds.
"""

import pytest

import crosstl
from crosstl.backend.HIP.HipCrossGLCodeGen import HipSurfaceByteOffsetError
from crosstl.backend.import_refusals import CudaImportRefused, HipImportRefused
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

LANGUAGES = {
    "cu": ("cudaSurfaceObject_t", CudaImportRefused, ""),
    "hip": ("hipSurfaceObject_t", HipImportRefused, "#include <hip/hip_runtime.h>\n"),
}

# The read-write-image coverage probe: the pointer form of surf2Dread, then a write of the same surface.
READ_WRITE = """{include}extern "C" __global__ void probe(float* fin, float* outp, {surface} img) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v;
    surf2Dread(&v, img, 0, 0);
    surf2Dwrite(v + fin[i], img, 0, 0);
    outp[i] = 0.0f;
}}
"""


def _source(tmp_path, suffix, template, **fields):
    surface, _, include = LANGUAGES[suffix]
    path = tmp_path / f"probe.{suffix}"
    path.write_text(template.format(include=include, surface=surface, **fields))
    return str(path)


def _import(tmp_path, suffix, template, **fields):
    code = crosstl.translate(
        _source(tmp_path, suffix, template, **fields), backend="cgl"
    )
    Parser(Lexer(code).tokens).parse()
    return code


@pytest.mark.parametrize("suffix", sorted(LANGUAGES))
def test_pointer_form_read_assigns_the_texel(tmp_path, suffix):
    code = _import(tmp_path, suffix, READ_WRITE)

    assert "@format(r32f) image2D img" in code
    assert "v = imageLoad(img, vec2<i32>(0, 0)).x;" in code
    assert "imageStore(img, vec2<i32>(0, 0), (v + fin[i]));" in code
    assert "surf2D" not in code


@pytest.mark.parametrize("suffix", sorted(LANGUAGES))
@pytest.mark.parametrize(
    ("backend", "declaration"),
    [
        ("opengl", "layout(r32f, CROSSGL_BINDING(0, 2, 0)) uniform image2D img;"),
        ("metal", "texture2d<float, access::read_write> img"),
        ("directx", "RWTexture2D<float> img : register(u2);"),
        # SPIR-V ids are names only where SPIRV-Tools is installed; the image type carries the format either way.
        ("vulkan", "2D 0 0 0 2 R32f"),
    ],
)
def test_read_write_surface_reaches_every_target(
    tmp_path, suffix, backend, declaration
):
    code = crosstl.translate(_source(tmp_path, suffix, READ_WRITE), backend=backend)

    assert declaration in code
    assert "surf2D" not in code


ELEMENT = """{include}extern "C" __global__ void probe(float* outp, {surface} img, int x) {{
    {element} v;
    surf2Dread(&v, img, x * sizeof({element}), 0);
    surf2Dwrite(v, img, x * sizeof({element}), 1);
    outp[0] = 0.0f;
}}
"""


@pytest.mark.parametrize("suffix", sorted(LANGUAGES))
@pytest.mark.parametrize(
    ("element", "declaration", "load", "store"),
    [
        (
            "float",
            "@format(r32f) image2D img",
            "imageLoad(img, vec2<i32>(x, 0)).x",
            "v",
        ),
        (
            "float2",
            "@format(rg32f) image2D img",
            "imageLoad(img, vec2<i32>(x, 0)).xy",
            None,
        ),
        (
            "float4",
            "@format(rgba32f) image2D img",
            "imageLoad(img, vec2<i32>(x, 0))",
            "v",
        ),
        ("int", "@format(r32i) iimage2D img", "imageLoad(img, vec2<i32>(x, 0)).x", "v"),
        (
            "uint2",
            "@format(rg32ui) uimage2D img",
            "imageLoad(img, vec2<i32>(x, 0)).xy",
            None,
        ),
        (
            "uchar4",
            "@format(rgba8ui) uimage2D img",
            "vec4<u8>(imageLoad(img, vec2<i32>(x, 0)))",
            "vec4<u32>(v)",
        ),
        (
            "short2",
            "@format(rg16i) iimage2D img",
            "vec2<i16>(imageLoad(img, vec2<i32>(x, 0)).xy)",
            None,
        ),
        (
            "unsigned short",
            "@format(r16ui) uimage2D img",
            "u16(imageLoad(img, vec2<i32>(x, 0)).x)",
            "u32(v)",
        ),
    ],
)
def test_element_type_sets_the_format(
    tmp_path, suffix, element, declaration, load, store
):
    code = _import(tmp_path, suffix, ELEMENT, element=element)

    assert declaration in code
    # x is a byte offset: sizeof(T) divides out, leaving the texel column.
    assert f"v = {load};" in code
    if store is not None:
        assert f"imageStore(img, vec2<i32>(x, 1), {store});" in code
    else:
        # A two-component element fills blue with 0 and alpha with 1.
        assert "imageStore(img, vec2<i32>(x, 1), vec4<" in code
    assert "sizeof" not in code


def test_two_component_store_fills_blue_and_alpha(tmp_path):
    code = _import(tmp_path, "cu", ELEMENT, element="float2")

    assert "imageStore(img, vec2<i32>(x, 1), vec4<f32>(v, 0.0, 1.0));" in code


GLOBAL = """{include}__device__ {surface} gimg;
extern "C" __global__ void probe(float* outp, int x) {{
    short2 b = surf2Dread<short2>(gimg, x * 4, 1);
    surf2Dwrite(b, gimg, x * 4, 1);
    outp[0] = (float)b.x;
}}
"""


@pytest.mark.parametrize("suffix", sorted(LANGUAGES))
def test_device_surface_handle_is_a_bound_image(tmp_path, suffix):
    # A __device__ surface handle is set by the host, as a bound image is: not a storage buffer of an image.
    code = _import(tmp_path, suffix, GLOBAL)

    assert "@format(rg16i) var gimg: iimage2D;" in code
    assert "var<storage" not in code.split("fn probe")[0]
    assert "var b: vec2<i16> = vec2<i16>(imageLoad(gimg, vec2<i32>(x, 1)).xy);" in code


REFERENCE = """{include}surface<void, {kind}> sref;
extern "C" __global__ void probe(float* outp, int x) {{
    float c = surf2Dread<float>(sref, x * 4, 2);
    surf2Dwrite(c + 1.0f, sref, x * 4, 2);
    outp[0] = c;
}}
"""

KINDS = {"cu": "cudaSurfaceType2D", "hip": "hipSurfaceType2D"}


@pytest.mark.parametrize("suffix", sorted(LANGUAGES))
def test_surface_reference_keeps_the_format(tmp_path, suffix):
    code = _import(tmp_path, suffix, REFERENCE, kind=KINDS[suffix])

    assert "@format(r32f) var sref: image2D;" in code
    assert "var c: f32 = imageLoad(sref, vec2<i32>(x, 2)).x;" in code
    # Both byte offsets divide by sizeof(float).
    assert "imageStore(sref, vec2<i32>(x, 2), (c + 1.0f));" in code


@pytest.mark.parametrize("suffix", sorted(LANGUAGES))
@pytest.mark.parametrize(
    ("backend", "declaration"),
    [
        ("opengl", "uniform iimage2D gimg;"),
        ("metal", "texture2d<int, access::read_write> gimg"),
        ("directx", "RWTexture2D<int2> gimg"),
        ("vulkan", "2D 0 0 0 2 Rg16i"),
    ],
)
def test_device_surface_handle_reaches_every_target(
    tmp_path, suffix, backend, declaration
):
    code = crosstl.translate(_source(tmp_path, suffix, GLOBAL), backend=backend)

    assert declaration in code


SHADOW = """{include}__device__ {surface} img;
__device__ float helper({surface} img, int x) {{
    return surf2Dread<float>(img, x * 4, 0);
}}
extern "C" __global__ void probe(float* outp, int x) {{
    uint4 v = surf2Dread<uint4>(img, x * 16, 0);
    outp[0] = helper(img, x) + (float)v.x;
}}
"""


@pytest.mark.parametrize("suffix", sorted(LANGUAGES))
def test_parameter_shadows_the_global_surface(tmp_path, suffix):
    # The helper's parameter is its own surface: its float accesses do not conflict with the global's uint4.
    code = _import(tmp_path, suffix, SHADOW)

    assert "@format(rgba32ui)" in code
    assert "@format(r32f) image2D img" in code


REFUSED = """{include}extern "C" __global__ void probe(float* outp, {surface} img, int x) {{
    {body}
    outp[0] = 0.0f;
}}
"""


@pytest.mark.parametrize("suffix", sorted(LANGUAGES))
@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (
            "float a = surf2Dread<float>(img, x * 4, 0);\n"
            "    float4 b = surf2Dread<float4>(img, x * 16, 0);",
            "accessed as float and float4",
        ),
        ("double a = surf2Dread<double>(img, x * 8, 0);", "64-bit components"),
        ("float3 a = surf2Dread<float3>(img, x * 12, 0);", "three components"),
        ("float a = surf2Dread(img, x * 4, 0);", "type of the value it reads"),
        ("float a = surf2Dread<float>(img, x * 3, 0);", "not provably a multiple"),
    ],
)
def test_surfaces_without_one_format_are_refused(tmp_path, suffix, body, reason):
    refusal = LANGUAGES[suffix][1]
    if suffix == "hip" and "provably" in reason:
        # HIP reports an unproven byte offset with its own diagnostic code.
        refusal = HipSurfaceByteOffsetError
    with pytest.raises(refusal, match=reason):
        crosstl.translate(_source(tmp_path, suffix, REFUSED, body=body), backend="cgl")
