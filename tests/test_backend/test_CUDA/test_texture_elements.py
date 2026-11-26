"""CUDA and HIP texture fetches import with the element type they return, and ``tex1Dfetch`` reads a texel buffer.

A texture object names its element ``T`` at each fetch (``tex2D<int>(tex, x, y)``) and a texture reference in its
declaration (``texture<int, 1> ref``). The importers wrote every texture as a float sampler and every fetch as the
sampler's 4-component result, so ``tex1Dfetch<float>`` assigned a ``vec4<f32>`` to an ``f32`` and ``tex2D<int>``
read an integer texture through a float sampler. ``tex1Dfetch`` reads a texture bound to linear memory, which they
wrote as a 1D texture read at mip level 0: Metal, HLSL, GLSL and SPIR-V then declared a 1D image, not the texel
buffer the kernel reads. A texture object passed to a helper takes the type the helper's fetch gives it.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

LANGUAGES = {
    "cu": ("cudaTextureObject_t", "cuda", ""),
    "hip": ("hipTextureObject_t", "hip", "#include <hip/hip_runtime.h>\n"),
}

# The texel-buffer coverage probe.
TEXEL_BUFFER = """{include}extern "C" __global__ void probe(float* fin, float* outp, {texture} tex) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    outp[i] = tex1Dfetch<float4>(tex, int(fin[i]) & 7).x;
}}
"""

TYPED_FETCHES = """{include}extern "C" __global__ void probe(float* outp, int* iout, unsigned int* uout,
        {texture} ftex, {texture} itex, {texture} utex, {texture} stex, {texture} ctex) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float s = tex1Dfetch<float>(ftex, i);
    int4 v = tex1Dfetch<int4>(itex, i);
    uout[i] = tex1Dfetch<unsigned int>(utex, i);
    iout[i] = v.y + tex2D<int>(stex, 0.5f, 0.5f);
    float2 c = tex2D<float2>(ctex, 0.25f, 0.75f);
    outp[i] = s + c.y;
}}
"""

REFERENCES = """{include}texture<float4, {prefix}TextureType1D, {prefix}ReadModeElementType> fref;
texture<int, 1, {prefix}ReadModeElementType> iref;
texture<unsigned char, 1, {prefix}ReadModeNormalizedFloat> nref;
texture<float, 2> sref;
__global__ void probe(float* outp, int* iout) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    outp[i] = tex1Dfetch(fref, i).x + tex1Dfetch(nref, i) + tex2D(sref, 0.5f, 0.5f);
    iout[i] = tex1Dfetch(iref, i);
}}
"""

HELPERS = """{include}__device__ unsigned int fetch({texture} t, int i) {{ return tex1Dfetch<unsigned int>(t, i); }}
__device__ unsigned int twice({texture} t, int i) {{ return fetch(t, i) + fetch(t, i + 1); }}
__global__ void probe(unsigned int* outp, {texture} tex) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    outp[i] = twice(tex, i);
}}
"""


def _source(tmp_path, suffix, template):
    texture, prefix, include = LANGUAGES[suffix]
    path = tmp_path / f"probe.{suffix}"
    path.write_text(template.format(include=include, texture=texture, prefix=prefix))
    return str(path)


def _import(tmp_path, suffix, template):
    code = crosstl.translate(_source(tmp_path, suffix, template), backend="cgl")
    Parser(Lexer(code).tokens).parse()
    return code


@pytest.mark.parametrize("suffix", sorted(LANGUAGES))
def test_texel_fetch_reads_a_texel_buffer(tmp_path, suffix):
    code = _import(tmp_path, suffix, TEXEL_BUFFER)

    assert "samplerBuffer tex" in code
    assert "texelFetch(tex, (i32(fin[i]) & 7)).x" in code
    assert "sampler1D" not in code


@pytest.mark.parametrize("suffix", sorted(LANGUAGES))
def test_fetches_return_their_element_type(tmp_path, suffix):
    code = _import(tmp_path, suffix, TYPED_FETCHES)

    for declaration in (
        "samplerBuffer ftex",
        "isamplerBuffer itex",
        "usamplerBuffer utex",
        "isampler2D stex",
        "sampler2D ctex",
    ):
        assert declaration in code
    assert "var s: f32 = texelFetch(ftex, i).x;" in code
    assert "var v: vec4<i32> = texelFetch(itex, i);" in code
    assert "uout[i] = texelFetch(utex, i).x;" in code
    assert "texture(stex, vec2<f32>(0.5f, 0.5f)).x" in code
    assert "var c: vec2<f32> = texture(ctex, vec2<f32>(0.25f, 0.75f)).xy;" in code


@pytest.mark.parametrize("suffix", sorted(LANGUAGES))
def test_texture_references_take_their_declared_element(tmp_path, suffix):
    code = _import(tmp_path, suffix, REFERENCES)

    # tex1Dfetch reads linear memory; a normalized-float read of an 8-bit texture returns a float.
    assert "var fref: samplerBuffer;" in code
    assert "var iref: isamplerBuffer;" in code
    assert "var nref: samplerBuffer;" in code
    assert "var sref: sampler2D;" in code
    assert "texelFetch(fref, i).x" in code
    assert "texelFetch(nref, i).x" in code
    assert "texture(sref, vec2<f32>(0.5f, 0.5f)).x" in code
    assert "iout[i] = texelFetch(iref, i).x;" in code


@pytest.mark.parametrize("suffix", sorted(LANGUAGES))
def test_object_passed_to_helpers_takes_the_helper_fetch_type(tmp_path, suffix):
    code = _import(tmp_path, suffix, HELPERS)
    texture = LANGUAGES[suffix][0]

    assert "u32 fetch(usamplerBuffer t, i32 i)" in code
    assert "u32 twice(usamplerBuffer t, i32 i)" in code
    assert "usamplerBuffer tex" in code
    assert texture not in code


def test_helpers_that_read_an_object_differently_leave_it_untyped(tmp_path):
    path = tmp_path / "probe.cu"
    path.write_text(
        """__device__ float a(cudaTextureObject_t t, int i) { return tex1Dfetch<float>(t, i); }
__device__ int b(cudaTextureObject_t t, int i) { return tex1Dfetch<int>(t, i); }
__global__ void probe(float* outp, cudaTextureObject_t tex) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    outp[i] = a(tex, i) + b(tex, i);
}
"""
    )
    code = crosstl.translate(str(path), backend="cgl")

    assert "f32 a(samplerBuffer t, i32 i)" in code
    assert "i32 b(isamplerBuffer t, i32 i)" in code
    assert "cudaTextureObject_t tex" in code


@pytest.mark.parametrize("suffix", sorted(LANGUAGES))
def test_texel_buffer_targets(tmp_path, suffix):
    source = _source(tmp_path, suffix, TYPED_FETCHES)

    metal = crosstl.translate(source, backend="metal")
    assert "texture_buffer<float, access::read> ftex" in metal
    assert "texture_buffer<int, access::read> itex" in metal
    assert "texture_buffer<uint, access::read> utex" in metal
    assert "texture2d<int> stex" in metal
    assert "float s = ftex.read(uint(i)).x;" in metal

    hlsl = crosstl.translate(source, backend="directx")
    for declaration in (
        "Buffer<float4> ftex",
        "Buffer<int4> itex",
        "Buffer<uint4> utex",
    ):
        assert declaration in hlsl

    glsl = crosstl.translate(source, backend="opengl")
    for declaration in (
        "uniform samplerBuffer ftex",
        "uniform isamplerBuffer itex",
        "uniform usamplerBuffer utex",
    ):
        assert declaration in glsl


@pytest.mark.skipif(
    shutil.which("glslangValidator") is None
    or shutil.which("spirv-as") is None
    or shutil.which("spirv-val") is None,
    reason="glslangValidator, spirv-as and spirv-val are not on PATH",
)
@pytest.mark.parametrize("template", [TEXEL_BUFFER, TYPED_FETCHES, REFERENCES])
def test_texel_buffer_targets_validate(tmp_path, template):
    source = _source(tmp_path, "cu", template)
    glsl = tmp_path / "k.comp"
    glsl.write_text(crosstl.translate(source, backend="opengl"))
    subprocess.run(
        ["glslangValidator", "-S", "comp", str(glsl)], check=True, capture_output=True
    )
    spvasm = tmp_path / "k.spvasm"
    spvasm.write_text(crosstl.translate(source, backend="vulkan"))
    spv = tmp_path / "k.spv"
    subprocess.run(
        ["spirv-as", "--target-env", "vulkan1.0", str(spvasm), "-o", str(spv)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["spirv-val", "--target-env", "vulkan1.0", str(spv)],
        check=True,
        capture_output=True,
    )
