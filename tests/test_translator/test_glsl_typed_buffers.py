"""CrossGL ``Buffer<T>``/``RWBuffer<T>`` in GLSL output are texel buffers (#3642).

GLSL output wrote ``uniform Buffer<float> values;``, which glslang rejects. The typed buffers now lower, through the
same rewrite Metal output uses, to ``samplerBuffer`` and ``imageBuffer`` declarations: element reads become
``texelFetch``/``imageLoad`` narrowed to ``T``, stores ``imageStore`` and element atomics the image atomics. A
storage image that a program function updates atomically now also gets the single-channel format image atomics
need, through any depth of calls, so HLSL helpers that take an ``RWBuffer`` or ``RWTexture2D`` compile as well.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.glsl_refusals import OpenGLLoweringRefused

KERNEL = """
shader TypedBuffers {
    Buffer<float> values;
    Buffer<int2> pairs;
    RWBuffer<float4> rgba;
    RWBuffer<uint> counts;
    RWBuffer<uint> total;
    RWBuffer<half2> halves;
    RWStructuredBuffer<float> outp;
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = values[i] * 2.0 + float(pairs[i].y - pairs[i].x) + float(buffer_dimensions(values));
            rgba[i] = vec4(values[i], float(pairs[i].x), 2.0, 3.0);
            rgba[i].y += 4.0;
            rgba[i] *= 2.0;
            counts[i]++;
            atomicAdd(total[0], 1u);
            halves[i] = halves[i] + half2(1.0, 2.0);
        }
    }
}
"""


def _translate(tmp_path, source, backend="opengl", name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def _validate(tmp_path, glsl, stage):
    """glslang accepts ``glsl`` for OpenGL and, with ``VULKAN`` defined, for Vulkan."""
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")
    path = tmp_path / f"shader.{stage}"
    path.write_text(glsl, encoding="utf-8")
    for flags in ([], ["-V", "-o", str(tmp_path / "shader.spv")]):
        result = subprocess.run(
            [glslang, *flags, "-S", stage, str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def _declaration(glsl, name):
    match = re.search(rf"^.*\buniform\s+\w+\s+{name};$", glsl, re.MULTILINE)
    assert match, glsl
    return match.group(0)


def test_typed_buffers_lower_to_glsl_texel_buffers(tmp_path):
    glsl = _translate(tmp_path, KERNEL)
    assert "Buffer<" not in glsl
    assert _declaration(glsl, "values").endswith("uniform samplerBuffer values;")
    assert _declaration(glsl, "pairs").endswith("uniform isamplerBuffer pairs;")
    assert _declaration(glsl, "rgba").endswith("uniform imageBuffer rgba;")
    assert _declaration(glsl, "counts").endswith("uniform uimageBuffer counts;")
    assert _declaration(glsl, "halves").endswith("uniform imageBuffer halves;")
    # Image atomics accept only single-channel 32-bit formats.
    assert "layout(r32ui" in _declaration(glsl, "total")
    assert "texelFetch(values, int(i)).x" in glsl
    assert "texelFetch(pairs, int(i)).xy.y" in glsl
    assert "uint(textureSize(values))" in glsl
    assert "imageAtomicAdd(total, 0, 1u);" in glsl
    assert re.search(
        r"imageStore\(counts, int\(i\), uvec4\(crosstl_texel_\d+, 0u, 0u, 0u\)\);", glsl
    )
    _validate(tmp_path, glsl, "comp")


def test_typed_buffer_helpers_and_dimensions_validate(tmp_path):
    source = """
shader P {
    Buffer<float4> src;
    RWBuffer<int> dst;
    RWStructuredBuffer<float> outp;
    float fetch(Buffer<float4> b, uint i) { return b[i].z; }
    void put(RWBuffer<int> d, uint i, int v) { d[i] = v; atomicMax(d[i], v + 1); }
    compute {
        layout(local_size_x = 8, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = fetch(src, i);
            put(dst, i, int(i));
            uint n;
            buffer_dimensions(dst, n);
            outp[0] = float(n);
        }
    }
}
"""
    glsl = _translate(tmp_path, source)
    assert "float fetch(samplerBuffer b, uint i)" in glsl
    assert "texelFetch(b, int(i)).z" in glsl
    # The helper updates ``dst`` atomically, so ``dst`` takes the format image atomics need.
    assert "layout(r32i" in _declaration(glsl, "dst")
    assert "n = uint(imageSize(dst));" in glsl
    _validate(tmp_path, glsl, "comp")


def test_rasterizer_ordered_typed_buffer_runs_inside_the_fragment_interlock(tmp_path):
    source = """
shader P {
    RasterizerOrderedBuffer<uint> counts;
    fragment {
        vec4 main(vec4 pos @ gl_FragCoord) @ gl_FragColor {
            uint k = uint(pos.x);
            counts[k] += 1u;
            return vec4(float(counts[k]));
        }
    }
}
"""
    glsl = _translate(tmp_path, source)
    assert "#extension GL_ARB_fragment_shader_interlock : require" in glsl
    assert "uniform uimageBuffer counts;" in glsl
    begin = glsl.index("beginInvocationInterlockARB();")
    assert (
        begin
        < glsl.index("imageStore(counts")
        < glsl.index("endInvocationInterlockARB();")
    )
    _validate(tmp_path, glsl, "frag")


def test_webgl_refuses_typed_buffers_with_a_typed_diagnostic(tmp_path):
    source = """
shader P {
    Buffer<float4> colors;
    fragment {
        vec4 main(vec4 pos @ gl_FragCoord) @ gl_FragColor {
            return colors[uint(pos.x)];
        }
    }
}
"""
    # WebGL 2 is GLSL ES 3.00, which has no texel buffers (GLSL ES 3.20 4.1.7.1 adds samplerBuffer).
    with pytest.raises(Exception, match=r"'samplerBuffer' requires GLSL ES 3\.20"):
        _translate(tmp_path, source, backend="webgl")
    glsl = _translate(tmp_path, source)
    assert "fragColor = texelFetch(colors, int(uint(gl_FragCoord.x)));" in glsl
    _validate(tmp_path, glsl, "frag")


REFUSED = {
    "read_only_store": ("Buffer<float> v;", "v[i] = 1.0;", "store to the read-only"),
    "double": (
        "Buffer<double> v;",
        "outp[i] = float(v[i]);",
        "GLSL texel buffers hold texels of one to four 16- or 32-bit",
    ),
    "side_effects": (
        "RWBuffer<float> v;",
        "v[i++] += 1.0;",
        "whose buffer or index has side effects",
    ),
    "nested_store": (
        "RWBuffer<float> v;",
        "outp[i] = (v[i] = 2.0);",
        "inside a larger expression",
    ),
    "local_alias": (
        "RWBuffer<float> v;",
        "RWBuffer<float> w = v; w[i] = 1.0;",
        "used other than through",
    ),
    "builtin_argument": (
        "RWBuffer<float> v;",
        "outp[i] = length(v);",
        "passed to 'length', which has no texel-buffer form in GLSL",
    ),
    "structure_member": (
        "struct S { Buffer<float> b; };",
        "outp[i] = 1.0;",
        "as a structure member",
    ),
}


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_typed_buffer_forms_without_a_glsl_lowering_are_refused(tmp_path, case):
    declaration, body, reason = REFUSED[case]
    source = f"""
shader P {{
    {declaration}
    RWStructuredBuffer<float> outp;
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            {body}
        }}
    }}
}}
"""
    with pytest.raises(OpenGLLoweringRefused, match=re.escape(reason)):
        _translate(tmp_path, source)


HLSL_HELPER_ATOMICS = """
RWBuffer<int> maxima : register(u0);
RWTexture2D<uint> hist : register(u1);
RWTexture2D<float4> img : register(u2);
void raise_to(RWBuffer<int> d, uint i, int v) { int o; InterlockedMax(d[i], v, o); }
void bump(RWTexture2D<uint> h, uint2 p) { InterlockedAdd(h[p], 1u); }
void count(RWTexture2D<uint> h, uint2 p) { bump(h, p); }
float4 peek(RWTexture2D<float4> t, uint2 p) { return t[p]; }
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    raise_to(maxima, id.x, int(id.y));
    count(hist, id.xy);
    img[id.xy] = peek(img, id.xy) * 2.0;
}
"""


def test_images_a_helper_updates_atomically_get_a_single_channel_format(tmp_path):
    glsl = _translate(tmp_path, HLSL_HELPER_ATOMICS, name="shader.hlsl")
    assert "layout(r32i" in _declaration(glsl, "maxima")
    # ``hist`` reaches the atomic through two calls.
    assert "layout(r32ui" in _declaration(glsl, "hist")
    # A helper only loads ``img``, which keeps the format of the bound image.
    assert not re.search(r"layout\(\w+32", _declaration(glsl, "img"))
    assert "#extension GL_EXT_shader_image_load_formatted : require" in glsl
    _validate(tmp_path, glsl, "comp")
