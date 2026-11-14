"""Typedefs that declare the ``xhalf`` names decide what they mean.

Apple's samples switch lighting precision with ``typedef half xhalf;`` and ``typedef half3 xhalf3;``. The importer
reads an undeclared ``xhalfN`` as ``halfN``, and two gaps followed from treating the names as built-in:

- the lexer read ``xhalf2`` as a vector keyword, so ``typedef half2 xhalf2;`` did not parse;
- the built-in reading won over a declared typedef, so after ``typedef float xhalf;`` an ``xhalf`` value was imported
  as ``half`` and lost precision without a diagnostic.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.backend.Metal.MetalLexer import MetalLexer

HALF_TYPEDEFS = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    typedef half  xhalf;
    typedef half2 xhalf2;
    typedef half3 xhalf3;
    typedef half4 xhalf4;
    typedef half3x3 xhalf3x3;

    xhalf3 unpremultiply(xhalf4 c) {
        return (xhalf3)(c.rgb * 1.0h / c.a);
    }

    kernel void k(device half4* out [[buffer(0)]],
                  uint i [[thread_position_in_grid]]) {
        xhalf2 uv = xhalf2(0.5h);
        xhalf3x3 m = xhalf3x3(1.0h);
        xhalf s = (xhalf)uv.x;
        out[i] = xhalf4(m * unpremultiply(out[i]), s);
    }
    """)

FLOAT_TYPEDEFS = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    typedef float  xhalf;
    typedef float3 xhalf3;

    kernel void k(device float4* out [[buffer(0)]],
                  uint i [[thread_position_in_grid]]) {
        xhalf3 v = xhalf3(out[i].xyz);
        xhalf s = (xhalf)out[i].w;
        out[i] = float4(v * s, s);
    }
    """)


def _translate(tmp_path, source, backend):
    path = tmp_path / "xhalf.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(tmp_path, *command):
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=120, cwd=tmp_path
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]


@pytest.mark.parametrize("name", ["xhalf2", "xhalf3", "xhalf4", "xhalf3x3"])
def test_lexer_reads_xhalf_names_as_identifiers(name):
    tokens = MetalLexer(f"typedef half3 {name};").tokenize()

    assert ("IDENTIFIER", name) in tokens


def test_half_typedefs_import_as_half(tmp_path):
    crossgl = _translate(tmp_path, HALF_TYPEDEFS, "cgl")

    assert "f16vec2 uv = f16vec2(float16(0.5));" in crossgl
    assert "float16 s = (float16)uv.x;" in crossgl
    assert "f16vec3 unpremultiply(f16vec4 c)" in crossgl


def test_half_typedefs_compile_as_metal(tmp_path):
    metal = _translate(tmp_path, HALF_TYPEDEFS, "metal")

    assert "half2 uv = half2(half(0.5));" in metal
    assert "half s = half(uv.x);" in metal
    if not shutil.which("xcrun"):
        pytest.skip("xcrun is not installed")
    (tmp_path / "a.metal").write_text(metal, encoding="utf-8")
    _run(tmp_path, "xcrun", "-sdk", "macosx", "metal", "-c", "a.metal", "-o", "a.air")


def test_half_typedefs_compile_as_glsl(tmp_path):
    glsl = _translate(tmp_path, HALF_TYPEDEFS, "opengl")

    assert "xhalf" not in glsl.split("void main", 1)[1]
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not installed")
    (tmp_path / "a.comp").write_text(glsl, encoding="utf-8")
    _run(tmp_path, "glslangValidator", "a.comp")


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("cgl", ["vec3 v = vec3(", "float s = float("]),
        ("metal", ["float3 v = float3(out_[i].xyz);", "float s = float(out_[i].w);"]),
        ("opengl", ["vec3 v = vec3(", "float s = float("]),
    ],
)
def test_float_typedefs_keep_float_precision(tmp_path, backend, expected):
    code = _translate(tmp_path, FLOAT_TYPEDEFS, backend)

    for fragment in expected:
        assert fragment in code
    assert "half s" not in code
    assert "float16 s" not in code
