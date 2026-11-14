"""A statement keyword followed by two parenthesised groups is not a temporary function object.

The Metal preprocessor refuses ``Type(...)(...)`` when ``Type`` does not name a struct with ``operator()``, because
the call could not be lowered. It read every identifier as a candidate type, so ``return (float)(x);``, a C-style
cast of a parenthesised expression, and ``if (c) (x) = y;`` were refused as unresolved function objects.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.backend.Metal.preprocessor import MetalPreprocessor, MetalStructMethodError

KEYWORD_CASTS = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    float geometry(float n, float a) {
        float b = n * n;
        return (float)(1.0 / (n + sqrt(a + b - a * b)));
    }

    half3 unpremultiply(half4 c) {
        return (half3)(c.rgb * 1.0h / c.a);
    }

    int pick(int mode, int x) {
        int y = 0;
        if (mode > 0) (y) = x;
        switch (mode) {
            case (1): y += (int)(x * 2); break;
            default: break;
        }
        while (y > 100) (y) -= 100;
        do (y) += 1; while (y < 0);
        return y;
    }

    kernel void k(device float* out [[buffer(0)]],
                  uint i [[thread_position_in_grid]]) {
        out[i] = geometry(out[i], 0.5) + float(unpremultiply(half4(1.0h)).x)
            + float(pick(int(i), 3));
    }
    """)


def _translate(tmp_path, source, backend):
    path = tmp_path / "casts.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(tmp_path, *command):
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=120, cwd=tmp_path
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]


@pytest.mark.parametrize(
    "statement",
    [
        "return (float)(x);",
        "return(float)(x);",
        "if (x > 0.0) (x) = 1.0;",
        "while (x > 1.0) (x) -= 1.0;",
        "for (int i = 0; i < 2; i++) (x) += 1.0;",
        "do (x) += 1.0; while (x < 0.0);",
    ],
)
def test_preprocessor_keeps_keyword_parenthesised_groups(statement):
    code = f"float f(float x) {{ {statement} return x; }}"

    output = MetalPreprocessor().preprocess(code)

    assert statement in output


def test_preprocessor_still_refuses_unresolved_temporary_functors():
    code = "float f(float x) { return Missing(x)(x); }"

    with pytest.raises(MetalStructMethodError, match="temporary function object"):
        MetalPreprocessor().preprocess(code)


def test_keyword_casts_import_as_constructor_casts(tmp_path):
    crossgl = _translate(tmp_path, KEYWORD_CASTS, "cgl")

    assert "return float(1.0 / (n + sqrt(a + b - a * b)));" in crossgl
    assert "return f16vec3(c.rgb * float16(1.0) / c.a);" in crossgl


def test_keyword_casts_compile_as_metal(tmp_path):
    metal = _translate(tmp_path, KEYWORD_CASTS, "metal")

    assert "return float(1.0 / (n + sqrt(a + b - a * b)));" in metal
    if not shutil.which("xcrun"):
        pytest.skip("xcrun is not installed")
    (tmp_path / "a.metal").write_text(metal, encoding="utf-8")
    _run(tmp_path, "xcrun", "-sdk", "macosx", "metal", "-c", "a.metal", "-o", "a.air")


def test_keyword_casts_compile_as_glsl(tmp_path):
    glsl = _translate(tmp_path, KEYWORD_CASTS, "opengl")

    assert "return float((1.0 / (n + sqrt(((a + b) - (a * b))))));" in glsl
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not installed")
    (tmp_path / "a.comp").write_text(glsl, encoding="utf-8")
    _run(tmp_path, "glslangValidator", "a.comp")


def test_keyword_casts_translate_to_hlsl(tmp_path):
    hlsl = _translate(tmp_path, KEYWORD_CASTS, "directx")

    assert "return float((1.0 / (n + sqrt(((a + b) - (a * b))))));" in hlsl
