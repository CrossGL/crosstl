"""Exact ``fmod``, ``frexp`` and ``modf`` in HLSL output, and ``fmod`` for a floating ``%`` in HLSL and Metal output.

DXC lowers ``fmod`` through a fast division and ``frac``, drops the sign of ``frexp``'s significand and computes
``modf`` as ``x - trunc(x)``; the helpers compute the C and GLSL 4.60 section 8.3 results from the operands' bits.
Execution on Direct3D 12 WARP is covered by the conformance cases ``m_fmod_sign``, ``u_fmodf_sign``,
``h_fmod_remainder`` and the ``*_frexp_ldexp_modf`` family.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

import crosstl

CONFORMANCE = Path(__file__).resolve().parents[1] / "conformance" / "cases"
NATIVE = re.compile(r"(?<![\w.])(fmod|frexp|modf)\(")


def _compute(body, declarations="float x = a[i];\n float y = a[i + 1u];"):
    return f"""
shader T {{
    compute {{
        buffer float a[];
        buffer float o[];
        layout(local_size_x = 64) in;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            {declarations}
            {body}
        }}
    }}
}}
"""


def _translate(tmp_path, source, backend="directx", suffix=".cgl", **options):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False, **options)


def _calls(code):
    """The native ``fmod``, ``frexp`` and ``modf`` calls left outside the helpers' own definitions."""
    body = code[code.index("CSMain") :] if "CSMain" in code else code
    return NATIVE.findall(body)


def _compile_hlsl(tmp_path, code, *flags):
    compiler = shutil.which("dxc")
    if compiler is None:
        return
    source = tmp_path / "out.hlsl"
    source.write_text(code)
    result = subprocess.run(
        [compiler, "-T", "cs_6_2", "-E", "CSMain", *flags, str(source)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "warning" not in result.stderr, result.stderr[-2000:]


def test_fmod_calls_the_exact_helper(tmp_path):
    code = _translate(tmp_path, _compute("o[i] = fmod(x, y);"))
    assert "o[i] = crosstl_fmod_float(x, y);" in code
    assert "float crosstl_fmod_float(float x, float y) {" in code
    assert not _calls(code)
    _compile_hlsl(tmp_path, code)


def test_fmod_divides_integer_significands():
    """The helper divides integer significands; no floating division or ``frac`` reaches DXC."""
    from crosstl.translator.codegen.hlsl_exact_float_builtins import _fmod_scalar

    class Holder:
        pass

    holder = Holder()
    text = "".join(
        holder.__dict__["hlsl_exact_float_helpers"][name]
        for name in [_fmod_scalar(holder, "float")]
    )
    assert "remainder -= divisor;" in text
    assert "frac(" not in text and " / " not in text
    assert "if (a >= infinity || b > infinity || b == ((uint)0)) {" in text


def test_floating_remainder_operator_calls_the_helper(tmp_path):
    code = _translate(
        tmp_path,
        _compute("o[i] = x % y + float(i % 3u);"),
    )
    assert "crosstl_fmod_float(x, y)" in code
    assert "(i % 3u)" in code
    _compile_hlsl(tmp_path, code)


def test_vector_fmod_with_a_scalar_divisor(tmp_path):
    code = _translate(
        tmp_path,
        _compute("vec3 v = vec3(x, -x, y);\n o[i] = fmod(v, 2.0).y + (v % y).z;"),
    )
    assert "crosstl_fmod_float3(v, 2.0)" in code
    assert "crosstl_fmod_float3(v, y)" in code
    assert (
        "return float3(crosstl_fmod_float(x.x, y.x), crosstl_fmod_float(x.y, y.y), "
        "crosstl_fmod_float(x.z, y.z));" in code
    )
    assert code.count("float crosstl_fmod_float(float x, float y) {") == 1
    _compile_hlsl(tmp_path, code)


def test_frexp_keeps_the_sign_and_the_exponent_type(tmp_path):
    code = _translate(
        tmp_path,
        _compute(
            "int e;\n float s = frexp(x, e);\n float fe;\n float t = frexp(y, fe);\n"
            " ivec3 e3;\n vec3 s3 = frexp(vec3(x, y, 0.5), e3);\n"
            " o[i] = s + float(e) + t + fe + s3.y + float(e3.z);"
        ),
    )
    assert "float s = crosstl_frexp_float(x, e);" in code
    assert "float t = crosstl_frexp_float_float(y, fe);" in code
    assert "crosstl_frexp_float3_int3(float3(x, y, 0.5), e3)" in code
    assert (
        "return asfloat((bits & sign_mask) | (((uint)126) << 23) | significand);"
        in code
    )
    assert not _calls(code)
    _compile_hlsl(tmp_path, code)


def test_modf_keeps_the_sign_of_a_zero_fraction(tmp_path):
    code = _translate(
        tmp_path,
        _compute(
            "float w;\n float f = modf(x, w);\n int iw;\n float g = modf(y, iw);\n"
            " o[i] = f + w + g + float(iw);"
        ),
    )
    assert "float f = crosstl_modf_float(x, w);" in code
    assert "float g = crosstl_modf_float_int(y, iw);" in code
    assert "return asfloat(asuint(fraction) | sign);" in code
    assert not _calls(code)
    _compile_hlsl(tmp_path, code)


def test_double_helpers_decode_64_bit_words(tmp_path):
    code = _translate(
        tmp_path,
        _compute(
            "int e;\n double w;\n"
            " o[i] = float(fmod(x, y) + frexp(x, e) + modf(x, w) + w + (x % y) + double(e));",
            "double x = double(a[i]);\n double y = double(a[i + 1u]);",
        ),
    )
    for helper in ("fmod", "frexp", "modf"):
        assert f"double crosstl_{helper}_double(double x, " in code
    assert "uint64_t crosstl_double_bits(double value) {" in code
    assert code.index("crosstl_double_bits(double value)") < code.index(
        "double crosstl_frexp_double("
    )
    assert not _calls(code)
    _compile_hlsl(tmp_path, code)


def test_half_helpers_compute_through_float(tmp_path):
    code = _translate(
        tmp_path,
        _compute(
            "int e;\n half w;\n"
            " o[i] = float(fmod(x, y) + frexp(x, e) + modf(x, w) + w + (x % y));",
            "half x = half(a[i]);\n half y = half(a[i + 1u]);",
        ),
    )
    assert "return (half)crosstl_fmod_float((float)x, (float)y);" in code
    assert "return (half)crosstl_frexp_float((float)x, exponent);" in code
    assert code.index("float crosstl_fmod_float(") < code.index(
        "half crosstl_fmod_half("
    )
    assert not _calls(code)
    _compile_hlsl(tmp_path, code, "-enable-16bit-types")


def test_a_user_function_named_fmod_is_called_unchanged(tmp_path):
    source = """
shader T {
    float fmod(float a, float b) { return a - b; }
    compute {
        buffer float a[];
        layout(local_size_x = 64) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            a[i] = fmod(a[i], 2.0);
        }
    }
}
"""
    code = _translate(tmp_path, source)
    assert "crosstl_fmod" not in code


def test_integer_operands_keep_the_native_remainder(tmp_path):
    code = _translate(
        tmp_path,
        _compute("int n = int(i);\n o[i] = float(n % 7);"),
    )
    assert "crosstl_fmod" not in code
    assert "(n % 7)" in code


@pytest.mark.parametrize(
    "case, suffix, source_backend",
    [
        ("m_fmod_sign", ".metal", "metal"),
        ("h_fmod_remainder", ".hlsl", "directx"),
        ("h_frexp_ldexp_modf", ".hlsl", "directx"),
        ("s_frexp_ldexp_modf", ".spvasm", "vulkan"),
        ("g_frexp_ldexp_modf", ".comp", "opengl"),
    ],
)
def test_imported_programs_call_the_helpers(tmp_path, case, suffix, source_backend):
    source = next((CONFORMANCE / case).glob("src.*"))
    code = _translate(
        tmp_path,
        source.read_text(),
        suffix=source.suffix if source.suffix else suffix,
        source_backend=source_backend,
    )
    entry = re.search(r"\[numthreads[^\]]*\]\s*void (\w+)", code).group(1)
    body = code[code.index(f"void {entry}(") :]
    assert not NATIVE.findall(body)
    assert "crosstl_" in body


def test_metal_floating_remainder_of_builtin_calls_uses_fmod(tmp_path):
    code = _translate(
        tmp_path,
        _compute("o[i] = abs(x) % abs(y);\n o[i + 1u] = float(int(i) % 3);"),
        backend="metal",
    )
    assert "fmod(abs(x), abs(y))" in code
    assert "% 3" in code
