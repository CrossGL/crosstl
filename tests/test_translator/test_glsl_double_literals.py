"""GLSL output writes floating literals in binary64 contexts as ``double`` constants.

GLSL reads an unsuffixed floating-point constant as a ``float`` (GLSL 4.60 section 4.1.4), so ``double(0.1)`` is the
binary32 value nearest 0.1 widened to ``double`` on a compiler that follows the specification, and ``double(1e-300)``
is 0. An unsuffixed CrossGL literal denotes its exact decimal value rounded once to its context's type, and ``0.1f``
is a binary32 value; glslang folds both spellings in double precision. GLSL output now writes an unsuffixed literal
where a ``double`` is required as ``0.1lf``, and a suffixed one as the ``double`` constant of its binary32 value.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl

DOUBLE_FROM_CUDA = """extern "C" __global__ void probe(double* fin, double* outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    outp[i] = fin[i] * 0.1 + fin[i] * 0.1f;
}
"""

CROSSGL = """shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer double* fin;
        buffer double* outp;
        buffer float* fout;
        double scale(double v) { return v * 0.3; }
        void main() {
            uint i = gl_GlobalInvocationID.x;
            float f = 0.1;
            double a = 0.1;
            double b = 0.2f;
            dvec2 v = dvec2(0.1, -0.7);
            dmat2 m = dmat2(0.5);
            a += 0.1;
            if (fin[i] < -0.25) { a = scale(0.5); }
            double c = fin[i] > 0.0 ? 0.1 : 0.2;
            fout[i] = f * 0.1;
            outp[i] = fin[i] * 0.1 + a + b + v.x + v.y + m[0].x + c + 1e-300 + double(f);
        }
    }
}
"""


def _translate(tmp_path, name, source, backend="opengl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _spirv_constants(tmp_path, glsl):
    """The ``float64_t`` constants glslang writes for ``glsl``, as (low, high) word pairs, or ``None``."""
    if shutil.which("glslangValidator") is None:
        return None
    path = tmp_path / "out.comp"
    path.write_text(glsl)
    result = subprocess.run(
        ["glslangValidator", "-V", "-H", "-o", str(tmp_path / "out.spv"), str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    constants = set()
    for line in result.stdout.splitlines():
        if "(float64_t) Constant" in line:
            low, high = line.split("Constant", 1)[1].split()[:2]
            constants.add((int(low), int(high)))
    return constants


def _words(value_bits):
    return (value_bits & 0xFFFFFFFF, value_bits >> 32)


def test_cuda_double_operands_keep_their_literal_values(tmp_path):
    glsl = _translate(tmp_path, "probe.cu", DOUBLE_FROM_CUDA)
    assert "(fin[i] * 0.1lf)" in glsl
    # 0.1f is the binary32 value nearest 0.1, written exactly as a double.
    assert "(fin[i] * 0.10000000149011612lf)" in glsl
    assert "double(0.1)" not in glsl
    constants = _spirv_constants(tmp_path, glsl)
    if constants is not None:
        assert _words(0x3FB999999999999A) in constants  # binary64 nearest 0.1
        assert _words(0x3FB99999A0000000) in constants  # binary32 nearest 0.1


@pytest.mark.parametrize(
    "expected",
    [
        "return (v * 0.3lf);",
        "double a = 0.1lf;",
        "double b = 0.20000000298023224lf;",
        "dvec2 v = dvec2(0.1lf, -0.7lf);",
        "dmat2 m = dmat2(0.5lf);",
        "a = (a + 0.1lf);",
        "if ((fin[i] < -0.25lf)) {",
        "a = scale(0.5lf);",
        "((fin[i] > 0.0lf) ? 0.1lf : 0.2lf)",
        "1e-300lf",
        "(fin[i] * 0.1lf)",
    ],
)
def test_binary64_contexts_write_double_constants(tmp_path, expected):
    glsl = _translate(tmp_path, "probe.cgl", CROSSGL)
    assert expected in glsl


def test_binary32_contexts_keep_float_constants(tmp_path):
    glsl = _translate(tmp_path, "probe.cgl", CROSSGL)
    assert "float f = 0.1;" in glsl
    assert "fout[i] = (f * 0.1);" in glsl
    assert "double(f)" in glsl
    assert "crossgl_select_bits_float" not in glsl


def test_binary64_constants_compile_with_their_values(tmp_path):
    glsl = _translate(tmp_path, "probe.cgl", CROSSGL)
    constants = _spirv_constants(tmp_path, glsl)
    if constants is None:
        pytest.skip("glslangValidator is not installed")
    assert _words(0x3FB999999999999A) in constants  # 0.1
    assert _words(0x3FC99999A0000000) in constants  # binary32 nearest 0.2
    assert _words(0x01A56E1FC2F8F359) in constants  # 1e-300


def test_glsl_double_constants_round_trip(tmp_path):
    source = textwrap.dedent("""\
        #version 450
        layout(local_size_x = 1) in;
        layout(std430, binding = 0) buffer Out { double values[]; };
        void main() {
            values[0] = 0.1lf * values[1];
        }
        """)
    glsl = _translate(tmp_path, "probe.comp", source)
    assert "(0.1lf * out_.values[1])" in glsl
    _spirv_constants(tmp_path, glsl)
