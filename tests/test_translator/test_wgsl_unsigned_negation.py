"""WGSL output negates unsigned integers as ``~x + 1u`` (issue #3283).

GLSL, HLSL, MSL, SPIR-V and CrossGL wrap ``-x`` for an unsigned ``x`` modulo 2^32, while WGSL defines unary ``-``
only for signed integers and floats, so naga rejected ``-a`` with ``a: u32``.
"""

import shutil
import subprocess

import pytest

import crosstl

SHADER = """shader Neg {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer uint* outp;
        buffer int* iout;
        buffer float* fout;
        void main() {
            uint a = outp[0];
            uvec2 v = uvec2(a, 3u);
            uvec2 w = -v;
            int s = iout[0];
            float f = fout[0];
            outp[0] = -a + w.x + (-(a + 1u)) + (-2u) + (-0u);
            iout[0] = -s;
            fout[0] = -f;
        }
    }
}
"""


@pytest.fixture
def wgsl(tmp_path):
    path = tmp_path / "neg.cgl"
    path.write_text(SHADER)
    return crosstl.translate(str(path), backend="wgsl")


@pytest.mark.parametrize(
    "fragment",
    [
        "var w: vec2<u32> = (~v + 1u);",
        "(~a + 1u)",
        "(~(a + 1u) + 1u)",
        "4294967294u",
        "+ 0u)",
        "iout[0] = -s;",
        "fout[0] = -f;",
    ],
)
def test_unsigned_negation_wraps(wgsl, fragment):
    assert fragment in wgsl


def test_no_unary_minus_reaches_an_unsigned_operand(wgsl):
    assert "-a" not in wgsl
    assert "-v" not in wgsl
    assert "-2u" not in wgsl
    assert "-(a" not in wgsl


@pytest.mark.skipif(shutil.which("naga") is None, reason="naga is not installed")
def test_naga_accepts_the_negations(wgsl, tmp_path):
    module = tmp_path / "neg.wgsl"
    module.write_text(wgsl)
    result = subprocess.run(["naga", str(module)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
