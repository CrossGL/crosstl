"""A conditional array extent is emitted as Metal code.

An extent such as ``tgp_mem_size == 0 ? 1 : tgp_mem_size`` (a common guard against a zero-length threadgroup
array) reached the Metal emitter's type printer, which has no case for a conditional expression and printed the AST
node. The output was then refused as an AST text leak although Metal, like HLSL and GLSL, accepts the conditional as
an integral constant expression.
"""

import shutil
import subprocess

import pytest

import crosstl

_CGL = """
shader T {
    const int TILE = 4;
    compute {
        buffer float data[];
        void main() @workgroup_size(4, 1, 1) {
            threadgroup float[TILE == 0 ? 1 : TILE] scratch;
            threadgroup float[2 * (TILE > 2 ? TILE : 2) + 1] padded;
            scratch[0] = 1.0;
            padded[0] = scratch[0];
            data[0] = padded[0];
        }
    }
}
"""


def _translate(tmp_path, backend):
    source = tmp_path / "extents.cgl"
    source.write_text(_CGL)
    return crosstl.translate(str(source), backend=backend, format_output=False)


def test_metal_emits_conditional_array_extents(tmp_path):
    output = _translate(tmp_path, "metal")

    assert "TernaryOpNode" not in output
    assert "scratch[TILE == 0 ? 1 : TILE];" in output
    assert "padded[2 * (TILE > 2 ? TILE : 2) + 1];" in output


@pytest.mark.skipif(shutil.which("xcrun") is None, reason="requires the Metal compiler")
def test_metal_conditional_array_extents_compile(tmp_path):
    source = tmp_path / "extents.metal"
    source.write_text(_translate(tmp_path, "metal"))
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "extents.air"),
        ],
        capture_output=True,
        text=True,
    )
    if "unable to find utility" in result.stderr:
        pytest.skip("the Metal toolchain is not installed")
    assert result.returncode == 0, result.stderr
