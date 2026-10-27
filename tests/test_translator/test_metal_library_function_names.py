"""Metal output renames a user function named like a Metal Standard Library function (issue #3156).

``using namespace metal;`` puts the library in scope, so ``float fwidth(float x)`` made ``fwidth(values[i])``
ambiguous for the Metal compiler.
"""

import shutil
import subprocess

import crosstl
from crosstl.translator.codegen.metal_library_functions import shadows_msl_library

SHADER = """
shader S {
    buffer float values[];
    float fwidth(float x) { return x * 0.5; }
    float clamp(float x) { return x + 1.0; }
    float clamp(float x, float y) { return x + y; }
    float simd_sum(float x) { return x * 3.0; }
    float helper(float x) { return x; }
    compute {
        void main() {
            uint i = gl_GlobalInvocationID.x;
            values[i] = fwidth(values[i]) + clamp(values[i]) + clamp(values[i], 2.0)
                + simd_sum(values[i]) + helper(max(values[i], 0.0));
        }
    }
}
"""


def _metal(tmp_path):
    path = tmp_path / "names.cgl"
    path.write_text(SHADER)
    return crosstl.translate(str(path), backend="metal", format_output=False)


def test_library_names_are_recognized():
    assert shadows_msl_library("fwidth")
    assert shadows_msl_library("simd_shuffle_down")
    assert not shadows_msl_library("helper")


def test_user_functions_named_like_library_functions_are_renamed(tmp_path):
    code = " ".join(_metal(tmp_path).split())
    assert "float fwidth_fn(float x)" in code
    assert "float clamp_fn(float x)" in code
    assert "float clamp_fn(float x, float y)" in code
    assert "float simd_sum_fn(float x)" in code
    assert "fwidth_fn(values[i])" in code
    assert "clamp_fn(values[i], 2.0)" in code
    # Other functions and library calls keep their names.
    assert "float helper(float x)" in code
    assert "max(values[i], 0.0)" in code


def test_metal_compiles(tmp_path):
    if shutil.which("xcrun") is None:
        return
    source = tmp_path / "names.metal"
    source.write_text(_metal(tmp_path))
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "n.air"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
