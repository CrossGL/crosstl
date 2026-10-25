"""Metal refuses array literals passed to array parameters.

A C array parameter in Metal is a pointer, and no Metal expression yields the
address of an array literal. ``sum2({1.0, 2.0})`` used to be emitted verbatim
and failed with "no matching function for call to 'sum2'". It is now refused
with a typed diagnostic, as HLSL already does. Passing a named array still
works.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused

PROGRAM = textwrap.dedent("""
    shader S {
        float sum2(float v[2]) { return v[0] + v[1]; }
        compute {
            buffer float data[];
            layout(local_size_x = 1) in;
            void main() {
                float t[2] = {1.0, 2.0};
                data[0] = sum2(%s);
            }
        }
    }
    """)


def _translate(tmp_path, argument):
    source = tmp_path / "arguments.cgl"
    source.write_text(PROGRAM % argument, encoding="utf-8")
    return crosstl.translate(str(source), backend="metal")


def test_array_literal_arguments_to_array_parameters_are_refused(tmp_path):
    with pytest.raises(MetalLoweringRefused, match="array literal argument to 'sum2'"):
        _translate(tmp_path, "{1.0, 2.0}")


def test_named_array_arguments_are_kept(tmp_path):
    code = _translate(tmp_path, "t")
    assert "sum2(t)" in code


def test_named_array_output_compiles(tmp_path):
    if shutil.which("xcrun") is None:
        pytest.skip("xcrun is not installed")
    (tmp_path / "arguments.metal").write_text(
        _translate(tmp_path, "t"), encoding="utf-8"
    )
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", "arguments.metal", "-o", "a.air"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
    )
    if "unable to find utility" in result.stderr:
        pytest.skip("the Metal toolchain is not installed")
    assert result.returncode == 0, result.stderr[-2000:]
