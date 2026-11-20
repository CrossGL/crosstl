"""Metal output for cooperative-matrix transfers through an element pointer."""

import shutil
import subprocess

import pytest

from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

SOURCE = """
shader ElementPointer {
    compute {
        layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
        layout(set = 0, binding = 0) buffer float* input;
        layout(set = 0, binding = 1) buffer float* output;
        void main() {
            CooperativeMatrix<float, 8, 8, subgroup, unspecified, unspecified> m;
            cooperative_matrix_load(m, input[64], 8);
            cooperative_matrix_store(output[128], m, 8);
            cooperative_matrix_store(output, m, 8);
        }
    }
}
"""


def _generate():
    return MetalCodeGen().generate(Parser(Lexer(SOURCE).tokens).parse())


def test_element_pointer_operands_take_the_element_address():
    generated = _generate()
    assert "simdgroup_load(m, &input[64], 8);" in generated
    assert "simdgroup_store(m, &output[128], 8);" in generated
    # A plain buffer pointer is already a pointer and stays unchanged.
    assert "simdgroup_store(m, output, 8);" in generated


def _metal_compiler_available():
    if shutil.which("xcrun") is None:
        return False
    probe = subprocess.run(
        ["xcrun", "-f", "metal"], capture_output=True, text=True, timeout=60
    )
    return probe.returncode == 0


@pytest.mark.skipif(not _metal_compiler_available(), reason="Metal toolchain missing")
def test_element_pointer_output_compiles(tmp_path):
    artifact = tmp_path / "translated.metal"
    artifact.write_text(_generate())
    compiled = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(artifact)]
        + ["-o", str(tmp_path / "translated.air")],
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stderr
