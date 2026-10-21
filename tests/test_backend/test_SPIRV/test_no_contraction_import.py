"""A SPIR-V NoContraction decoration survives import as a precise local.

NoContraction forbids fusing the decorated operation with others, for example ``a * b - c`` into an FMA (SPIR-V
1.6 section 3.20). The importer used to inline the decorated results and drop the decoration, so ``a * b - c`` could
be contracted on every target. Each decorated result is now assigned to a ``@precise`` local where it is defined,
which GLSL and HLSL print as ``precise``, Metal as ``#pragma clang fp contract(off)`` and SPIR-V as NoContraction
again. Execution on every runtime is covered by the ``s_nocontraction`` conformance case.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

import crosstl

FIXTURE = (
    Path(__file__).parents[2]
    / "conformance"
    / "cases"
    / "s_nocontraction"
    / "src.spvasm"
)


def _translate(backend):
    return crosstl.translate(str(FIXTURE), backend=backend, format_output=False)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_decorated_results_become_precise_locals():
    crossgl = re.sub(r"\s+", " ", _translate("cgl"))
    product = re.search(r"float (\w+) @precise = \(a \* b\);", crossgl)
    assert product is not None, crossgl
    difference = re.search(
        rf"float (\w+) @precise = \({product.group(1)} - c\);", crossgl
    )
    assert difference is not None, crossgl
    assert f"r = {difference.group(1)};" in crossgl
    # Loads carry no decoration and stay plain assignments.
    assert crossgl.count("@precise") == 2


@pytest.mark.parametrize("backend", ("opengl", "directx"))
def test_glsl_and_hlsl_declare_the_locals_precise(backend):
    assert len(re.findall(r"precise float \w+ = ", _translate(backend))) == 2


def test_metal_turns_contraction_off():
    assert "#pragma clang fp contract(off)" in _translate("metal")


def test_spirv_round_trip_decorates_both_operations_and_validates(tmp_path):
    assembly = _translate("vulkan")
    decorated = set(re.findall(r"OpDecorate (%\w+) NoContraction", assembly))
    operations = dict(re.findall(r"(%\w+) = (OpFMul|OpFSub) ", assembly))
    assert sorted(operations[result] for result in decorated) == ["OpFMul", "OpFSub"]
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    path = tmp_path / "round_trip.spvasm"
    path.write_text(assembly, encoding="utf-8")
    binary = tmp_path / "round_trip.spv"
    _run(["spirv-as", "--target-env", "vulkan1.1", str(path), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])


LOOP = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer B { float v[]; };
void main() {
    precise float acc;
    for (int k = 0; k < 4; ++k) {
        acc = v[k] * v[k + 4] - v[8];
        if (acc > 0.0) break;
    }
    v[9] = acc;
}
"""


def test_precise_results_inside_a_loop_stay_in_the_loop(tmp_path):
    if not (shutil.which("glslangValidator") and shutil.which("spirv-dis")):
        pytest.skip("glslangValidator or spirv-dis is not on PATH")
    source = tmp_path / "loop.comp"
    source.write_text(LOOP, encoding="utf-8")
    binary = tmp_path / "loop.spv"
    assembly = tmp_path / "loop.spvasm"
    _run(
        [
            "glslangValidator",
            "-V",
            "--target-env",
            "vulkan1.1",
            str(source),
            "-o",
            str(binary),
        ]
    )
    _run(["spirv-dis", str(binary), "-o", str(assembly)])
    crossgl = re.sub(
        r"\s+",
        " ",
        crosstl.translate(str(assembly), backend="cgl", format_output=False),
    )
    loop = crossgl[crossgl.index("while (") : crossgl.index("break;")]
    assert len(re.findall(r"float \w+ @precise = ", loop)) == 2, crossgl
