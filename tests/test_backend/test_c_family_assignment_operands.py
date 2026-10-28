"""CUDA, HIP and OpenCL C import keep an assignment that is an operand as one group.

The importers wrote ``(t -= 3) > 0`` as ``t -= 3 > 0``, which CrossGL, like C, reads
as ``t -= (3 > 0)``, so the surrounding operator bound to the right-hand side of the
assignment and every target computed something else.
"""

import shutil
import subprocess

import pytest

import crosstl

BODY = """
    int t = values[i];
    int n = 0;
    while ((t -= 3) > 0) {
        n += 1;
    }
    int m = (n = n + 1) * 2;
    int q = (t = t - 3) > 0 ? (n = 5) : -(m = 2);
    int s = n + m + q + (t = 1);
    for (int j = 0; j < 2; j += 1) { n = n + 1; }
    values[i] = n + m + q + s;
}
"""

SOURCES = {
    "cu": (
        "__global__ void k(int* values) {\n"
        "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + BODY
    ),
    "hip": (
        "__global__ void k(int* values) {\n"
        "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + BODY
    ),
    "cl": (
        "__kernel void k(__global int* values) {\n"
        "    int i = get_global_id(0);" + BODY
    ),
}


def _translate(tmp_path, extension, backend):
    path = tmp_path / f"kernel.{extension}"
    path.write_text(SOURCES[extension])
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return " ".join(code.split())


@pytest.mark.parametrize("extension", sorted(SOURCES))
def test_binary_operands_keep_the_assignment_grouped(tmp_path, extension):
    code = _flat(_translate(tmp_path, extension, "cgl"))
    assert "while (((t -= 3) > 0))" in code
    assert "((n = (n + 1)) * 2);" in code
    assert "+ (t = 1))" in code
    assert "+ t = 1" not in code


@pytest.mark.parametrize("extension", sorted(SOURCES))
def test_unary_and_conditional_operands_keep_the_assignment_grouped(
    tmp_path, extension
):
    code = _flat(_translate(tmp_path, extension, "cgl"))
    assert "(((t = (t - 3)) > 0) ? (n = 5) : (-(m = 2)));" in code


@pytest.mark.parametrize("extension", sorted(SOURCES))
def test_statement_assignments_stay_unparenthesized(tmp_path, extension):
    code = _flat(_translate(tmp_path, extension, "cgl"))
    assert "n = (n + 1);" in code
    assert "(n = (n + 1));" not in code


@pytest.mark.parametrize("extension", sorted(SOURCES))
def test_grouped_assignments_reach_glsl(tmp_path, extension):
    code = _translate(tmp_path, extension, "opengl")
    flat = _flat(code)
    assert "while (((t -= 3) > 0))" in flat
    assert "(-(m = 2))" in flat
    validator = shutil.which("glslangValidator")
    if validator:
        shader = tmp_path / "kernel.comp"
        shader.write_text(code)
        checked = subprocess.run(
            [validator, "-S", "comp", str(shader)], capture_output=True, text=True
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr
