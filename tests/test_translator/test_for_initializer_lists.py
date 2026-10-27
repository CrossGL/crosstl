"""``for (int a = 0, b = n; a < b; a++, b--)`` declares both counters in one initializer.

The CrossGL parser keeps the two declarations as a list. GLSL, MSL and Slang output wrote that list's Python repr
into the loop header, SPIR-V output refused the program ("Unknown variable a"), and the GLSL interval analysis kept
``a`` at its initial value inside the loop, so it dropped ``if (a == 1) continue;`` as unreachable.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.for_initializers import (
    ForInitializerError,
    join_for_initializer,
)

TWO_COUNTERS = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* iin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            float acc = 0.0;
            int n = 6 + (iin[i] & 3);
            for (int a = 0, b = n; a < b; a++, b--) {
                if (a == 1) {
                    continue;
                }
                acc += float(a * 16 + b);
            }
            outp[i] = acc;
        }
    }
}
"""


def _translate(tmp_path, backend):
    path = tmp_path / "loop.cgl"
    path.write_text(TWO_COUNTERS)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_declarations_of_one_type_share_one_declaration():
    assert (
        join_for_initializer(["int a = 0", "int b = n"], [True, True], "opengl")
        == "int a = 0, b = n"
    )
    assert (
        join_for_initializer(["float w[2] = x", "float v[2]"], [True, True], "metal")
        == "float w[2] = x, v[2]"
    )
    assert join_for_initializer(["a = 0", "b = n"], [False, False], "opengl") == (
        "a = 0, b = n"
    )


@pytest.mark.parametrize(
    ("parts", "declarations"),
    [
        (["int a = 0", "float b = 1.0"], [True, True]),
        (["int a = 0", "b = 1"], [True, False]),
        (["int a = 0", "int t = 1; int b = t"], [True, True]),
    ],
)
def test_initializers_one_declaration_cannot_write_are_refused(parts, declarations):
    with pytest.raises(ForInitializerError):
        join_for_initializer(parts, declarations, "opengl")


@pytest.mark.parametrize("backend", ["opengl", "metal", "slang", "directx"])
def test_loop_header_declares_both_counters(tmp_path, backend):
    code = " ".join(_translate(tmp_path, backend).split())
    assert "for (int a = 0, b = n;" in code
    assert "VariableNode" not in code
    assert "continue;" in code


def test_opengl_output_validates(tmp_path):
    code = _translate(tmp_path, "opengl")
    if shutil.which("glslangValidator"):
        shader = tmp_path / "loop.comp"
        shader.write_text(code)
        _run(["glslangValidator", str(shader)])


def test_spirv_runs_both_declarations(tmp_path):
    code = _translate(tmp_path, "vulkan")
    assert "OpLoopMerge" in code
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        asm = tmp_path / "loop.spvasm"
        asm.write_text(code)
        binary = tmp_path / "loop.spv"
        _run(["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)])
        _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])
