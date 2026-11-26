"""Assignments of a CrossGL ``T[N]`` call result to an array variable in the generated HLSL.

HLSL functions cannot return arrays, so the emitter holds a ``T[N]`` call result as a vector (``N`` in 2..4) or as a
``CrossGLFixedArray`` struct, and a local initialized from such a call takes that type. A local declared as
``T a[N]`` and assigned later stays an array, and ``a = spread(x);`` used to assign the vector to it, which DXC
rejects ("cannot implicitly convert from 'float3' to 'float [3]'"). SPIR-V imports write every local this way. The
value is now flat-cast to the array through a typedef, as by-value array arguments are.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl

SOURCE = """
shader ArrayValueAssignments {
    struct B { float outp[]; };
    RWStructuredBuffer<B> o @set(0) @binding(0);
    compute {
        layout(local_size_x = 1) in;
        void main() {
            float a[3];
            float b[6];
            a = spread(2.0);
            float held[3] = spread(3.0);
            float c[3] = {0.0, 0.0, 0.0};
            c = held;
            b = wide(1.0);
            o[0].outp[0] = a[0] + a[1] * 16.0 + a[2] * 256.0 + b[5] + c[2];
        }
    }
    float[3] spread(float x) { return {x, (x * 2.0), (x + 1.0)}; }
    float[6] wide(float x) { float r[6] = {x, x, x, x, x, 6.0}; return r; }
}
"""


def _generate(tmp_path):
    path = tmp_path / "assign.cgl"
    path.write_text(SOURCE, encoding="utf-8")
    return crosstl.translate(str(path), backend="directx", format_output=False)


def test_lowered_call_results_assigned_to_arrays_are_flat_cast(tmp_path):
    code = _generate(tmp_path)
    assert "typedef float CrossGLArray_float_3[3];" in code
    assert "a = (CrossGLArray_float_3)(spread(2.0));" in code
    assert "b = (CrossGLArray_float_6)(wide(1.0));" in code
    # A local held as the lowered vector is converted the same way.
    assert "float3 held = spread(3.0);" in code
    assert "c = (CrossGLArray_float_3)(held);" in code


def test_array_to_array_assignment_is_unchanged(tmp_path):
    source = SOURCE.replace("float held[3] = spread(3.0);", "float held[3] = a;")
    path = tmp_path / "copy.cgl"
    path.write_text(source, encoding="utf-8")
    code = crosstl.translate(str(path), backend="directx", format_output=False)
    assert "c = held;" in code


@pytest.mark.skipif(shutil.which("dxc") is None, reason="dxc is not available")
@pytest.mark.parametrize("spirv", [False, True], ids=["dxil", "spirv"])
def test_dxc_compiles_the_array_assignments(tmp_path, spirv):
    shader = tmp_path / "array_value_assignments.hlsl"
    shader.write_text(_generate(tmp_path), encoding="utf-8")
    command = [shutil.which("dxc"), "-T", "cs_6_0", "-E", "CSMain", str(shader)]
    command += ["-spirv"] if spirv else []
    command += ["-Fo", str(tmp_path / "out.bin")]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
