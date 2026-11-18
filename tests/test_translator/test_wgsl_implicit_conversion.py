"""WGSL implicit conversions in mixed integer/float expressions.

C-style languages convert an integer operand to the floating-point type when
the other operand of ``+``, ``-``, ``*``, ``/`` or a comparison is a float
(GLSL 4.60 section 5.9, "Implicit Conversions"). WGSL has no implicit
conversions, and an abstract float literal next to an ``i32`` or ``u32``
operand is one naga rejects outside a constant expression, so the generator
must write the conversion itself (issue #2384).
"""

import shutil
import subprocess

import pytest

import crosstl

IMPLICIT_CONVERSION = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* iin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int a = iin[i];
            uint u = 3u;
            float x = a;
            float y = a * 0.5 + 1;
            float z = x / 4 + u;
            outp[i] = x + y * 16.0 + z * 256.0;
        }
    }
}
"""

MIXED_COMPARISON = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* iin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int a = iin[i];
            float y = 0.0;
            if (a < 0.5) {
                y = 1.0;
            }
            if (a == 1.0) {
                y = 2.0;
            }
            outp[i] = y;
        }
    }
}
"""

MIXED_VECTOR = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* iin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int a = iin[i];
            vec3 v = vec3(1.0, 2.0, 3.0) * a;
            vec3 w = v + a;
            outp[i] = w.x + w.y + w.z;
        }
    }
}
"""


def _translate(tmp_path, source):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend="wgsl", format_output=False)


def _naga_validates(tmp_path, code):
    validator = shutil.which("naga")
    if validator is None:
        return
    (tmp_path / "out.wgsl").write_text(code)
    result = subprocess.run(
        [validator, "out.wgsl"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr + code


def test_integer_operand_of_float_arithmetic_converts_to_f32(tmp_path):
    wgsl = _translate(tmp_path, IMPLICIT_CONVERSION)
    assert "f32(a) * 0.5" in wgsl, wgsl
    assert "((x / 4) + f32(u))" in wgsl, wgsl
    assert "a * 0.5" not in wgsl, wgsl
    _naga_validates(tmp_path, wgsl)


def test_integer_operand_of_float_comparison_converts_to_f32(tmp_path):
    wgsl = _translate(tmp_path, MIXED_COMPARISON)
    assert "f32(a) < 0.5" in wgsl, wgsl
    assert "f32(a) == 1.0" in wgsl, wgsl
    _naga_validates(tmp_path, wgsl)


def test_integer_operand_of_float_vector_arithmetic_converts(tmp_path):
    wgsl = _translate(tmp_path, MIXED_VECTOR)
    assert "vec3<f32>(a)" in wgsl, wgsl
    _naga_validates(tmp_path, wgsl)
