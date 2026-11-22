"""WGSL implicit conversions in mixed integer/float expressions.

C-style languages convert an integer operand to the floating-point type when
the other operand of ``+``, ``-``, ``*``, ``/``, ``%`` or a comparison is a float
(GLSL 4.60 section 5.9, "Implicit Conversions"). WGSL has no implicit
conversions, and an abstract float literal next to an ``i32`` or ``u32``
operand is one naga rejects outside a constant expression, so the generator
must write the conversion itself (issue #2384).
"""

import shutil
import subprocess

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


MIXED_VECTOR_SCALAR = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* iin;
        buffer uint* uin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int a = iin[i];
            uint u = uin[i];
            vec3 v = vec3(1.0, 2.0, 3.0);
            vec3 r = v * a;
            vec3 s = v * u;
            outp[i] = r.x + r.y + r.z + s.x + s.y + s.z;
        }
    }
}
"""

MIXED_MATRIX_AND_REMAINDER = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* iin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int a = iin[i];
            float x = 5.0;
            mat3 m = mat3(1.0);
            mat3 t = m * a;
            float q = x % a;
            outp[i] = t[0].x + t[1].y + t[2].z + q;
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
    assert "f32(f32(a))" not in wgsl, wgsl
    _naga_validates(tmp_path, wgsl)


def test_integer_operand_of_float_comparison_converts_to_f32(tmp_path):
    wgsl = _translate(tmp_path, MIXED_COMPARISON)
    assert "f32(a) < 0.5" in wgsl, wgsl
    assert "f32(a) == 1.0" in wgsl, wgsl
    assert "f32(f32(a))" not in wgsl, wgsl
    _naga_validates(tmp_path, wgsl)


def test_integer_operand_of_float_vector_arithmetic_converts(tmp_path):
    wgsl = _translate(tmp_path, MIXED_VECTOR)
    assert "f32(a)" in wgsl, wgsl
    assert "vec3<f32>(a)" not in wgsl, wgsl
    assert "f32(f32(a))" not in wgsl, wgsl
    _naga_validates(tmp_path, wgsl)


def test_float_vector_times_integer_scalar_converts_only_the_scalar(tmp_path):
    # WGSL's splat constructor vecN<f32>(e) needs an f32 argument, so the
    # scalar converts to f32 alone; vecN<f32>(i) is one naga rejects.
    wgsl = _translate(tmp_path, MIXED_VECTOR_SCALAR)
    assert "(v * f32(a))" in wgsl, wgsl
    assert "(v * f32(u))" in wgsl, wgsl
    assert "vec3<f32>(a)" not in wgsl, wgsl
    assert "vec3<f32>(u)" not in wgsl, wgsl
    _naga_validates(tmp_path, wgsl)


def test_integer_scalar_beside_float_matrix_and_remainder_convert(tmp_path):
    # WGSL defines % for f32, and a scalar beside a matrix converts the
    # same way as beside a vector: m * f32(i).
    wgsl = _translate(tmp_path, MIXED_MATRIX_AND_REMAINDER)
    assert "(m * f32(a))" in wgsl, wgsl
    assert "(x % f32(a))" in wgsl, wgsl
    assert "f32(f32(a))" not in wgsl, wgsl
    _naga_validates(tmp_path, wgsl)
