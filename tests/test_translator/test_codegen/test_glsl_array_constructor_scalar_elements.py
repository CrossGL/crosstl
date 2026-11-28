"""GLSL array constructors wrap scalar elements in the element type (#3636).

A GLSL array constructor takes one value of the element type per element
(GLSL 4.60 section 5.4.4), so a scalar where a vector or matrix element is
expected is rejected by glslang with ``'constructor' : not enough data
provided for construction``. Each scalar element must be written as the
element type's constructor: ``mat4[3](mat4(1.0), mat4(0.0), mat4(0.0))``.
"""

import shutil
import subprocess

import crosstl.translator
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen

SCALAR_ELEMENTS_CGL = """
shader Probe3636 {
    buffer float* outp;
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            float k = 2.0;
            mat4 a[3] = mat4[3](1.0);
            mat2 b[2] = mat2[2](1.0, 2.0);
            vec3 v[2];
            v = vec3[2](1.0, 2.0);
            vec3 w[2] = vec3[2](k, 1.0);
            mat3 z[2] = mat3[2](0.0);
            mat3 m[2];
            m = mat3[2](0.0);
            vec3 u[2];
            u = vec3[2](1.0);
            vec3 already[2] = vec3[2](vec3(1.0), vec3(2.0));
            float f[2] = float[2](1.0, 2.0);
            outp[i] = a[0][0][0] + b[1][0][0] + v[1].x + w[0].x + z[0][0][0]
                + m[0][0][0] + u[0].x + already[0].x + f[0];
        }
    }
}
"""


def _generate():
    return GLSLCodeGen().generate_stage(
        crosstl.translator.parse(SCALAR_ELEMENTS_CGL), "compute"
    )


def test_glsl_array_constructors_wrap_scalar_matrix_elements():
    code = _generate()

    assert "mat4 a[3] = mat4[3](mat4(1.0), mat4(0.0), mat4(0.0));" in code
    assert "mat2 b[2] = mat2[2](mat2(1.0), mat2(2.0));" in code
    assert "mat3 z[2] = mat3[2](mat3(0.0), mat3(0.0));" in code


def test_glsl_array_constructors_wrap_scalar_vector_elements_outside_declarations():
    code = _generate()

    assert "v = vec3[2](vec3(1.0), vec3(2.0));" in code
    assert "vec3 w[2] = vec3[2](vec3(k), vec3(1.0));" in code


def test_glsl_array_constructors_wrap_scalar_elements_of_padded_constructors():
    # #3624 pads a short constructor outside a declaration; the given scalar
    # and the padded zero element must both be the element type (#3636).
    code = _generate()

    assert "m = mat3[2](mat3(0.0), mat3(0.0));" in code
    assert "u = vec3[2](vec3(1.0), vec3(0.0));" in code


def test_glsl_array_constructors_leave_typed_elements_and_scalar_arrays_alone():
    code = _generate()

    assert "vec3[2](vec3(1.0), vec3(2.0))" in code
    assert "vec3(vec3(1.0))" not in code
    assert "float f[2] = float[2](1.0, 2.0);" in code


NON_ARRAY_SCALAR_TO_MATRIX_CGL = """
shader Probe5077 {
    buffer float* outp;
    mat4 scale(mat4 m) { return m; }
    mat4 pick() { return 5.0; }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            mat4 m;
            m = 2.0;
            mat4 d = 2.0;
            mat4 k = scale(3.0);
            mat4 r = pick();
            outp[i] = m[0][1] + d[0][1] + k[0][1] + r[0][1];
        }
    }
}
"""


def test_glsl_scalar_to_matrix_stays_bare_outside_array_constructors():
    # #5077 review: #3636 settles what a scalar means only for array
    # elements (``mat4(1.0)`` is the identity there). CrossGL has no rule
    # for an implicit scalar-to-matrix conversion anywhere else, so a scalar
    # assigned to a matrix variable, used as a matrix initializer, returned
    # from a matrix function, or passed to a matrix parameter must be left
    # exactly as written -- glslang rejects it, so the problem stays visible
    # instead of silently compiling with diagonal semantics nobody chose.
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(NON_ARRAY_SCALAR_TO_MATRIX_CGL), "compute"
    )

    assert "m = 2.0;" in code
    assert "mat4 d = 2.0;" in code
    assert "scale(3.0)" in code
    assert "return 5.0;" in code
    assert "mat4(2.0)" not in code
    assert "mat4(3.0)" not in code
    assert "mat4(5.0)" not in code


def test_glsl_array_constructor_scalar_elements_validate_if_available(tmp_path):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        return
    code = _generate()
    source_path = tmp_path / "scalar_array_elements.comp"
    output_path = tmp_path / "scalar_array_elements.spv"
    source_path.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [glslang, "-S", "comp", str(source_path), "-o", str(output_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    diagnostics = "\n".join(
        part for part in (result.stdout, result.stderr) if part.strip()
    )
    assert result.returncode == 0, diagnostics
    assert output_path.is_file()
