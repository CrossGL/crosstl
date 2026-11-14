"""GLSL output declares a double-precision matrix with its ``dmat`` type (#3729).

``dmat3 m = dmat3(double(x));`` used to be written ``mat3 m = dmat3(double(x));``,
which glslang rejects: a double matrix cannot initialize a float one.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl

PROGRAM = textwrap.dedent("""
    shader P {
        buffer float* outp;
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() {
                uint i = gl_GlobalInvocationID.x;
                double d = double(outp[i]);
                %s
            }
        }
    }
""")

# Parameters, return types, structure members and uniform block members take the
# declared type the same way a local does.
MEMBERS_PROGRAM = textwrap.dedent("""
    shader P {
        struct Frame { dmat3 basis; dmat4x3 xf; };
        cbuffer Params @binding(1) { dmat3 rot; dmat2x3 skew; };
        buffer float* outp;
        dmat3 scale(dmat3 a, double s) { return a * s; }
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() {
                uint i = gl_GlobalInvocationID.x;
                double d = double(outp[i]);
                Frame f;
                f.basis = scale(rot, d);
                f.xf = dmat4x3(d);
                dvec3 s = skew * dvec2(d);
                outp[i] = float(f.basis[1][1] + f.xf[3][2] + s.y);
            }
        }
    }
""")


def translate_source(tmp_path, source):
    path = tmp_path / "probe.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend="glsl", format_output=False)


def translate(tmp_path, body):
    return translate_source(tmp_path, PROGRAM % body)


@pytest.mark.parametrize(
    "matrix_type", ["dmat2", "dmat3", "dmat4", "dmat2x3", "dmat4x2"]
)
def test_double_matrix_local_keeps_its_double_type(tmp_path, matrix_type):
    output = translate(
        tmp_path,
        f"{matrix_type} m = {matrix_type}(d); outp[i] = float(m[1][1]);",
    )
    assert f"{matrix_type} m = {matrix_type}(d);" in output


def test_double_matrix_parameters_and_members_keep_their_double_type(tmp_path):
    output = translate_source(tmp_path, MEMBERS_PROGRAM)
    assert "dmat3 scale(dmat3 a, double s) {" in output
    assert "dmat3 basis;" in output
    assert "dmat4x3 xf;" in output
    # A float matrix here would be laid out with std140 offsets 0 and 48, not the
    # 0 and 96 of the host's doubles.
    assert "dmat3 rot;" in output
    assert "dmat2x3 skew;" in output


def test_float_matrix_and_double_vector_locals_are_unchanged(tmp_path):
    output = translate(
        tmp_path,
        "mat3 f = mat3(outp[i]); mat2x3 g = mat2x3(outp[i]); dvec3 v = dvec3(d); "
        "outp[i] = f[0][0] + g[1][2] + float(v.x);",
    )
    assert "mat3 f = mat3(outp[i]);" in output
    assert "mat2x3 g = mat2x3(outp[i]);" in output
    assert "dvec3 v = dvec3(d);" in output


def test_double_matrix_output_compiles_with_glslang(tmp_path):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")
    sources = {
        "local": PROGRAM % "dmat3 m = dmat3(d); outp[i] = float(m[1][1]);",
        "members": MEMBERS_PROGRAM,
    }
    for name, source in sources.items():
        shader = tmp_path / f"{name}.comp"
        shader.write_text(translate_source(tmp_path, source))
        result = subprocess.run(
            [glslang, "-S", "comp", str(shader)],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
