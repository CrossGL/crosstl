"""WGSL output drops a numeric unary plus, because WGSL has no unary ``+`` operator."""

import re
import shutil
import subprocess

import pytest

import crosstl

PRELUDE = """shader Probe {
    const float OFFSET = +1.5;
    buffer float* values;
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            float x = values[i];
            int k = int(i);
            uint u = i;
            vec2 v = vec2(x, 2.0);
            mat2 m = mat2(1.0, 2.0, 3.0, 4.0);
            bool b = x > 0.0;
"""
EPILOGUE = """
        }
    }
}
"""


def translate(tmp_path, body):
    path = tmp_path / "probe.cgl"
    path.write_text(PRELUDE + body + EPILOGUE)
    output = crosstl.translate(str(path), backend="wgsl", format_output=False)
    if shutil.which("naga"):
        (tmp_path / "out.wgsl").write_text(output)
        checked = subprocess.run(
            ["naga", "out.wgsl"], cwd=tmp_path, capture_output=True, text=True
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr
    return output


def assignment(output, target):
    match = re.search(rf"{re.escape(target)} = (.*);", output)
    assert match, output
    return match.group(1)


def test_scalar_unary_plus_is_dropped(tmp_path):
    output = translate(tmp_path, "values[i] = +x;")
    assert assignment(output, "values[i]") == "x"


def test_unary_plus_inside_negation_keeps_tokens_apart(tmp_path):
    output = translate(tmp_path, "values[i] = -(+x) - (-(+(-x)));")
    rhs = assignment(output, "values[i]")
    # ``-(-x)`` keeps its parentheses so the two minus signs never form ``--``.
    assert rhs == "(-x - -(-x))"


def test_integer_and_unsigned_unary_plus(tmp_path):
    output = translate(
        tmp_path,
        "int kk = +k + +(k * 2);\n uint uu = +u;\n values[i] = float(kk + int(uu));",
    )
    assert assignment(output, "var kk: i32") == "(k + (k * 2))"
    assert assignment(output, "var uu: u32") == "u"


def test_vector_and_member_access(tmp_path):
    output = translate(
        tmp_path, "vec2 w = +v;\n values[i] = (+w).y + (+vec2(x, 1.0)).x;"
    )
    assert assignment(output, "var w: vec2<f32>") == "v"
    assert assignment(output, "values[i]") == "(w.y + vec2<f32>(x, 1.0).x)"


def test_matrix_unary_plus(tmp_path):
    output = translate(tmp_path, "mat2 n = +m;\n values[i] = n[1][0];")
    assert assignment(output, "var n: mat2x2<f32>") == "m"


def test_global_constant_initializer(tmp_path):
    output = translate(tmp_path, "values[i] = OFFSET;")
    assert re.search(r"const OFFSET: f32 = 1\.5;", output), output
    assert "+1.5" not in output


def test_boolean_unary_plus_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unary plus"):
        translate(tmp_path, "bool c = +b;\n values[i] = c ? 1.0 : 0.0;")
