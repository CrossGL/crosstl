"""WGSL output types an untyped mixed-type scalar built-in call by its converted type (#4508)."""

import textwrap
import crosstl

PROGRAM = textwrap.dedent("""
    shader main {
        const int K = 3;
        const uint U = 4u;
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            buffer float data[];
            void main() {
                float f = data[1];
                vec3 v = vec3(data[0], data[1], data[2]);
                %s
            }
        }
    }
""")


def translate(tmp_path, body):
    path = tmp_path / "probe.cgl"
    path.write_text(PROGRAM % body)
    return crosstl.translate(str(path), backend="wgsl", format_output=False)


def test_mixed_scalar_builtin_types_converted(tmp_path):
    output = translate(tmp_path, "var x = max(1, f); data[0] = float(x);")
    assert "var x: f32" in output, f"expected var x: f32 in output:\n{output}"

    output = translate(tmp_path, "var x = min(2, U); data[0] = float(x);")
    assert "var x: u32" in output, f"expected var x: u32 in output:\n{output}"

    output = translate(tmp_path, "var x = pow(2, f); data[0] = float(x);")
    assert "var x: f32" in output, f"expected var x: f32 in output:\n{output}"

    output = translate(tmp_path, "var x = mod(1, f); data[0] = float(x);")
    assert "var x: f32" in output, f"expected var x: f32 in output:\n{output}"

    output = translate(tmp_path, "var x = atan(1, f); data[0] = float(x);")
    assert "var x: f32" in output, f"expected var x: f32 in output:\n{output}"

    output = translate(tmp_path, "var x = clamp(1, f, 3.0); data[0] = float(x);")
    assert "var x: f32" in output, f"expected var x: f32 in output:\n{output}"


def test_unchanged_builtin_types_preserved(tmp_path):
    output = translate(tmp_path, "var x = max(f, 1); data[0] = float(x);")
    assert "var x: f32" in output, f"expected var x: f32 in output:\n{output}"

    output = translate(tmp_path, "var x = abs(K); data[0] = float(x);")
    assert "var x: i32" in output, f"expected var x: i32 in output:\n{output}"

    output = translate(tmp_path, "var x = max(1, v); data[0] = x.x;")
    assert "var x: vec3<f32>" in output, f"expected var x: vec3<f32> in output:\n{output}"

    output = translate(tmp_path, "var x = min(U, K); data[0] = float(x);")
    assert "var x: u32" in output, f"expected var x: u32 in output:\n{output}"
