"""WGSL output resolves mixed scalar built-in calls by GLSL overload rules (#4508)."""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator.refusals import WGSLRefusal

PROGRAM = textwrap.dedent("""
    shader main {
        const int K = 3;
        const uint U = 4u;
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            buffer float data[];
            void main() {
                float f = data[1];
                bool b = true;
                vec3 v = vec3(data[0], data[1], data[2]);
                %s
            }
        }
    }
    """)


def translate(tmp_path, body):
    path = tmp_path / "probe.cgl"
    path.write_text(PROGRAM % body, encoding="utf-8")
    return crosstl.translate(str(path), backend="wgsl", format_output=False)


def validate_wgsl_with_naga(tmp_path, code):
    naga = shutil.which("naga")
    if naga is None:
        pytest.skip("naga is not installed")
    source = tmp_path / "shader.wgsl"
    source.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [naga, str(source)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    diagnostics = "\n".join(
        part for part in (result.stdout, result.stderr) if part.strip()
    )
    assert result.returncode == 0, diagnostics


@pytest.mark.parametrize(
    ("expression", "expected_type"),
    [
        ("max(1, f)", "f32"),
        ("min(2, U)", "u32"),
        ("pow(2, f)", "f32"),
        ("mod(1, f)", "f32"),
        ("atan(1, f)", "f32"),
        ("clamp(1, f, 3.0)", "f32"),
        ("mix(1, f, 0.5)", "f32"),
        ("mix(1, 2, b)", "i32"),
        ("mix(U, U, b)", "u32"),
        ("mix(f, f, b)", "f32"),
        ("step(1, f)", "f32"),
        ("smoothstep(0, 1, f)", "f32"),
        ("max(1, 2u)", "u32"),
        ("clamp(1, U, 7)", "u32"),
        ("max(1, max(2, f))", "f32"),
    ],
)
def test_mixed_scalar_builtin_types_validate_with_naga(
    tmp_path, expression, expected_type
):
    output = translate(tmp_path, f"var x = {expression}; data[0] = float(x);")
    assert f"var x: {expected_type}" in output, output
    validate_wgsl_with_naga(tmp_path, output)


@pytest.mark.parametrize(
    ("expression", "expected_type"),
    [
        ("max(f, 1)", "f32"),
        ("abs(K)", "i32"),
        ("max(1, v)", "vec3<f32>"),
        ("min(U, K)", "u32"),
    ],
)
def test_unchanged_builtin_types_validate_with_naga(
    tmp_path, expression, expected_type
):
    assignment = (
        "data[0] = x.x;" if expression == "max(1, v)" else "data[0] = float(x);"
    )
    output = translate(tmp_path, f"var x = {expression}; {assignment}")
    assert f"var x: {expected_type}" in output, output
    validate_wgsl_with_naga(tmp_path, output)


@pytest.mark.parametrize(
    "expression",
    [
        "max(fma(f, f, f), 1)",
        "max(determinant(mat2(1.0)), 1)",
    ],
)
def test_unresolved_scalar_builtin_types_are_refused(tmp_path, expression):
    with pytest.raises(WGSLRefusal, match="cannot infer a concrete type"):
        translate(tmp_path, f"var x = {expression}; data[0] = float(x);")
