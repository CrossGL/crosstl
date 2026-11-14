"""A shader-scope ``let`` or ``var`` with no written type takes its WGSL type from the initializer.

Regression for #2444: WGSL output declared ``var<private> scale: void = 2.0;``, which naga rejects, because the
global-variable path printed the missing type as ``void``.
"""

import shutil
import subprocess

import pytest

import crosstl


def translate(tmp_path, declaration, use="scale"):
    path = tmp_path / "shader.cgl"
    path.write_text(
        f"shader main {{ {declaration} compute {{ buffer float data[]; "
        f"void main() {{ data[0] = {use}; }} }} }}",
        encoding="utf-8",
    )
    output = crosstl.translate(str(path), backend="wgsl")
    if shutil.which("naga"):
        (tmp_path / "out.wgsl").write_text(output, encoding="utf-8")
        checked = subprocess.run(
            ["naga", "out.wgsl"], cwd=tmp_path, capture_output=True, text=True
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr
    return output


@pytest.mark.parametrize("binding", ["let", "var"])
@pytest.mark.parametrize(
    "value, declaration, use",
    [
        ("2.0", "var<private> scale: f32 = 2.0;", "scale"),
        ("2", "var<private> scale: i32 = 2;", "float(scale)"),
        ("2u", "var<private> scale: u32 = 2u;", "float(scale)"),
        ("true", "var<private> scale: bool = true;", "scale ? 1.0 : 0.0"),
        ("-2.5 * 2.0", "var<private> scale: f32 = (-2.5 * 2.0);", "scale"),
        (
            "vec3(1.0, 2.0, 3.0)",
            "var<private> scale: vec3<f32> = vec3<f32>(1.0, 2.0, 3.0);",
            "scale.y",
        ),
        (
            "ivec2(1, 2)",
            "var<private> scale: vec2<i32> = vec2<i32>(1, 2);",
            "float(scale.x)",
        ),
        (
            "mat2(1.0, 0.0, 0.0, 1.0)",
            "var<private> scale: mat2x2<f32> = mat2x2<f32>(1.0, 0.0, 0.0, 1.0);",
            "scale[1].y",
        ),
    ],
)
def test_untyped_global_takes_the_initializer_type(
    tmp_path, binding, value, declaration, use
):
    output = translate(tmp_path, f"{binding} scale = {value};", use)

    assert declaration in output
    assert ": void" not in output


def test_untyped_global_with_no_inferable_type_is_refused(tmp_path):
    with pytest.raises(
        ValueError,
        match="cannot infer the type of untyped global scale from its initializer",
    ):
        translate(tmp_path, "let scale = missing_symbol;")
