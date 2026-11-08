"""WGSL output gives a shader-scope array variable's brace list its declared type (#3676).

``float g[3] = float[3](1.0);`` used to be written ``var<private> g: array<f32, 3> = array(1.0);``, a one-element
array that naga rejects. A short list is zero-filled, as it is for a local declaration and a shader-scope constant,
and a list longer than its array is refused.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator.codegen.wgsl_codegen import WGSLAggregateInitializerError

PROGRAM = textwrap.dedent("""
    shader Probe {
        buffer float* outp;
        %s
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() {
                uint i = gl_GlobalInvocationID.x;
                outp[i] = g[0] + g[2];
            }
        }
    }
""")


def translate(tmp_path, declaration):
    path = tmp_path / "probe.cgl"
    path.write_text(PROGRAM % declaration)
    output = crosstl.translate(str(path), backend="wgsl", format_output=False)
    if shutil.which("naga"):
        (tmp_path / "out.wgsl").write_text(output)
        checked = subprocess.run(
            ["naga", "out.wgsl"], cwd=tmp_path, capture_output=True, text=True
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr
    return output


@pytest.mark.parametrize("initializer", ["float[3](1.0)", "{1.0}"])
def test_short_shader_scope_array_initializer_is_zero_filled(tmp_path, initializer):
    output = translate(tmp_path, f"float g[3] = {initializer};")
    assert "var<private> g: array<f32, 3> = array<f32, 3>(1.0, f32(), f32());" in output


def test_full_shader_scope_array_initializer_is_unchanged(tmp_path):
    output = translate(tmp_path, "float g[3] = {1.0, 2.0, 3.0};")
    assert "var<private> g: array<f32, 3> = array(1.0, 2.0, 3.0);" in output


@pytest.mark.parametrize(
    "declaration, array_type",
    [
        ("float g[3] = {1.0, 2.0, 3.0, 4.0};", "array<f32, 3>"),
        (
            "const int COUNT = 3;\n    float g[COUNT] = {1.0, 2.0, 3.0, 4.0};",
            "array<f32, COUNT>",
        ),
    ],
)
def test_long_shader_scope_array_initializer_is_refused(
    tmp_path, declaration, array_type
):
    with pytest.raises(
        WGSLAggregateInitializerError,
        match=f"'{array_type}' expects at most 3 elements, got 4",
    ):
        translate(tmp_path, declaration)


def test_short_shader_scope_constant_array_is_still_zero_filled(tmp_path):
    output = translate(tmp_path, "const float g[3] = {1.0};")
    assert "const g: array<f32, 3> = array<f32, 3>(1.0, f32(), f32());" in output
