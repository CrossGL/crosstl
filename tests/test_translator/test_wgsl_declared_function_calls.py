"""WGSL calls of functions the shader declares resolve to them before any built-in or built-in alias."""

import re
import shutil
import subprocess

import crosstl

PROGRAM = """shader Probe {{
    buffer float values[];
{functions}
    compute {{
        layout(local_size_x = 1) in;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            float x = values[i];
            values[i] = {value};
        }}
    }}
}}
"""


def translate(tmp_path, functions, value):
    path = tmp_path / "probe.cgl"
    path.write_text(PROGRAM.format(functions=functions, value=value))
    output = crosstl.translate(str(path), backend="wgsl", format_output=False)
    if shutil.which("naga"):
        (tmp_path / "out.wgsl").write_text(output)
        checked = subprocess.run(
            ["naga", "out.wgsl"], cwd=tmp_path, capture_output=True, text=True
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr
    return output


def stored_value(output):
    match = re.search(r"values\[i\] = (.*);", output)
    assert match, output
    return match.group(1)


def test_functions_named_like_hlsl_builtins_keep_their_calls(tmp_path):
    # CrossGL spells these built-ins mix and inversesqrt; lerp and rsqrt are only aliases.
    output = translate(
        tmp_path,
        """
    float lerp(float a, float b) { return a * 2.0 - b; }
    float rsqrt(float x) { return x + 0.25; }
    float fabs(float x) { return x * x; }""",
        "lerp(x, 1.0) + rsqrt(x) + fabs(x)",
    )
    assert "fn lerp(a: f32, b: f32) -> f32" in output
    assert "fn rsqrt(x: f32) -> f32" in output
    value = stored_value(output)
    assert "lerp(x, 1.0)" in value
    assert "rsqrt(x)" in value
    assert "mix" not in value
    assert "inverseSqrt" not in value


def test_function_named_like_a_wgsl_builtin_is_renamed(tmp_path):
    output = translate(
        tmp_path,
        """
    float clamp(float x) { return x * 3.0; }
    float saturate(float x) { return x + 7.0; }""",
        "clamp(x) + saturate(x) + clamp(x, 0.0, 1.0)",
    )
    assert "fn clamp_(x: f32) -> f32" in output
    assert "fn saturate_(x: f32) -> f32" in output
    value = stored_value(output)
    assert "clamp_(x)" in value
    assert "saturate_(x)" in value
    # The three-argument call matches no declared function, so it stays the built-in.
    assert "clamp(x, 0.0, 1.0)" in value


def test_declared_two_argument_atan_is_not_atan2(tmp_path):
    output = translate(
        tmp_path,
        """
    float atan(float y, float x) { return y * x; }""",
        "atan(x, 2.0)",
    )
    assert stored_value(output) == "atan_(x, 2.0)"


def test_builtin_alias_without_declaration_still_maps(tmp_path):
    output = translate(tmp_path, "", "lerp(x, 1.0, 0.5) + rsqrt(x)")
    value = stored_value(output)
    assert "mix(x, 1.0, 0.5)" in value
    assert "inverseSqrt(x)" in value


def test_declared_function_and_alias_of_other_arity(tmp_path):
    # A declared two-argument lerp leaves the three-argument call to the built-in alias.
    output = translate(
        tmp_path,
        """
    float lerp(float a, float b) { return a - b; }""",
        "lerp(x, 1.0) + lerp(x, 1.0, 0.5)",
    )
    value = stored_value(output)
    assert "lerp(x, 1.0)" in value
    assert "mix(x, 1.0, 0.5)" in value
