"""WGSL output lowers isnan, isinf, isfinite and signbit to bit tests.

WGSL has none of these built-in functions, so emitting the call verbatim produced a module that failed validation.
WGSL also lets an implementation assume that no NaN or infinity occurs (WGSL section 15.7), so the lowering tests the
binary32 bits through ``bitcast`` rather than comparing the value with itself.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator.codegen.wgsl_float_classification import (
    WGSLFloatClassificationError,
)

PROGRAM = textwrap.dedent("""
    shader Probe {
        buffer float values[];
        buffer uint flags[];
        %s
        compute {
            layout(local_size_x = 1) in;
            void main() {
                uint i = gl_GlobalInvocationID.x;
                %s
            }
        }
    }
""")


def wgsl(tmp_path, body, helpers=""):
    """WGSL output for a compute body, checked with naga when it is installed."""
    path = tmp_path / "probe.cgl"
    path.write_text(PROGRAM % (helpers, body))
    output = crosstl.translate(str(path), backend="wgsl", format_output=False)
    if shutil.which("naga"):
        (tmp_path / "out.wgsl").write_text(output)
        checked = subprocess.run(
            ["naga", "out.wgsl"], cwd=tmp_path, capture_output=True, text=True
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr
    return output


@pytest.mark.parametrize(
    ("function", "test"),
    [
        ("isnan", "((bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u)"),
        ("isinf", "((bitcast<u32>(x) & 0x7fffffffu) == 0x7f800000u)"),
        ("isfinite", "((bitcast<u32>(x) & 0x7fffffffu) < 0x7f800000u)"),
        ("signbit", "((bitcast<u32>(x) & 0x80000000u) != 0u)"),
    ],
)
def test_scalar_classification_is_a_bit_test(tmp_path, function, test):
    output = wgsl(
        tmp_path,
        f"float x = values[i]; bool b = {function}(x); flags[i] = b ? 1u : 0u;",
    )
    assert f"var b: bool = {test};" in output
    assert f"{function}(" not in output


@pytest.mark.parametrize("width", [2, 3, 4])
def test_vector_classification_tests_every_component(tmp_path, width):
    output = wgsl(
        tmp_path,
        f"vec{width} v = vec{width}(values[i]); bvec{width} n = isnan(v); "
        f"bvec{width} s = signbit(v); flags[i] = any(n) || all(s) ? 1u : 0u;",
    )
    bits = f"vec{width}<u32>"
    assert (
        f"var n: vec{width}<bool> = ((bitcast<{bits}>(v) & {bits}(0x7fffffffu)) > "
        f"{bits}(0x7f800000u));"
    ) in output
    assert (
        f"var s: vec{width}<bool> = ((bitcast<{bits}>(v) & {bits}(0x80000000u)) != "
        f"{bits}(0u));"
    ) in output


def test_classification_in_conditions_and_reductions(tmp_path):
    output = wgsl(
        tmp_path,
        "float x = values[i]; vec2 v = vec2(x, -x); uint bits = 0u; "
        "if (isinf(x)) { bits = bits | 1u; } "
        "if (all(isfinite(v))) { bits = bits | 2u; } "
        "flags[i] = isnan(x) ? bits : 0u;",
    )
    assert "isinf(" not in output and "isfinite(" not in output
    assert "all(((bitcast<vec2<u32>>(v) & vec2<u32>(0x7fffffffu)) < " in output


def test_operand_with_side_effects_is_evaluated_once(tmp_path):
    output = wgsl(
        tmp_path,
        "flags[i] = isnan(next(i)) ? 1u : 0u;",
        helpers="float next(uint k) { flags[k] = flags[k] + 1u; return values[k]; }",
    )
    assert output.count("next(i)") == 1
    assert "bitcast<u32>(next(i))" in output


def test_half_operand_is_tested_as_f32(tmp_path):
    output = wgsl(
        tmp_path,
        "half h = half(values[i]); flags[i] = isinf(h) ? 1u : 0u;",
    )
    assert "((bitcast<u32>(h) & 0x7fffffffu) == 0x7f800000u)" in output


def test_user_function_of_the_same_name_is_called(tmp_path):
    output = wgsl(
        tmp_path,
        "flags[i] = isnan(values[i]) ? 1u : 0u;",
        helpers="bool isnan(float x) { return x > 1.0; }",
    )
    assert "bitcast<u32>" not in output
    assert "isnan(values[i])" in output


@pytest.mark.parametrize(
    ("operand", "described"),
    [("int(values[i])", "i32"), ("uvec2(1u, 2u)", "vec2<u32>")],
)
def test_non_float_operand_is_refused(tmp_path, operand, described):
    with pytest.raises(WGSLFloatClassificationError, match=described) as raised:
        wgsl(tmp_path, f"flags[i] = isfinite({operand}) ? 1u : 0u;")
    assert raised.value.operation == "isfinite"
    assert raised.value.project_diagnostic_code == (
        "project.translate.wgsl-float-classification-unsupported"
    )
