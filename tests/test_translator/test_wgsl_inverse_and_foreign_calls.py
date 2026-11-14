"""WGSL: ``inverse`` is lowered to a helper, and a call to another language's built-in function is refused."""

import math
import shutil
import subprocess

import pytest

from crosstl.translator.codegen.wgsl_codegen import WGSLCodeGen
from crosstl.translator.codegen.wgsl_foreign_calls import WGSLForeignBuiltinCallError
from crosstl.translator.codegen.wgsl_matrix_inverse import WGSLMatrixInverseError
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

PROGRAM = """shader Probe {{
{functions}
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* values;
        void main(uvec3 gid @builtin(global_invocation_id)) {{
            float x = values[gid.x];
{body}
        }}
    }}
}}
"""


def generate(body, functions=""):
    source = PROGRAM.format(functions=functions, body=body)
    return WGSLCodeGen().generate(Parser(Lexer(source).get_tokens()).parse())


def validate(tmp_path, wgsl):
    if shutil.which("naga") is None:
        return
    path = tmp_path / "out.wgsl"
    path.write_text(wgsl)
    checked = subprocess.run(["naga", str(path)], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stdout + checked.stderr


def test_inverse_is_lowered_to_one_helper_per_matrix_type(tmp_path):
    # WGSL declares no inverse (WGSL section 17.5); the call was written unchanged and failed validation.
    wgsl = generate(
        """            mat2 a = mat2(x, 2.0, 3.0, 4.0);
            mat3 b = mat3(x);
            mat4 c = mat4(x);
            mat3 d = inverse(inverse(b));
            values[gid.x] = inverse(a)[0][0] + d[1][1] + inverse(c)[3][3] + inverse(b)[2][2];"""
    )
    for size in (2, 3, 4):
        assert wgsl.count(f"fn crosstl_inverse_mat{size}x{size}_f32(") == 1
    assert "crosstl_inverse_mat3x3_f32(crosstl_inverse_mat3x3_f32(b))" in wgsl
    assert " inverse(" not in wgsl
    validate(tmp_path, wgsl)


def test_inverse_helper_pivots_on_the_largest_column_entry():
    wgsl = generate("            values[gid.x] = inverse(mat2(x))[0][0];")
    helper = wgsl[wgsl.index("fn crosstl_inverse_mat2x2_f32(") :]
    assert "if (candidate > pivotAbs)" in helper
    assert "var inv = mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));" in helper


def test_inverse_of_a_non_square_matrix_is_refused():
    with pytest.raises(WGSLMatrixInverseError, match="only square f32 matrices"):
        generate("            mat2x3 m = mat2x3(x); values[gid.x] = inverse(m)[0][0];")


def test_a_declared_inverse_function_is_called():
    wgsl = generate(
        "            values[gid.x] = inverse(x);",
        functions="    float inverse(float v) { return 1.0 / v; }",
    )
    assert "inverse(x)" in wgsl
    assert "crosstl_inverse_" not in wgsl


@pytest.mark.parametrize("call", ["lit(x, x, x).y", "simd_sum(x)", "rcp(x)"])
def test_another_languages_built_in_function_is_refused(call):
    # lit and rcp are HLSL, simd_sum is MSL: WGSL declares none of them, so the call cannot be written.
    with pytest.raises(WGSLForeignBuiltinCallError, match="WGSL declares no built-in"):
        generate(f"            values[gid.x] = {call};")


def test_a_declared_function_with_a_built_in_name_is_called(tmp_path):
    wgsl = generate(
        "            values[gid.x] = lit(x);",
        functions="    float lit(float v) { return v * 2.0; }",
    )
    assert "values[gid.x] = lit(x);" in wgsl
    validate(tmp_path, wgsl)


def test_log10_is_lowered_through_log2(tmp_path):
    # WGSL declares log2 but not log10, so the call cannot be written as named.
    wgsl = generate(
        "            vec3 lanes = vec3(x, x + 1.0, x + 2.0);\n"
        "            values[gid.x] = log10(x) + log10(lanes).y;"
    )
    assert "log10(" not in wgsl
    assert "(log2(x) * 0.3010299956639812)" in wgsl
    assert "(log2(lanes) * 0.3010299956639812)" in wgsl
    # The constant is log10(2) to binary64 precision.
    assert float("0.3010299956639812") == pytest.approx(math.log10(2.0), abs=1e-16)
    validate(tmp_path, wgsl)


def test_a_declared_log10_function_is_called(tmp_path):
    wgsl = generate(
        "            values[gid.x] = log10(x);",
        functions="    float log10(float v) { return v + 1.0; }",
    )
    assert "values[gid.x] = log10(x);" in wgsl
    assert "log2(" not in wgsl
