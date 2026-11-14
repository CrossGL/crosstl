"""Componentwise matrix arithmetic: ``+``, ``-`` and ``/`` with a scalar or a matrix of one shape, and ``-m``.

GLSL 4.60 section 5.9 makes each of them a matrix of the matrix operand's type, computed one component at a
time. HLSL has every operator. MSL has ``+`` and ``-`` on two matrices and ``/`` by a scalar only, and SPIR-V
has no componentwise matrix instruction, so both are lowered one column at a time.
"""

import shutil
import subprocess

import pytest

from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.metal_codegen import (
    MetalCodeGen,
    UnsupportedMetalFeatureError,
)
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.matrix_shape import crossgl_matrix_componentwise_type
from crosstl.translator.parser import Parser

SHADER = """
shader Componentwise {
    compute {
        layout(std430, binding = 0) buffer Out { float data[]; } outBuf;
        void main() {
            int k = int(outBuf.data[0]);
            mat2x3 m = mat2x3(1.0, 2.0, 3.0, 4.0, 5.0, outBuf.data[1]);
            mat2x3 n = mat2x3(outBuf.data[2]);
            vec2 r = vec2(1.0, 10.0);
            vec3 q = (m / 2.0)[1];
            vec3 s = (m + 1.0)[k];
            vec3 t = (10.0 - m)[1];
            vec3 x = (m - 1.0) * r;
            mat2x3 p = -m;
            mat2x3 h = m;
            h += 1.0;
            h -= n;
            h /= n;
            mat2x3 e = 4.0 / n;
            mat2x3 f = m / n;
            outBuf.data[3] = q.x + s.y + t.z + x.x + p[1].x + h[k].y + e[0].z + f[1].y;
        }
    }
}
"""


def parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _check(command, cwd):
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr + result.stdout


@pytest.mark.parametrize(
    ("operator", "left", "right", "result"),
    [
        ("+", "mat2x3", "float", "mat2x3"),
        ("-", "float", "mat3", "mat3"),
        ("/", "mat4x2", "int", "mat4x2"),
        ("/", "dmat2", "dmat2x2", "dmat2"),
        ("+", "mat2x3", "mat3x2", None),
        ("-", "mat2x3", "vec3", None),
        ("*", "mat2x3", "float", None),
        ("+", "vec3", "float", None),
    ],
)
def test_componentwise_result_types(operator, left, right, result):
    assert crossgl_matrix_componentwise_type(operator, left, right) == result


def test_glsl_types_a_matrix_minus_a_scalar_as_a_matrix():
    glsl = GLSLCodeGen().generate(parse(SHADER))

    assert "vec3 x = ((m - 1.0) * r);" in glsl
    assert "vec3 t = (10.0 - m)[1];" in glsl


def test_metal_lowers_the_operators_msl_lacks_column_by_column(tmp_path):
    metal = MetalCodeGen().generate(parse(SHADER))

    assert "float3 q = (m / 2.0)[1];" in metal
    assert "float3 s = (__crossgl_add_float2x3_float(m, 1.0))[k];" in metal
    assert "float3 t = (__crossgl_sub_float_float2x3(10.0, m))[1];" in metal
    assert "float3 x = (__crossgl_sub_float2x3_float(m, 1.0)) * r;" in metal
    assert "float2x3 p = __crossgl_negate_float2x3(m);" in metal
    assert "h = __crossgl_add_float2x3_float(h, 1.0);" in metal
    assert "h -= n;" in metal
    assert "h = __crossgl_div_float2x3_float2x3(h, n);" in metal
    assert "float2x3 e = __crossgl_div_float_float2x3(4.0, n);" in metal
    assert "return float2x3(x - y[0], x - y[1]);" in metal
    assert "return float2x3(-x[0], -x[1]);" in metal
    if shutil.which("xcrun"):
        (tmp_path / "out.metal").write_text(metal, encoding="utf-8")
        _check(
            ["xcrun", "-sdk", "macosx", "metal", "-c", "out.metal", "-o", "out.air"],
            tmp_path,
        )


def test_metal_refuses_a_rewritten_update_of_a_side_effecting_target():
    source = SHADER.replace("h += 1.0;", "mat2x3 hs[2] = {h, h}; hs[k++] += 1.0;")
    with pytest.raises(UnsupportedMetalFeatureError, match="side effects"):
        MetalCodeGen().generate(parse(source))


def test_hlsl_keeps_the_native_operators(tmp_path):
    hlsl = HLSLCodeGen().generate(parse(SHADER))

    assert "float3 q = ((m / 2.0))._m01_m11_m21;" in hlsl
    assert "float3 x = mul((m - 1.0), r);" in hlsl
    assert "float3 s = transpose(((m + 1.0)))[k];" in hlsl
    assert "h += 1.0;" in hlsl
    assert "h /= n;" in hlsl
    dxc = shutil.which("dxc")
    if dxc:
        (tmp_path / "out.hlsl").write_text(hlsl, encoding="utf-8")
        _check([dxc, "-T", "cs_6_0", "-E", "CSMain", "out.hlsl"], tmp_path)


def test_spirv_lowers_matrix_arithmetic_one_column_at_a_time(tmp_path):
    assembly = VulkanSPIRVCodeGen().generate(parse(SHADER))

    for line in assembly.splitlines():
        if any(op in line for op in ("OpFAdd", "OpFSub", "OpFDiv", "OpFNegate")):
            assert "%mat" not in line, line
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        (tmp_path / "out.spvasm").write_text(assembly, encoding="utf-8")
        _check(
            ["spirv-as", "--target-env", "vulkan1.1", "out.spvasm", "-o", "out.spv"],
            tmp_path,
        )
        _check(["spirv-val", "--target-env", "vulkan1.1", "out.spv"], tmp_path)
