"""SPIR-V lowering of the GLSL matrix functions and of named workgroup sizes."""

import re

import pytest

import crosstl.translator
from crosstl.translator.codegen.SPIRV_codegen import (
    SPIRVBuiltinLoweringError,
    VulkanSPIRVCodeGen,
)
from tests.test_translator.test_codegen.test_shader_validation import (
    validate_spirv_shader_source,
)

MATRIX_FUNCTIONS = """
shader MatrixFunctions {
    fragment {
        vec4 main(vec3 a, vec2 b) @ gl_FragColor {
            mat3 m3 = mat3(a, a.yzx, a.zxy);
            mat4 m4 = mat4(2.0);
            mat2x3 m23 = mat2x3(a, a.zyx);
            mat3 t = transpose(m3);
            mat3x2 tt = transpose(m23);
            mat4 i = inverse(m4);
            float d = determinant(m3);
            mat2x3 o = outerProduct(a, b);
            mat3 c = matrixCompMult(m3, t);
            return vec4(t[0] + o[1] + c[2] + vec3(tt[1], d), i[3][3]);
        }
    }
}
"""


def generate(source):
    return VulkanSPIRVCodeGen().generate(crosstl.translator.parse(source))


def test_matrix_functions_lower_to_core_and_glsl_std450_instructions():
    code = generate(MATRIX_FUNCTIONS)

    assert len(re.findall(r"= OpTranspose %\d+ %\d+$", code, re.M)) == 2
    assert (
        len(re.findall(r"= OpExtInst %\d+ %\d+ MatrixInverse %\d+$", code, re.M)) == 1
    )
    assert len(re.findall(r"= OpExtInst %\d+ %\d+ Determinant %\d+$", code, re.M)) == 1
    assert len(re.findall(r"= OpOuterProduct %\d+ %\d+ %\d+$", code, re.M)) == 1
    # matrixCompMult multiplies column by column; spirv-val checks the result shapes below.
    assert len(re.findall(r"= OpFMul ", code)) == 3


def test_matrix_functions_module_validates(tmp_path):
    validate_spirv_shader_source(
        tmp_path, "matrix_functions", MATRIX_FUNCTIONS, target_env="vulkan1.0"
    )


@pytest.mark.parametrize(
    "call, detail",
    [
        ("determinant(m23)", "requires a square matrix"),
        ("inverse(m23)[0].x", "requires a square matrix"),
        ("transpose(a)[0].x", "requires a float matrix operand"),
        ("outerProduct(a, ia)[0].x", "two float vectors"),
        ("matrixCompMult(m23, m3)[0].x", "two float matrices of the same type"),
        ("determinant(m3, m3)", "requires 1 operand"),
    ],
)
def test_matrix_functions_refuse_invalid_operands(call, detail):
    source = f"""
shader BadMatrixCall {{
    fragment {{
        vec4 main(vec3 a) @ gl_FragColor {{
            ivec3 ia = ivec3(a);
            mat3 m3 = mat3(1.0);
            mat2x3 m23 = mat2x3(a, a);
            float value = {call};
            return vec4(value);
        }}
    }}
}}
"""
    with pytest.raises(SPIRVBuiltinLoweringError, match=detail):
        generate(source)


def test_compute_local_size_resolves_named_integer_constants(tmp_path):
    source = """
shader NamedWorkgroupSize {
    const int TILE = 16;
    const int DEPTH = TILE / 8;
    compute {
        layout(local_size_x = TILE, local_size_y = TILE, local_size_z = DEPTH) in;
        void main() {
        }
    }
}
"""
    code = generate(source)

    assert re.search(r"OpExecutionMode %\w+ LocalSize 16 16 2\b", code)
    validate_spirv_shader_source(tmp_path, "named_local_size", source)
