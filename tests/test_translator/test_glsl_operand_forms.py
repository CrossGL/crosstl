"""GLSL output writes CrossGL operands in the types GLSL declares, and refuses built-in calls with no GLSL overload.

See ``crosstl/translator/codegen/opengl_operand_forms.py``: case labels take the type of a 32-bit switch selector,
the offset and bit count of ``bitfieldExtract`` and ``bitfieldInsert`` become ``int``, ``cross`` of operands that are
not 3-vectors and the float-only functions of GLSL 4.60 sections 8.1 and 8.2 with ``double`` operands are refused.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.opengl_operand_forms import (
    FLOAT_ONLY_FUNCTIONS,
    OpenGLBuiltinOperandError,
)
from crosstl.translator.refusals import OpenGLRefusal

GLSLANG = shutil.which("glslangValidator")

SWITCH_LABELS = """
shader SwitchLabels {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* data;
        void main() {
            uint u = uint(data[0]);
            int s = data[1];
            int k = data[2];
            float r = 0.0;
            switch (u) {
                case 0: r = 1.0; break;
                case 4294967295: r = 2.0; break;
                case 7u: r = 3.0; break;
                case -8: r = 5.0; break;
                default: r = 4.0; break;
            }
            switch (s) {
                case -1: r = r + 1.0; break;
                case 4294967294: r = r + 2.0; break;
                case 2147483648: r = r + 3.0; break;
                case 9u: r = r + 4.0; break;
                default: break;
            }
            data[3] = int(r) + k;
        }
    }
}
"""

BITFIELDS = """
shader Bitfields {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* values;
        buffer uint* fields;
        buffer ivec3* outputs;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            ivec3 a = ivec3(values[i], values[i + 1u], values[i + 2u]);
            ivec3 b = ivec3(values[i + 3u], values[i + 4u], values[i + 5u]);
            uint offset = fields[i];
            uint count = fields[i + 1u];
            int signedCount = values[i + 6u];
            outputs[i] = bitfieldExtract(a, offset, count) + bitfieldInsert(a, b, offset, count)
                + bitfieldExtract(a, 1, signedCount);
        }
    }
}
"""

CROSS_FOUR = """
shader CrossFour {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer vec4* points;
        void main() {
            points[2] = cross(points[0], points[1]);
        }
    }
}
"""

DOUBLE_CALL = """
shader DoubleCall {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* values;
        void main() {
            double x = double(values[0]);
            double y = double(values[1]);
            values[2] = float(CALL);
        }
    }
}
"""


def translate(tmp_path, source, backend="opengl", name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def assert_glslang_accepts(tmp_path, code, stage="comp"):
    if GLSLANG is None:
        return
    path = tmp_path / f"out.{stage}"
    path.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [GLSLANG, str(path)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def case_labels(code):
    return [
        line.strip() for line in code.splitlines() if line.strip().startswith("case ")
    ]


def test_case_labels_take_the_type_of_the_selector(tmp_path):
    code = translate(tmp_path, SWITCH_LABELS)
    assert case_labels(code) == [
        "case 0u: {",
        "case 4294967295u: {",
        "case 7u: {",
        "case 4294967288u: {",
        "case -1: {",
        "case -2: {",
        "case (-2147483647 - 1): {",
        "case 9: {",
    ]
    assert_glslang_accepts(tmp_path, code)


def test_unsigned_case_labels_compile_without_implicit_conversions(tmp_path):
    # GLSL ES 3.00 has no implicit int-to-uint conversion, so a label of a uint selector must be a uint.
    source = """
shader UnsignedSelector {
    fragment {
        uniform uint mode;
        vec4 main() @ gl_FragColor {
            float r = 0.0;
            switch (mode) {
                case 1: r = 0.5; break;
                default: r = 1.0; break;
            }
            return vec4(r);
        }
    }
}
"""
    code = translate(tmp_path, source, backend="webgl")
    assert "case 1u:" in code
    assert_glslang_accepts(tmp_path, code, stage="frag")


def test_bitfield_offsets_and_counts_become_int(tmp_path):
    code = translate(tmp_path, BITFIELDS)
    assert "bitfieldExtract(a, int(offset), int(count))" in code
    assert "bitfieldInsert(a, b, int(offset), int(count))" in code
    assert "bitfieldExtract(a, 1, signedCount)" in code
    assert_glslang_accepts(tmp_path, code)


def test_cross_of_four_component_vectors_is_refused(tmp_path):
    with pytest.raises(OpenGLBuiltinOperandError) as caught:
        translate(tmp_path, CROSS_FOUR)
    error = caught.value
    assert isinstance(error, OpenGLRefusal)
    assert error.operation == "cross"
    assert error.operand_types == ("vec4", "vec4")
    assert "vec3 and dvec3" in str(error)


@pytest.mark.parametrize(
    "call", ["acos(x)", "sinh(x)", "asin(x)", "atan2(y, x)", "log10(x)", "tanh(x)"]
)
def test_double_operands_of_float_only_functions_are_refused(tmp_path, call):
    with pytest.raises(OpenGLBuiltinOperandError) as caught:
        translate(tmp_path, DOUBLE_CALL.replace("CALL", call))
    error = caught.value
    assert error.operation == call.split("(")[0]
    assert "double" in error.operand_types
    assert error.diagnostic_reason == "double-precision-operand"


@pytest.mark.parametrize(
    "call", ["sqrt(x)", "inversesqrt(x)", "abs(x) + floor(y)", "fma(x, y, x)"]
)
def test_double_functions_glsl_declares_still_translate(tmp_path, call):
    code = translate(tmp_path, DOUBLE_CALL.replace("CALL", call))
    assert call.split("(")[0] in code
    assert_glslang_accepts(tmp_path, code)


def test_float_operands_of_float_only_functions_still_translate(tmp_path):
    source = DOUBLE_CALL.replace(
        "double x = double(values[0]);", "float x = values[0];"
    )
    source = source.replace("double y = double(values[1]);", "float y = values[1];")
    code = translate(
        tmp_path, source.replace("CALL", "acos(x) + pow(x, y) + atan2(y, x)")
    )
    assert "acos(x)" in code and "pow(x, y)" in code
    assert_glslang_accepts(tmp_path, code)


def test_float_only_table_names_the_functions_without_double_overloads():
    assert {"sqrt", "inversesqrt", "abs", "fma"}.isdisjoint(FLOAT_ONLY_FUNCTIONS)
    assert {"sin", "cos", "exp", "log", "pow", "atan2"} <= FLOAT_ONLY_FUNCTIONS


SHARED_VECTOR_ARRAY = """
shader SharedVectorArray {
    groupshared float4 tile[2];
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* data;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            float f = data[i];
            tile[0] = vec4(f, 1.0, 2.0, 3.0);
            tile[1] = tile[0];
            tile[1][0] = tile[1][0] * 0.5;
            tile[1][1] = tile[1][1] + f;
            barrier();
            data[i] = tile[1][0] + tile[1][1] + tile[0].z;
        }
    }
}
"""


def test_component_of_a_shared_vector_array_element_is_not_an_element_offset(tmp_path):
    # tile[1][1] is component 1 of element 1. It was written as tile[(1 + 1)], element 2 of a 2-element array.
    code = translate(tmp_path, SHARED_VECTOR_ARRAY)
    assert "tile[1][0] = (tile[1][0] * 0.5);" in code
    assert "tile[1][1] = (tile[1][1] + f);" in code
    assert "tile[(1 + 1)]" not in code
    assert_glslang_accepts(tmp_path, code)
