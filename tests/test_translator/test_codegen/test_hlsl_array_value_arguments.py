"""Array values passed to by-value array parameters in the generated HLSL.

HLSL has no array-valued expression other than an lvalue, and a CrossGL ``T[N]`` call result is held as a vector
or as a ``CrossGLFixedArray`` struct. The emitter used to refuse array literals passed to a ``T v[N]`` parameter
and to pass lowered call results unchanged, which DXC rejects ("no matching function for call"). Literals are now
built through a struct whose array member is the argument, and lowered values are flat-cast through a typedef.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen

SOURCE = """
shader ArrayValueArguments {
    struct P {
        vec2 q;
        int n;
    };

    struct S {
        float a[4];
        float k;
    };

    layout(std430, binding = 0) buffer Dst {
        float dst[];
    } out_data;

    compute {
        layout(local_size_x = 1) in;

        float sum4(float v[4]) {
            v[1] = 7.0;
            return v[0] + v[1] + v[3];
        }

        float sum5(float v[5]) {
            v[0] = 100.0;
            return v[0] + v[4];
        }

        float sumv(vec3 v[2]) {
            return v[0].x + v[1].z;
        }

        float sump(P v[3]) {
            v[1].n = 9;
            return v[0].q.x + float(v[2].n) + float(v[1].n);
        }

        float sums(S x) {
            return x.a[1] + x.a[2] + x.k;
        }

        float summ(float m[2][2]) {
            return m[0][1] + m[1][0];
        }

        float[4] make4(float k) {
            return { k, k + 1.0, k + 2.0, k + 3.0 };
        }

        float[5] make5(float k) {
            return { k, k + 1.0, k + 2.0, k + 3.0, k + 4.0 };
        }

        vec3[2] makev(float k) {
            return { vec3(k), vec3(k + 1.0) };
        }

        void main() {
            uint i = uint(out_data.dst[9]);
            float k = out_data.dst[3];
            float a[5] = make5(2.0);
            float b[4] = make4(1.0);
            out_data.dst[0] = sum5(make5(3.0)) + sum4(make4(1.0)) + a[i] + b[i];
            out_data.dst[1] = sumv(makev(2.0)) + sumv({ vec3(1.0), vec3(k) });
            out_data.dst[2] = sump({ P(vec2(1.0), 2), P(vec2(3.0), 4), P(vec2(k), 6) });
            out_data.dst[4] = sum5({ 1.0, 2.0, 3.0, 4.0, k }) + sum4({ 1.0, 2.0, 3.0, 4.0 });
            out_data.dst[5] = sum4(b) + sum5(a);
            out_data.dst[6] = summ({ { 1.0, 2.0 }, { k, 4.0 } });
            out_data.dst[7] = sums(S({ k, 2.0, 3.0, 4.0 }, 5.0)) + sums(S(b, 6.0));
        }
    }
}
"""


def _generate(source=SOURCE):
    return HLSLCodeGen().generate(crosstl.translator.parse(source))


def _main_body(code):
    return code[code.index("void CSMain") :]


def test_lowered_call_results_and_locals_are_flat_cast_to_the_parameter_array():
    code = _generate()
    body = _main_body(code)
    assert "typedef float CrossGLArray_float_4[4];" in code
    assert "typedef float CrossGLArray_float_5[5];" in code
    assert "typedef float3 CrossGLArray_float3_2[2];" in code
    assert "sum5((CrossGLArray_float_5)(make5(3.0)))" in body
    assert "sum4((CrossGLArray_float_4)(make4(1.0)))" in body
    assert "sumv((CrossGLArray_float3_2)(makev(2.0)))" in body
    assert "sum4((CrossGLArray_float_4)(b))" in body
    assert "sum5((CrossGLArray_float_5)(a))" in body


def test_array_literals_become_struct_member_rvalues():
    code = _generate()
    body = _main_body(code)
    assert re.search(r"struct CrossGLArrayValue_float_5\s*\{\s*float v\[5\];", code)
    assert "sum5(crossgl_array_value_float_5(1.0, 2.0, 3.0, 4.0, k).v)" in body
    assert "sum4(crossgl_array_value_float_4(1.0, 2.0, 3.0, 4.0).v)" in body
    assert re.search(r"sump\(crossgl_array_value_P_3\(__crossgl_construct_P\(", body)
    # Nested literals build the inner arrays first and pass them as array arguments.
    assert (
        "summ(crossgl_array_value_float_2_2(crossgl_array_value_float_2(1.0, 2.0).v, "
        "crossgl_array_value_float_2(k, 4.0).v).v)"
    ) in body
    assert re.search(
        r"CrossGLArrayValue_float_2_2 crossgl_array_value_float_2_2\(float e0\[2\], float e1\[2\]\)",
        code,
    )
    # The inner value struct is declared before the outer one, whose constructor takes its array.
    assert code.index("struct CrossGLArrayValue_float_2\n") < code.index(
        "struct CrossGLArrayValue_float_2_2\n"
    )


def test_struct_constructor_array_fields_take_literals_and_lowered_values():
    body = _main_body(_generate())
    assert re.search(
        r"__crossgl_construct_S\(crossgl_array_value_float_4\(k, 2\.0, 3\.0, 4\.0\)\.v, 5\.0\)",
        body,
    )
    assert "__crossgl_construct_S((CrossGLArray_float_4)(b), 6.0)" in body


def test_output_is_deterministic():
    assert _generate() == _generate()


def test_effectful_literal_elements_are_refused():
    source = SOURCE.replace(
        "sum4({ 1.0, 2.0, 3.0, 4.0 })", "sum4({ 1.0, out_data.dst[8]++, k, 4.0 })"
    )
    with pytest.raises(Exception, match="evaluated as helper arguments"):
        _generate(source)


def test_lowered_value_to_writable_array_parameter_is_refused():
    source = SOURCE.replace(
        "float summ(float m[2][2]) {",
        "void fill(inout float v[5]) { v[0] = 1.0; }\n        float summ(float m[2][2]) {",
    ).replace(
        "out_data.dst[5] = sum4(b) + sum5(a);", "fill(a); out_data.dst[5] = a[0];"
    )
    with pytest.raises(
        Exception, match="writable 'float\\[5\\]' parameter needs an array lvalue"
    ):
        _generate(source)


@pytest.mark.skipif(shutil.which("dxc") is None, reason="dxc is not available")
@pytest.mark.parametrize("spirv", [False, True], ids=["dxil", "spirv"])
def test_dxc_compiles_the_array_arguments(tmp_path, spirv):
    shader = tmp_path / "array_value_arguments.hlsl"
    shader.write_text(_generate(), encoding="utf-8")
    command = [shutil.which("dxc"), "-T", "cs_6_0", "-E", "CSMain", str(shader)]
    command += ["-spirv"] if spirv else []
    command += ["-Fo", str(tmp_path / "out.bin")]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
