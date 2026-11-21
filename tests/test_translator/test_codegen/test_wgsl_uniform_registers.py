"""WGSL output of uniform blocks whose declared layout WGSL's own uniform layout does not reproduce.

A CrossGL constant block is laid out by std140 unless it declares another layout; WGSL lays out a ``var<uniform>``
struct by its own rules (WGSL section 14.4). The generator kept the typed struct for every block, so a ``mat2``
read its second column from byte 8 instead of 16, a ``@ row_major`` matrix read its rows as columns, an HLSL-packed
``vec3`` after a ``float`` moved from byte 4 to byte 16, and a ``vec2`` array was rejected by naga. Such blocks are
now declared as 16-byte registers and every read is built from the words at the declared offsets
(:mod:`crosstl.translator.codegen.wgsl_uniform_registers`).
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.glsl_block_layout import (
    STD140_LAYOUT,
    block_struct_placements,
)
from crosstl.translator.codegen.hlsl_buffer_layout import (
    array_type,
    oriented_matrix_type,
    scalar_type,
    struct_type,
    vector_type,
)
from crosstl.translator.codegen.wgsl_uniform_registers import typed_layout_matches
from crosstl.translator.refusals import WGSLRefusal


def translate_wgsl(tmp_path, source):
    path = tmp_path / "shader.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="wgsl")


def validate_with_naga(tmp_path, code):
    naga = shutil.which("naga")
    if naga is None:
        pytest.skip("naga is not installed")
    path = tmp_path / "shader.wgsl"
    path.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [naga, str(path)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def compute(block, body, structs=""):
    return f"""
shader Probe {{
    {structs}
    {block}
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        buffer int* iin;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            int k = iin[i];
            {body}
        }}
    }}
}}
"""


def test_std140_mat2_reads_columns_sixteen_bytes_apart(tmp_path):
    code = translate_wgsl(
        tmp_path,
        compute(
            "cbuffer Params @binding(3) { mat2 m; };",
            "vec2 t = m * vec2(1.0, -0.5); outp[i] = t.x + m[1][0];",
        ),
    )
    assert "registers: array<vec4<f32>, 2>," in code
    assert "mat2x2<f32>(_Params.registers[0].xy, _Params.registers[1].xy)" in code
    assert "m: mat2x2<f32>" not in code
    validate_with_naga(tmp_path, code)


def test_row_major_matrices_read_rows_through_transpose(tmp_path):
    code = translate_wgsl(
        tmp_path,
        compute(
            "cbuffer Params @binding(3) { @ row_major mat3 m; @ row_major mat2x3 n; vec4 v; };",
            "vec3 t = m * v.xyz; vec3 u = n * v.xy; outp[i] = t.x + u.y + m[2][1] + n[1][2];",
        ),
    )
    rows = (
        "_Params.registers[0].xyz, _Params.registers[1].xyz, _Params.registers[2].xyz"
    )
    assert f"transpose(mat3x3<f32>({rows}))" in code
    # mat2x3 row-major: three rows of two floats, transposed back to two columns of three.
    assert (
        "transpose(mat3x2<f32>(_Params.registers[3].xy, _Params.registers[4].xy, "
        "_Params.registers[5].xy))"
    ) in code
    # Column 2 of a row-major mat3 is component z of each row.
    assert (
        "vec3<f32>(_Params.registers[0].z, _Params.registers[1].z, _Params.registers[2].z)"
        in code
    )
    validate_with_naga(tmp_path, code)


def test_dynamic_index_into_row_major_matrix_array_uses_a_helper(tmp_path):
    code = translate_wgsl(
        tmp_path,
        compute(
            "cbuffer Params @binding(3) { @ row_major mat2 m[2]; };",
            "vec2 t = m[k & 1] * vec2(1.0, -0.5); outp[i] = t.x;",
        ),
    )
    assert "load_Params_mat2x2f32(u32((k & 1)) * 2u)" in code
    assert (
        "fn load_Params_mat2x2f32(r: u32) -> mat2x2<f32> {\n"
        "    return transpose(mat2x2<f32>(_Params.registers[r].xy, "
        "_Params.registers[r + 1u].xy));\n}"
    ) in code
    validate_with_naga(tmp_path, code)


def test_hlsl_cbuffer_packing_reads_a_vec3_at_byte_four(tmp_path):
    code = translate_wgsl(
        tmp_path,
        compute(
            "cbuffer Params @binding(3) @buffer_layout(hlsl_cbuffer) "
            "{ float a; vec3 b; float c[2]; float d; vec2 g; };",
            "outp[i] = a + b.y + c[1] + d + g.y;",
        ),
    )
    assert "_Params.registers[0].yzw.y" in code
    # c[1] is the first word of register 2; d packs into register 2's second word, after the array.
    assert "_Params.registers[2].x" in code
    assert "_Params.registers[2].y" in code
    # g is 8-byte aligned in the same register, at bytes 40 to 48.
    assert "_Params.registers[2].zw.y" in code
    validate_with_naga(tmp_path, code)


def test_mixed_scalar_types_bitcast_from_u32_registers(tmp_path):
    code = translate_wgsl(
        tmp_path,
        compute(
            "cbuffer Params @binding(3) { mat2 m; int n; uint u; bool flag; };",
            "outp[i] = m[0][0] + float(n) + float(u) + (flag ? 1.0 : 0.0);",
        ),
    )
    assert "registers: array<vec4<u32>, 3>," in code
    assert "bitcast<vec2<f32>>(_Params.registers[0].xy)" in code
    assert "bitcast<i32>(_Params.registers[2].x)" in code
    assert "(_Params.registers[2].z != 0u)" in code
    validate_with_naga(tmp_path, code)


def test_std430_float_array_with_dynamic_index_reads_words(tmp_path):
    code = translate_wgsl(
        tmp_path,
        compute(
            "cbuffer Params @binding(3) @buffer_layout(std430) { float w[8]; vec4 v; };",
            "outp[i] = w[k] + w[5] + v.w;",
        ),
    )
    assert "registers: array<vec4<f32>, 3>," in code
    assert "fn load_Params_f32(w: u32) -> f32" in code
    assert "_Params.registers[w / 4u][w % 4u]" in code
    assert "load_Params_f32(u32(k))" in code
    assert "_Params.registers[1].y" in code
    assert "_Params.registers[2].w" in code
    validate_with_naga(tmp_path, code)


def test_struct_members_and_whole_values_are_rebuilt(tmp_path):
    code = translate_wgsl(
        tmp_path,
        compute(
            "cbuffer Params @binding(3) { vec2 va[3]; Light lights[2]; float s; };",
            "Light l = lights[k & 1]; vec2 all_[3] = va; "
            "outp[i] = l.r + lights[1].pos.y + all_[2].x + va[k & 1].y + s;",
            structs="struct Light { vec3 pos; float r; mat2 m; };",
        ),
    )
    # vec2 array elements are 16 bytes apart in std140.
    assert "fn load_Params_array3_vec2f32(r: u32) -> array<vec2<f32>, 3>" in code
    assert "value[i] = _Params.registers[r + i].xy;" in code
    assert "load_Params_Light(u32((k & 1)) * 3u + 3u)" in code
    assert "fn load_Params_Light(r: u32) -> Light" in code
    # lights[1].pos.y reads register 6 directly.
    assert "_Params.registers[6].xyz.y" in code
    assert "_Params.registers[u32((k & 1))].xy.y" in code
    validate_with_naga(tmp_path, code)


def test_push_constant_block_follows_std430(tmp_path):
    code = translate_wgsl(
        tmp_path,
        compute(
            "cbuffer Push @push_constant { float scale; float offsets[3]; };",
            "outp[i] = scale + offsets[2];",
        ),
    )
    # std430 puts offsets[2] at byte 12.
    assert "_Push.registers[0].w" in code
    validate_with_naga(tmp_path, code)


def test_blocks_wgsl_already_lays_out_keep_their_typed_struct(tmp_path):
    code = translate_wgsl(
        tmp_path,
        compute(
            "cbuffer Params @binding(3) { mat4 m; vec3 v; float s; float a[2]; Inner inner; };",
            "outp[i] = m[1][2] + v.y + s + a[1] + inner.p.x;",
            structs="struct Inner { vec4 p; };",
        ),
    )
    assert "registers" not in code
    assert "m: mat4x4<f32>," in code
    validate_with_naga(tmp_path, code)


def test_constant_index_outside_the_array_is_refused(tmp_path):
    with pytest.raises(WGSLRefusal, match="element 3 of a uniform value of 2"):
        translate_wgsl(
            tmp_path,
            compute(
                "cbuffer Params @binding(3) { mat2 m[2]; };",
                "outp[i] = m[3][0][0];",
            ),
        )


def test_typed_layout_model_agrees_with_std140_where_wgsl_does():
    def matches(*members):
        layout_type = struct_type("B", members)
        placements, _size = block_struct_placements(
            layout_type, STD140_LAYOUT, root=True
        )
        return typed_layout_matches(layout_type, placements, STD140_LAYOUT)

    vec3 = vector_type("float", 3)
    assert matches(("v", vec3), ("s", scalar_type("float")))
    assert matches(("m", oriented_matrix_type("float", 4, 3)))
    assert matches(("a", array_type(scalar_type("float"), 4)))
    assert not matches(("m", oriented_matrix_type("float", 2, 2)))
    assert not matches(("m", oriented_matrix_type("float", 3, 3, row_major=True)))
    assert not matches(("a", array_type(vector_type("float", 2), 2)))
    # A struct of one float takes 16 bytes in std140; WGSL rejects the next member at byte 4.
    small = struct_type("S", [("x", scalar_type("float"))])
    assert not matches(("s", small), ("t", scalar_type("float")))


def test_struct_with_scalar_array_and_bool_is_read_from_registers(tmp_path):
    # The typed struct would keep 4-byte array strides and a bool member, which the uniform address space rejects.
    code = translate_wgsl(
        tmp_path,
        compute(
            "cbuffer Params @binding(3) { S st; };",
            "S local = st; outp[i] = local.a[1] + st.a[k & 1] + (st.b ? 1.0 : 0.0) + st.v.w;",
            structs="struct S { float a[2]; bool b; vec4 v; };",
        ),
    )
    assert "registers: array<vec4<u32>, 4>," in code
    assert (
        "S(load_Params_array2_f32(0u), (_Params.registers[2].x != 0u), "
        "bitcast<vec4<f32>>(_Params.registers[3]))"
    ) in code
    assert "bitcast<f32>(_Params.registers[u32((k & 1))].x)" in code
    validate_with_naga(tmp_path, code)


def test_writes_to_register_block_members_stay_refused(tmp_path):
    with pytest.raises(WGSLRefusal, match="read-only"):
        translate_wgsl(
            tmp_path,
            compute(
                "cbuffer Params @binding(3) { mat2 m; };",
                "m[0][0] = 1.0; outp[i] = m[1][1];",
            ),
        )
