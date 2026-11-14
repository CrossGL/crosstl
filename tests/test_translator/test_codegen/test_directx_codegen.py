
# Tests for matrix aggregate initializer column-vector path (issue #3718)
# -----------------------------------------------------------------


def _write_shader(tmp_path, source, name):
    path = tmp_path / name
    path.write_text(source)
    return path


_MAT2_COL_VEC_CGL = """
shader MatColVec {{
    buffer float* out;
    compute {{
        layout(local_size_x = 1) in;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            float x = out[i];
            mat2 m = {{vec2(x, 1.0), vec2(2.0, x)}};
            out[i] = m[0][0] + m[1][1];
        }}
    }}
}}
"""

_MAT2X3_COL_VEC_CGL = """
shader MatColVec2x3 {{
    buffer float* out;
    compute {{
        layout(local_size_x = 1) in;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            float x = out[i];
            mat2x3 m = {{vec3(x, 1.0, 2.0), vec3(3.0, x, 4.0)}};
            out[i] = m[0][0] + m[1][2];
        }}
    }}
}}
"""

_MAT4_COL_VEC_CGL = """
shader Mat4ColVec {{
    buffer float* out;
    compute {{
        layout(local_size_x = 1) in;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            float x = out[i];
            mat4 m = {{vec4(x), vec4(1.0), vec4(2.0), vec4(3.0)}};
            out[i] = m[0][0] + m[3][3];
        }}
    }}
}}
"""

_MAT2_MIXED_CGL = """
shader MatMixed {{
    buffer float* out;
    compute {{
        layout(local_size_x = 1) in;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            float x = out[i];
            mat2 m = {{vec2(x), 1.0, 2.0}};
            out[i] = m[0][0];
        }}
    }}
}}
"""


def test_directx_mat2_column_vector_initializer(tmp_path):
    """mat2 initialized with two vec2 column vectors translates to HLSL."""
    import crosstl
    code = crosstl.translate(
        str(_write_shader(tmp_path, _MAT2_COL_VEC_CGL, "mat2_col.cgl")),
        backend="directx",
    )
    # The column-vector path wraps each element in float2(...) and uses transpose()
    # so the HLSL constructor receives row vectors instead of scalar zero-padding.
    assert "float2(" in code
    assert "transpose(" in code
    assert "(float2)0" not in code


def test_directx_mat4_column_vector_initializer(tmp_path):
    """mat4 initialized with four vec4 column vectors translates to HLSL."""
    import crosstl
    code = crosstl.translate(
        str(_write_shader(tmp_path, _MAT4_COL_VEC_CGL, "mat4_col.cgl")),
        backend="directx",
    )
    assert "float4(" in code
    assert "transpose(" in code
    assert "(float4)0" not in code


def test_directx_mixed_matrix_initializer_raises(tmp_path):
    """mat2 with a mix of vector and scalar elements raises an error."""
    import crosstl
    from crosstl.translator.codegen.directx_codegen import DirectXAggregateInitializerError
    import pytest
    with pytest.raises(DirectXAggregateInitializerError) as exc_info:
        crosstl.translate(
            str(_write_shader(tmp_path, _MAT2_MIXED_CGL, "mat2_mixed.cgl")),
            backend="directx",
        )
    assert exc_info.value.reason == "element-count-mismatch"
