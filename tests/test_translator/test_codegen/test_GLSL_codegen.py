
# Tests for matrix aggregate initializer column-vector path (issue #3718)
# -----------------------------------------------------------------

_MAT2_COL_VEC_GLSL_CGL = """
shader MatColVecGLSL {
    buffer float* out;
    compute {
        layout(local_size_x = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            float x = out[i];
            mat2 m = {vec2(x, 1.0), vec2(2.0, x)};
            out[i] = m[0][0] + m[1][1];
        }
    }
}
"""

_MAT2X3_COL_VEC_GLSL_CGL = """
shader MatColVec2x3GLSL {
    buffer float* out;
    compute {
        layout(local_size_x = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            float x = out[i];
            mat2x3 m = {vec3(x, 1.0, 2.0), vec3(3.0, x, 4.0)};
            out[i] = m[0][0] + m[1][2];
        }
    }
}
"""

_MAT4_COL_VEC_GLSL_CGL = """
shader Mat4ColVecGLSL {
    buffer float* out;
    compute {
        layout(local_size_x = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            float x = out[i];
            mat4 m = {vec4(x), vec4(1.0), vec4(2.0), vec4(3.0)};
            out[i] = m[0][0] + m[3][3];
        }
    }
}
"""

_MAT2_IVEC2_GLSL_CGL = """
shader MatIvec2GLSL {
    buffer float* out;
    buffer int* k_buf;
    compute {
        layout(local_size_x = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            float x = out[i];
            int k = k_buf[i];
            mat2 m = {ivec2(k), vec2(x)};
            out[i] = m[0][0] + m[1][1];
        }
    }
}
"""

_MAT2_MIXED_GLSL_CGL = """
shader MatMixedGLSL {
    buffer float* out;
    compute {
        layout(local_size_x = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            float x = out[i];
            mat2 m = {vec2(x), 1.0, 2.0};
            out[i] = m[0][0];
        }
    }
}
"""


def _write_glsl_shader(tmp_path, source, name):
    path = tmp_path / name
    path.write_text(source)
    return path


def test_glsl_mat2_column_vector_initializer(tmp_path):
    """mat2 initialized with two vec2 column vectors translates to GLSL without error."""
    import crosstl
    code = crosstl.translate(
        str(_write_glsl_shader(tmp_path, _MAT2_COL_VEC_GLSL_CGL, "mat2_col.cgl")),
        backend="opengl",
    )
    # Should produce a mat2 constructor with vec2 column arguments, not scalar components.
    assert "mat2(" in code
    assert "vec2(" in code
    # Should not have a zero-padding float scalar inserted next to a vec2
    assert "float(0)" not in code


def test_glsl_mat2x3_column_vector_initializer(tmp_path):
    """mat2x3 initialized with two vec3 column vectors translates to GLSL without error."""
    import crosstl
    code = crosstl.translate(
        str(_write_glsl_shader(tmp_path, _MAT2X3_COL_VEC_GLSL_CGL, "mat2x3_col.cgl")),
        backend="opengl",
    )
    assert "mat2x3(" in code
    assert "vec3(" in code
    assert "float(0)" not in code


def test_glsl_mat2_ivec2_column_vector_initializer(tmp_path):
    """mat2 initialized with ivec2 column vectors uses vec2 (column_type) as expected type."""
    import crosstl
    code = crosstl.translate(
        str(_write_glsl_shader(tmp_path, _MAT2_IVEC2_GLSL_CGL, "mat2_ivec2.cgl")),
        backend="opengl",
    )
    # The column type (vec2) must be used as the expected type for each element so that
    # an ivec2 argument is converted to vec2, not treated as two scalars.
    assert "mat2(" in code
    assert "vec2(" in code
    assert "float(0)" not in code


def test_glsl_mat4_column_vector_initializer(tmp_path):
    """mat4 initialized with four vec4 column vectors translates to GLSL without error."""
    import crosstl
    code = crosstl.translate(
        str(_write_glsl_shader(tmp_path, _MAT4_COL_VEC_GLSL_CGL, "mat4_col.cgl")),
        backend="opengl",
    )
    assert "mat4(" in code
    assert "vec4(" in code
    assert "float(0)" not in code


def test_glsl_mixed_matrix_initializer_raises(tmp_path):
    """mat2 with a mix of vector and scalar elements raises an OpenGL aggregate error."""
    import crosstl
    import pytest
    from crosstl.translator.codegen.GLSL_codegen import OpenGLAggregateInitializerError
    with pytest.raises(OpenGLAggregateInitializerError) as exc_info:
        crosstl.translate(
            str(_write_glsl_shader(tmp_path, _MAT2_MIXED_GLSL_CGL, "mat2_mixed.cgl")),
            backend="opengl",
        )
    assert exc_info.value.reason == "element-count-mismatch"
