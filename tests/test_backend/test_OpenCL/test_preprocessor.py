from crosstl.backend.OpenCL.OpenCLLexer import OpenCLLexer
from crosstl.backend.OpenCL.OpenCLParser import OpenCLParser
from crosstl.backend.OpenCL.preprocessor import OpenCLPreprocessor


def parse_code(source):
    tokens = OpenCLLexer(source).tokenize()
    return OpenCLParser(tokens).parse()


def test_einstein_vector_defaults_avoid_error_and_expand_token_paste():
    source = r"""
        #if VECTOR_SIZE_I == 1
        #define doubleV double
        #elif VECTOR_SIZE_I == 2
        #define doubleV double2
        #else
        #error vector size
        #endif
        #if VECTOR_SIZE_J != 1 || VECTOR_SIZE_K != 1
        #error vector size
        #endif
        #define LC_SET_VECTOR_VARS(IND,D)                                \
          ptrdiff_t const IND =                                          \
            (lc_off##D + VECTOR_SIZE_##D);                               \
          bool const lc_vec_any_##D =                                    \
            1 /*TODO lc_vec_trivial_##D ||                               \
                (IND+VECTOR_SIZE_##D-1 >= lc_##D##min && IND < lc_##D##max)*/; \
          bool const lc_vec_lo_##D = IND >= lc_##D##min;
        kernel void rhs(global double *out) {
          LC_SET_VECTOR_VARS(i,I);
          out[0] = (doubleV)i;
        }
        """

    processed = OpenCLPreprocessor().preprocess(source)

    assert "lc_vec_lo_I" in processed
    assert "##D" not in processed
    assert "VECTOR_SIZE_I" not in processed
    assert parse_code(source).statements[0].name == "rhs"


def test_kernel_header_newline_and_variable_attribute_prefix_parse():
    ast = parse_code("""
        kernel
        __attribute__((vec_type_hint(double)))
        __attribute__((reqd_work_group_size(1, 1, 1)))
        void rhs
        (global double *out) {
          double __attribute__((__unused__)) detgt = ((double)(double)(1));
          out[0] = detgt;
        }
        """)

    assert ast.statements[0].name == "rhs"


def test_opencv_mixchannels_build_macros_default_to_empty():
    source = r"""
        #define DECLARE_INPUT_MAT(i) \
            __global const uchar * src##i##ptr, int src##i##_step, int src##i##_offset,
        #define DECLARE_OUTPUT_MAT(i) \
            __global uchar * dst##i##ptr, int dst##i##_step, int dst##i##_offset,
        #define DECLARE_INDEX(i) \
            int src##i##_index = mad24(src##i##_step, y0, mad24(x, (int)sizeof(T) * scn##i, src##i##_offset)); \
            int dst##i##_index = mad24(dst##i##_step, y0, mad24(x, (int)sizeof(T) * dcn##i, dst##i##_offset));
        #define PROCESS_ELEM(i) \
            __global const T * src##i = (__global const T *)(src##i##ptr + src##i##_index); \
            __global T * dst##i = (__global T *)(dst##i##ptr + dst##i##_index); \
            dst##i[0] = src##i[0]; \
            src##i##_index += src##i##_step; \
            dst##i##_index += dst##i##_step;

        __kernel void mixChannels(DECLARE_INPUT_MAT_N DECLARE_OUTPUT_MAT_N int rows,
                                  int cols,
                                  int rowsPerWI) {
            int x = get_global_id(0);
            int y0 = get_global_id(1) * rowsPerWI;
            if (x < cols) {
                DECLARE_INDEX_N
                for (int y = y0, y1 = min(y0 + rowsPerWI, rows); y < y1; ++y) {
                    PROCESS_ELEM_N
                }
            }
        }
        """

    processed = OpenCLPreprocessor().preprocess(source)

    assert "DECLARE_INPUT_MAT_N" not in processed
    assert "DECLARE_OUTPUT_MAT_N" not in processed
    assert "DECLARE_INDEX_N" not in processed
    assert "PROCESS_ELEM_N" not in processed
    kernel = parse_code(source).statements[0]
    assert kernel.name == "mixChannels"
    assert [param["name"] for param in kernel.params] == [
        "rows",
        "cols",
        "rowsPerWI",
    ]
