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
