import re

import pytest

from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.backend.Metal.preprocessor import (
    MetalPreprocessor,
    MetalStructMethodError,
    MetalTemplateSpecializationError,
)


def token_values(code, **lexer_options):
    return [value for _, value in MetalLexer(code, **lexer_options).tokenize()]


def test_containing_span_preserves_sorted_lookup_boundaries():
    preprocessor = MetalPreprocessor()
    spans = [(2, 5), (8, 13), (20, 24)]

    assert preprocessor._containing_span(1, spans) is None
    assert preprocessor._containing_span(2, spans) == (2, 5)
    assert preprocessor._containing_span(4, spans) == (2, 5)
    assert preprocessor._containing_span(5, spans) is None
    assert preprocessor._containing_span(12, spans) == (8, 13)


def test_containing_span_preserves_unsorted_overlap_order():
    preprocessor = MetalPreprocessor()
    spans = [(10, 20), (0, 15), (30, 40)]

    assert preprocessor._containing_span(12, spans) == (10, 20)
    assert preprocessor._containing_span(5, spans) == (0, 15)


def test_preprocessor_conditional_expansion():
    code = """
    #define ENABLED 1
    #if ENABLED
    int only_enabled = 1;
    #else
    int only_disabled = ;
    #endif
    """

    values = token_values(code)

    assert "only_enabled" in values
    assert "only_disabled" not in values


def test_preprocessor_function_like_macro_expansion():
    code = """
    #define MAKE_FLOAT2(a, b) float2((a), (b))
    float2 value = MAKE_FLOAT2(1.0, 2.0);
    """

    values = token_values(code)

    assert "MAKE_FLOAT2" not in values
    assert "float2" in values
    assert "1.0" in values
    assert "2.0" in values


def test_preprocessor_materializes_mlx_instantiate_kernel_macros():
    code = """
    #define instantiate_arange(name, itype, offset) \\
        instantiate_kernel("arange" #name, arange, itype, offset)

    template <typename T, int Offset>
    [[kernel]] void arange(
        device T* out [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {
        out[gid] = T(gid + Offset);
    }

    instantiate_arange(float32, float, 7)
    instantiate_arange(float16, half, 3)
    """

    output = MetalPreprocessor().preprocess(code)

    assert "template <typename T" not in output
    assert "instantiate_kernel" not in output
    assert '[[host_name("arangefloat32")]]' in output
    assert "void arangefloat32(" in output
    assert "device float* out" in output
    assert "out[gid] = float(gid + 7);" in output
    assert '[[host_name("arangefloat16")]]' in output
    assert "void arangefloat16(" in output
    assert "device half* out" in output
    assert "out[gid] = half(gid + 3);" in output


def test_preprocessor_materializes_mlx_instantiations_with_template_defaults():
    code = """
    #define instantiate_rope(name, type, idx_type) \\
        instantiate_kernel("rope_" #name, rope, type, idx_type)

    template <typename T, typename IdxT, int N = 4>
    [[kernel]] void rope(
        device const T* src [[buffer(0)]],
        device T* dst [[buffer(1)]],
        device const IdxT* positions [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
        IdxT pos = positions[gid];
        dst[gid] = src[pos] + T(N);
    }

    instantiate_rope(float32, float, uint)
    """

    output = MetalPreprocessor().preprocess(code)

    assert "template <typename T" not in output
    assert "instantiate_kernel" not in output
    assert '[[host_name("rope_float32")]]' in output
    assert "void rope_float32(" in output
    assert "device const float* src" in output
    assert "device const uint* positions" in output
    assert "uint pos = positions[gid];" in output
    assert "dst[gid] = src[pos] + float(4);" in output


def test_preprocessor_materializes_multiple_mlx_instantiations_from_one_macro():
    code = """
    #define instantiate_copy_pair(name, type) \\
        instantiate_kernel("s_copy" #name, copy_s, type, type, 1) \\
        instantiate_kernel("v_copy" #name, copy_v, type, type, 1)

    template <typename T, typename U, int N>
    [[kernel]] void copy_s(
        device const T* src [[buffer(0)]],
        device U* dst [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
        dst[gid] = U(src[0] + N);
    }

    template <typename T, typename U, int N>
    [[kernel]] void copy_v(
        device const T* src [[buffer(0)]],
        device U* dst [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
        dst[gid] = U(src[gid] + N);
    }

    instantiate_copy_pair(float32, float)
    """

    output = MetalPreprocessor().preprocess(code)

    assert "instantiate_kernel" not in output
    assert '[[host_name("s_copyfloat32")]]' in output
    assert "void s_copyfloat32(" in output
    assert "device const float* src" in output
    assert "dst[gid] = float(src[0] + 1);" in output
    assert '[[host_name("v_copyfloat32")]]' in output
    assert "void v_copyfloat32(" in output
    assert "dst[gid] = float(src[gid] + 1);" in output


def test_preprocessor_materializes_mlx_decltype_instantiation_entries():
    code = """
    template <typename T>
    [[kernel]] void arange(
        device T* out [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {
        out[gid] = T(gid);
    }

    template [[host_name("arangefloat32")]] [[kernel]]
    decltype(arange<float>) arange<float>;
    """

    output = MetalPreprocessor().preprocess(code)

    assert "decltype(arange<float>)" not in output
    assert '[[host_name("arangefloat32")]]' in output
    assert "void arangefloat32(" in output
    assert "device float* out" in output
    assert "out[gid] = float(gid);" in output


def test_preprocessor_materializes_decltype_instantiation_with_joined_host_name():
    code = """
    template <typename T>
    [[kernel]] void arange(
        device T* out [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {
        out[gid] = T(gid);
    }

    template [[host_name("arange" "uint32")]] [[kernel]]
    decltype(arange<uint>) arange<uint>;
    """

    output = MetalPreprocessor().preprocess(code)

    assert "decltype(arange<uint>)" not in output
    assert '[[host_name("arangeuint32")]]' in output
    assert "void arangeuint32(" in output
    assert "device uint* out" in output
    assert "out[gid] = uint(gid);" in output


def test_preprocessor_materializes_signature_only_template_instantiation_entries():
    code = """
    template <int N>
    struct ConvParams {
        int shape[N];
    };

    template <typename T, int N>
    [[kernel]] void naive_unfold_Nd(
        const device T* in [[buffer(0)]],
        device T* out [[buffer(1)]],
        const constant ConvParams<N>* params [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
        out[gid] = in[gid] + T(N + params->shape[0]);
    }

    template <typename T, int N>
    [[kernel]] void naive_unfold_transpose_Nd(
        const device T* in [[buffer(0)]],
        device T* out [[buffer(1)]],
        const constant ConvParams<N>* params [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
        out[gid] = in[gid] - T(N + params->shape[0]);
    }

    #define instantiate_naive_unfold_nd(name, itype, n) \\
        template [[host_name("naive_unfold_nd_" #name "_" #n)]] [[kernel]] void \\
        naive_unfold_Nd( \\
            const device itype* in [[buffer(0)]], \\
            device itype* out [[buffer(1)]], \\
            const constant ConvParams<n>* params [[buffer(2)]], \\
            uint gid [[thread_position_in_grid]]); \\
        template [[host_name("naive_unfold_transpose_nd_" #name "_" #n)]] [[kernel]] void \\
        naive_unfold_transpose_Nd( \\
            const device itype* in [[buffer(0)]], \\
            device itype* out [[buffer(1)]], \\
            const constant ConvParams<n>* params [[buffer(2)]], \\
            uint gid [[thread_position_in_grid]]);

    instantiate_naive_unfold_nd(float32, float, 2) instantiate_naive_unfold_nd(float16, half, 3)
    """

    output = MetalPreprocessor().preprocess(code)

    assert '[[host_name("naive_unfold_nd_float32_2")]]' in output
    assert "void naive_unfold_nd_float32_2(" in output
    assert "const device float* in" in output
    assert "device float* out" in output
    assert "const constant ConvParams_2* params" in output
    assert "out[gid] = in[gid] + float(2 + params->shape[0]);" in output
    assert '[[host_name("naive_unfold_transpose_nd_float32_2")]]' in output
    assert "void naive_unfold_transpose_nd_float32_2(" in output
    assert "out[gid] = in[gid] - float(2 + params->shape[0]);" in output
    assert '[[host_name("naive_unfold_nd_float16_3")]]' in output
    assert "void naive_unfold_nd_float16_3(" in output
    assert "const device half* in" in output
    assert "const constant ConvParams_3* params" in output
    assert '[[host_name("naive_unfold_transpose_nd_float16_3")]]' in output
    assert "void naive_unfold_transpose_nd_float16_3(" in output
    assert "template [[host_name" not in output
    assert "naive_unfold_Nd(" not in output
    assert "naive_unfold_transpose_Nd(" not in output
    assert "ConvParams<n>" not in output
    # The ConvParams<N> struct template referenced by the kernel signatures is
    # materialized into concrete structs, with no residual instantiations left.
    assert "struct ConvParams_2 {" in output
    assert "int shape[2];" in output
    assert "struct ConvParams_3 {" in output
    assert "int shape[3];" in output
    assert "ConvParams<2>" not in output
    assert "ConvParams<3>" not in output


def test_preprocessor_materialized_numeric_specialization_preserves_member_names():
    code = """
    struct PackedScale {
        uint8_t bits;
    };

    template <typename T, const int group_size, const int bits>
    [[kernel]] void nvfp4_quantize(
        const device T* in [[buffer(0)]],
        device uint8_t* out [[buffer(1)]]) {
        PackedScale s;
        T sample = in[0];
        uint8_t q_scale = s.bits;
        uint8_t output = q_scale + bits;
        out[0] = output;
    }

    template [[host_name("nvfp4_quantize_float_gs_16_b_4")]] [[kernel]]
    decltype(nvfp4_quantize<float, 16, 4>) nvfp4_quantize<float, 16, 4>;
    """

    output = MetalPreprocessor().preprocess(code)
    ast = MetalParser(MetalLexer(output, preprocess=False).tokenize()).parse()

    assert "decltype(nvfp4_quantize<float, 16, 4>)" not in output
    assert "void nvfp4_quantize_float_gs_16_b_4(" in output
    assert "const device float* in" in output
    assert "float sample = in[0];" in output
    assert "uint8_t q_scale = s.bits;" in output
    assert "uint8_t q_scale = s.4;" not in output
    assert "uint8_t output = q_scale + 4;" in output
    assert [function.name for function in ast.functions] == [
        "nvfp4_quantize_float_gs_16_b_4"
    ]


def test_preprocessor_materializes_explicit_template_helper_calls():
    code = """
    template <typename T, typename IdxT, int Offset>
    T load_with_offset(device const T* src, IdxT index) {
        return src[index] + T(Offset);
    }

    kernel void copy(device const float* src [[buffer(0)]],
                     device float* dst [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
        dst[gid] = load_with_offset<float, uint, 7>(src, gid);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "load_with_offset<float, uint, 7>" not in output
    assert "load_with_offset_float_uint_7(src, gid)" in output
    assert "float load_with_offset_float_uint_7(" in output
    assert "device const float* src" in output
    assert "uint index" in output
    assert "return src[index] + float(7);" in output


def test_preprocessor_leaves_large_decltype_instantiation_families_bounded():
    declarations = "\n".join(
        f'template [[host_name("kernel" "{index}")]] [[kernel]] '
        f"decltype(arange<uint>) arange<uint>;"
        for index in range(4)
    )
    code = f"""
    template <typename T>
    [[kernel]] void arange(
        device T* out [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {{
        out[gid] = T(gid);
    }}

    {declarations}
    """

    output = MetalPreprocessor(max_template_specializations=3).preprocess(code)

    assert "decltype(arange<uint>)" in output
    assert "void kernel0(" not in output
    assert "template <typename T>" in output


def test_preprocessor_materializes_nested_explicit_template_helper_calls():
    code = """
    template <typename T>
    T cast_value(float value) {
        return T(value);
    }

    template <typename T>
    T twice(float value) {
        return cast_value<T>(value) + cast_value<T>(value);
    }

    kernel void copy(device float* dst [[buffer(0)]]) {
        dst[0] = twice<float>(1.0);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "twice<float>" not in output
    assert "cast_value<float>" not in output
    assert "twice_float(1.0)" in output
    assert "float twice_float(float value)" in output
    assert "cast_value_float(value) + cast_value_float(value)" in output
    assert "float cast_value_float(float value)" in output


def test_preprocessor_preserves_operator_comparison_overloads():
    code = """
    struct complex64_t {
        float real;
        float imag;
    };

    template <typename T>
    T operator+(T lhs, T rhs) {
        return lhs + rhs;
    }

    constexpr bool operator>(complex64_t a, complex64_t b) {
        return a.real > b.real;
    }

    constexpr bool operator<=(complex64_t a, complex64_t b) {
        return operator>(b, a);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "operator<=(complex64_t a, complex64_t b)" in output
    assert "return operator>(b, a);" in output
    assert "operator_complex64_t_a_complex64_t_b" not in output


def test_preprocessor_materializes_only_reachable_explicit_template_helper_calls():
    code = """
    template <typename T>
    T cast_value(float value) {
        return T(value);
    }

    void write_value(device float* dst) {
        dst[0] = cast_value<float>(1.0);
    }

    void unused(device half* dst) {
        dst[0] = cast_value<half>(2.0);
    }

    kernel void copy(device float* dst [[buffer(0)]]) {
        write_value(dst);
    }
    """

    output = MetalPreprocessor(max_template_specializations=1).preprocess(code)

    assert "cast_value_float(1.0)" in output
    assert "float cast_value_float(float value)" in output
    assert "cast_value_half" not in output
    assert "cast_value<half>(2.0)" in output


def test_preprocessor_preserves_explicit_template_specialization_calls():
    code = """
    template <typename T>
    T convert_type(float value) {
        return T(value);
    }

    template <> uint convert_type<uint>(float value) {
        return uint(value + 1.0);
    }

    kernel void copy(device uint* dst [[buffer(0)]]) {
        dst[0] = convert_type<uint>(1.0);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "convert_type<uint>(1.0)" in output
    assert "convert_type_uint(1.0)" not in output
    assert "uint convert_type<uint>(float value)" in output
    assert "uint convert_type_uint(float value)" not in output


def test_preprocessor_reports_explicit_template_specialization_limit():
    code = """
    template <typename T>
    T cast_value(float value) {
        return T(value);
    }

    kernel void copy(device float* dst [[buffer(0)]]) {
        dst[0] = cast_value<float>(1.0);
        dst[1] = cast_value<half>(2.0);
    }
    """

    with pytest.raises(MetalTemplateSpecializationError) as exc_info:
        MetalPreprocessor(max_template_specializations=1).preprocess(code)

    assert "template specialization limit exceeded" in str(exc_info.value)
    assert (
        getattr(exc_info.value, "project_diagnostic_code", None)
        == "project.translate.metal-template-specialization"
    )
    assert getattr(exc_info.value, "missing_capabilities", None) == (
        "template.specialization",
    )


def test_preprocessor_dedupes_equivalent_explicit_template_helper_signatures():
    code = """
    template <typename T, typename IdxT, int Width>
    T load_value(device const T* src, IdxT index) {
        return src[index + Width];
    }

    kernel void copy(device const float* src [[buffer(0)]],
                     device float* dst [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
        dst[0] = load_value<float, uint, 4>(src, gid);
        dst[1] = load_value< float, uint, 4 >(src, gid);
        dst[2] = load_value<float /* same concrete type */, uint, 4>(src, gid);
    }
    """

    output = MetalPreprocessor(max_template_specializations=1).preprocess(code)

    assert output.count("float load_value_float_uint_4(") == 1
    assert output.count("load_value_float_uint_4(src, gid)") == 3
    assert "load_value<" not in output


def test_preprocessor_dedupes_scan_style_repeated_helper_calls_before_budget():
    calls = "\n".join(
        f"        acc = scan_step<float, int, 4, true>(acc, {index});"
        for index in range(24)
    )
    code = f"""
    template <typename T, typename OffsetT, int Width, bool Inclusive>
    T scan_step(T value, OffsetT offset) {{
        return value + T(offset) + T(Width);
    }}

    kernel void scan(device float* dst [[buffer(0)]]) {{
        float acc = 0.0;
{calls}
        dst[0] = acc;
    }}
    """

    output = MetalPreprocessor(max_template_specializations=1).preprocess(code)

    assert output.count("float scan_step_float_int_4_true(") == 1
    assert output.count("scan_step_float_int_4_true(acc,") == 24
    assert "scan_step<" not in output


def test_preprocessor_reports_template_specialization_limit_details():
    code = """
    template <typename T>
    T cast_value(float value) {
        return T(value);
    }

    kernel void copy(device float* dst [[buffer(0)]]) {
        dst[0] = cast_value<float>(1.0);
        dst[1] = cast_value<half>(2.0);
    }
    """

    with pytest.raises(MetalTemplateSpecializationError) as exc_info:
        MetalPreprocessor(
            max_template_specializations=1,
            template_specialization_limit_source=(
                "project.source_options.metal.max_template_specializations"
            ),
        ).preprocess(code)

    error = exc_info.value
    assert error.limit == 1
    assert error.limit_source == (
        "project.source_options.metal.max_template_specializations"
    )
    assert error.unique_specialization_count == 2
    assert error.requested_signature == "cast_value<half>"
    assert "2 unique concrete signatures requested" in str(error)
    assert "limit 1 from project.source_options.metal.max_template_specializations" in (
        str(error)
    )
    assert "Suggested action:" in str(error)


def test_preprocessor_preserves_incomplete_multiline_function_macro_invocation():
    code = """
    #define DECLARE_TYPED(name, type) type name;
    DECLARE_TYPED(
        value,
        float);
    """

    output = MetalPreprocessor().preprocess(code)

    assert "DECLARE_TYPED(" in output
    assert "float value" not in output


def test_preprocessor_include_with_search_path(tmp_path):
    include_file = tmp_path / "constants.metal"
    include_file.write_text("constant int from_include = 7;\n", encoding="utf-8")
    code = """
    #include "constants.metal"
    int value = from_include;
    """

    values = token_values(
        code,
        include_paths=[str(tmp_path)],
        file_path=str(tmp_path / "main.metal"),
    )

    assert "from_include" in values
    assert '"constants.metal"' not in values


def test_preprocessor_include_honors_pragma_once(tmp_path):
    include_file = tmp_path / "constants.metal"
    include_file.write_text(
        "#pragma once\nconstant int guarded = 7;\n", encoding="utf-8"
    )
    code = """
    #include "constants.metal"
    #include "constants.metal"
    int value = guarded;
    """

    output = MetalPreprocessor(include_paths=[str(tmp_path)]).preprocess(
        code,
        file_path=str(tmp_path / "main.metal"),
    )

    assert output.count("guarded") == 2
    assert "#pragma once" not in output


def test_unresolved_system_include_is_preserved():
    code = """
    #include <metal_stdlib>
    using namespace metal;
    """

    output = MetalPreprocessor().preprocess(code)
    values = token_values(code)

    assert "#include <metal_stdlib>" in output
    assert "#include <metal_stdlib>" in values
    assert "using" in values


def test_dawn_tint_warning_prologue_is_stripped_before_msl():
    # Reduced from:
    # Repo: https://dawn.googlesource.com/dawn
    # Commit: 78a171ad2ed7f7265cfc3dd52e4e7a637a099df0
    # Path: test/tint/bug/tint/2201.wgsl.expected.msl
    code = """
    <dawn>/test/tint/bug/tint/2201.wgsl:9:9 warning: code is unreachable
            let _e16_ = vec2(false, false);
            ^^^^^^^^^

    #include <metal_stdlib>
    using namespace metal;
    kernel void v() {}
    """

    output = MetalPreprocessor().preprocess(code)

    assert output.lstrip().startswith("#include <metal_stdlib>")
    assert "warning: code is unreachable" not in output


def test_diagnostic_error_prologue_is_preserved():
    code = """
    <dawn>/test/tint/broken.wgsl:1:1 error: invalid shader

    #include <metal_stdlib>
    using namespace metal;
    kernel void v() {}
    """

    output = MetalPreprocessor().preprocess(code)

    assert "error: invalid shader" in output


def test_target_conditionals_simulator_macro_defaults_to_device_import():
    code = """
    #include <TargetConditionals.h>
    #ifndef TARGET_OS_SIMULATOR
    #error TARGET_OS_SIMULATOR not defined. Check <TargetConditionals.h>
    #endif
    #if TARGET_OS_SIMULATOR
    int simulator_only = ;
    #else
    int device_import = 1;
    #endif
    """

    values = token_values(code)

    assert "device_import" in values
    assert "simulator_only" not in values


def test_has_include_checks_include_paths_from_metalpetal_header(tmp_path):
    # Reduced from:
    # Repo: https://github.com/MetalPetal/MetalPetal
    # Commit: f9b78897bd4214bb097f352a1bde0a4f4a1e2ddb
    # Path: Frameworks/MetalPetal/MTIContext+Internal.h
    package_header = tmp_path / "MetalPetal" / "MetalPetal.h"
    package_header.parent.mkdir()
    package_header.write_text("// package umbrella\n", encoding="utf-8")
    code = """
    #if __has_include(<MetalPetal/MetalPetal.h>)
    int package_layout = 1;
    #else
    int local_layout = 1;
    #endif
    """

    values = token_values(code, include_paths=[str(tmp_path)])

    assert "package_layout" in values
    assert "local_layout" not in values


def test_strict_missing_include_raises(tmp_path):
    code = '#include "missing.metal"\nint value = 1;'

    with pytest.raises(FileNotFoundError, match="Include not found: missing.metal"):
        MetalLexer(
            code,
            include_paths=[str(tmp_path)],
            strict_preprocessor=True,
            file_path=str(tmp_path / "main.metal"),
        ).tokenize()


def test_parser_uses_preprocessed_conditionals():
    code = """
    #define ENABLE_BODY 1
    #if ENABLE_BODY
    void main() {
        int value = 1;
    }
    #else
    void broken() {
        int value = ;
    }
    #endif
    """

    tokens = MetalLexer(code).tokenize()
    ast = MetalParser(tokens).parse()

    assert ast is not None
    assert [func.name for func in ast.functions] == ["main"]


def test_preprocessor_can_be_disabled_for_raw_directive_tokens():
    code = """
    #define VALUE 1
    int value = VALUE;
    """

    values = token_values(code, preprocess=False)

    assert "#define VALUE 1" in values
    assert "VALUE" in values


def test_find_and_materialize_template_struct():
    # Foundation for the struct-template materializer (issue #1354): detect a
    # struct template and materialize a concrete instantiation by substituting
    # both type and non-type parameters and renaming the declaration.
    code = """
    template <typename T, int N>
    struct Tile {
        T data[N];
        T sum() {
            T total = T(0);
            for (int i = 0; i < N; ++i) {
                total += data[i];
            }
            return total;
        }
    };
    """
    preprocessor = MetalPreprocessor()
    structs = preprocessor._find_template_structs(code)
    assert len(structs) == 1
    struct = structs[0]
    assert struct.name == "Tile"
    assert struct.template_parameters == ["T", "N"]

    materialized = preprocessor._materialize_template_struct_with_name(
        struct, ["float", "4"], "Tile_float_4"
    )
    assert "struct Tile_float_4" in materialized
    assert "float data[4];" in materialized
    assert "for (int i = 0; i < 4; ++i)" in materialized
    assert "float total = float(0);" in materialized
    # Template parameters and the template<> header must not leak through.
    assert "[N]" not in materialized
    assert "template" not in materialized


def test_materialize_template_struct_requires_all_parameters():
    code = """
    template <typename T>
    struct Box {
        T value;
    };
    """
    preprocessor = MetalPreprocessor()
    struct = preprocessor._find_template_structs(code)[0]
    # No arguments for a required parameter -> not materialized.
    assert (
        preprocessor._materialize_template_struct_with_name(struct, [], "Box_x") == ""
    )


def test_preprocessor_materializes_struct_template_instantiation():
    # Wiring the struct-template materializer (issue #1354): a concrete
    # `StructName<args>` type reference is rewritten to a materialized concrete
    # struct, the counterpart of the function-template call materializer.
    code = """
    template <typename T, int N>
    struct Tile {
        T data[N];
    };

    [[kernel]] void k(
        device float* out [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Tile<float, 4> tile;
        tile.data[0] = out[i];
        out[i] = tile.data[0];
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "Tile_float_4 tile;" in output
    assert "struct Tile_float_4 {" in output
    assert "float data[4];" in output
    # The materialized concrete struct is terminated with a semicolon.
    assert output.rstrip().endswith(";")
    # No residual instantiation survives.
    assert "Tile<float, 4>" not in output


def test_preprocessor_materializes_nested_struct_template_instantiations():
    # Nested instantiations resolve through iteration: materializing Outer
    # surfaces a concrete Inner<float, 8> reference, which is then materialized.
    code = """
    template <typename T, int N> struct Inner { T vals[N]; };
    template <typename T, int N> struct Outer { Inner<T, N> inner; };

    [[kernel]] void k(
        device float* out [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Outer<float, 8> o;
        out[i] = o.inner.vals[0];
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "Outer_float_8 o;" in output
    assert "struct Outer_float_8 {" in output
    assert "Inner_float_8 inner;" in output
    assert "struct Inner_float_8 {" in output
    assert "float vals[8];" in output
    assert "Outer<float, 8>" not in output
    assert "Inner<float, 8>" not in output


def test_preprocessor_leaves_specialized_struct_templates_unmaterialized():
    # A struct template with an explicit specialization is left for the future
    # specialization-aware path rather than incorrectly materializing only the
    # primary template; the residual instantiation falls back to the diagnostic.
    code = """
    template <typename T> struct Trait { T value; };
    template <> struct Trait<bool> { int value; };

    [[kernel]] void k(
        device int* out [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Trait<bool> t;
        out[i] = t.value;
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "Trait<bool>" in output
    assert "struct Trait_bool {" not in output


def test_preprocessor_struct_materializer_ignores_non_template_structs():
    # Non-template structs are untouched; the materializer only rewrites concrete
    # references to struct templates.
    code = """
    struct Plain { float x; };

    [[kernel]] void k(
        device float* out [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Plain p;
        p.x = out[i];
        out[i] = p.x;
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "struct Plain {" in output
    assert "Plain_" not in output


def test_preprocessor_lowers_instance_method_with_member_reference():
    # CrossGL structs are data-only, so an instance member function is lowered to
    # a free function taking the receiver `self` by value, and bare references to
    # the struct's data members inside the body are rewritten to `self.member`.
    code = """
    struct Adder { float bias; float add(float a, float b){ return a + b + bias; } };

    [[kernel]] void k(
        device const float* in [[buffer(0)]],
        device float* out [[buffer(1)]],
        uint i [[thread_position_in_grid]]) {
        Adder adder;
        adder.bias = 1.0;
        out[i] = adder.add(in[i], 2.0);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    # The struct keeps only its data member; the method is gone from the struct.
    assert "struct Adder { float bias;" in output
    assert "float add(" not in output.split("struct Adder")[1].split("}")[0]
    # The method is re-emitted as a free function with a by-value `self` receiver
    # and the member reference qualified.
    assert "float Adder__add(Adder self, float a, float b)" in output
    assert "a + b + self.bias" in output
    # The call site is rewritten to pass the receiver as the first argument.
    assert "Adder__add(adder, in[i], 2.0)" in output
    # A bare data-member access is left untouched.
    assert "adder.bias = 1.0;" in output
    # No dangling method call survives.
    assert "adder.add(" not in output


def test_preprocessor_lowers_static_method_without_self_parameter():
    # A static member function is lowered to a free function with NO `self`
    # parameter, and a qualified `S::m(args)` call becomes `S__m(args)`.
    code = """
    struct Maths { static float twice(float a){ return a * 2.0; } };

    [[kernel]] void k(
        device float* out [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        out[i] = Maths::twice(out[i]);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "float Maths__twice(float a)" in output
    # No `self` is introduced for a static method.
    assert "Maths self" not in output
    assert "Maths__twice(out[i])" in output
    assert "Maths::twice" not in output


def test_preprocessor_lowers_call_operator_functor():
    # An `operator()` functor is lowered to a free function named
    # `S__operator_call`, and a `var(args)` call where `var` has type S is
    # rewritten to `S__operator_call(var, args)`.
    code = """
    struct Sum { float operator()(float a, float b){ return a + b; } };

    [[kernel]] void k(
        device const float* in [[buffer(0)]],
        device float* out [[buffer(1)]],
        uint i [[thread_position_in_grid]]) {
        Sum op;
        out[i] = op(in[i], out[i]);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "float Sum__operator_call(Sum self, float a, float b)" in output
    assert "Sum__operator_call(op, in[i], out[i])" in output


def test_preprocessor_lowers_materialized_template_functor():
    # After the struct-template materializer produces a concrete `Sum_float`, the
    # member-function lowering pass lowers its `operator()` and rewrites the call.
    code = """
    template <typename U>
    struct Sum { static constexpr constant U init = U(0); U operator()(U a, U b){ return a + b; } };

    [[kernel]] void k(
        device const float* in [[buffer(0)]],
        device float* out [[buffer(1)]],
        uint i [[thread_position_in_grid]]) {
        Sum<float> op;
        float acc = op(in[i], out[i]);
        out[i] = acc;
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "struct Sum_float {" in output
    assert "float Sum_float__operator_call(Sum_float self, float a, float b)" in output
    assert "Sum_float__operator_call(op, in[i], out[i])" in output
    # The primary template is left untouched (template methods are out of scope).
    assert "template <typename U>" in output
    # No dangling functor call survives on the concrete instance.
    assert "= op(" not in output


def test_preprocessor_lowering_qualifies_member_only_when_not_shadowed():
    # A member reference shadowed by a parameter or a local stays bare; the
    # `this->member` spelling is always rewritten to `self.member`.
    code = """
    struct C { float scale; float apply(float scale){ return this->scale * scale; } };

    [[kernel]] void k(
        device float* out [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        C c;
        c.scale = 2.0;
        out[i] = c.apply(out[i]);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    # `this->scale` -> `self.scale`; the parameter `scale` is NOT rewritten.
    assert "return self.scale * scale;" in output
    assert "C__apply(c, out[i])" in output


def test_preprocessor_lowering_is_noop_without_struct_methods():
    # Regression-safety: a struct with no member functions (and the surrounding
    # kernel) must be byte-identical after the lowering pass.
    code = (
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct Plain { float x; float y; };\n"
        "kernel void k(device float* o [[buffer(0)]],"
        " uint i [[thread_position_in_grid]]) {\n"
        "    Plain p;\n"
        "    p.x = 1.0;\n"
        "    o[i] = p.x + p.y;\n"
        "}\n"
    )
    preprocessor = MetalPreprocessor()
    assert preprocessor._lower_struct_member_functions(code) == code


def test_preprocessor_lowering_leaves_method_free_source_untouched():
    # Regression-safety: a kernel with no structs at all is byte-identical.
    code = (
        "kernel void k(device float* o [[buffer(0)]],"
        " uint i [[thread_position_in_grid]]) {\n"
        "    o[i] = o[i] * 2.0;\n"
        "}\n"
    )
    preprocessor = MetalPreprocessor()
    assert preprocessor._lower_struct_member_functions(code) == code


def test_preprocessor_instantiates_template_method_from_buffer_element_arg():
    # A CALLED template member method is instantiated from a buffer-element
    # argument (`in[i]` whose element type is float) and lowered to a concrete
    # free function `S__m__float(S self, float val)`; the call is rewritten.
    code = """
    struct Sum { template <typename T> T reduce(T val){ return val + val; } };

    kernel void k(
        device const float* in [[buffer(0)]],
        device float* out [[buffer(1)]],
        uint i [[thread_position_in_grid]]) {
        Sum op;
        out[i] = op.reduce(in[i]);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "float Sum__reduce__float(Sum self, float val)" in output
    assert "Sum__reduce__float(op, in[i])" in output
    # The struct is data-only; the template method is gone from it.
    assert "template" not in output.split("struct Sum")[1].split("}")[0]
    # No dangling template-method call survives.
    assert "op.reduce(" not in output


def test_preprocessor_instantiates_template_method_from_typed_local_arg():
    # The argument is a bare local declared `half v = ...;` so the method
    # instantiates with T=half.
    code = """
    struct Sum { template <typename T> T reduce(T val){ return val; } };

    kernel void k(
        device half* out [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Sum op;
        half v = out[i];
        out[i] = op.reduce(v);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "half Sum__reduce__half(Sum self, half val)" in output
    assert "Sum__reduce__half(op, v)" in output


def test_preprocessor_instantiates_template_operator_call_from_typed_local():
    # A template `operator()` is instantiated from the call-site argument types;
    # the struct template parameter U is materialized first, the method's own
    # template parameter T binds from the second argument.
    code = """
    template <typename U>
    struct Op { template <typename T> U operator()(U a, T b){ return a + b; } };

    [[kernel]] void k(
        device const float* in [[buffer(0)]],
        device float* out [[buffer(1)]],
        uint i [[thread_position_in_grid]]) {
        Op<float> op;
        float acc = out[i];
        float x = in[i];
        out[i] = op(acc, x);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "struct Op_float {" in output
    assert (
        "float Op_float__operator_call__float(Op_float self, float a, float b)"
        in output
    )
    assert "Op_float__operator_call__float(op, acc, x)" in output
    # No dangling functor call survives.
    assert "= op(" not in output


def test_preprocessor_instantiates_static_template_method_from_literal():
    # A static template member method called as `S::m(literal)` instantiates from
    # the literal type and lowers WITHOUT a `self` parameter.
    code = """
    struct Maths { template <typename T> static T twice(T a){ return a + a; } };

    [[kernel]] void k(
        device float* out [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        out[i] = Maths::twice(2.5f);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "float Maths__twice__float(float a)" in output
    assert "Maths self" not in output
    assert "Maths__twice__float(2.5f)" in output
    assert "Maths::twice" not in output


def test_preprocessor_template_method_instantiations_are_deduplicated():
    # Two call sites with the same inferred argument type share ONE concrete
    # free function (deduplicated by struct/method/bindings).
    code = """
    struct Sum { template <typename T> T reduce(T val){ return val; } };

    kernel void k(
        device const float* in [[buffer(0)]],
        device float* out [[buffer(1)]],
        uint i [[thread_position_in_grid]]) {
        Sum op;
        float a = op.reduce(in[i]);
        out[i] = op.reduce(a);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert output.count("float Sum__reduce__float(Sum self, float val)") == 1
    assert "Sum__reduce__float(op, in[i])" in output
    assert "Sum__reduce__float(op, a)" in output


def test_preprocessor_template_method_uninferable_argument_clean_fails():
    # An un-inferable argument shape (a function-call result) raises a clean
    # MetalStructMethodError that PROPAGATES out of preprocess rather than
    # leaving a dangling `op.reduce(...)` / broken output.
    code = """
    struct Sum { template <typename T> T reduce(T val){ return val; } };
    float helper(uint i){ return 1.0; }

    kernel void k(
        device float* out [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Sum op;
        out[i] = op.reduce(helper(i));
    }
    """
    with pytest.raises(MetalStructMethodError) as excinfo:
        MetalPreprocessor().preprocess(code)
    assert excinfo.value.struct_name == "Sum"
    assert excinfo.value.method_name == "reduce"
    assert (
        excinfo.value.project_diagnostic_code == "project.translate.metal-struct-method"
    )


def test_preprocessor_template_method_conflicting_binding_clean_fails():
    # The single template parameter T appears in two parameters whose call-site
    # arguments infer to DIFFERENT concrete types (float and int). Rather than
    # silently keeping the first guess, this clean-fails (never guess).
    code = """
    struct D { template <typename T> T pick(T a, T b){ return a; } };

    kernel void k(
        device float* o [[buffer(0)]],
        device int* q [[buffer(1)]],
        uint i [[thread_position_in_grid]]) {
        D d;
        o[i] = d.pick(o[i], q[i]);
    }
    """
    with pytest.raises(MetalStructMethodError):
        MetalPreprocessor().preprocess(code)


def test_preprocessor_template_method_consistent_multi_param_instantiates():
    # A multi-parameter template method whose parameters consistently bind T to
    # one type instantiates to a single concrete free function.
    code = """
    struct D { template <typename T> T pick(T a, T b){ return a; } };

    kernel void k(
        device float* o [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        D d;
        o[i] = d.pick(o[i], o[i]);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "float D__pick__float(D self, float a, float b)" in output
    assert "D__pick__float(d, o[i], o[i])" in output


def test_preprocessor_template_method_unbound_parameter_clean_fails():
    # A template parameter that cannot be deduced from any argument (here T only
    # appears in the return type) clean-fails rather than guessing.
    code = """
    struct Maker { template <typename T> T make(float a){ return T(a); } };

    kernel void k(
        device float* out [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Maker m;
        out[i] = m.make(out[i]);
    }
    """
    with pytest.raises(MetalStructMethodError):
        MetalPreprocessor().preprocess(code)


def test_preprocessor_template_method_lowering_is_noop_without_template_calls():
    # Regression-safety: a struct that DECLARES a template method but never CALLS
    # it is left untouched (the uninstantiated template would be dropped by the
    # parser exactly as before; no clean-fail, no spurious free function).
    code = (
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct Sum { float x; template <typename T> T reduce(T v){ return v; } };\n"
        "kernel void k(device float* o [[buffer(0)]],"
        " uint i [[thread_position_in_grid]]) {\n"
        "    o[i] = o[i] * 2.0;\n"
        "}\n"
    )
    output = MetalPreprocessor().preprocess(code)
    # No instantiation emitted and no exception raised; the data member survives.
    assert "Sum__reduce" not in output
    assert "float x;" in output


def test_preprocessor_template_method_full_pipeline_to_hlsl():
    # End-to-end: the Sum reduction example translates to valid HLSL with the
    # correct T=float instantiation (simd_sum -> WaveActiveSum) and no dangling
    # simd_reduce call.
    from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter
    from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
    from crosstl.translator.lexer import Lexer as CrossGLLexer
    from crosstl.translator.parser import Parser as CrossGLParser

    code = (
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "template <typename U> struct Sum {"
        " template <typename T> T simd_reduce(T val){ return simd_sum(val); }"
        " U operator()(U a, U b){ return a + b; } };\n"
        "kernel void reduce(device const float* in [[buffer(0)]],"
        " device float* out [[buffer(1)]],"
        " uint i [[thread_position_in_grid]]) {\n"
        "    Sum<float> op;\n"
        "    float v = op.simd_reduce(in[i]);\n"
        "    out[i] = op(v, 0.0);\n"
        "}\n"
    )
    pre = MetalPreprocessor().preprocess(code)
    assert "float Sum_float__simd_reduce__float(Sum_float self, float val)" in pre
    tokens = MetalLexer(pre).tokenize()
    ast = MetalParser(tokens).parse()
    crossgl = MetalToCrossGLConverter().generate(ast)
    parsed = CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse()
    hlsl = HLSLCodeGen().generate(parsed)
    # No dangling receiver call and no bare Metal `simd_reduce(` builtin survive
    # (the lowered free-function NAME legitimately contains the substring).
    assert "op.simd_reduce(" not in hlsl
    assert " simd_reduce(" not in hlsl
    assert "WaveActiveSum" in hlsl
    assert "Sum_float__simd_reduce__float(op, in_.Load(i))" in hlsl


def test_preprocessor_instantiates_template_method_from_local_array_subscript():
    # A CALLED template member method whose argument is a LOCAL ARRAY subscript
    # (`totals[i]`, `T totals[N]`) instantiates with T = the array element type.
    # This is the dominant argument shape in MLX reduce, where reduction
    # accumulators are stack arrays.
    code = """
    struct Sum { template <typename T> T reduce(T val){ return val; } };

    kernel void k(
        device float* out [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Sum op;
        float totals[4];
        totals[i] = out[i];
        out[i] = op.reduce(totals[i]);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "float Sum__reduce__float(Sum self, float val)" in output
    assert "Sum__reduce__float(op, totals[i])" in output


def test_preprocessor_instantiates_template_method_from_threadgroup_array_subscript():
    # A `threadgroup U sh[N]` array local subscript is element-typed too.
    code = """
    struct Sum { template <typename T> T reduce(T val){ return val; } };

    kernel void k(
        device half* out [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Sum op;
        threadgroup half shared_vals[32];
        out[i] = op.reduce(shared_vals[i]);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "half Sum__reduce__half(Sum self, half val)" in output
    assert "Sum__reduce__half(op, shared_vals[i])" in output


def test_preprocessor_instantiates_template_method_from_member_access_subscript():
    # A member-access subscript `obj.member[i]` is element-typed via the struct
    # field's element type, even when the carrier struct (`Buf`) has NO methods.
    code = """
    struct Buf { device float* data; };
    struct Sum { template <typename T> T reduce(T val){ return val; } };

    kernel void k(
        device float* o [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Sum op;
        Buf init;
        o[i] = op.reduce(init.data[i]);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "float Sum__reduce__float(Sum self, float val)" in output
    assert "Sum__reduce__float(op, init.data[i])" in output


def test_preprocessor_instantiates_template_method_from_member_access():
    # A bare member access `obj.member` (non-subscript) resolves to the field's
    # declared type.
    code = """
    struct Pt { float x; };
    struct Sum { template <typename T> T reduce(T val){ return val; } };

    kernel void k(
        device float* o [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Sum op;
        Pt p;
        o[i] = op.reduce(p.x);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "float Sum__reduce__float(Sum self, float val)" in output
    assert "Sum__reduce__float(op, p.x)" in output


def test_preprocessor_instantiates_template_method_from_union_local():
    # A union-typed local (`bool4_or_uint update;`) is recognized by the local
    # scanner, so a bare such local is an inferable call argument.
    code = """
    union bool4_or_uint { bool4 b; unsigned int i; };
    struct Sum { template <typename T> T reduce(T val){ return val; } };

    kernel void k(
        device uint* o [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Sum op;
        bool4_or_uint update;
        o[i] = op.reduce(update).i;
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert (
        "bool4_or_uint Sum__reduce__bool4_or_uint(Sum self, bool4_or_uint val)"
        in output
    )
    assert "Sum__reduce__bool4_or_uint(op, update)" in output


def test_preprocessor_instantiates_template_method_from_struct_local():
    # A struct-typed local of a method-less carrier is an inferable bare argument.
    code = """
    struct Pair { float a; float b; };
    struct Sum { template <typename T> T reduce(T val){ return val; } };

    kernel void k(
        device float* o [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Sum op;
        Pair p;
        o[i] = op.reduce(p).a;
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "Pair Sum__reduce__Pair(Sum self, Pair val)" in output
    assert "Sum__reduce__Pair(op, p)" in output


def test_preprocessor_instantiates_template_method_from_function_pointer_parameter():
    # A call argument that subscripts a POINTER PARAMETER of the enclosing
    # function (`device T* p` -> `p[i]` is T) is inferable.
    code = """
    struct Sum { template <typename T> T reduce(T val){ return val; } };

    float helper(device float* p, uint i) {
        Sum op;
        return op.reduce(p[i]);
    }

    kernel void k(
        device float* o [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        o[i] = helper(o, i);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "float Sum__reduce__float(Sum self, float val)" in output
    assert "Sum__reduce__float(op, p[i])" in output


def test_preprocessor_instantiates_template_method_from_function_scalar_parameter():
    # A call argument that is a bare SCALAR PARAMETER of the enclosing function is
    # inferable (the local-declaration scanner alone misses comma-terminated
    # parameters).
    code = """
    struct Sum { template <typename T> T reduce(T val){ return val; } };

    half helper(half v) {
        Sum op;
        return op.reduce(v);
    }

    kernel void k(
        device half* o [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        o[i] = helper(o[i]);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "half Sum__reduce__half(Sum self, half val)" in output
    assert "Sum__reduce__half(op, v)" in output


def test_preprocessor_template_member_lowering_is_deterministic_across_kernels():
    # Regression for a PYTHONHASHSEED-dependent ordering bug: two kernels in one
    # file that each declare a variable named `op` of a DIFFERENT concrete functor
    # type must resolve `op` per-kernel (nearest preceding declaration) instead of
    # collapsing to one set-iteration-order winner. The lowered output is the same
    # regardless of declaration discovery order.
    code = """
    struct SumF { template <typename T> T reduce(T val){ return val + val; } };
    struct MaxF { template <typename T> T reduce(T val){ return val; } };

    kernel void k_sum(
        device float* o [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        SumF op;
        float totals[4];
        o[i] = op.reduce(totals[i]);
    }

    kernel void k_max(
        device int* o [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        MaxF op;
        int totals[4];
        o[i] = op.reduce(totals[i]);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    # Each kernel binds `op` to its OWN struct type and element type.
    assert "float SumF__reduce__float(SumF self, float val)" in output
    assert "int MaxF__reduce__int(MaxF self, int val)" in output
    assert "SumF__reduce__float(op, totals[i])" in output
    assert "MaxF__reduce__int(op, totals[i])" in output
    # No cross-kernel mis-binding.
    assert "MaxF__reduce__float" not in output
    assert "SumF__reduce__int" not in output


def test_infer_argument_type_local_array_and_member_subscript():
    # Direct unit coverage of the conservative inference contract for the new
    # shapes, using flat (position-resolved) views like the call site builds.
    pp = MetalPreprocessor()
    buffers = {"totals": "float", "buf": "uint"}
    locals_ = {"acc": "half", "update": "bool4_or_uint"}
    fields = {"init": {"data": "float", "scale": "half"}}
    # Local array / buffer subscript -> element type.
    assert pp._infer_argument_type("totals[i]", buffers, locals_, fields) == "float"
    assert pp._infer_argument_type("buf[i + 1]", buffers, locals_, fields) == "uint"
    # Member-access subscript -> struct field element type.
    assert (
        pp._infer_argument_type("init.data[elem]", buffers, locals_, fields) == "float"
    )
    # Bare member access -> field type.
    assert pp._infer_argument_type("init.scale", buffers, locals_, fields) == "half"
    # Union / struct local -> its declared type.
    assert (
        pp._infer_argument_type("update", buffers, locals_, fields) == "bool4_or_uint"
    )
    # Unknown shapes stay un-inferable (never guess).
    assert pp._infer_argument_type("missing[i]", buffers, locals_, fields) is None
    assert pp._infer_argument_type("init.unknown", buffers, locals_, fields) is None
    assert pp._infer_argument_type("foo()", buffers, locals_, fields) is None


def test_collect_buffer_element_types_rejects_non_declaration_subscripts():
    # The array-declaration scanner must NEVER record a bogus element type from a
    # statement that merely matches the `<token> name[` shape — e.g.
    # `return totals[i]` (leading token `return`) or a control-flow subscript.
    # Only a recognized scalar/vector or struct/union element type is accepted.
    pp = MetalPreprocessor()
    code = (
        "struct Pair { float a; };\n"
        "kernel void k(device float* o [[buffer(0)]],"
        " uint i [[thread_position_in_grid]]) {\n"
        "    float totals[4];\n"
        "    Pair items[4];\n"
        "    return totals[i];\n"
        "    if (i) block[i];\n"
        "}\n"
    )
    element_types = pp._collect_buffer_element_types(code, [])
    # totals is a real `float totals[4]` array; the only recorded type is float.
    assert [t for _, t in element_types.get("totals", [])] == ["float"]
    # A struct array records its element struct type.
    assert [t for _, t in element_types.get("items", [])] == ["Pair"]
    # `return`/`if` tokens never become element types.
    assert "block" not in element_types
    assert all(
        t not in {"return", "if", "else", "for", "while"}
        for entries in element_types.values()
        for _, t in entries
    )


# --------------------------------------------------------------------------- #
# SFINAE-overloaded template member methods (MLX `simd_reduce`, issue #1354).  #
# --------------------------------------------------------------------------- #

# A synthetic struct mirroring MLX's DEFINE_SIMD_REDUCE() macro expansion plus a
# return-type-SFINAE `simd_reduce_impl` (Min/Max style), used across the tests
# below. It exercises BOTH SFINAE layers: the non-type constraint on
# `simd_reduce` (sizeof) and the return-type constraint on `simd_reduce_impl`
# (is_integral_v).
_SFINAE_SIMD_REDUCE_STRUCT = (
    "static constant constexpr const uint8_t simd_size = 32;\n"
    "template <typename U>\n"
    "struct Red {\n"
    "  template <typename T, metal::enable_if_t<sizeof(T) < 8, bool> = true>\n"
    "  T simd_reduce(T val) { return simd_reduce_impl(val); }\n"
    "  template <typename T, metal::enable_if_t<sizeof(T) == 8, bool> = true>\n"
    "  T simd_reduce(T val) {\n"
    "    for (short i = simd_size / 2; i > 0; i /= 2) {\n"
    "      val = operator()(val, simd_shuffle_down(val, i));\n"
    "    }\n"
    "    return val;\n"
    "  }\n"
    "  template <typename T>\n"
    "  metal::enable_if_t<metal::is_integral_v<T>, T> simd_reduce_impl(T val) {\n"
    "    return simd_sum(val);\n"
    "  }\n"
    "  template <typename T>\n"
    "  metal::enable_if_t<!metal::is_integral_v<T>, T> simd_reduce_impl(T val) {\n"
    "    return simd_sum(val);\n"
    "  }\n"
    "  U operator()(U a, U b) { return a + b; }\n"
    "};\n"
)


def _sfinae_reduce_source(elem):
    return (
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        + _SFINAE_SIMD_REDUCE_STRUCT
        + f"kernel void reduce(device const {elem}* in [[buffer(0)]],\n"
        f"  device {elem}* out [[buffer(1)]],\n"
        "  uint i [[thread_position_in_grid]]) {\n"
        f"    Red<{elem}> op;\n"
        f"    {elem} total = in[i];\n"
        "    out[i] = op.simd_reduce(total);\n"
        "}\n"
    )


def test_sfinae_template_param_list_parses_constraint_separately():
    # The `<` disambiguation: a SFINAE non-type parameter whose constraint embeds
    # a comparison (`sizeof(T) < 8`) is parsed as ONE parameter; the bindable type
    # parameter list is just `T` and the constraint text is recorded apart.
    pp = MetalPreprocessor()
    text = "typename T, metal::enable_if_t<sizeof(T) < 8, bool> = true"
    assert pp._template_parameter_names(text) == ["T"]
    assert pp._template_parameter_constraints(text) == [
        "metal::enable_if_t<sizeof(T) < 8, bool>"
    ]
    # The matching angle for the parameter list is found despite the comparison.
    header = "<" + text + ">"
    assert pp._find_matching_template_param_angle(header, 0) == len(header) - 1
    # The `== 8` variant likewise yields a single bindable `T`, not a spurious one.
    text_eq = "typename T, metal::enable_if_t<sizeof(T) == 8, bool> = true"
    assert pp._template_parameter_names(text_eq) == ["T"]
    # Existing parameter shapes are unchanged.
    assert pp._template_parameter_names("typename U = bool") == ["U"]
    assert pp._template_parameter_defaults("typename U = bool") == {"U": "bool"}
    assert pp._template_parameter_names("typename T, int N") == ["T", "N"]
    assert pp._variadic_template_parameter_names("typename... Args") == {"Args"}


def test_sfinae_simd_reduce_recognized_as_template_methods():
    # Both `simd_reduce` overloads and both `simd_reduce_impl` overloads are
    # recognized as template member methods (bindable `T`), with their SFINAE
    # constraints captured and the return-type SFINAE unwrapped to the value type.
    pp = MetalPreprocessor()
    body = _SFINAE_SIMD_REDUCE_STRUCT.split("struct Red {", 1)[1].rsplit("};", 1)[0]
    _names, _types, concrete, templates = pp._split_struct_body("Red", body, 0)
    by_name = {}
    for method in templates:
        by_name.setdefault(method.name, []).append(method)
    assert len(by_name.get("simd_reduce", [])) == 2
    assert len(by_name.get("simd_reduce_impl", [])) == 2
    for method in templates:
        assert method.template_parameters == ["T"]
    reduce_constraints = {
        c for m in by_name["simd_reduce"] for c in m.template_constraints
    }
    assert reduce_constraints == {
        "metal::enable_if_t<sizeof(T) < 8, bool>",
        "metal::enable_if_t<sizeof(T) == 8, bool>",
    }
    for method in by_name["simd_reduce_impl"]:
        # Return-type SFINAE unwrapped to plain `T`; constraint recorded.
        assert method.return_type == "T"
        assert method.return_type_constraint in {
            "metal::is_integral_v<T>",
            "!metal::is_integral_v<T>",
        }


def test_sfinae_constraint_evaluation_size_and_integral_tables():
    # Direct coverage of the size / is_integral tables and constraint evaluation.
    pp = MetalPreprocessor()
    assert pp._sizeof_concrete_type("float") == 4
    assert pp._sizeof_concrete_type("int") == 4
    assert pp._sizeof_concrete_type("long") == 8
    assert pp._sizeof_concrete_type("double") == 8
    assert pp._sizeof_concrete_type("half") == 2
    assert pp._sizeof_concrete_type("float4") == 16
    assert pp._is_integral_concrete_type("int") is True
    assert pp._is_integral_concrete_type("uint") is True
    assert pp._is_integral_concrete_type("float") is False
    assert pp._is_integral_concrete_type("half") is False
    less8 = "metal::enable_if_t<sizeof(T) < 8, bool>"
    eq8 = "metal::enable_if_t<sizeof(T) == 8, bool>"
    assert pp._evaluate_template_constraint(less8, {"T": "float"}) is True
    assert pp._evaluate_template_constraint(less8, {"T": "long"}) is False
    assert pp._evaluate_template_constraint(eq8, {"T": "double"}) is True
    assert pp._evaluate_template_constraint(eq8, {"T": "int"}) is False
    assert (
        pp._evaluate_template_constraint("metal::is_integral_v<T>", {"T": "int"})
        is True
    )
    assert (
        pp._evaluate_template_constraint("!metal::is_integral_v<T>", {"T": "float"})
        is True
    )
    # An unrecognized constraint / type clean-fails (raises, never guesses).
    with pytest.raises(MetalPreprocessor._UnrecognizedConstraint):
        pp._evaluate_template_constraint(
            "metal::is_floating_point_v<T>", {"T": "float"}
        )
    with pytest.raises(MetalPreprocessor._UnrecognizedConstraint):
        pp._evaluate_template_constraint(less8, {"T": "complex64_t"})


@pytest.mark.parametrize(
    "elem,impl_suffix",
    [
        ("float", "float"),  # sizeof 4 -> <8 ; non-integral simd_reduce_impl
        ("int", "int"),  # sizeof 4 -> <8 ; integral simd_reduce_impl
    ],
)
def test_sfinae_simd_reduce_selects_less8_overload_and_lowers_chain(elem, impl_suffix):
    # `T` of size 4 selects the sizeof<8 overload, whose body calls
    # `simd_reduce_impl` — the second SFINAE layer is resolved too, so the whole
    # chain lowers to free functions with no dangling call.
    output = MetalPreprocessor().preprocess(_sfinae_reduce_source(elem))
    reduce_fn = f"{elem} Red_{elem}__simd_reduce__{elem}(Red_{elem} self, {elem} val)"
    assert reduce_fn in output
    assert f"Red_{elem}__simd_reduce_impl__{impl_suffix}" in output
    assert f"Red_{elem}__simd_reduce__{elem}(op, total)" in output
    # No dangling receiver call survives.
    assert "op.simd_reduce(" not in output
    # The lowered `simd_reduce` free function calls the LOWERED impl, not a bare
    # `simd_reduce_impl(...)`. (A bare `simd_reduce_impl(` legitimately remains in
    # the leftover PRIMARY template body, which the downstream parser drops.)
    reduce_body = output.split(reduce_fn, 1)[1].split("}", 1)[0]
    assert "self" in reduce_body  # receiver threaded into the internal call
    assert re.search(r"(?<![\w_])simd_reduce_impl\s*\(", reduce_body) is None
    # The only call sites of the bare impl name are inside `template <` blocks.
    for match in re.finditer(r"(?<![\w_])simd_reduce_impl\s*\(", output):
        preceding = output[: match.start()]
        assert "template <" in preceding.rsplit("struct", 1)[-1]


@pytest.mark.parametrize("elem", ["long", "double"])
def test_sfinae_simd_reduce_selects_eq8_overload(elem):
    # `T` of size 8 selects the sizeof==8 overload, whose body combines elements
    # with `operator()` (lowered to the concrete call operator) and never calls
    # `simd_reduce_impl`.
    output = MetalPreprocessor().preprocess(_sfinae_reduce_source(elem))
    reduce_fn = f"{elem} Red_{elem}__simd_reduce__{elem}(Red_{elem} self, {elem} val)"
    assert reduce_fn in output
    assert f"Red_{elem}__simd_reduce__{elem}(op, total)" in output
    # The ==8 body's `operator()(...)` is lowered to the concrete free function.
    assert f"Red_{elem}__operator_call(self," in output
    # The ==8 overload does not call simd_reduce_impl, so the lowered free
    # function body contains no impl call and NO lowered impl is emitted.
    reduce_body = output.split(reduce_fn, 1)[1].split("}", 1)[0]
    assert "simd_reduce_impl" not in reduce_body
    assert f"Red_{elem}__simd_reduce_impl" not in output
    assert "op.simd_reduce(" not in output


def test_sfinae_simd_reduce_unrecognized_type_clean_fails():
    # A call whose concrete `T` is a non-scalar type (`complex64_t`) has no
    # recognized sizeof, so no overload's constraint can be evaluated: the method
    # clean-fails with MetalStructMethodError rather than guessing an overload.
    code = (
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct complex64_t { float real; float imag; };\n"
        + _SFINAE_SIMD_REDUCE_STRUCT
        + "kernel void reduce(device const complex64_t* in [[buffer(0)]],\n"
        "  device complex64_t* out [[buffer(1)]],\n"
        "  uint i [[thread_position_in_grid]]) {\n"
        "    Red<complex64_t> op;\n"
        "    complex64_t total = in[i];\n"
        "    out[i] = op.simd_reduce(total);\n"
        "}\n"
    )
    with pytest.raises(MetalStructMethodError) as excinfo:
        MetalPreprocessor().preprocess(code)
    assert excinfo.value.method_name == "simd_reduce"
    assert (
        excinfo.value.project_diagnostic_code == "project.translate.metal-struct-method"
    )
    call_text = "simd_reduce(total)"
    call_offset = code.index(call_text)
    call_line_start = code.rfind("\n", 0, call_offset)
    location = excinfo.value.source_location
    assert location["line"] == code.count("\n", 0, call_offset) + 1
    assert location["column"] == call_offset - call_line_start
    assert location["length"] == len(call_text)
    assert location["endLine"] == location["line"]
    assert location["endColumn"] == location["column"] + len(call_text)


def test_sfinae_simd_reduce_full_pipeline_to_hlsl():
    # End-to-end: the sizeof<8 + is_integral chain for T=float translates to HLSL
    # with `simd_sum` -> `WaveActiveSum` and no dangling simd_reduce calls.
    from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter
    from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
    from crosstl.translator.lexer import Lexer as CrossGLLexer
    from crosstl.translator.parser import Parser as CrossGLParser

    pre = MetalPreprocessor().preprocess(_sfinae_reduce_source("float"))
    assert "Red_float__simd_reduce__float(Red_float self, float val)" in pre
    tokens = MetalLexer(pre).tokenize()
    ast = MetalParser(tokens).parse()
    crossgl = MetalToCrossGLConverter().generate(ast)
    parsed = CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse()
    hlsl = HLSLCodeGen().generate(parsed)
    assert "op.simd_reduce(" not in hlsl
    assert re.search(r"(?<![\w_])simd_reduce\s*\(", hlsl) is None
    assert "WaveActiveSum" in hlsl
