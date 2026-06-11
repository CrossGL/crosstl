import pytest

from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.backend.Metal.preprocessor import (
    MetalPreprocessor,
    MetalTemplateSpecializationError,
)


def token_values(code, **lexer_options):
    return [value for _, value in MetalLexer(code, **lexer_options).tokenize()]


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
