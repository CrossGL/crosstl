import pytest

from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.backend.Metal.preprocessor import MetalPreprocessor


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
