import pytest

from crosstl.backend.DirectX.preprocessor import HLSLPreprocessor


def test_object_like_macro_expansion():
    code = """
    #define PI 3.14
    float value = PI;
    """
    output = HLSLPreprocessor().preprocess(code)
    assert "PI" not in output
    assert "3.14" in output


def test_self_referential_object_macro_member_access_does_not_recurse():
    code = """
    #define PERSISTENCE global.PERSISTENCE / 100
    float value = PERSISTENCE;
    """
    output = HLSLPreprocessor().preprocess(code)

    assert "float value = global.PERSISTENCE / 100;" in output


def test_macro_definition_strips_trailing_comment_before_expansion():
    code = """
    #define FLT_MIN 1.175494351e-38F // min positive value
    float value = max(range, FLT_MIN);
    """
    output = HLSLPreprocessor().preprocess(code)
    assert "float value = max(range, 1.175494351e-38F);" in output


def test_function_like_macro_expansion():
    code = """
    #define MUL(a, b) ((a) * (b))
    float value = MUL(2, 3);
    """
    output = HLSLPreprocessor().preprocess(code)
    assert "MUL" not in output
    assert "2" in output and "3" in output


def test_multiline_function_like_macro_arguments_from_raytracing_fallback():
    code = """
    #define SIZE 116
    #define GetBVHMetadataAddress(ptr, offset, leafIndex) offset + leafIndex * SIZE
    #define GetBVHMetadataFromLeafIndex(ptr, offset, leafIndex) LoadBVHMetadata(ptr.buffer, GetBVHMetadataAddress(ptr, offset, leafIndex))

    BVHMetadata metadata = GetBVHMetadataFromLeafIndex(
        topLevelAccelerationStructure, offsetToInstanceLevelDesc, leafIndex);
    """

    output = HLSLPreprocessor().preprocess(code)

    assert (
        "LoadBVHMetadata(topLevelAccelerationStructure.buffer, "
        "offsetToInstanceLevelDesc + leafIndex * 116)"
    ) in output


def test_self_referential_function_macro_does_not_reexpand_recursively():
    code = """
    #define GetBVHSize(ID) case ID: size = GetBVHSize(BVH##ID); break
    switch (id) {
        GetBVHSize(1);
    }
    """

    output = HLSLPreprocessor().preprocess(code)

    assert "case 1: size = GetBVHSize/**/(BVH1); break;" in output


def test_variadic_stringize_and_paste():
    code = """
    #define LOG(fmt, ...) printf(fmt, __VA_ARGS__)
    #define STR(x) #x
    #define CAT(a, b) a##b
    int CAT(foo, bar) = 0;
    const char* s = STR(hello world);
    LOG("%d", 42);
    """
    output = HLSLPreprocessor().preprocess(code)
    assert "foobar" in output
    assert '"hello world"' in output
    assert "printf" in output and "42" in output


def test_include_with_search_path(tmp_path):
    header = tmp_path / "include.hlsl"
    header.write_text("float4 includedValue;\n", encoding="utf-8")
    code = '#include "include.hlsl"\nfloat4 main() { return includedValue; }'
    output = HLSLPreprocessor(include_paths=[str(tmp_path)]).preprocess(
        code, file_path=str(tmp_path / "main.hlsl")
    )
    assert "includedValue" in output


def test_include_honors_pragma_once(tmp_path):
    header = tmp_path / "guarded.hlsl"
    header.write_text("#pragma once\nint guardedValue;\n", encoding="utf-8")
    code = '#include "guarded.hlsl"\n#include "guarded.hlsl"\n'

    output = HLSLPreprocessor(include_paths=[str(tmp_path)]).preprocess(
        code,
        file_path=str(tmp_path / "main.hlsl"),
    )

    assert output.count("guardedValue") == 1
    assert "#pragma once" not in output


def test_include_decodes_non_utf8_bytes_with_replacement(tmp_path):
    header = tmp_path / "legacy_comment.hlsl"
    header.write_bytes(b"#define VALUE 7 // legacy dash: \x96\n")
    code = '#include "legacy_comment.hlsl"\nint value = VALUE;\n'

    output = HLSLPreprocessor(include_paths=[str(tmp_path)]).preprocess(
        code, file_path=str(tmp_path / "main.hlsl")
    )

    assert "int value = 7;" in output


def test_nested_include_resolves_relative_to_included_file(tmp_path):
    shader_dir = tmp_path / "shaders"
    include_dir = shader_dir / "include"
    include_dir.mkdir(parents=True)
    (include_dir / "constants.hlsl").write_text(
        "#define BASE_COLOR float4(1.0, 0.0, 0.0, 1.0)\n",
        encoding="utf-8",
    )
    (include_dir / "material.hlsl").write_text(
        '#include "constants.hlsl"\nfloat4 materialColor = BASE_COLOR;\n',
        encoding="utf-8",
    )
    main_path = shader_dir / "main.hlsl"
    main_path.write_text(
        '#include "include/material.hlsl"\nfloat4 main() { return materialColor; }\n',
        encoding="utf-8",
    )

    output = HLSLPreprocessor().preprocess(
        main_path.read_text(encoding="utf-8"),
        file_path=str(main_path),
    )

    assert "materialColor" in output
    assert "BASE_COLOR" not in output
    assert "float4(1.0, 0.0, 0.0, 1.0)" in output


def test_include_handler_accepts_legacy_text_result():
    class LegacyIncludePreprocessor(HLSLPreprocessor):
        def _handle_include(self, rest, file_path):
            return "#define INCLUDED_VALUE 7\n"

    code = '#include "generated.hlsl"\nint value = INCLUDED_VALUE;\n'
    output = LegacyIncludePreprocessor().preprocess(code)

    assert "INCLUDED_VALUE" not in output
    assert "int value = 7;" in output


def test_conditional_compilation_and_undef():
    code = """
    #define ENABLED 1
    #if ENABLED
    float ok = 1.0;
    #else
    float bad = ;
    #endif
    #undef ENABLED
    #ifdef ENABLED
    float broken = ;
    #else
    float still_ok = 2.0;
    #endif
    """
    output = HLSLPreprocessor().preprocess(code)
    assert "bad" not in output
    assert "still_ok" in output


def test_ifdef_trailing_comment_uses_identifier_only_from_directx_samples():
    code = """
    #ifdef _WAVE_OP // SM 6.0
    uint WaveOr(uint mask) { return mask; }
    #endif
    """
    output = HLSLPreprocessor(defines={"_WAVE_OP": "1"}).preprocess(code)

    assert "WaveOr" in output


def test_ifndef_trailing_comment_uses_identifier_only():
    code = """
    #ifndef DISABLED /* optional fallback */
    uint Fallback() { return 1; }
    #endif
    """
    output = HLSLPreprocessor().preprocess(code)

    assert "Fallback" in output


def test_line_directive_is_consumed():
    code = """
    #line 200
    float value = 1.0;
    """
    output = HLSLPreprocessor().preprocess(code)
    assert "#line" not in output
    assert "float value" in output


def test_builtin_line_and_file_macros_respect_line_directive(tmp_path):
    source_path = tmp_path / "shader.hlsl"
    code = (
        "int firstLine = __LINE__;\n"
        '#line 200 "virtual/path.hlsl"\n'
        "int remappedLine = __LINE__;\n"
        "const char* remappedFile = __FILE__;\n"
    )

    output = HLSLPreprocessor().preprocess(code, file_path=str(source_path))

    assert "int firstLine = 1;" in output
    assert "int remappedLine = 200;" in output
    assert 'const char* remappedFile = "virtual/path.hlsl";' in output


def test_builtin_file_macro_uses_nested_include_path(tmp_path):
    include_path = tmp_path / "include.hlsl"
    include_path.write_text(
        "const char* includedFile = __FILE__;\n" "int includedLine = __LINE__;\n",
        encoding="utf-8",
    )
    main_path = tmp_path / "main.hlsl"
    main_path.write_text('#include "include.hlsl"\n', encoding="utf-8")

    output = HLSLPreprocessor().preprocess(
        main_path.read_text(encoding="utf-8"),
        file_path=str(main_path),
    )

    escaped_include_path = str(include_path).replace("\\", "\\\\").replace('"', '\\"')
    assert f'const char* includedFile = "{escaped_include_path}";' in output
    assert "int includedLine = 2;" in output


def test_error_and_warning_directives():
    error_code = "#error fail"
    with pytest.raises(SyntaxError):
        HLSLPreprocessor().preprocess(error_code)

    warning_code = "#warning warn"
    HLSLPreprocessor().preprocess(warning_code)

    with pytest.raises(SyntaxError):
        HLSLPreprocessor(strict=True).preprocess(warning_code)
