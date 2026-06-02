import pytest

from crosstl import translate
from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.GLSL.OpenglParser import GLSLParser


def test_preprocessor_conditional_expansion():
    code = """
    #version 450 core
    #define ENABLE_FEATURE 1
    #if ENABLE_FEATURE
    int only_enabled = 1;
    #else
    int only_disabled = 1;
    #endif
    void main() { }
    """
    tokens = GLSLLexer(code).tokenize()
    values = [tok[1] for tok in tokens]
    assert "only_enabled" in values
    assert "only_disabled" not in values


def test_preprocessor_ifdef_trailing_comment_uses_identifier_only():
    code = """
    #version 450 core
    #ifdef ENABLE_FEATURE // enabled by pipeline
    int only_enabled = 1;
    #endif
    void main() { }
    """
    tokens = GLSLLexer(code, defines={"ENABLE_FEATURE": "1"}).tokenize()
    values = [tok[1] for tok in tokens]

    assert "only_enabled" in values


def test_preprocessor_include(tmp_path):
    include_file = tmp_path / "inc.glsl"
    include_file.write_text("int from_include = 1;")

    code = """
    #version 450 core
    #include <inc.glsl>
    void main() { }
    """
    tokens = GLSLLexer(code, include_paths=[str(tmp_path)]).tokenize()
    values = [tok[1] for tok in tokens]
    assert "from_include" in values


def test_translate_resolves_quoted_include_from_source_file_dir(tmp_path):
    include_file = tmp_path / "blur_common.h"
    include_file.write_text("const float INCLUDED_SCALE = 2.0;\n")
    shader_file = tmp_path / "blur_up.comp.glsl"
    shader_file.write_text("""
        #version 450
        #include "blur_common.h"

        void main()
        {
            float value = INCLUDED_SCALE;
        }
        """)

    cgl = translate(str(shader_file), backend="cgl", format_output=False)
    directx = translate(str(shader_file), backend="directx", format_output=False)

    assert "INCLUDED_SCALE" in cgl
    assert "INCLUDED_SCALE" in directx


def test_preprocessor_resolves_angle_include_from_source_file_dir(tmp_path):
    include_file = tmp_path / "tonemapping.glsl"
    include_file.write_text("vec3 tone_map(vec3 color) { return color; }\n")
    shader_file = tmp_path / "pbr.frag"
    shader_file.write_text("""
        precision highp float;

        #include <tonemapping.glsl>

        void main()
        {
            vec3 color = tone_map(vec3(1.0));
        }
        """)

    tokens = GLSLLexer.from_file(str(shader_file)).tokenize()
    ast = GLSLParser(tokens, "fragment").parse()

    assert any(function.name == "tone_map" for function in ast.functions)


def test_preprocessor_resolves_vulkan_samples_shared_glsl_include(tmp_path):
    shader_root = tmp_path / "shaders"
    include_dir = shader_root / "includes" / "glsl"
    include_dir.mkdir(parents=True)
    (include_dir / "lighting.h").write_text("""
        struct Light
        {
            vec4 direction;
            vec4 color;
        };

        vec3 apply_directional_light(Light light, vec3 normal)
        {
            vec3 world_to_light = -light.direction.xyz;
            float ndotl = clamp(dot(normal, world_to_light), 0.0, 1.0);
            return ndotl * light.color.rgb;
        }
        """)

    shader_file = shader_root / "constant_data" / "ubo.frag"
    shader_file.parent.mkdir()
    shader_file.write_text("""
        #version 320 es
        precision highp float;

        layout(location = 0) out vec4 o_color;

        #include "lighting.h"

        void main(void)
        {
            Light light;
            vec3 color = apply_directional_light(light, vec3(0.0));
            o_color = vec4(color, 1.0);
        }
        """)

    tokens = GLSLLexer.from_file(str(shader_file)).tokenize()
    ast = GLSLParser(tokens, "fragment").parse()

    assert any(function.name == "apply_directional_light" for function in ast.functions)
    assert any(struct.name == "Light" for struct in ast.structs)


def test_preprocessor_resolves_project_common_shader_include(tmp_path):
    shader_root = tmp_path / "vk_mini_samples"
    common_shader_dir = shader_root / "common" / "common_shaders"
    common_shader_dir.mkdir(parents=True)
    (common_shader_dir / "palette.h").write_text("""
        vec3 palette(float t)
        {
            return vec3(t);
        }
        """)

    shader_file = (
        shader_root / "samples" / "compute_only" / "shaders" / "main.comp.glsl"
    )
    shader_file.parent.mkdir(parents=True)
    shader_file.write_text("""
        #version 450
        #include "common_shaders/palette.h"

        void main()
        {
            vec3 color = palette(1.0);
        }
        """)

    tokens = GLSLLexer.from_file(str(shader_file)).tokenize()
    ast = GLSLParser(tokens, "compute").parse()

    assert any(function.name == "palette" for function in ast.functions)


def test_preprocessor_uses_glsl_profile_macros_and_ignores_commented_directives(
    tmp_path,
):
    include_file = tmp_path / "shared_types.h"
    include_file.write_text("""
        #ifndef SHARED_TYPES_H
        #define SHARED_TYPES_H

        /*
        #include <missing_from_doc_comment.glsl>
        */

        #ifdef __cplusplus
        #error "C++ branch should be inactive"
        #elif defined(GL_core_profile)  // GLSL
        #define SHARED_VECTOR vec3
        #else
        #error "Unknown language environment"
        #endif

        #endif
        """)

    code = """
    #version 450
    #include "shared_types.h"

    void main()
    {
        SHARED_VECTOR color = vec3(1.0);
    }
    """

    tokens = GLSLLexer(code, include_paths=[str(tmp_path)]).tokenize()
    values = [tok[1] for tok in tokens]

    assert "missing_from_doc_comment" not in values
    assert "SHARED_VECTOR" not in values
    assert "vec3" in values


def test_preprocessor_version_must_be_first():
    code = """
    #define SOMETHING 1
    #version 450 core
    void main() { }
    """
    with pytest.raises(SyntaxError):
        GLSLLexer(code).tokenize()


def test_preprocessor_function_like_macro():
    code = """
    #version 450 core
    #define ADD(a, b) ((a) + (b))
    int value = ADD(1, 2);
    void main() { }
    """
    tokens = GLSLLexer(code).tokenize()
    values = [tok[1] for tok in tokens]
    assert "value" in values
    assert "1" in values
    assert "2" in values


def test_preprocessor_variadic_macro_and_paste():
    code = """
    #version 450 core
    #define JOIN(a, b) a##b
    #define MAKE(name, ...) int JOIN(name, _value) = __VA_ARGS__;
    MAKE(test, 7)
    void main() { }
    """
    tokens = GLSLLexer(code).tokenize()
    values = [tok[1] for tok in tokens]
    assert "test_value" in values
    assert "7" in values


def test_preprocessor_stringize():
    code = """
    #version 450 core
    #define STR(x) #x
    const char* name = STR(hello);
    void main() { }
    """
    tokens = GLSLLexer(code).tokenize()
    values = [tok[1] for tok in tokens]
    assert '"hello"' in values


def test_preprocessor_extension_directive():
    code = """
    #version 450 core
    #extension GL_ARB_separate_shader_objects : enable
    void main() { }
    """
    tokens = GLSLLexer(code).tokenize()
    values = [tok[1] for tok in tokens]
    assert "extension" in values
    assert "GL_ARB_separate_shader_objects" in values


def test_preprocessor_error_directive_raises():
    code = """
    #version 450 core
    #error stop
    void main() { }
    """
    with pytest.raises(SyntaxError):
        GLSLLexer(code).tokenize()


def test_preprocessor_line_directive():
    code = """
    #version 450 core
    #line 42
    int value = 1;
    """
    tokens = GLSLLexer(code).tokenize()
    values = [tok[1] for tok in tokens]
    assert "value" in values


if __name__ == "__main__":
    pytest.main()
