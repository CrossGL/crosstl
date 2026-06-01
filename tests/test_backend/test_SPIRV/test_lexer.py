from typing import List

import pytest

from crosstl.backend.SPIRV.VulkanLexer import VulkanLexer


def tokenize_code(code: str) -> List:
    lexer = VulkanLexer(code)
    return lexer.tokenize()


def test_mod_tokenization():
    code = """
        int a = 10 % 3;  // Basic modulus
    """
    tokens = tokenize_code(code)

    # Find the modulus operator in tokens
    has_mod = False
    for token in tokens:
        if token == ("MOD", "%"):
            has_mod = True
            break

    assert has_mod, "Modulus operator (%) not tokenized correctly"


def test_bitwise_not_tokenization():
    code = """
        int a = ~5;  // Bitwise NOT
    """
    tokens = tokenize_code(code)
    has_not = False
    for token in tokens:
        if token == ("BITWISE_NOT", "~"):
            has_not = True
            break
    assert has_not, "Bitwise NOT operator (~) not tokenized correctly"


def test_hex_integer_literal_tokenization():
    tokens = tokenize_code("""
        uint mask = 0xFFu;
        uint upper = 0X10U;
    """)

    assert ("NUMBER", "0xFFu") in tokens
    assert ("NUMBER", "0X10U") in tokens
    assert ("IDENTIFIER", "xFFu") not in tokens
    assert ("IDENTIFIER", "X10U") not in tokens


def test_array_bracket_tokenization():
    code = """
        mat4 transforms[64];
    """
    tokens = tokenize_code(code)

    assert ("LBRACKET", "[") in tokens
    assert ("RBRACKET", "]") in tokens


def test_array_postfix_update_tokenization():
    tokens = tokenize_code("items[i]++; values[index]--;")

    assert ("POST_INCREMENT", "++") in tokens
    assert ("POST_DECREMENT", "--") in tokens
    assert ("PLUS", "+") not in tokens
    assert ("MINUS", "-") not in tokens


def test_preprocessor_version_directive_tokenization():
    tokens = tokenize_code("""
        #version 450
        void main() {}
        """)

    assert tokens[:4] == [
        ("VOID", "void"),
        ("IDENTIFIER", "main"),
        ("LPAREN", "("),
        ("RPAREN", ")"),
    ]
    assert all(token_type != "PREPROCESSOR" for token_type, _ in tokens)


def test_preprocessor_extension_directive_tokenization():
    tokens = tokenize_code("""
        #extension GL_EXT_nonuniform_qualifier : enable
        layout(set = 0, binding = 0) uniform sampler2D albedoTex;
        """)

    assert tokens[:2] == [("LAYOUT", "layout"), ("LPAREN", "(")]
    assert ("IDENTIFIER", "GL_EXT_nonuniform_qualifier") not in tokens
    assert all(token_type != "PREPROCESSOR" for token_type, _ in tokens)


def test_one_dimensional_sampler_tokenization():
    tokens = tokenize_code("""
        uniform sampler1D ramp;
        uniform sampler1DArray ramps;
        """)

    assert ("SAMPLER1D", "sampler1D") in tokens
    assert ("SAMPLER1DARRAY", "sampler1DArray") in tokens


def test_integer_one_dimensional_sampler_tokenization():
    tokens = tokenize_code("""
        uniform isampler1D signedRamp;
        uniform usampler1D unsignedRamp;
        uniform isampler1DArray signedRamps;
        uniform usampler1DArray unsignedRamps;
        """)

    assert ("ISAMPLER1D", "isampler1D") in tokens
    assert ("USAMPLER1D", "usampler1D") in tokens
    assert ("ISAMPLER1DARRAY", "isampler1DArray") in tokens
    assert ("USAMPLER1DARRAY", "usampler1DArray") in tokens


def test_integer_2d_sampler_tokenization():
    tokens = tokenize_code("""
        uniform isampler2D signedTex;
        uniform usampler2D unsignedTex;
        """)

    assert ("ISAMPLER2D", "isampler2D") in tokens
    assert ("USAMPLER2D", "usampler2D") in tokens


def test_integer_2d_array_sampler_tokenization():
    tokens = tokenize_code("""
        uniform isampler2DArray signedLayers;
        uniform usampler2DArray unsignedLayers;
        """)

    assert ("ISAMPLER2DARRAY", "isampler2DArray") in tokens
    assert ("USAMPLER2DARRAY", "usampler2DArray") in tokens


def test_integer_multisample_sampler_tokenization():
    tokens = tokenize_code("""
        uniform isampler2DMS signedSamples;
        uniform usampler2DMS unsignedSamples;
        uniform isampler2DMSArray signedSampleLayers;
        uniform usampler2DMSArray unsignedSampleLayers;
        """)

    assert ("ISAMPLER2DMS", "isampler2DMS") in tokens
    assert ("USAMPLER2DMS", "usampler2DMS") in tokens
    assert ("ISAMPLER2DMSARRAY", "isampler2DMSArray") in tokens
    assert ("USAMPLER2DMSARRAY", "usampler2DMSArray") in tokens


def test_integer_3d_sampler_tokenization():
    tokens = tokenize_code("""
        uniform isampler3D signedVolume;
        uniform usampler3D unsignedVolume;
        """)

    assert ("ISAMPLER3D", "isampler3D") in tokens
    assert ("USAMPLER3D", "usampler3D") in tokens


def test_integer_cube_sampler_tokenization():
    tokens = tokenize_code("""
        uniform isamplerCube signedCube;
        uniform usamplerCube unsignedCube;
        """)

    assert ("ISAMPLERCUBE", "isamplerCube") in tokens
    assert ("USAMPLERCUBE", "usamplerCube") in tokens


def test_integer_cube_array_sampler_tokenization():
    tokens = tokenize_code("""
        uniform isamplerCubeArray signedCubeLayers;
        uniform usamplerCubeArray unsignedCubeLayers;
        """)

    assert ("ISAMPLERCUBEARRAY", "isamplerCubeArray") in tokens
    assert ("USAMPLERCUBEARRAY", "usamplerCubeArray") in tokens


def test_integer_sampler_buffer_tokenization():
    tokens = tokenize_code("""
        uniform isamplerBuffer signedData;
        uniform usamplerBuffer unsignedData;
        """)

    assert ("ISAMPLERBUFFER", "isamplerBuffer") in tokens
    assert ("USAMPLERBUFFER", "usamplerBuffer") in tokens


def test_typed_image_buffer_tokenization():
    tokens = tokenize_code("""
        uniform iimageBuffer signedData;
        uniform uimageBuffer unsignedData;
        """)

    assert ("IIMAGEBUFFER", "iimageBuffer") in tokens
    assert ("UIMAGEBUFFER", "uimageBuffer") in tokens


def test_shadow_sampler_tokenization():
    tokens = tokenize_code("""
        uniform sampler1DShadow shadowRamp;
        uniform sampler1DArrayShadow shadowRamps;
        uniform sampler2DShadow shadowMap;
        uniform sampler2DArrayShadow shadowArray;
        uniform samplerCubeShadow shadowCube;
        uniform samplerCubeArrayShadow shadowCubeArray;
        """)

    assert ("SAMPLER1DSHADOW", "sampler1DShadow") in tokens
    assert ("SAMPLER1DARRAYSHADOW", "sampler1DArrayShadow") in tokens
    assert ("SAMPLER2DSHADOW", "sampler2DShadow") in tokens
    assert ("SAMPLER2DARRAYSHADOW", "sampler2DArrayShadow") in tokens
    assert ("SAMPLERCUBESHADOW", "samplerCubeShadow") in tokens
    assert ("SAMPLERCUBEARRAYSHADOW", "samplerCubeArrayShadow") in tokens


def test_subpass_input_tokenization():
    tokens = tokenize_code("""
        uniform subpassInput colorInput;
        uniform subpassInputMS msColorInput;
        """)

    assert ("SUBPASSINPUT", "subpassInput") in tokens
    assert ("SUBPASSINPUTMS", "subpassInputMS") in tokens


def test_atomic_uint_tokenization():
    tokens = tokenize_code("""
        uniform atomic_uint counter;
        """)

    assert ("ATOMICUINT", "atomic_uint") in tokens


def test_standalone_sampler_tokenization():
    tokens = tokenize_code("""
        uniform sampler compareSampler;
        uniform sampler samplers[4];
        """)

    assert ("SAMPLER", "sampler") in tokens


def test_typed_2d_image_tokenization():
    tokens = tokenize_code("""
        uniform iimage2D signedImage;
        uniform uimage2D counters;
        """)

    assert ("IIMAGE2D", "iimage2D") in tokens
    assert ("UIMAGE2D", "uimage2D") in tokens


def test_typed_1d_image_tokenization():
    tokens = tokenize_code("""
        uniform iimage1D signedLine;
        uniform uimage1D counters;
        uniform iimage1DArray signedLayers;
        uniform uimage1DArray layerCounters;
        """)

    assert ("IIMAGE1D", "iimage1D") in tokens
    assert ("UIMAGE1D", "uimage1D") in tokens
    assert ("IIMAGE1DARRAY", "iimage1DArray") in tokens
    assert ("UIMAGE1DARRAY", "uimage1DArray") in tokens


def test_typed_3d_image_tokenization():
    tokens = tokenize_code("""
        uniform iimage3D signedVolume;
        uniform uimage3D counters;
        """)

    assert ("IIMAGE3D", "iimage3D") in tokens
    assert ("UIMAGE3D", "uimage3D") in tokens


def test_typed_cube_image_tokenization():
    tokens = tokenize_code("""
        uniform iimageCube signedCube;
        uniform uimageCube cubeCounters;
        uniform iimageCubeArray signedCubeLayers;
        uniform uimageCubeArray cubeLayerCounters;
        """)

    assert ("IIMAGECUBE", "iimageCube") in tokens
    assert ("UIMAGECUBE", "uimageCube") in tokens
    assert ("IIMAGECUBEARRAY", "iimageCubeArray") in tokens
    assert ("UIMAGECUBEARRAY", "uimageCubeArray") in tokens


def test_typed_2d_array_image_tokenization():
    tokens = tokenize_code("""
        uniform iimage2DArray signedLayers;
        uniform uimage2DArray counters;
        """)

    assert ("IIMAGE2DARRAY", "iimage2DArray") in tokens
    assert ("UIMAGE2DARRAY", "uimage2DArray") in tokens


def test_typed_2d_ms_image_tokenization():
    tokens = tokenize_code("""
        uniform iimage2DMS signedSamples;
        uniform uimage2DMS sampleCounters;
        """)

    assert ("IIMAGE2DMS", "iimage2DMS") in tokens
    assert ("UIMAGE2DMS", "uimage2DMS") in tokens


def test_typed_2d_ms_array_image_tokenization():
    tokens = tokenize_code("""
        uniform iimage2DMSArray signedLayers;
        uniform uimage2DMSArray sampleCounters;
        """)

    assert ("IIMAGE2DMSARRAY", "iimage2DMSArray") in tokens
    assert ("UIMAGE2DMSARRAY", "uimage2DMSArray") in tokens


if __name__ == "__main__":
    pytest.main()
