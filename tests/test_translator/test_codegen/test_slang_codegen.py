from crosstl.translator.lexer import Lexer
import pytest
from typing import List
from crosstl.translator.parser import Parser
from crosstl.translator.ast import LiteralNode, PrimitiveType
from crosstl.translator.codegen.slang_codegen import SlangCodeGen


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = Lexer(code)
    return lexer.get_tokens()


def parse_code(tokens: List):
    """Helper function to parse tokens into an AST.

    Args:
        tokens (List): The list of tokens to parse
    Returns:
        AST: The abstract syntax tree generated from the parser
    """
    parser = Parser(tokens)
    return parser.parse()


def generate_code(ast_node):
    """Test the code generator
    Args:
        ast_node: The abstract syntax tree generated from the code
    Returns:
        str: The generated code from the abstract syntax tree
    """
    codegen = SlangCodeGen()
    return codegen.generate(ast_node)


def test_struct():
    code = """
    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };
    struct VSOutput {
        vec4 color @ COLOR;
    };
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        assert code is not None
    except SyntaxError:
        pytest.fail("Slang struct codegen not implemented.")


def test_basic_shader():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = vec4(input.texCoord, 0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return input.color;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        assert code is not None
    except SyntaxError:
        pytest.fail("Slang basic shader codegen not implemented.")


def test_stage_only_shader_emits_slang_entry_point():
    code = """
    shader main {
        vertex {
            void main() {
                gl_Position = vec4(1.0, 0.0, 0.0, 1.0);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "// Vertex Shader" in generated_code
    assert '[shader("vertex")]' in generated_code
    assert "void main()" in generated_code
    assert "gl_Position = float4(1.0, 0.0, 0.0, 1.0);" in generated_code


def test_stage_local_functions_and_initialized_variables_emit():
    code = """
    shader main {
        vertex {
            float helper(float x) {
                return x;
            }

            void main() {
                float y = helper(1.0);
                gl_Position = vec4(y, y, y, 1.0);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float helper(float x)" in generated_code
    assert "return x;" in generated_code
    assert "float y = helper(1.0);" in generated_code
    assert "gl_Position = float4(y, y, y, 1.0);" in generated_code


def test_multiple_stage_entry_points_emit():
    code = """
    shader main {
        vertex {
            void main() {
            }
        }
        fragment {
            void main() {
            }
        }
        compute {
            void main() {
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert '[shader("vertex")]' in generated_code
    assert '[shader("fragment")]' in generated_code
    assert '[shader("compute")]' in generated_code


def test_if_else_control_flow_emits_slang_blocks():
    code = """
    shader main {
        vertex {
            void main() {
                float y = 1.0;
                if (y > 0.0) {
                    y = y + 1.0;
                } else {
                    y = 0.0;
                }
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "if (y > 0.0)" in generated_code
    assert "y = y + 1.0;" in generated_code
    assert "else" in generated_code
    assert "y = 0.0;" in generated_code


def test_for_loop_control_flow_emits_slang_loop():
    code = """
    shader main {
        vertex {
            void main() {
                float y = 0.0;
                for (int i = 0; i < 4; i = i + 1) {
                    y = y + 1.0;
                }
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for (int i = 0; i < 4; i = i + 1)" in generated_code
    assert "y = y + 1.0;" in generated_code


def test_while_loop_control_flow_emits_slang_loop():
    code = """
    shader main {
        vertex {
            void main() {
                int i = 0;
                while (i < 4) {
                    i = i + 1;
                }
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "while (i < 4)" in generated_code
    assert "i = i + 1;" in generated_code
    assert "WhileNode(" not in generated_code


def test_control_flow_blocks_are_indented_consistently():
    code = """
    shader main {
        vertex {
            void main() {
                float y = 0.0;
                while (y < 4.0) {
                    if (y > 1.0) {
                        y = y + 2.0;
                    } else {
                        y = y + 1.0;
                    }
                }
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "\n    while (y < 4.0)\n    {\n" in generated_code
    assert "\n        if (y > 1.0)\n        {\n" in generated_code
    assert "\n        else\n        {\n" in generated_code
    assert "\n            y = y + 2.0;\n" in generated_code


def test_switch_break_continue_and_void_return_emit_slang_syntax():
    code = """
    shader main {
        vertex {
            void main() {
                int i = 0;
                while (i < 4) {
                    switch (i) {
                        case 0:
                            i = i + 1;
                            continue;
                        default:
                            break;
                    }
                }
                return;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "switch (i)" in generated_code
    assert "case 0:" in generated_code
    assert "default:" in generated_code
    assert "continue;" in generated_code
    assert "break;" in generated_code
    assert "return;" in generated_code
    assert "SwitchNode(" not in generated_code
    assert "BreakNode(" not in generated_code
    assert "ContinueNode(" not in generated_code
    assert "return None;" not in generated_code


def test_array_access_and_ternary_expressions_emit_slang_syntax():
    code = """
    shader main {
        vertex {
            void main() {
                float values[2];
                values[0] = 1.0;
                values[1] = 2.0;
                float chosen = values[0] > 0.0 ? values[0] : values[1];
                gl_Position = vec4(chosen, chosen, chosen, 1.0);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float values[2];" in generated_code
    assert "values[0] = 1.0;" in generated_code
    assert "values[1] = 2.0;" in generated_code
    assert "float chosen = (values[0] > 0.0 ? values[0] : values[1]);" in generated_code
    assert "ArrayType(" not in generated_code
    assert "ArrayAccessNode(" not in generated_code
    assert "TernaryOpNode(" not in generated_code


def test_matrix_and_non_float_vector_types_emit_slang_names():
    code = """
    shader main {
        struct Transform {
            mat4 model;
            mat3 normal;
            ivec2 tile;
            uvec4 mask;
            bvec3 flags;
        };

        vertex {
            vec4 transformPosition(mat4 m, vec4 p) {
                return m * p;
            }

            void main() {
                mat4 model = mat4(1.0);
                mat3 normal;
                mat2x3 nonSquare;
                ivec2 tile;
                uvec4 mask;
                bvec3 flags;
                bvec2 mask = bvec2(true, false);
                dvec2 preciseUV = dvec2(1.0, 2.0);
                dmat2 precise = dmat2(1.0, 0.0, 0.0, 1.0);
                dmat4x3 jacobian;
                vec4 p = transformPosition(model, vec4(1.0, 2.0, 3.0, 1.0));
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float4x4 model;" in generated_code
    assert "float3x3 normal;" in generated_code
    assert "int2 tile;" in generated_code
    assert "uint4 mask;" in generated_code
    assert "bool3 flags;" in generated_code
    assert "bool2 mask = bool2(true, false);" in generated_code
    assert "float4 transformPosition(float4x4 m, float4 p)" in generated_code
    assert "float4x4 model = float4x4(1.0);" in generated_code
    assert "float2x3 nonSquare;" in generated_code
    assert "double2 preciseUV = double2(1.0, 2.0);" in generated_code
    assert "double2x2 precise = double2x2(1.0, 0.0, 0.0, 1.0);" in generated_code
    assert "double4x3 jacobian;" in generated_code
    assert "dvec2(" not in generated_code
    assert "bvec2(" not in generated_code
    assert "dmat2(" not in generated_code
    assert "MatrixType(" not in generated_code


def test_generic_vector_constructors_emit_slang_names():
    code = """
    shader main {
        compute {
            void main() {
                vec2<f64> precise = vec2<f64>(1.0, 2.0);
                vec3<i32> index = vec3<i32>(1, 2, 3);
                vec4<u32> mask = vec4<u32>(1, 2, 3, 4);
                vec2<bool> flags = vec2<bool>(true, false);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "double2 precise = double2(1.0, 2.0);" in generated_code
    assert "int3 index = int3(1, 2, 3);" in generated_code
    assert "uint4 mask = uint4(1, 2, 3, 4);" in generated_code
    assert "bool2 flags = bool2(true, false);" in generated_code
    assert "vec2<" not in generated_code
    assert "vec3<" not in generated_code
    assert "vec4<" not in generated_code


def test_resource_types_emit_slang_texture_names():
    code = """
    shader Resources {
        sampler2d colorMap;
        sampler2dms msTex;
        sampler2dmsarray msArray;
        image2D colorImage;
        image2DMS msColor;
        image2DMSArray msLayers;
        iimage2DMSArray signedLayers;
        uimage2DMS counters;

        compute {
            void main() {}
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "Sampler2D<float4> colorMap;" in generated_code
    assert "Sampler2DMS<float4> msTex;" in generated_code
    assert "Sampler2DMSArray<float4> msArray;" in generated_code
    assert "RWTexture2D<float4> colorImage;" in generated_code
    assert "RWTexture2DMS<float4> msColor;" in generated_code
    assert "RWTexture2DMSArray<float4> msLayers;" in generated_code
    assert "RWTexture2DMSArray<int> signedLayers;" in generated_code
    assert "RWTexture2DMS<uint> counters;" in generated_code
    assert "sampler2d" not in generated_code
    assert "image2DMS" not in generated_code
    assert "iimage2DMSArray" not in generated_code
    assert "uimage2DMS" not in generated_code


def test_sampled_texture_builtins_emit_slang_methods():
    code = """
    shader Resources {
        sampler2d colorMap;
        sampler2darray layerMap;
        sampler2dms msTex;
        sampler2dmsarray msArray;

        compute {
            vec4 sampleTexture(
                sampler2d tex,
                sampler2darray layers,
                sampler2dms ms,
                sampler2dmsarray msLayers,
                vec2 uv,
                vec3 uvLayer,
                ivec2 pixel,
                ivec3 pixelLayer,
                int sampleIndex
            ) {
                vec4 color = texture(tex, uv);
                vec4 biased = texture(tex, uv, 0.5);
                vec4 mip = textureLod(tex, uv, 2.0);
                vec4 grad = textureGrad(
                    tex,
                    uv,
                    vec2(0.1, 0.0),
                    vec2(0.0, 0.1)
                );
                vec4 fetched = texelFetch(tex, pixel, 1);
                vec4 fetchedLayer = texelFetch(layers, pixelLayer, 1);
                vec4 msColor = texelFetch(ms, pixel, sampleIndex);
                vec4 msLayer = texelFetch(msLayers, pixelLayer, sampleIndex);
                return color + biased + mip + grad + fetched
                    + fetchedLayer + msColor + msLayer;
            }

            void main() {}
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "Sampler2D<float4> colorMap;" in generated_code
    assert "Sampler2DArray<float4> layerMap;" in generated_code
    assert "Sampler2DMS<float4> msTex;" in generated_code
    assert "Sampler2DMSArray<float4> msArray;" in generated_code
    assert "Sampler2D<float4> tex" in generated_code
    assert "Sampler2DArray<float4> layers" in generated_code
    assert "Sampler2DMS<float4> ms" in generated_code
    assert "Sampler2DMSArray<float4> msLayers" in generated_code
    assert "float4 color = tex.Sample(uv);" in generated_code
    assert "float4 biased = tex.SampleBias(uv, 0.5);" in generated_code
    assert "float4 mip = tex.SampleLevel(uv, 2.0);" in generated_code
    assert (
        "float4 grad = tex.SampleGrad(uv, float2(0.1, 0.0), float2(0.0, 0.1));"
        in generated_code
    )
    assert "float4 fetched = tex.Load(int3(pixel, 1));" in generated_code
    assert "float4 fetchedLayer = layers.Load(int4(pixelLayer, 1));" in generated_code
    assert "float4 msColor = ms[pixel, sampleIndex];" in generated_code
    assert "float4 msLayer = msLayers[pixelLayer, sampleIndex];" in generated_code
    assert "texture(" not in generated_code
    assert "textureLod(" not in generated_code
    assert "textureGrad(" not in generated_code
    assert "texelFetch(" not in generated_code


def test_resource_query_builtins_emit_slang_get_dimensions_helpers():
    code = """
    shader Resources {
        sampler2d colorMap;
        sampler2darray layers;
        sampler3d volumeTex;
        sampler2dms msTex;
        sampler2dmsarray msLayers;
        image2D colorImage;
        image3D volumeImage;
        image2DArray layerImage;
        image2DMS msImage;
        image2DMSArray msImageLayers;
        uimage2DMS counters;

        compute {
            void main() {
                ivec2 texSize = textureSize(colorMap, 2);
                ivec3 layerSize = textureSize(layers, 1);
                ivec3 volumeSize = textureSize(volumeTex, 0);
                ivec2 msSize = textureSize(msTex, 0);
                ivec3 msLayerSize = textureSize(msLayers, 0);
                ivec2 imageSize2d = imageSize(colorImage);
                ivec3 imageSize3d = imageSize(volumeImage);
                ivec3 imageSizeLayer = imageSize(layerImage);
                ivec2 msImageSize = imageSize(msImage);
                ivec3 msImageLayerSize = imageSize(msImageLayers);
                int texSamples = textureSamples(msTex);
                int texLayerSamples = textureSamples(msLayers);
                int imageSamplesValue = imageSamples(msImage);
                int counterSamples = imageSamples(counters);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "int2 cgl_textureSize_sampler2D(Sampler2D<float4> tex, uint mipLevel)"
        in generated_code
    )
    assert "tex.GetDimensions(mipLevel, width, height, levels);" in generated_code
    assert (
        "int3 cgl_textureSize_sampler2DArray("
        "Sampler2DArray<float4> tex, uint mipLevel)" in generated_code
    )
    assert (
        "tex.GetDimensions(mipLevel, width, height, elements, levels);"
        in generated_code
    )
    assert (
        "int3 cgl_textureSize_sampler3D(Sampler3D<float4> tex, uint mipLevel)"
        in generated_code
    )
    assert (
        "tex.GetDimensions(mipLevel, width, height, depth, levels);" in generated_code
    )
    assert "int2 cgl_textureSize_sampler2DMS(Sampler2DMS<float4> tex)" in generated_code
    assert (
        "int3 cgl_textureSize_sampler2DMSArray(Sampler2DMSArray<float4> tex)"
        in generated_code
    )
    assert "tex.GetDimensions(width, height, samples);" in generated_code
    assert "tex.GetDimensions(width, height, elements, samples);" in generated_code
    assert "int2 cgl_imageSize_image2D(RWTexture2D<float4> tex)" in generated_code
    assert "int3 cgl_imageSize_image3D(RWTexture3D<float4> tex)" in generated_code
    assert (
        "int3 cgl_imageSize_image2DArray(RWTexture2DArray<float4> tex)"
        in generated_code
    )
    assert "int2 cgl_imageSize_image2DMS(RWTexture2DMS<float4> tex)" in generated_code
    assert (
        "int3 cgl_imageSize_image2DMSArray(RWTexture2DMSArray<float4> tex)"
        in generated_code
    )
    assert (
        "int cgl_textureSamples_sampler2DMS(Sampler2DMS<float4> tex)" in generated_code
    )
    assert (
        "int cgl_textureSamples_sampler2DMSArray(Sampler2DMSArray<float4> tex)"
        in generated_code
    )
    assert "int cgl_imageSamples_image2DMS(RWTexture2DMS<float4> tex)" in generated_code
    assert "int cgl_imageSamples_uimage2DMS(RWTexture2DMS<uint> tex)" in generated_code
    assert "return samples;" in generated_code
    assert "int2 texSize = cgl_textureSize_sampler2D(colorMap, 2);" in generated_code
    assert (
        "int3 layerSize = cgl_textureSize_sampler2DArray(layers, 1);" in generated_code
    )
    assert (
        "int3 volumeSize = cgl_textureSize_sampler3D(volumeTex, 0);" in generated_code
    )
    assert "int2 msSize = cgl_textureSize_sampler2DMS(msTex);" in generated_code
    assert (
        "int3 msLayerSize = cgl_textureSize_sampler2DMSArray(msLayers);"
        in generated_code
    )
    assert "int2 imageSize2d = cgl_imageSize_image2D(colorImage);" in generated_code
    assert "int3 imageSize3d = cgl_imageSize_image3D(volumeImage);" in generated_code
    assert (
        "int3 imageSizeLayer = cgl_imageSize_image2DArray(layerImage);"
        in generated_code
    )
    assert "int2 msImageSize = cgl_imageSize_image2DMS(msImage);" in generated_code
    assert (
        "int3 msImageLayerSize = cgl_imageSize_image2DMSArray(msImageLayers);"
        in generated_code
    )
    assert "int texSamples = cgl_textureSamples_sampler2DMS(msTex);" in generated_code
    assert (
        "int texLayerSamples = cgl_textureSamples_sampler2DMSArray(msLayers);"
        in generated_code
    )
    assert (
        "int imageSamplesValue = cgl_imageSamples_image2DMS(msImage);" in generated_code
    )
    assert (
        "int counterSamples = cgl_imageSamples_uimage2DMS(counters);" in generated_code
    )
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "textureSamples(" not in generated_code
    assert "imageSamples(" not in generated_code


def test_storage_image_load_store_emit_slang_subscript_access():
    code = """
    shader Resources {
        image2D colorImage;
        image2DMS msColor;
        image2DMSArray msLayers;
        uimage2DMS counters;
        iimage2DMSArray signedLayers;

        compute {
            void main() {
                int sampleIndex = 2;
                ivec2 pixel = ivec2(4, 8);
                ivec3 pixelLayer = ivec3(4, 8, 1);
                vec4 regular = imageLoad(colorImage, pixel);
                imageStore(colorImage, pixel, regular);
                vec4 color = imageLoad(msColor, pixel, sampleIndex);
                imageStore(msColor, pixel, sampleIndex, color);
                vec4 layer = imageLoad(msLayers, pixelLayer, sampleIndex);
                imageStore(msLayers, pixelLayer, sampleIndex, layer);
                uint count = imageLoad(counters, pixel, sampleIndex);
                imageStore(counters, pixel, sampleIndex, count);
                int signedValue = imageLoad(signedLayers, pixelLayer, sampleIndex);
                imageStore(signedLayers, pixelLayer, sampleIndex, signedValue);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float4 regular = colorImage[pixel];" in generated_code
    assert "colorImage[pixel] = regular;" in generated_code
    assert "float4 color = msColor[pixel, sampleIndex];" in generated_code
    assert "msColor[pixel, sampleIndex] = color;" in generated_code
    assert "float4 layer = msLayers[pixelLayer, sampleIndex];" in generated_code
    assert "msLayers[pixelLayer, sampleIndex] = layer;" in generated_code
    assert "uint count = counters[pixel, sampleIndex];" in generated_code
    assert "counters[pixel, sampleIndex] = count;" in generated_code
    assert "int signedValue = signedLayers[pixelLayer, sampleIndex];" in generated_code
    assert "signedLayers[pixelLayer, sampleIndex] = signedValue;" in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_bool_string_and_char_literals_emit_slang_syntax():
    code = """
    shader main {
        vertex {
            void main() {
                bool enabled = true;
                bool disabled = false;
                string label = "debug";
                char marker = 'x';
                if (enabled && !disabled) {
                    label = "active";
                    marker = 'y';
                }
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    entry_point = next(iter(ast.stages.values())).entry_point

    assert entry_point.body.statements[0].initial_value.value is True
    assert entry_point.body.statements[1].initial_value.value is False

    generated_code = generate_code(ast)

    assert "bool enabled = true;" in generated_code
    assert "bool disabled = false;" in generated_code
    assert 'string label = "debug";' in generated_code
    assert "char marker = 'x';" in generated_code
    assert 'label = "active";' in generated_code
    assert "marker = 'y';" in generated_code
    assert "True" not in generated_code
    assert "False" not in generated_code


def test_direct_literal_nodes_emit_slang_escaping():
    codegen = SlangCodeGen()

    assert (
        codegen.generate_expression(LiteralNode(True, PrimitiveType("bool"))) == "true"
    )
    assert (
        codegen.generate_expression(LiteralNode('debug"name', PrimitiveType("string")))
        == '"debug\\"name"'
    )
    assert (
        codegen.generate_expression(LiteralNode("'", PrimitiveType("char"))) == "'\\''"
    )


def test_prefix_and_postfix_unary_operators_preserve_position():
    code = """
    shader main {
        vertex {
            void main() {
                int i = 0;
                i++;
                ++i;
                i--;
                --i;
                for (int j = 0; j < 2; j++) {
                    i++;
                }
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    entry_point = next(iter(ast.stages.values())).entry_point
    first_postfix = entry_point.body.statements[1].expression
    first_prefix = entry_point.body.statements[2].expression
    for_loop = entry_point.body.statements[5]

    assert first_postfix.is_postfix is True
    assert first_prefix.is_postfix is False
    assert for_loop.update.is_postfix is True

    generated_code = generate_code(ast)

    assert "\n    i++;\n" in generated_code
    assert "\n    ++i;\n" in generated_code
    assert "\n    i--;\n" in generated_code
    assert "\n    --i;\n" in generated_code
    assert "for (int j = 0; j < 2; j++)" in generated_code
    assert "++j" not in generated_code


if __name__ == "__main__":
    pytest.main()
