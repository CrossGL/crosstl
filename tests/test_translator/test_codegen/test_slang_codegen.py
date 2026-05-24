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


def test_mix_builtin_lowers_to_lerp_without_affecting_resources_or_constructors():
    code = """
    shader MixBug {
        sampler2d tex;

        compute {
            void main() {
                vec2 uv = vec2(0.5, 0.5);
                float x = mix(0.0, 1.0, 0.25);
                vec4 c = vec4(x, x, x, 1.0);
                vec4 sampled = texture(tex, uv);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float x = lerp(0.0, 1.0, 0.25);" in generated_code
    assert "float4 c = float4(x, x, x, 1.0);" in generated_code
    assert "float4 sampled = tex.Sample(uv);" in generated_code
    assert "mix(" not in generated_code
    assert "texture(" not in generated_code


def test_user_defined_texture_function_shadows_resource_lowering():
    code = """
    shader TextureShadow {
        sampler2d tex;

        fragment {
            vec4 texture(sampler2d src, vec2 uv) {
                return vec4(uv, 0.0, 1.0);
            }

            void main() {
                vec2 uv = vec2(0.25, 0.5);
                vec4 color = texture(tex, uv);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float4 texture(Sampler2D<float4> src, float2 uv)" in generated_code
    assert "float4 color = texture(tex, uv);" in generated_code
    assert "float4 color = tex.Sample(uv);" not in generated_code


def test_mod_builtin_lowers_to_slang_fmod():
    code = """
    shader BuiltinGap {
        compute {
            void main() {
                float wrapped = mod(5.0, 2.0);
                vec2 v = vec2(5.0, 7.0);
                vec2 wrappedVec = mod(v, vec2(2.0));
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float wrapped = fmod(5.0, 2.0);" in generated_code
    assert "float2 wrappedVec = fmod(v, float2(2.0));" in generated_code
    assert "float2 v = float2(5.0, 7.0);" in generated_code
    assert "wrapped = mod(" not in generated_code
    assert "wrappedVec = mod(" not in generated_code


def test_fract_builtin_lowers_to_slang_frac():
    code = """
    shader BuiltinGap {
        compute {
            void main() {
                float wrapped = fract(1.25);
                vec2 v = fract(vec2(1.25, 2.5));
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float wrapped = frac(1.25);" in generated_code
    assert "float2 v = frac(float2(1.25, 2.5));" in generated_code
    assert "fract(" not in generated_code


def test_inversesqrt_builtin_lowers_to_slang_rsqrt():
    code = """
    shader BuiltinGap {
        compute {
            void main() {
                float inv = inversesqrt(x);
                vec2 invVec = inversesqrt(vec2(4.0, 9.0));
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float inv = rsqrt(x);" in generated_code
    assert "float2 invVec = rsqrt(float2(4.0, 9.0));" in generated_code
    assert "inversesqrt(" not in generated_code


def test_saturate_builtin_lowers_to_slang_clamp():
    code = """
    shader BuiltinGap {
        compute {
            void main() {
                float saturated = saturate(1.25);
                vec2 saturatedVec = saturate(vec2(-1.0, 2.0));
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float saturated = clamp(1.25, 0.0, 1.0);" in generated_code
    assert "float2 saturatedVec = clamp(float2(-1.0, 2.0), 0.0, 1.0);" in generated_code
    assert "saturate(" not in generated_code


def test_user_defined_saturate_function_is_not_lowered_to_clamp():
    code = """
    shader BuiltinGap {
        compute {
            float saturate(float x) {
                return x + 1.0;
            }

            void main() {
                float adjusted = saturate(0.5);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float saturate(float x)" in generated_code
    assert "float adjusted = saturate(0.5);" in generated_code
    assert "float adjusted = clamp(0.5, 0.0, 1.0);" not in generated_code


def test_user_defined_mix_function_is_not_lowered_to_lerp():
    code = """
    shader BuiltinGap {
        compute {
            float mix(float x, float y, float t) {
                return x + y + t;
            }

            void main() {
                float adjusted = mix(0.0, 1.0, 0.25);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float mix(float x, float y, float t)" in generated_code
    assert "float adjusted = mix(0.0, 1.0, 0.25);" in generated_code
    assert "float adjusted = lerp(0.0, 1.0, 0.25);" not in generated_code


def test_workgroup_barrier_lowers_to_slang_group_sync():
    code = """
    shader BarrierGap {
        compute {
            void main() {
                workgroupBarrier();
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "GroupMemoryBarrierWithGroupSync();" in generated_code
    assert "workgroupBarrier();" not in generated_code


def test_user_defined_workgroup_barrier_is_not_lowered_to_group_sync():
    code = """
    shader BarrierGap {
        compute {
            void workgroupBarrier() {
                return;
            }

            void main() {
                workgroupBarrier();
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "void workgroupBarrier()" in generated_code
    assert "workgroupBarrier();" in generated_code
    assert "GroupMemoryBarrierWithGroupSync();" not in generated_code


def test_floating_binary_modulo_lowers_to_slang_fmod():
    code = """
    shader BinaryModuloGap {
        compute {
            void main() {
                float a = 5.0;
                float wrapped = a % 2.0;
                a %= 3.0;
                vec2 v = vec2(5.0, 7.0);
                vec2 wrappedVec = v % vec2(2.0);
                v %= vec2(3.0);
                double d = 5.0;
                double wrappedDouble = d % 2.0;
                d %= 3.0;
                int i = 5 % 2;
                i %= 2;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float wrapped = fmod(a, 2.0);" in generated_code
    assert "a = fmod(a, 3.0);" in generated_code
    assert "float2 wrappedVec = fmod(v, float2(2.0));" in generated_code
    assert "v = fmod(v, float2(3.0));" in generated_code
    assert "double wrappedDouble = fmod(d, 2.0);" in generated_code
    assert "d = fmod(d, 3.0);" in generated_code
    assert "int i = 5 % 2;" in generated_code
    assert "i %= 2;" in generated_code
    assert "float wrapped = a % 2.0;" not in generated_code
    assert "float2 wrappedVec = v % float2(2.0);" not in generated_code
    assert "double wrappedDouble = d % 2.0;" not in generated_code
    assert "a %= 3.0;" not in generated_code
    assert "v %= float2(3.0);" not in generated_code
    assert "d %= 3.0;" not in generated_code


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


def test_optional_for_loop_clauses_emit_empty_slang_slots():
    code = """
    shader main {
        compute {
            void main() {
                int i = 0;
                for (; ; ) {
                    i = i + 1;
                    if (i > 3) {
                        break;
                    }
                }
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for (; ; )" in generated_code
    assert "None" not in generated_code
    assert "break;" in generated_code


def test_for_in_statement_lowers_to_counted_slang_loops():
    code = """
    shader main {
        compute {
            void main() {
                int total = 0;
                for i in 4 {
                    total = total + i;
                }
                for j in 2..5 {
                    total = total + j;
                }
                for k in 1..=4 {
                    total = total + k;
                }
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for (int i = 0; i < 4; ++i)" in generated_code
    assert "for (int j = 2; j < 5; ++j)" in generated_code
    assert "for (int k = 1; k <= 4; ++k)" in generated_code
    assert "total = total + i;" in generated_code
    assert "total = total + j;" in generated_code
    assert "total = total + k;" in generated_code
    assert "ForInNode" not in generated_code
    assert "RangeNode" not in generated_code


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


def test_do_while_loop_control_flow_emits_slang_loop():
    code = """
    shader main {
        vertex {
            void main() {
                int i = 0;
                do {
                    i = i + 1;
                } while (i < 4);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "\n    do\n    {\n" in generated_code
    assert "i = i + 1;" in generated_code
    assert "\n    } while (i < 4);" in generated_code
    assert "DoWhileNode(" not in generated_code


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


def test_match_literal_and_wildcard_arms_lower_to_slang_switch():
    code = """
    shader main {
        compute {
            int main(int mode) {
                int value = 0;
                match mode {
                    0 => {
                        value = 1;
                    }
                    _ => {
                        value = 2;
                    }
                }
                return value;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "switch (mode)" in generated_code
    assert "case 0:" in generated_code
    assert "value = 1;" in generated_code
    assert "default:" in generated_code
    assert "value = 2;" in generated_code
    assert generated_code.count("break;") == 2
    assert "MatchNode" not in generated_code


def test_match_guarded_arm_rejected_for_slang_switch_lowering():
    code = """
    shader main {
        compute {
            int main(int mode) {
                int value = 0;
                match mode {
                    0 if mode > 0 => {
                        value = 1;
                    }
                    _ => {
                        value = 2;
                    }
                }
                return value;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    with pytest.raises(ValueError, match="Unsupported match arm for Slang"):
        generate_code(ast)


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


def test_non_resource_arrays_preserve_expression_sizes():
    code = """
    shader ArraySizes {
        vec3 colors[(2 + 1) * 2];

        float accumulate(float values[(2 + 1) * 2], vec3 normals[+6]) {
            float localWeights[(2 + 1) * 2];
            return values[0] + localWeights[1] + normals[0].x;
        }

        compute {
            void main() {}
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float3 colors[((2 + 1) * 2)];" in generated_code
    assert (
        "float accumulate(float values[((2 + 1) * 2)], float3 normals[+6])"
        in generated_code
    )
    assert "float localWeights[((2 + 1) * 2)];" in generated_code
    assert "2 + 1 * 2" not in generated_code


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


def test_explicit_sampler_texture_builtins_emit_combined_slang_methods():
    code = """
    shader Resources {
        sampler linearSampler;
        sampler2d colorMap;
        sampler2darray layerMap;
        sampler3d volumeMap;

        compute {
            vec4 sampleExplicit(
                sampler2d tex,
                sampler2darray layers,
                sampler3d volume,
                sampler sampleState,
                vec2 uv,
                vec3 uvw,
                vec2 ddx,
                vec2 ddy,
                vec3 ddx3,
                vec3 ddy3
            ) {
                vec4 color = texture(tex, sampleState, uv);
                vec4 biased = texture(tex, sampleState, uv, 0.5);
                vec4 mip = textureLod(tex, sampleState, uv, 2.0);
                vec4 grad = textureGrad(tex, sampleState, uv, ddx, ddy);
                vec4 layer = texture(layers, sampleState, uvw);
                vec4 volumeGrad = textureGrad(volume, sampleState, uvw, ddx3, ddy3);
                return color + biased + mip + grad + layer + volumeGrad;
            }

            void main() {}
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "SamplerState linearSampler;" in generated_code
    assert "SamplerState sampleState" in generated_code
    assert "float4 color = tex.Sample(uv);" in generated_code
    assert "float4 biased = tex.SampleBias(uv, 0.5);" in generated_code
    assert "float4 mip = tex.SampleLevel(uv, 2.0);" in generated_code
    assert "float4 grad = tex.SampleGrad(uv, ddx, ddy);" in generated_code
    assert "float4 layer = layers.Sample(uvw);" in generated_code
    assert "float4 volumeGrad = volume.SampleGrad(uvw, ddx3, ddy3);" in generated_code
    assert "tex.Sample(sampleState" not in generated_code
    assert "tex.SampleBias(sampleState" not in generated_code
    assert "tex.SampleLevel(sampleState" not in generated_code
    assert "tex.SampleGrad(sampleState" not in generated_code
    assert "texture(" not in generated_code
    assert "textureLod(" not in generated_code
    assert "textureGrad(" not in generated_code


def test_texture_and_shadow_arrays_preserve_expression_sizes_and_group_indices():
    code = """
    shader TextureArrays {
        sampler2d textures[(2 + 1) * 2];
        sampler samplers[(2 + 1) * 2];
        sampler2d unaryTextures[+6];
        sampler2dshadow shadowMaps[(2 + 1) * 2];
        sampler shadowSamplers[(2 + 1) * 2];

        compute {
            vec4 sampleLayer(
                sampler2d textures[(2 + 1) * 2],
                sampler samplers[(2 + 1) * 2],
                sampler2d unaryTextures[+6],
                vec2 uv,
                vec2 ddx,
                vec2 ddy
            ) {
                vec4 color = texture(textures[1 + 2], samplers[1 + 2], uv);
                vec4 lodColor = textureLod(textures[2], samplers[2], uv, 1.0);
                vec4 gradColor = textureGrad(textures[2], samplers[2], uv, ddx, ddy);
                vec4 gathered = textureGather(textures[1 + 2], samplers[1 + 2], uv);
                vec4 unaryColor = texture(unaryTextures[2], uv);
                return color + lodColor + gradColor + gathered + unaryColor;
            }

            float shadowLayer(
                sampler2dshadow shadowMaps[(2 + 1) * 2],
                sampler shadowSamplers[(2 + 1) * 2],
                vec2 uv,
                float depth
            ) {
                float compared = textureCompare(
                    shadowMaps[1 + 2],
                    shadowSamplers[1 + 2],
                    uv,
                    depth
                );
                vec4 gathered = textureGatherCompare(
                    shadowMaps[2],
                    shadowSamplers[2],
                    uv,
                    depth
                );
                return compared + gathered.x;
            }

            void main() {}
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "Sampler2D<float4> textures[((2 + 1) * 2)];" in generated_code
    assert "SamplerState samplers[((2 + 1) * 2)];" in generated_code
    assert "Sampler2D<float4> unaryTextures[+6];" in generated_code
    assert "Sampler2DShadow shadowMaps[((2 + 1) * 2)];" in generated_code
    assert "SamplerState shadowSamplers[((2 + 1) * 2)];" in generated_code
    assert (
        "float4 sampleLayer(Sampler2D<float4> textures[((2 + 1) * 2)], "
        "SamplerState samplers[((2 + 1) * 2)], "
        "Sampler2D<float4> unaryTextures[+6], float2 uv, float2 ddx, float2 ddy)"
        in generated_code
    )
    assert (
        "float shadowLayer(Sampler2DShadow shadowMaps[((2 + 1) * 2)], "
        "SamplerState shadowSamplers[((2 + 1) * 2)], float2 uv, float depth)"
        in generated_code
    )
    assert "float4 color = textures[(1 + 2)].Sample(uv);" in generated_code
    assert "float4 lodColor = textures[2].SampleLevel(uv, 1.0);" in generated_code
    assert "float4 gradColor = textures[2].SampleGrad(uv, ddx, ddy);" in generated_code
    assert "float4 gathered = textures[(1 + 2)].Gather(uv);" in generated_code
    assert "float4 unaryColor = unaryTextures[2].Sample(uv);" in generated_code
    assert (
        "float compared = shadowMaps[(1 + 2)].SampleCmp(uv, depth);" in generated_code
    )
    assert "float4 gathered = shadowMaps[2].GatherCmp(uv, depth);" in generated_code
    assert "1 + 2]." not in generated_code
    assert "2 + 1 * 2" not in generated_code
    assert ".Sample(samplers" not in generated_code
    assert ".SampleCmp(shadowSamplers" not in generated_code
    assert "texture(" not in generated_code
    assert "textureLod(" not in generated_code
    assert "textureGrad(" not in generated_code
    assert "textureGather(" not in generated_code
    assert "textureCompare(" not in generated_code


def test_texture_offset_builtins_emit_slang_offset_methods():
    code = """
    shader Resources {
        sampler linearSampler;
        sampler2d colorMap;
        sampler2darray layerMap;

        compute {
            vec4 offsetOps(
                sampler2d tex,
                sampler2darray layers,
                sampler sampleState,
                vec2 uv,
                vec3 uvw,
                vec2 ddx,
                vec2 ddy,
                ivec2 offset
            ) {
                vec4 offsetColor = textureOffset(tex, uv, offset);
                vec4 explicitOffset = textureOffset(
                    tex,
                    sampleState,
                    uv,
                    offset
                );
                vec4 arrayOffset = textureOffset(layers, sampleState, uvw, offset);
                vec4 lodOffset = textureLodOffset(tex, uv, 2.0, offset);
                vec4 explicitLodOffset = textureLodOffset(
                    tex,
                    sampleState,
                    uv,
                    3.0,
                    offset
                );
                vec4 gradOffset = textureGradOffset(tex, uv, ddx, ddy, offset);
                vec4 explicitGradOffset = textureGradOffset(
                    tex,
                    sampleState,
                    uv,
                    ddx,
                    ddy,
                    offset
                );
                return offsetColor
                    + explicitOffset
                    + arrayOffset
                    + lodOffset
                    + explicitLodOffset
                    + gradOffset
                    + explicitGradOffset;
            }

            void main() {}
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "Sampler2D<float4> tex" in generated_code
    assert "Sampler2DArray<float4> layers" in generated_code
    assert "SamplerState sampleState" in generated_code
    assert "float4 offsetColor = tex.Sample(uv, offset);" in generated_code
    assert "float4 explicitOffset = tex.Sample(uv, offset);" in generated_code
    assert "float4 arrayOffset = layers.Sample(uvw, offset);" in generated_code
    assert "float4 lodOffset = tex.SampleLevel(uv, 2.0, offset);" in generated_code
    assert (
        "float4 explicitLodOffset = tex.SampleLevel(uv, 3.0, offset);" in generated_code
    )
    assert "float4 gradOffset = tex.SampleGrad(uv, ddx, ddy, offset);" in generated_code
    assert (
        "float4 explicitGradOffset = tex.SampleGrad(uv, ddx, ddy, offset);"
        in generated_code
    )
    assert "textureOffset(" not in generated_code
    assert "textureLodOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code
    assert ".Sample(sampleState" not in generated_code
    assert ".SampleLevel(sampleState" not in generated_code
    assert ".SampleGrad(sampleState" not in generated_code


def test_texture_offset_invalid_slang_calls_emit_diagnostic_stubs():
    code = """
    shader Resources {
        sampler2d colorMap;

        compute {
            void main() {
                vec2 uv = vec2(0.25, 0.75);
                vec2 ddx = vec2(0.1, 0.0);
                ivec2 offset = ivec2(1, 0);
                vec4 missingOffset = textureOffset(colorMap, uv);
                vec4 missingLodOffset = textureLodOffset(colorMap, uv, offset);
                vec4 missingGradOffset = textureGradOffset(colorMap, uv, ddx, offset);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "float4 missingOffset = /* unsupported Slang texture offset: "
        "textureOffset requires one offset argument */ float4(0.0);" in generated_code
    )
    assert (
        "float4 missingLodOffset = /* unsupported Slang texture offset: "
        "textureLodOffset requires lod and offset arguments */ float4(0.0);"
        in generated_code
    )
    assert (
        "float4 missingGradOffset = /* unsupported Slang texture offset: "
        "textureGradOffset requires gradient x, gradient y, and offset arguments */ "
        "float4(0.0);" in generated_code
    )
    assert "textureOffset(" not in generated_code
    assert "textureLodOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code


def test_projected_texture_builtins_emit_slang_projected_samples():
    code = """
    shader Resources {
        sampler linearSampler;
        sampler2d colorMap;
        sampler3d volumeMap;

        compute {
            vec4 projectedOps(
                sampler2d tex,
                sampler3d volume,
                sampler sampleState,
                vec3 uvq,
                vec4 uvqw,
                vec4 xyzq,
                vec2 ddx,
                vec2 ddy,
                ivec2 offset
            ) {
                vec4 projected = textureProj(tex, uvq);
                vec4 explicitProjected = textureProj(tex, sampleState, uvqw, 0.25);
                vec4 volumeProjected = textureProj(volume, xyzq);
                vec4 projectedOffset = textureProjOffset(tex, uvq, offset);
                vec4 projectedOffsetBias = textureProjOffset(
                    tex,
                    sampleState,
                    uvq,
                    offset,
                    0.5
                );
                vec4 projectedLod = textureProjLod(tex, sampleState, uvq, 2.0);
                vec4 projectedLodOffset = textureProjLodOffset(
                    tex,
                    uvq,
                    3.0,
                    offset
                );
                vec4 projectedGrad = textureProjGrad(tex, uvq, ddx, ddy);
                vec4 projectedGradOffset = textureProjGradOffset(
                    tex,
                    sampleState,
                    uvq,
                    ddx,
                    ddy,
                    offset
                );
                return projected
                    + explicitProjected
                    + volumeProjected
                    + projectedOffset
                    + projectedOffsetBias
                    + projectedLod
                    + projectedLodOffset
                    + projectedGrad
                    + projectedGradOffset;
            }

            void main() {}
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "Sampler2D<float4> tex" in generated_code
    assert "Sampler3D<float4> volume" in generated_code
    assert "SamplerState sampleState" in generated_code
    assert "float4 projected = tex.Sample(uvq.xy / uvq.z);" in generated_code
    assert (
        "float4 explicitProjected = tex.SampleBias(uvqw.xy / uvqw.w, 0.25);"
        in generated_code
    )
    assert (
        "float4 volumeProjected = volume.Sample(xyzq.xyz / xyzq.w);" in generated_code
    )
    assert (
        "float4 projectedOffset = tex.Sample(uvq.xy / uvq.z, offset);" in generated_code
    )
    assert (
        "float4 projectedOffsetBias = tex.SampleBias("
        "uvq.xy / uvq.z, 0.5, offset);" in generated_code
    )
    assert (
        "float4 projectedLod = tex.SampleLevel(uvq.xy / uvq.z, 2.0);" in generated_code
    )
    assert (
        "float4 projectedLodOffset = tex.SampleLevel("
        "uvq.xy / uvq.z, 3.0, offset);" in generated_code
    )
    assert (
        "float4 projectedGrad = tex.SampleGrad(uvq.xy / uvq.z, ddx, ddy);"
        in generated_code
    )
    assert (
        "float4 projectedGradOffset = tex.SampleGrad("
        "uvq.xy / uvq.z, ddx, ddy, offset);" in generated_code
    )
    assert "textureProj(" not in generated_code
    assert "textureProjOffset(" not in generated_code
    assert "textureProjLod(" not in generated_code
    assert "textureProjLodOffset(" not in generated_code
    assert "textureProjGrad(" not in generated_code
    assert "textureProjGradOffset(" not in generated_code
    assert ".Sample(sampleState" not in generated_code
    assert ".SampleBias(sampleState" not in generated_code
    assert ".SampleLevel(sampleState" not in generated_code
    assert ".SampleGrad(sampleState" not in generated_code


def test_projected_texture_invalid_slang_calls_emit_diagnostic_stubs():
    code = """
    shader Resources {
        sampler2d colorMap;

        compute {
            void main() {
                vec2 uv = vec2(0.25, 0.75);
                vec3 uvq = vec3(0.25, 0.75, 1.0);
                vec2 ddx = vec2(0.1, 0.0);
                ivec2 offset = ivec2(1, 0);
                vec4 badCoord = textureProj(colorMap, uv);
                vec4 missingLod = textureProjLod(colorMap, uvq);
                vec4 missingGrad = textureProjGrad(colorMap, uvq, ddx);
                vec4 missingOffset = textureProjGradOffset(
                    colorMap,
                    uvq,
                    ddx,
                    ddx
                );
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "float4 badCoord = /* unsupported Slang projected texture: "
        "textureProj requires sampler1D/2D/3D projection coordinates */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 missingLod = /* unsupported Slang projected texture: "
        "textureProjLod requires one lod argument */ float4(0.0);" in generated_code
    )
    assert (
        "float4 missingGrad = /* unsupported Slang projected texture: "
        "textureProjGrad requires gradient x and gradient y arguments */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 missingOffset = /* unsupported Slang projected texture: "
        "textureProjGradOffset requires gradient x, gradient y, and offset arguments */ "
        "float4(0.0);" in generated_code
    )
    assert "textureProj(" not in generated_code
    assert "textureProjLod(" not in generated_code
    assert "textureProjGrad(" not in generated_code
    assert "textureProjGradOffset(" not in generated_code


def test_explicit_sampler_texel_fetch_emits_combined_slang_methods():
    code = """
    shader Resources {
        sampler linearSampler;
        sampler2d colorMap;
        sampler2darray layerMap;
        sampler3d volumeMap;
        sampler2dms msTex;
        sampler2dmsarray msArray;

        compute {
            vec4 fetchExplicit(
                sampler2d tex,
                sampler2darray layers,
                sampler3d volume,
                sampler2dms ms,
                sampler2dmsarray msLayers,
                sampler sampleState,
                ivec2 pixel,
                ivec3 pixelLayer,
                int lod,
                int sampleIndex
            ) {
                vec4 fetched = texelFetch(tex, sampleState, pixel, lod);
                vec4 fetchedLayer = texelFetch(
                    layers,
                    sampleState,
                    pixelLayer,
                    lod
                );
                vec4 fetchedVolume = texelFetch(volume, sampleState, pixelLayer, lod);
                vec4 msColor = texelFetch(ms, sampleState, pixel, sampleIndex);
                vec4 msLayer = texelFetch(
                    msLayers,
                    sampleState,
                    pixelLayer,
                    sampleIndex
                );
                return fetched + fetchedLayer + fetchedVolume + msColor + msLayer;
            }

            void main() {}
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "SamplerState linearSampler;" in generated_code
    assert "SamplerState sampleState" in generated_code
    assert "float4 fetched = tex.Load(int3(pixel, lod));" in generated_code
    assert "float4 fetchedLayer = layers.Load(int4(pixelLayer, lod));" in generated_code
    assert (
        "float4 fetchedVolume = volume.Load(int4(pixelLayer, lod));" in generated_code
    )
    assert "float4 msColor = ms[pixel, sampleIndex];" in generated_code
    assert "float4 msLayer = msLayers[pixelLayer, sampleIndex];" in generated_code
    assert "tex.Load(int3(sampleState" not in generated_code
    assert "layers.Load(int4(sampleState" not in generated_code
    assert "volume.Load(int4(sampleState" not in generated_code
    assert "ms[sampleState" not in generated_code
    assert "msLayers[sampleState" not in generated_code
    assert "texelFetch(" not in generated_code


def test_texture_gather_builtins_emit_slang_gather_methods():
    code = """
    shader Resources {
        sampler linearSampler;
        sampler2d colorMap;
        sampler2darray layerMap;

        compute {
            vec4 gatherOps(
                sampler2d tex,
                sampler2darray layers,
                sampler sampleState,
                vec2 uv,
                vec3 uvLayer,
                ivec2 offset,
                ivec2 offsets[4],
                int component
            ) {
                vec4 gathered = textureGather(tex, uv);
                vec4 explicitGathered = textureGather(tex, sampleState, uv);
                vec4 greenGather = textureGather(tex, uv, 1);
                vec4 blueExplicitGather = textureGather(
                    tex,
                    sampleState,
                    uv,
                    2
                );
                vec4 offsetGather = textureGatherOffset(tex, uv, offset);
                vec4 alphaOffsetGather = textureGatherOffset(
                    tex,
                    sampleState,
                    uv,
                    offset,
                    3
                );
                vec4 offsetsGather = textureGatherOffsets(
                    layers,
                    sampleState,
                    uvLayer,
                    offsets
                );
                vec4 dynamicGather = textureGather(tex, uv, component);
                vec4 dynamicOffsetGather = textureGatherOffset(
                    tex,
                    sampleState,
                    uv,
                    offset,
                    component
                );
                vec4 dynamicOffsetsGather = textureGatherOffsets(
                    layers,
                    sampleState,
                    uvLayer,
                    offsets,
                    component
                );
                return gathered
                    + explicitGathered
                    + greenGather
                    + blueExplicitGather
                    + offsetGather
                    + alphaOffsetGather
                    + offsetsGather
                    + dynamicGather
                    + dynamicOffsetGather
                    + dynamicOffsetsGather;
            }

            void main() {}
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "Sampler2D<float4> tex" in generated_code
    assert "Sampler2DArray<float4> layers" in generated_code
    assert "SamplerState sampleState" in generated_code
    assert "float4 gathered = tex.Gather(uv);" in generated_code
    assert "float4 explicitGathered = tex.Gather(uv);" in generated_code
    assert "float4 greenGather = tex.GatherGreen(uv);" in generated_code
    assert "float4 blueExplicitGather = tex.GatherBlue(uv);" in generated_code
    assert "float4 offsetGather = tex.Gather(uv, offset);" in generated_code
    assert "float4 alphaOffsetGather = tex.GatherAlpha(uv, offset);" in generated_code
    assert (
        "float4 offsetsGather = layers.Gather("
        "uvLayer, offsets[0], offsets[1], offsets[2], offsets[3]);" in generated_code
    )
    assert (
        "float4 dynamicGather = (component == 0 ? tex.GatherRed(uv) : "
        "component == 1 ? tex.GatherGreen(uv) : "
        "component == 2 ? tex.GatherBlue(uv) : tex.GatherAlpha(uv));" in generated_code
    )
    assert (
        "float4 dynamicOffsetGather = ("
        "component == 0 ? tex.GatherRed(uv, offset) : "
        "component == 1 ? tex.GatherGreen(uv, offset) : "
        "component == 2 ? tex.GatherBlue(uv, offset) : "
        "tex.GatherAlpha(uv, offset));" in generated_code
    )
    assert (
        "float4 dynamicOffsetsGather = ("
        "component == 0 ? layers.GatherRed("
        "uvLayer, offsets[0], offsets[1], offsets[2], offsets[3]) : "
        "component == 1 ? layers.GatherGreen("
        "uvLayer, offsets[0], offsets[1], offsets[2], offsets[3]) : "
        "component == 2 ? layers.GatherBlue("
        "uvLayer, offsets[0], offsets[1], offsets[2], offsets[3]) : "
        "layers.GatherAlpha("
        "uvLayer, offsets[0], offsets[1], offsets[2], offsets[3]));" in generated_code
    )
    assert "textureGather" not in generated_code
    assert ".Gather(sampleState" not in generated_code
    assert ".GatherBlue(sampleState" not in generated_code
    assert ".GatherAlpha(sampleState" not in generated_code


def test_texture_gather_invalid_slang_calls_emit_diagnostic_stubs():
    code = """
    shader Resources {
        sampler2d colorMap;

        compute {
            vec4 gatherDiagnostics(
                sampler2d tex,
                vec2 uv,
                ivec2 offset
            ) {
                vec4 badComponent = textureGather(tex, uv, 4);
                vec4 missingOffset = textureGatherOffset(tex, uv);
                vec4 badOffsets = textureGatherOffsets(tex, uv, offset);
                return badComponent + missingOffset + badOffsets;
            }

            void main() {}
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "float4 badComponent = /* unsupported Slang texture gather: "
        "textureGather component literal must be 0, 1, 2, or 3 */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 missingOffset = /* unsupported Slang texture gather: "
        "textureGatherOffset requires offset and optional component arguments */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 badOffsets = /* unsupported Slang texture gather: "
        "textureGatherOffsets requires a typed offsets array or four offset arguments */ "
        "float4(0.0);" in generated_code
    )
    assert "textureGather(" not in generated_code
    assert "textureGatherOffset(" not in generated_code
    assert "textureGatherOffsets(" not in generated_code


def test_shadow_compare_builtins_emit_slang_compare_methods():
    code = """
    shader Resources {
        sampler compareSampler;
        sampler2dshadow shadowMap;
        sampler2darrayshadow shadowLayers;
        samplercubeshadow cubeShadow;

        compute {
            float compareOps(
                sampler2DShadow tex,
                sampler2DArrayShadow layers,
                samplerCubeShadow cube,
                sampler compareState,
                vec2 uv,
                vec3 uvLayer,
                vec3 direction,
                float depth,
                vec2 ddx,
                vec2 ddy,
                ivec2 offset
            ) {
                float direct = textureCompare(tex, uv, depth);
                float explicitCompare = textureCompare(
                    tex,
                    compareState,
                    uv,
                    depth
                );
                float arrayCompare = textureCompare(
                    layers,
                    compareState,
                    uvLayer,
                    depth
                );
                float cubeCompare = textureCompare(cube, direction, depth);
                float lod = textureCompareLod(tex, compareState, uv, depth, 2.0);
                float grad = textureCompareGrad(
                    tex,
                    compareState,
                    uv,
                    depth,
                    ddx,
                    ddy
                );
                float offsetCompare = textureCompareOffset(tex, uv, depth, offset);
                vec4 gathered = textureGatherCompare(tex, compareState, uv, depth);
                vec4 gatheredOffset = textureGatherCompareOffset(
                    tex,
                    uv,
                    depth,
                    offset
                );
                return direct
                    + explicitCompare
                    + arrayCompare
                    + cubeCompare
                    + lod
                    + grad
                    + offsetCompare
                    + gathered.x
                    + gatheredOffset.x;
            }

            void main() {}
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "Sampler2DShadow tex" in generated_code
    assert "Sampler2DArrayShadow layers" in generated_code
    assert "SamplerCubeShadow cube" in generated_code
    assert "SamplerState compareState" in generated_code
    assert "float direct = tex.SampleCmp(uv, depth);" in generated_code
    assert "float explicitCompare = tex.SampleCmp(uv, depth);" in generated_code
    assert "float arrayCompare = layers.SampleCmp(uvLayer, depth);" in generated_code
    assert "float cubeCompare = cube.SampleCmp(direction, depth);" in generated_code
    assert "float lod = tex.SampleCmpLevel(uv, depth, 2.0);" in generated_code
    assert "float grad = tex.SampleCmpGrad(uv, depth, ddx, ddy);" in generated_code
    assert "float offsetCompare = tex.SampleCmp(uv, depth, offset);" in generated_code
    assert "float4 gathered = tex.GatherCmp(uv, depth);" in generated_code
    assert "float4 gatheredOffset = tex.GatherCmp(uv, depth, offset);" in generated_code
    assert "textureCompare" not in generated_code
    assert "textureGatherCompare" not in generated_code
    assert ".SampleCmp(compareState" not in generated_code
    assert ".GatherCmp(compareState" not in generated_code


def test_shadow_compare_invalid_slang_calls_emit_diagnostic_stubs():
    code = """
    shader Resources {
        sampler2d colorMap;
        sampler2dshadow shadowMap;

        compute {
            void main() {
                vec2 uv = vec2(0.25, 0.75);
                vec2 ddx = vec2(0.1, 0.0);
                float badResource = textureCompare(colorMap, uv, 0.5);
                float missingDepth = textureCompare(shadowMap, uv);
                float missingLod = textureCompareLod(shadowMap, uv, 0.5);
                float missingGrad = textureCompareGrad(shadowMap, uv, 0.5, ddx);
                vec4 missingGatherOffset = textureGatherCompareOffset(
                    shadowMap,
                    uv,
                    0.5
                );
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "float badResource = /* unsupported Slang shadow compare: "
        "textureCompare requires a shadow sampler resource */ 0.0;" in generated_code
    )
    assert (
        "float missingDepth = /* unsupported Slang shadow compare: "
        "textureCompare requires texture, coordinate, and compare arguments */ 0.0;"
        in generated_code
    )
    assert (
        "float missingLod = /* unsupported Slang shadow compare: "
        "textureCompareLod requires one lod argument */ 0.0;" in generated_code
    )
    assert (
        "float missingGrad = /* unsupported Slang shadow compare: "
        "textureCompareGrad requires gradient x and gradient y arguments */ 0.0;"
        in generated_code
    )
    assert (
        "float4 missingGatherOffset = /* unsupported Slang shadow gather compare: "
        "textureGatherCompareOffset requires one offset argument */ float4(0.0);"
        in generated_code
    )
    assert "textureCompare(" not in generated_code
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareGrad(" not in generated_code
    assert "textureGatherCompareOffset(" not in generated_code


def test_texture_query_lod_emits_slang_lod_methods():
    code = """
    shader Resources {
        sampler querySampler;
        sampler2d colorMap;
        sampler2darray layers;
        sampler3d volumeTex;
        samplercube cubeTex;

        compute {
            void main() {
                vec2 uv = vec2(0.25, 0.5);
                vec3 uvw = vec3(0.25, 0.5, 0.75);
                vec3 direction = vec3(1.0, 0.0, 0.0);
                vec2 lod = textureQueryLod(colorMap, uv);
                vec2 explicitLod = textureQueryLod(colorMap, querySampler, uv);
                vec2 arrayLod = textureQueryLod(layers, querySampler, uvw);
                vec2 volumeLod = textureQueryLod(volumeTex, uvw);
                vec2 cubeLod = textureQueryLod(cubeTex, direction);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "float2 lod = float2("
        "colorMap.CalculateLevelOfDetailUnclamped(uv), "
        "colorMap.CalculateLevelOfDetail(uv));" in generated_code
    )
    assert (
        "float2 explicitLod = float2("
        "colorMap.CalculateLevelOfDetailUnclamped(uv), "
        "colorMap.CalculateLevelOfDetail(uv));" in generated_code
    )
    assert (
        "float2 arrayLod = float2("
        "layers.CalculateLevelOfDetailUnclamped(uvw), "
        "layers.CalculateLevelOfDetail(uvw));" in generated_code
    )
    assert (
        "float2 volumeLod = float2("
        "volumeTex.CalculateLevelOfDetailUnclamped(uvw), "
        "volumeTex.CalculateLevelOfDetail(uvw));" in generated_code
    )
    assert (
        "float2 cubeLod = float2("
        "cubeTex.CalculateLevelOfDetailUnclamped(direction), "
        "cubeTex.CalculateLevelOfDetail(direction));" in generated_code
    )
    assert "textureQueryLod(" not in generated_code
    assert "CalculateLevelOfDetail(querySampler" not in generated_code


def test_resource_queries_on_texture_arrays_emit_slang_helpers():
    code = """
    shader QueryArrays {
        sampler querySampler;
        sampler2d textures[(2 + 1) * 2];
        sampler2dms msTextures[(2 + 1) * 2];
        sampler2dmsarray msLayers[(2 + 1) * 2];
        sampler2dshadow shadowMaps[(2 + 1) * 2];

        compute {
            void main() {
                vec2 uv = vec2(0.25, 0.5);
                ivec3 pixelLayer = ivec3(1, 2, 3);
                ivec2 sizeValue = textureSize(textures[1 + 2], 0);
                int levelsValue = textureQueryLevels(textures[1 + 2]);
                vec2 lodValue = textureQueryLod(textures[1 + 2], uv);
                vec2 explicitLodValue = textureQueryLod(
                    textures[1 + 2],
                    querySampler,
                    uv
                );
                ivec2 msSize = textureSize(msTextures[1 + 2], 0);
                int msSamples = textureSamples(msTextures[1 + 2]);
                ivec3 msLayerSize = textureSize(msLayers[1 + 2], 0);
                int msLayerSamples = textureSamples(msLayers[1 + 2]);
                ivec2 shadowSize = textureSize(shadowMaps[1 + 2], 0);
                int shadowLevels = textureQueryLevels(shadowMaps[1 + 2]);
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
    assert (
        "int cgl_textureQueryLevels_sampler2D(Sampler2D<float4> tex)" in generated_code
    )
    assert "int2 cgl_textureSize_sampler2DMS(Sampler2DMS<float4> tex)" in generated_code
    assert (
        "int cgl_textureSamples_sampler2DMS(Sampler2DMS<float4> tex)" in generated_code
    )
    assert (
        "int3 cgl_textureSize_sampler2DMSArray(Sampler2DMSArray<float4> tex)"
        in generated_code
    )
    assert (
        "int cgl_textureSamples_sampler2DMSArray(Sampler2DMSArray<float4> tex)"
        in generated_code
    )
    assert (
        "int2 cgl_textureSize_sampler2DShadow(Sampler2DShadow tex, uint mipLevel)"
        in generated_code
    )
    assert (
        "int cgl_textureQueryLevels_sampler2DShadow(Sampler2DShadow tex)"
        in generated_code
    )
    assert (
        "int2 sizeValue = cgl_textureSize_sampler2D(textures[(1 + 2)], 0);"
        in generated_code
    )
    assert (
        "int levelsValue = cgl_textureQueryLevels_sampler2D(textures[(1 + 2)]);"
        in generated_code
    )
    assert (
        "float2 lodValue = float2("
        "textures[(1 + 2)].CalculateLevelOfDetailUnclamped(uv), "
        "textures[(1 + 2)].CalculateLevelOfDetail(uv));" in generated_code
    )
    assert (
        "float2 explicitLodValue = float2("
        "textures[(1 + 2)].CalculateLevelOfDetailUnclamped(uv), "
        "textures[(1 + 2)].CalculateLevelOfDetail(uv));" in generated_code
    )
    assert (
        "int2 msSize = cgl_textureSize_sampler2DMS(msTextures[(1 + 2)]);"
        in generated_code
    )
    assert (
        "int msSamples = cgl_textureSamples_sampler2DMS(msTextures[(1 + 2)]);"
        in generated_code
    )
    assert (
        "int3 msLayerSize = cgl_textureSize_sampler2DMSArray(msLayers[(1 + 2)]);"
        in generated_code
    )
    assert (
        "int msLayerSamples = cgl_textureSamples_sampler2DMSArray(msLayers[(1 + 2)]);"
        in generated_code
    )
    assert (
        "int2 shadowSize = cgl_textureSize_sampler2DShadow(shadowMaps[(1 + 2)], 0);"
        in generated_code
    )
    assert (
        "int shadowLevels = cgl_textureQueryLevels_sampler2DShadow("
        "shadowMaps[(1 + 2)]);" in generated_code
    )
    assert "1 + 2]." not in generated_code
    assert "textureSize(" not in generated_code
    assert "textureQueryLevels(" not in generated_code
    assert "textureSamples(" not in generated_code
    assert "textureQueryLod(" not in generated_code
    assert "CalculateLevelOfDetail(querySampler" not in generated_code


def test_nested_resource_arrays_emit_slang_queries_and_operations():
    code = """
    shader NestedResources {
        sampler querySampler;
        sampler2d textureGrid[2][3];
        sampler2dms msGrid[2][2];
        sampler2dshadow shadowGrid[2][3];
        image2D imageGrid @rgba16f[2][4];
        uimage2D counterGrid @r32ui[2][2];

        vec4 sampleGrid(sampler2d paramGrid[2][3], int layer, int slot, vec2 uv) {
            return texture(paramGrid[layer][slot], uv);
        }

        float compareGrid(
            sampler2dshadow paramShadow[2][3],
            int layer,
            int slot,
            vec2 uv,
            float depth
        ) {
            return textureCompare(paramShadow[layer][slot], uv, depth);
        }

        compute {
            void main() {
                int layer = 1;
                int slot = 1;
                vec2 uv = vec2(0.5, 0.25);
                ivec2 pixel = ivec2(0, 0);
                float depth = 0.5;
                ivec2 sizeValue = textureSize(textureGrid[layer][slot], 0);
                int levelsValue = textureQueryLevels(textureGrid[layer][slot]);
                vec2 lodValue = textureQueryLod(
                    textureGrid[layer][slot],
                    querySampler,
                    uv
                );
                int msSamples = textureSamples(msGrid[layer][slot]);
                ivec2 imageSizeValue = imageSize(imageGrid[layer][slot]);
                vec4 sampled = texture(textureGrid[layer][slot], uv);
                vec4 sampledParam = sampleGrid(textureGrid, layer, slot, uv);
                float compared = textureCompare(shadowGrid[layer][slot], uv, depth);
                float comparedParam = compareGrid(shadowGrid, layer, slot, uv, depth);
                vec4 gathered = textureGather(textureGrid[layer][slot], uv, 1);
                vec4 gatheredShadow = textureGatherCompare(
                    shadowGrid[layer][slot],
                    uv,
                    depth
                );
                vec4 fetched = texelFetch(textureGrid[layer][slot], pixel, 0);
                vec4 fetchedMs = texelFetch(msGrid[layer][slot], pixel, 1);
                vec4 color = imageLoad(imageGrid[layer][slot], pixel);
                imageStore(imageGrid[layer][slot], pixel, color);
                uint count = imageLoad(counterGrid[layer][slot], pixel);
                imageStore(counterGrid[layer][slot], pixel, count);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "Sampler2D<float4> textureGrid[2][3];" in generated_code
    assert "Sampler2DMS<float4> msGrid[2][2];" in generated_code
    assert "Sampler2DShadow shadowGrid[2][3];" in generated_code
    assert "RWTexture2D<float4> imageGrid[2][4];" in generated_code
    assert "RWTexture2D<uint> counterGrid[2][2];" in generated_code
    assert (
        "float4 sampleGrid(Sampler2D<float4> paramGrid[2][3], "
        "int layer, int slot, float2 uv)" in generated_code
    )
    assert (
        "float compareGrid(Sampler2DShadow paramShadow[2][3], "
        "int layer, int slot, float2 uv, float depth)" in generated_code
    )
    assert "return paramGrid[layer][slot].Sample(uv);" in generated_code
    assert "return paramShadow[layer][slot].SampleCmp(uv, depth);" in generated_code
    assert (
        "int2 sizeValue = cgl_textureSize_sampler2D"
        "(textureGrid[layer][slot], 0);" in generated_code
    )
    assert (
        "int levelsValue = cgl_textureQueryLevels_sampler2D"
        "(textureGrid[layer][slot]);" in generated_code
    )
    assert (
        "float2 lodValue = float2("
        "textureGrid[layer][slot].CalculateLevelOfDetailUnclamped(uv), "
        "textureGrid[layer][slot].CalculateLevelOfDetail(uv));" in generated_code
    )
    assert (
        "int msSamples = cgl_textureSamples_sampler2DMS(msGrid[layer][slot]);"
        in generated_code
    )
    assert (
        "int2 imageSizeValue = cgl_imageSize_image2D(imageGrid[layer][slot]);"
        in generated_code
    )
    assert "float4 sampled = textureGrid[layer][slot].Sample(uv);" in generated_code
    assert (
        "float4 sampledParam = sampleGrid(textureGrid, layer, slot, uv);"
        in generated_code
    )
    assert (
        "float compared = shadowGrid[layer][slot].SampleCmp(uv, depth);"
        in generated_code
    )
    assert (
        "float comparedParam = compareGrid(shadowGrid, layer, slot, uv, depth);"
        in generated_code
    )
    assert (
        "float4 gathered = textureGrid[layer][slot].GatherGreen(uv);" in generated_code
    )
    assert (
        "float4 gatheredShadow = shadowGrid[layer][slot].GatherCmp(uv, depth);"
        in generated_code
    )
    assert (
        "float4 fetched = textureGrid[layer][slot].Load(int3(pixel, 0));"
        in generated_code
    )
    assert "float4 fetchedMs = msGrid[layer][slot][pixel, 1];" in generated_code
    assert "float4 color = imageGrid[layer][slot][pixel];" in generated_code
    assert "imageGrid[layer][slot][pixel] = color;" in generated_code
    assert "uint count = counterGrid[layer][slot][pixel];" in generated_code
    assert "counterGrid[layer][slot][pixel] = count;" in generated_code
    assert "textureSize(" not in generated_code
    assert "textureQueryLevels(" not in generated_code
    assert "textureQueryLod(" not in generated_code
    assert "textureSamples(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "textureGather(" not in generated_code
    assert "textureGatherCompare(" not in generated_code
    assert "texelFetch(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_dynamic_resource_array_params_emit_slang_queries_and_operations():
    code = """
    shader DynamicResources {
        sampler2d textureGrid[2][3];
        image2D imageGrid @rgba16f[2][3];
        uimage2D counterGrid @r32ui[2][2];

        vec4 sampleDynamic(
            sampler2d dynGrid[][3],
            int layer,
            int slot,
            vec2 uv
        ) {
            return texture(dynGrid[layer][slot], uv);
        }

        vec4 readDynamic(
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            ivec2 pixel
        ) {
            return imageLoad(dynImages[layer][slot], pixel);
        }

        uint readCounter(
            uimage2D dynCounters[][2] @r32ui,
            int layer,
            int slot,
            ivec2 pixel
        ) {
            return imageLoad(dynCounters[layer][slot], pixel);
        }

        void queryDynamic(
            sampler2d dynGrid[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot
        ) {
            ivec2 texSize = textureSize(dynGrid[layer][slot], 0);
            ivec2 imageSizeValue = imageSize(dynImages[layer][slot]);
        }

        compute {
            void main() {
                vec2 uv = vec2(0.5, 0.25);
                ivec2 pixel = ivec2(0, 0);
                vec4 sampled = sampleDynamic(textureGrid, 1, 2, uv);
                vec4 color = readDynamic(imageGrid, 1, 2, pixel);
                uint count = readCounter(counterGrid, 1, 1, pixel);
                queryDynamic(textureGrid, imageGrid, 1, 2);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "Sampler2D<float4> textureGrid[2][3];" in generated_code
    assert "RWTexture2D<float4> imageGrid[2][3];" in generated_code
    assert "RWTexture2D<uint> counterGrid[2][2];" in generated_code
    assert (
        "float4 sampleDynamic(Sampler2D<float4> dynGrid[][3], "
        "int layer, int slot, float2 uv)" in generated_code
    )
    assert (
        "float4 readDynamic(RWTexture2D<float4> dynImages[][3], "
        "int layer, int slot, int2 pixel)" in generated_code
    )
    assert (
        "uint readCounter(RWTexture2D<uint> dynCounters[][2], "
        "int layer, int slot, int2 pixel)" in generated_code
    )
    assert (
        "void queryDynamic(Sampler2D<float4> dynGrid[][3], "
        "RWTexture2D<float4> dynImages[][3], int layer, int slot)" in generated_code
    )
    assert "return dynGrid[layer][slot].Sample(uv);" in generated_code
    assert "return dynImages[layer][slot][pixel];" in generated_code
    assert "return dynCounters[layer][slot][pixel];" in generated_code
    assert (
        "int2 texSize = cgl_textureSize_sampler2D(dynGrid[layer][slot], 0);"
        in generated_code
    )
    assert (
        "int2 imageSizeValue = cgl_imageSize_image2D(dynImages[layer][slot]);"
        in generated_code
    )
    assert "float4 sampled = sampleDynamic(textureGrid, 1, 2, uv);" in generated_code
    assert "float4 color = readDynamic(imageGrid, 1, 2, pixel);" in generated_code
    assert "uint count = readCounter(counterGrid, 1, 1, pixel);" in generated_code
    assert "queryDynamic(textureGrid, imageGrid, 1, 2);" in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "unsupported Slang" not in generated_code


def test_forwarded_dynamic_resource_array_params_emit_slang_operations():
    code = """
    shader ForwardedDynamicResources {
        sampler2d textureGrid[2][3];
        image2D imageGrid @rgba16f[2][3];
        uimage2D counterGrid @r32ui[2][2];

        vec4 leafSample(sampler2d dynGrid[][3], int layer, int slot, vec2 uv) {
            return texture(dynGrid[layer][slot], uv);
        }

        vec4 forwardSample(sampler2d dynGrid[][3], int layer, int slot, vec2 uv) {
            return leafSample(dynGrid, layer, slot, uv);
        }

        vec4 leafImage(
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            ivec2 pixel
        ) {
            vec4 color = imageLoad(dynImages[layer][slot], pixel);
            imageStore(dynImages[layer][slot], pixel, color);
            return color;
        }

        vec4 forwardImage(
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            ivec2 pixel
        ) {
            return leafImage(dynImages, layer, slot, pixel);
        }

        uint leafCounter(
            uimage2D dynCounters[][2] @r32ui,
            int layer,
            int slot,
            ivec2 pixel
        ) {
            uint count = imageLoad(dynCounters[layer][slot], pixel);
            imageStore(dynCounters[layer][slot], pixel, count);
            return count;
        }

        uint forwardCounter(
            uimage2D dynCounters[][2] @r32ui,
            int layer,
            int slot,
            ivec2 pixel
        ) {
            return leafCounter(dynCounters, layer, slot, pixel);
        }

        void leafQuery(
            sampler2d dynGrid[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot
        ) {
            ivec2 texSize = textureSize(dynGrid[layer][slot], 0);
            ivec2 imageSizeValue = imageSize(dynImages[layer][slot]);
        }

        void forwardQuery(
            sampler2d dynGrid[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot
        ) {
            leafQuery(dynGrid, dynImages, layer, slot);
        }

        compute {
            void main() {
                vec2 uv = vec2(0.5, 0.25);
                ivec2 pixel = ivec2(0, 0);
                vec4 sampled = forwardSample(textureGrid, 1, 2, uv);
                vec4 color = forwardImage(imageGrid, 1, 2, pixel);
                uint count = forwardCounter(counterGrid, 1, 1, pixel);
                forwardQuery(textureGrid, imageGrid, 1, 2);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "float4 leafSample(Sampler2D<float4> dynGrid[][3], "
        "int layer, int slot, float2 uv)" in generated_code
    )
    assert (
        "float4 forwardSample(Sampler2D<float4> dynGrid[][3], "
        "int layer, int slot, float2 uv)" in generated_code
    )
    assert "return leafSample(dynGrid, layer, slot, uv);" in generated_code
    assert (
        "float4 leafImage(RWTexture2D<float4> dynImages[][3], "
        "int layer, int slot, int2 pixel)" in generated_code
    )
    assert "return leafImage(dynImages, layer, slot, pixel);" in generated_code
    assert (
        "uint leafCounter(RWTexture2D<uint> dynCounters[][2], "
        "int layer, int slot, int2 pixel)" in generated_code
    )
    assert "return leafCounter(dynCounters, layer, slot, pixel);" in generated_code
    assert (
        "void leafQuery(Sampler2D<float4> dynGrid[][3], "
        "RWTexture2D<float4> dynImages[][3], int layer, int slot)" in generated_code
    )
    assert "leafQuery(dynGrid, dynImages, layer, slot);" in generated_code
    assert "float4 sampled = forwardSample(textureGrid, 1, 2, uv);" in generated_code
    assert "float4 color = forwardImage(imageGrid, 1, 2, pixel);" in generated_code
    assert "uint count = forwardCounter(counterGrid, 1, 1, pixel);" in generated_code
    assert "forwardQuery(textureGrid, imageGrid, 1, 2);" in generated_code
    assert "return dynGrid[layer][slot].Sample(uv);" in generated_code
    assert "float4 color = dynImages[layer][slot][pixel];" in generated_code
    assert "dynImages[layer][slot][pixel] = color;" in generated_code
    assert "uint count = dynCounters[layer][slot][pixel];" in generated_code
    assert "dynCounters[layer][slot][pixel] = count;" in generated_code
    assert (
        "int2 texSize = cgl_textureSize_sampler2D(dynGrid[layer][slot], 0);"
        in generated_code
    )
    assert (
        "int2 imageSizeValue = cgl_imageSize_image2D(dynImages[layer][slot]);"
        in generated_code
    )
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code
    assert "unsupported Slang" not in generated_code


def test_mixed_fixed_dynamic_resource_array_params_emit_slang_operations():
    code = """
    shader MixedResourceForwarding {
        sampler2d textureGrid[2][3];
        image2D imageGrid @rgba16f[2][3];

        void leafMixed(
            sampler2d dynamicGrid[][3],
            sampler2d fixedRow[3],
            image2D dynamicImages[][3] @rgba16f,
            image2D fixedImages[3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            vec4 dynamicSample = texture(dynamicGrid[layer][slot], uv);
            vec4 fixedSample = texture(fixedRow[slot], uv);
            vec4 dynamicColor = imageLoad(dynamicImages[layer][slot], pixel);
            vec4 fixedColor = imageLoad(fixedImages[slot], pixel);
            imageStore(dynamicImages[layer][slot], pixel, dynamicColor);
            imageStore(fixedImages[slot], pixel, fixedColor);
            ivec2 dynamicSize = textureSize(dynamicGrid[layer][slot], 0);
            ivec2 fixedSize = textureSize(fixedRow[slot], 0);
            ivec2 dynamicImageSize = imageSize(dynamicImages[layer][slot]);
            ivec2 fixedImageSize = imageSize(fixedImages[slot]);
        }

        void forwardMixed(
            sampler2d dynamicGrid[][3],
            sampler2d fixedRow[3],
            image2D dynamicImages[][3] @rgba16f,
            image2D fixedImages[3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            leafMixed(
                dynamicGrid,
                fixedRow,
                dynamicImages,
                fixedImages,
                layer,
                slot,
                uv,
                pixel
            );
        }

        compute {
            void main() {
                vec2 uv = vec2(0.5, 0.25);
                ivec2 pixel = ivec2(0, 0);
                forwardMixed(
                    textureGrid,
                    textureGrid[1],
                    imageGrid,
                    imageGrid[1],
                    1,
                    2,
                    uv,
                    pixel
                );
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "void leafMixed(Sampler2D<float4> dynamicGrid[][3], "
        "Sampler2D<float4> fixedRow[3], "
        "RWTexture2D<float4> dynamicImages[][3], "
        "RWTexture2D<float4> fixedImages[3], "
        "int layer, int slot, float2 uv, int2 pixel)" in generated_code
    )
    assert (
        "void forwardMixed(Sampler2D<float4> dynamicGrid[][3], "
        "Sampler2D<float4> fixedRow[3], "
        "RWTexture2D<float4> dynamicImages[][3], "
        "RWTexture2D<float4> fixedImages[3], "
        "int layer, int slot, float2 uv, int2 pixel)" in generated_code
    )
    assert (
        "leafMixed(dynamicGrid, fixedRow, dynamicImages, fixedImages, "
        "layer, slot, uv, pixel);" in generated_code
    )
    assert (
        "forwardMixed(textureGrid, textureGrid[1], imageGrid, imageGrid[1], "
        "1, 2, uv, pixel);" in generated_code
    )
    assert (
        "float4 dynamicSample = dynamicGrid[layer][slot].Sample(uv);" in generated_code
    )
    assert "float4 fixedSample = fixedRow[slot].Sample(uv);" in generated_code
    assert "float4 dynamicColor = dynamicImages[layer][slot][pixel];" in generated_code
    assert "float4 fixedColor = fixedImages[slot][pixel];" in generated_code
    assert "dynamicImages[layer][slot][pixel] = dynamicColor;" in generated_code
    assert "fixedImages[slot][pixel] = fixedColor;" in generated_code
    assert (
        "int2 dynamicSize = cgl_textureSize_sampler2D"
        "(dynamicGrid[layer][slot], 0);" in generated_code
    )
    assert (
        "int2 fixedSize = cgl_textureSize_sampler2D(fixedRow[slot], 0);"
        in generated_code
    )
    assert (
        "int2 dynamicImageSize = cgl_imageSize_image2D"
        "(dynamicImages[layer][slot]);" in generated_code
    )
    assert (
        "int2 fixedImageSize = cgl_imageSize_image2D(fixedImages[slot]);"
        in generated_code
    )
    assert "texture(" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code
    assert "unsupported Slang" not in generated_code


def test_multihop_reordered_resource_array_params_emit_slang_operations():
    code = """
    shader MultiHopResourceForwarding {
        sampler2d textureGrid[2][3];
        image2D imageGrid @rgba16f[2][3];

        void leafShuffle(
            sampler2d dynA[][3],
            image2D fixedImageRow[3] @rgba16f,
            sampler2d fixedTexRow[3],
            image2D dynImageGrid[][3] @rgba16f,
            sampler2d dynAlias[][3],
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            vec4 sampledA = texture(dynA[layer][slot], uv);
            vec4 sampledFixed = texture(fixedTexRow[slot], uv);
            vec4 sampledAlias = texture(dynAlias[layer][slot], uv);
            vec4 dynamicColor = imageLoad(dynImageGrid[layer][slot], pixel);
            vec4 fixedColor = imageLoad(fixedImageRow[slot], pixel);
            imageStore(dynImageGrid[layer][slot], pixel, dynamicColor);
            imageStore(fixedImageRow[slot], pixel, fixedColor);
            ivec2 dynSize = textureSize(dynA[layer][slot], 0);
            ivec2 fixedTexSize = textureSize(fixedTexRow[slot], 0);
            ivec2 aliasSize = textureSize(dynAlias[layer][slot], 0);
            ivec2 dynamicImageSize = imageSize(dynImageGrid[layer][slot]);
            ivec2 fixedImageSize = imageSize(fixedImageRow[slot]);
        }

        void midForward(
            sampler2d firstDyn[][3],
            sampler2d firstFixed[3],
            image2D firstDynImages[][3] @rgba16f,
            image2D firstFixedImages[3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            leafShuffle(
                firstDyn,
                firstFixedImages,
                firstFixed,
                firstDynImages,
                firstDyn,
                layer,
                slot,
                uv,
                pixel
            );
        }

        void topForward(
            sampler2d topFixedTex[3],
            image2D topDynImages[][3] @rgba16f,
            sampler2d topDynTex[][3],
            image2D topFixedImages[3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            midForward(
                topDynTex,
                topFixedTex,
                topDynImages,
                topFixedImages,
                layer,
                slot,
                uv,
                pixel
            );
        }

        compute {
            void main() {
                vec2 uv = vec2(0.5, 0.25);
                ivec2 pixel = ivec2(0, 0);
                topForward(
                    textureGrid[1],
                    imageGrid,
                    textureGrid,
                    imageGrid[1],
                    1,
                    2,
                    uv,
                    pixel
                );
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "void leafShuffle(Sampler2D<float4> dynA[][3], "
        "RWTexture2D<float4> fixedImageRow[3], "
        "Sampler2D<float4> fixedTexRow[3], "
        "RWTexture2D<float4> dynImageGrid[][3], "
        "Sampler2D<float4> dynAlias[][3], "
        "int layer, int slot, float2 uv, int2 pixel)" in generated_code
    )
    assert (
        "void midForward(Sampler2D<float4> firstDyn[][3], "
        "Sampler2D<float4> firstFixed[3], "
        "RWTexture2D<float4> firstDynImages[][3], "
        "RWTexture2D<float4> firstFixedImages[3], "
        "int layer, int slot, float2 uv, int2 pixel)" in generated_code
    )
    assert (
        "void topForward(Sampler2D<float4> topFixedTex[3], "
        "RWTexture2D<float4> topDynImages[][3], "
        "Sampler2D<float4> topDynTex[][3], "
        "RWTexture2D<float4> topFixedImages[3], "
        "int layer, int slot, float2 uv, int2 pixel)" in generated_code
    )
    assert (
        "leafShuffle(firstDyn, firstFixedImages, firstFixed, firstDynImages, "
        "firstDyn, layer, slot, uv, pixel);" in generated_code
    )
    assert (
        "midForward(topDynTex, topFixedTex, topDynImages, topFixedImages, "
        "layer, slot, uv, pixel);" in generated_code
    )
    assert (
        "topForward(textureGrid[1], imageGrid, textureGrid, imageGrid[1], "
        "1, 2, uv, pixel);" in generated_code
    )
    assert "float4 sampledA = dynA[layer][slot].Sample(uv);" in generated_code
    assert "float4 sampledFixed = fixedTexRow[slot].Sample(uv);" in generated_code
    assert "float4 sampledAlias = dynAlias[layer][slot].Sample(uv);" in generated_code
    assert "float4 dynamicColor = dynImageGrid[layer][slot][pixel];" in generated_code
    assert "float4 fixedColor = fixedImageRow[slot][pixel];" in generated_code
    assert "dynImageGrid[layer][slot][pixel] = dynamicColor;" in generated_code
    assert "fixedImageRow[slot][pixel] = fixedColor;" in generated_code
    assert (
        "int2 dynSize = cgl_textureSize_sampler2D(dynA[layer][slot], 0);"
        in generated_code
    )
    assert (
        "int2 fixedTexSize = cgl_textureSize_sampler2D(fixedTexRow[slot], 0);"
        in generated_code
    )
    assert (
        "int2 aliasSize = cgl_textureSize_sampler2D(dynAlias[layer][slot], 0);"
        in generated_code
    )
    assert (
        "int2 dynamicImageSize = cgl_imageSize_image2D(dynImageGrid[layer][slot]);"
        in generated_code
    )
    assert (
        "int2 fixedImageSize = cgl_imageSize_image2D(fixedImageRow[slot]);"
        in generated_code
    )
    assert "texture(" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code
    assert "unsupported Slang" not in generated_code


def test_resource_calls_in_control_flow_emit_slang_operations():
    code = """
    shader ControlFlowResourceForwarding {
        sampler2d textureGrid[2][3];
        image2D imageGrid @rgba16f[2][3];

        vec4 sampleDynamic(
            sampler2d dynTex[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            vec4 sampled = texture(dynTex[layer][slot], uv);
            vec4 loaded = imageLoad(dynImages[layer][slot], pixel);
            ivec2 texSize = textureSize(dynTex[layer][slot], 0);
            ivec2 imageSizeValue = imageSize(dynImages[layer][slot]);
            return sampled + loaded;
        }

        vec4 sampleFixed(
            sampler2d fixedTex[3],
            image2D fixedImages[3] @rgba16f,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            vec4 sampled = texture(fixedTex[slot], uv);
            vec4 loaded = imageLoad(fixedImages[slot], pixel);
            ivec2 texSize = textureSize(fixedTex[slot], 0);
            ivec2 imageSizeValue = imageSize(fixedImages[slot]);
            return sampled + loaded;
        }

        vec4 chooseBranch(
            sampler2d dynTex[][3],
            sampler2d fixedTex[3],
            image2D dynImages[][3] @rgba16f,
            image2D fixedImages[3] @rgba16f,
            bool useFixed,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            vec4 result = sampleDynamic(dynTex, dynImages, layer, slot, uv, pixel);
            if (useFixed) {
                result = sampleFixed(fixedTex, fixedImages, slot, uv, pixel);
            }
            return result;
        }

        vec4 chooseReturn(
            sampler2d dynTex[][3],
            sampler2d fixedTex[3],
            image2D dynImages[][3] @rgba16f,
            image2D fixedImages[3] @rgba16f,
            bool useFixed,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            if (useFixed) {
                return sampleFixed(fixedTex, fixedImages, slot, uv, pixel);
            }
            return sampleDynamic(dynTex, dynImages, layer, slot, uv, pixel);
        }

        vec4 chooseTernary(
            sampler2d dynTex[][3],
            sampler2d fixedTex[3],
            image2D dynImages[][3] @rgba16f,
            image2D fixedImages[3] @rgba16f,
            bool useFixed,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            return useFixed
                ? sampleFixed(fixedTex, fixedImages, slot, uv, pixel)
                : sampleDynamic(dynTex, dynImages, layer, slot, uv, pixel);
        }

        compute {
            void main() {
                vec2 uv = vec2(0.5, 0.25);
                ivec2 pixel = ivec2(0, 0);
                vec4 branchValue = chooseBranch(
                    textureGrid,
                    textureGrid[1],
                    imageGrid,
                    imageGrid[1],
                    true,
                    1,
                    2,
                    uv,
                    pixel
                );
                vec4 returnValue = chooseReturn(
                    textureGrid,
                    textureGrid[1],
                    imageGrid,
                    imageGrid[1],
                    false,
                    1,
                    2,
                    uv,
                    pixel
                );
                vec4 ternaryValue = chooseTernary(
                    textureGrid,
                    textureGrid[1],
                    imageGrid,
                    imageGrid[1],
                    true,
                    1,
                    2,
                    uv,
                    pixel
                );
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "float4 chooseBranch(Sampler2D<float4> dynTex[][3], "
        "Sampler2D<float4> fixedTex[3], "
        "RWTexture2D<float4> dynImages[][3], "
        "RWTexture2D<float4> fixedImages[3], "
        "bool useFixed, int layer, int slot, float2 uv, int2 pixel)" in generated_code
    )
    assert (
        "float4 result = sampleDynamic(dynTex, dynImages, layer, slot, uv, pixel);"
        in generated_code
    )
    assert (
        "result = sampleFixed(fixedTex, fixedImages, slot, uv, pixel);"
        in generated_code
    )
    assert (
        "return sampleFixed(fixedTex, fixedImages, slot, uv, pixel);" in generated_code
    )
    assert (
        "return sampleDynamic(dynTex, dynImages, layer, slot, uv, pixel);"
        in generated_code
    )
    assert (
        "return (useFixed ? sampleFixed(fixedTex, fixedImages, slot, uv, pixel) : "
        "sampleDynamic(dynTex, dynImages, layer, slot, uv, pixel));" in generated_code
    )
    assert (
        "chooseBranch(textureGrid, textureGrid[1], imageGrid, imageGrid[1], "
        "true, 1, 2, uv, pixel);" in generated_code
    )
    assert (
        "chooseTernary(textureGrid, textureGrid[1], imageGrid, imageGrid[1], "
        "true, 1, 2, uv, pixel);" in generated_code
    )
    assert "float4 sampled = dynTex[layer][slot].Sample(uv);" in generated_code
    assert "float4 loaded = dynImages[layer][slot][pixel];" in generated_code
    assert "float4 sampled = fixedTex[slot].Sample(uv);" in generated_code
    assert "float4 loaded = fixedImages[slot][pixel];" in generated_code
    assert (
        "int2 texSize = cgl_textureSize_sampler2D(dynTex[layer][slot], 0);"
        in generated_code
    )
    assert (
        "int2 imageSizeValue = cgl_imageSize_image2D(dynImages[layer][slot]);"
        in generated_code
    )
    assert (
        "int2 texSize = cgl_textureSize_sampler2D(fixedTex[slot], 0);" in generated_code
    )
    assert (
        "int2 imageSizeValue = cgl_imageSize_image2D(fixedImages[slot]);"
        in generated_code
    )
    assert "texture(" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "unsupported Slang" not in generated_code


def test_resource_calls_in_loops_emit_slang_operations():
    code = """
    shader LoopResourceForwarding {
        sampler2d textureGrid[2][3];
        image2D imageGrid @rgba16f[2][3];

        vec4 sampleLoop(
            sampler2d dynTex[][3],
            sampler2d fixedTex[3],
            image2D dynImages[][3] @rgba16f,
            image2D fixedImages[3] @rgba16f,
            int layer,
            vec2 uv,
            ivec2 pixel
        ) {
            vec4 accum = vec4(0.0);
            for (int slot = 0; slot < 3; slot++) {
                if (slot == 1) {
                    continue;
                }
                vec4 sampled = texture(dynTex[layer][slot], uv);
                vec4 loaded = imageLoad(dynImages[layer][slot], pixel);
                ivec2 texSize = textureSize(dynTex[layer][slot], 0);
                ivec2 imageSizeValue = imageSize(dynImages[layer][slot]);
                accum = accum + sampled + loaded;
                if (slot == 2) {
                    break;
                }
            }
            int fixedSlot = 0;
            while (fixedSlot < 3) {
                accum = accum + texture(fixedTex[fixedSlot], uv);
                accum = accum + imageLoad(fixedImages[fixedSlot], pixel);
                ivec2 fixedTexSize = textureSize(fixedTex[fixedSlot], 0);
                ivec2 fixedImageSize = imageSize(fixedImages[fixedSlot]);
                fixedSlot++;
            }
            return accum;
        }

        compute {
            void main() {
                vec2 uv = vec2(0.5, 0.25);
                ivec2 pixel = ivec2(0, 0);
                vec4 value = sampleLoop(
                    textureGrid,
                    textureGrid[1],
                    imageGrid,
                    imageGrid[1],
                    1,
                    uv,
                    pixel
                );
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "float4 sampleLoop(Sampler2D<float4> dynTex[][3], "
        "Sampler2D<float4> fixedTex[3], "
        "RWTexture2D<float4> dynImages[][3], "
        "RWTexture2D<float4> fixedImages[3], int layer, float2 uv, int2 pixel)"
        in generated_code
    )
    assert "for (int slot = 0; slot < 3; slot++)" in generated_code
    assert "continue;" in generated_code
    assert "break;" in generated_code
    assert "while (fixedSlot < 3)" in generated_code
    assert "float4 sampled = dynTex[layer][slot].Sample(uv);" in generated_code
    assert "float4 loaded = dynImages[layer][slot][pixel];" in generated_code
    assert (
        "int2 texSize = cgl_textureSize_sampler2D(dynTex[layer][slot], 0);"
        in generated_code
    )
    assert (
        "int2 imageSizeValue = cgl_imageSize_image2D(dynImages[layer][slot]);"
        in generated_code
    )
    assert "accum = accum + fixedTex[fixedSlot].Sample(uv);" in generated_code
    assert "accum = accum + fixedImages[fixedSlot][pixel];" in generated_code
    assert (
        "int2 fixedTexSize = cgl_textureSize_sampler2D(fixedTex[fixedSlot], 0);"
        in generated_code
    )
    assert (
        "int2 fixedImageSize = cgl_imageSize_image2D(fixedImages[fixedSlot]);"
        in generated_code
    )
    assert (
        "sampleLoop(textureGrid, textureGrid[1], imageGrid, imageGrid[1], "
        "1, uv, pixel);" in generated_code
    )
    assert "texture(" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "unsupported Slang" not in generated_code


def test_nested_loop_resource_indices_emit_slang_operations():
    code = """
    shader NestedLoopResourceForwarding {
        sampler2d textureGrid[2][3];
        image2D imageGrid @rgba16f[2][3];

        vec4 sampleNestedLoops(
            sampler2d dynTex[][3],
            image2D dynImages[][3] @rgba16f,
            vec2 uv,
            ivec2 pixel
        ) {
            vec4 accum = vec4(0.0);
            for (int layer = 0; layer < 2; layer++) {
                if (layer == 1) {
                    break;
                }
                for (int slot = 0; slot < 3; slot++) {
                    if (slot == 1) {
                        continue;
                    }
                    vec4 sampled = texture(dynTex[layer][slot], uv);
                    vec4 loaded = imageLoad(dynImages[layer][slot], pixel);
                    imageStore(dynImages[layer][slot], pixel, loaded);
                    ivec2 texSize = textureSize(dynTex[layer][slot], 0);
                    ivec2 imageSizeValue = imageSize(dynImages[layer][slot]);
                    accum = accum + sampled + loaded;
                }
            }
            return accum;
        }

        compute {
            void main() {
                vec2 uv = vec2(0.5, 0.25);
                ivec2 pixel = ivec2(0, 0);
                vec4 value = sampleNestedLoops(textureGrid, imageGrid, uv, pixel);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "float4 sampleNestedLoops(Sampler2D<float4> dynTex[][3], "
        "RWTexture2D<float4> dynImages[][3], float2 uv, int2 pixel)" in generated_code
    )
    assert "for (int layer = 0; layer < 2; layer++)" in generated_code
    assert "for (int slot = 0; slot < 3; slot++)" in generated_code
    assert "break;" in generated_code
    assert "continue;" in generated_code
    assert "float4 sampled = dynTex[layer][slot].Sample(uv);" in generated_code
    assert "float4 loaded = dynImages[layer][slot][pixel];" in generated_code
    assert "dynImages[layer][slot][pixel] = loaded;" in generated_code
    assert (
        "int2 texSize = cgl_textureSize_sampler2D(dynTex[layer][slot], 0);"
        in generated_code
    )
    assert (
        "int2 imageSizeValue = cgl_imageSize_image2D(dynImages[layer][slot]);"
        in generated_code
    )
    assert "sampleNestedLoops(textureGrid, imageGrid, uv, pixel);" in generated_code
    assert "texture(" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code
    assert "unsupported Slang" not in generated_code


def test_struct_and_scalar_params_preserve_slang_resource_argument_order():
    code = """
    struct SampleParams {
        vec2 uv;
        ivec2 pixel;
        int layer;
        int slot;
    };

    shader PackedResourceForwarding {
        sampler2d textureGrid[2][3];
        image2D imageGrid @rgba16f[2][3];

        vec4 samplePacked(
            sampler2d dynTex[][3],
            SampleParams params,
            image2D dynImages[][3] @rgba16f,
            float weight,
            sampler2d fixedTex[3],
            image2D fixedImages[3] @rgba16f
        ) {
            vec4 dynamicSample = texture(
                dynTex[params.layer][params.slot],
                params.uv
            );
            vec4 fixedSample = texture(fixedTex[params.slot], params.uv);
            vec4 dynamicColor = imageLoad(
                dynImages[params.layer][params.slot],
                params.pixel
            );
            vec4 fixedColor = imageLoad(fixedImages[params.slot], params.pixel);
            ivec2 dynSize = textureSize(dynTex[params.layer][params.slot], 0);
            ivec2 fixedSize = textureSize(fixedTex[params.slot], 0);
            ivec2 dynImageSize = imageSize(dynImages[params.layer][params.slot]);
            ivec2 fixedImageSize = imageSize(fixedImages[params.slot]);
            vec4 combined = dynamicSample + fixedSample + dynamicColor + fixedColor;
            return combined * weight;
        }

        vec4 passPacked(
            SampleParams params,
            sampler2d firstDyn[][3],
            float weight,
            image2D firstImages[][3] @rgba16f,
            sampler2d fixedTex[3],
            image2D fixedImages[3] @rgba16f
        ) {
            return samplePacked(
                firstDyn,
                params,
                firstImages,
                weight,
                fixedTex,
                fixedImages
            );
        }

        compute {
            void main() {
                SampleParams params;
                params.uv = vec2(0.5, 0.25);
                params.pixel = ivec2(0, 0);
                params.layer = 1;
                params.slot = 2;
                vec4 value = passPacked(
                    params,
                    textureGrid,
                    0.5,
                    imageGrid,
                    textureGrid[1],
                    imageGrid[1]
                );
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "struct SampleParams" in generated_code
    assert (
        "float4 samplePacked(Sampler2D<float4> dynTex[][3], "
        "SampleParams params, RWTexture2D<float4> dynImages[][3], "
        "float weight, Sampler2D<float4> fixedTex[3], "
        "RWTexture2D<float4> fixedImages[3])" in generated_code
    )
    assert (
        "float4 passPacked(SampleParams params, "
        "Sampler2D<float4> firstDyn[][3], float weight, "
        "RWTexture2D<float4> firstImages[][3], "
        "Sampler2D<float4> fixedTex[3], "
        "RWTexture2D<float4> fixedImages[3])" in generated_code
    )
    assert (
        "return samplePacked(firstDyn, params, firstImages, weight, fixedTex, "
        "fixedImages);" in generated_code
    )
    assert (
        "passPacked(params, textureGrid, 0.5, imageGrid, textureGrid[1], "
        "imageGrid[1]);" in generated_code
    )
    assert (
        "float4 dynamicSample = dynTex[params.layer][params.slot].Sample(params.uv);"
        in generated_code
    )
    assert (
        "float4 fixedSample = fixedTex[params.slot].Sample(params.uv);"
        in generated_code
    )
    assert (
        "float4 dynamicColor = dynImages[params.layer][params.slot][params.pixel];"
        in generated_code
    )
    assert (
        "float4 fixedColor = fixedImages[params.slot][params.pixel];" in generated_code
    )
    assert (
        "int2 dynSize = cgl_textureSize_sampler2D"
        "(dynTex[params.layer][params.slot], 0);" in generated_code
    )
    assert (
        "int2 fixedSize = cgl_textureSize_sampler2D(fixedTex[params.slot], 0);"
        in generated_code
    )
    assert (
        "int2 dynImageSize = cgl_imageSize_image2D"
        "(dynImages[params.layer][params.slot]);" in generated_code
    )
    assert (
        "int2 fixedImageSize = cgl_imageSize_image2D(fixedImages[params.slot]);"
        in generated_code
    )
    assert "texture(" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "unsupported Slang" not in generated_code


def test_resource_values_in_struct_returns_emit_slang_operations():
    code = """
    struct SampleResult {
        vec4 sampled;
        vec4 loaded;
        ivec2 texSize;
        ivec2 imageSizeValue;
    };

    shader Resources {
        sampler2d textureGrid[2][3];
        image2D imageGrid @rgba16f[2][3];

        SampleResult buildAssigned(
            sampler2d dynTex[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            SampleResult result;
            result.sampled = texture(dynTex[layer][slot], uv);
            result.loaded = imageLoad(dynImages[layer][slot], pixel);
            result.texSize = textureSize(dynTex[layer][slot], 0);
            result.imageSizeValue = imageSize(dynImages[layer][slot]);
            return result;
        }

        SampleResult buildConstructed(
            sampler2d dynTex[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            return SampleResult(
                texture(dynTex[layer][slot], uv),
                imageLoad(dynImages[layer][slot], pixel),
                textureSize(dynTex[layer][slot], 0),
                imageSize(dynImages[layer][slot])
            );
        }

        vec4 consumeResults(
            sampler2d dynTex[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            SampleResult assigned = buildAssigned(
                dynTex,
                dynImages,
                layer,
                slot,
                uv,
                pixel
            );
            SampleResult constructed = buildConstructed(
                dynTex,
                dynImages,
                layer,
                slot,
                uv,
                pixel
            );
            return assigned.sampled + assigned.loaded +
                constructed.sampled + constructed.loaded;
        }

        compute {
            void main() {
                vec2 uv = vec2(0.5, 0.25);
                ivec2 pixel = ivec2(0, 0);
                vec4 value = consumeResults(textureGrid, imageGrid, 1, 2, uv, pixel);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "struct SampleResult" in generated_code
    assert (
        "SampleResult buildAssigned(Sampler2D<float4> dynTex[][3], "
        "RWTexture2D<float4> dynImages[][3], int layer, int slot, "
        "float2 uv, int2 pixel)" in generated_code
    )
    assert (
        "SampleResult buildConstructed(Sampler2D<float4> dynTex[][3], "
        "RWTexture2D<float4> dynImages[][3], int layer, int slot, "
        "float2 uv, int2 pixel)" in generated_code
    )
    assert (
        "float4 consumeResults(Sampler2D<float4> dynTex[][3], "
        "RWTexture2D<float4> dynImages[][3], int layer, int slot, "
        "float2 uv, int2 pixel)" in generated_code
    )
    assert "result.sampled = dynTex[layer][slot].Sample(uv);" in generated_code
    assert "result.loaded = dynImages[layer][slot][pixel];" in generated_code
    assert (
        "result.texSize = cgl_textureSize_sampler2D(dynTex[layer][slot], 0);"
        in generated_code
    )
    assert (
        "result.imageSizeValue = cgl_imageSize_image2D(dynImages[layer][slot]);"
        in generated_code
    )
    assert (
        "return SampleResult(dynTex[layer][slot].Sample(uv), "
        "dynImages[layer][slot][pixel], "
        "cgl_textureSize_sampler2D(dynTex[layer][slot], 0), "
        "cgl_imageSize_image2D(dynImages[layer][slot]));" in generated_code
    )
    assert (
        "SampleResult assigned = buildAssigned(dynTex, dynImages, layer, slot, "
        "uv, pixel);" in generated_code
    )
    assert (
        "SampleResult constructed = buildConstructed(dynTex, dynImages, layer, slot, "
        "uv, pixel);" in generated_code
    )
    assert "consumeResults(textureGrid, imageGrid, 1, 2, uv, pixel);" in generated_code
    assert "texture(" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "unsupported Slang" not in generated_code


def test_nested_struct_resource_assignments_emit_slang_operations():
    code = """
    struct SampleResult {
        vec4 sampled;
        vec4 loaded;
        ivec2 texSize;
        ivec2 imageSizeValue;
    };

    struct SampleEnvelope {
        SampleResult result;
        vec4 bias;
    };

    shader Resources {
        sampler2d textureGrid[2][3];
        image2D imageGrid @rgba16f[2][3];

        SampleEnvelope buildNestedAssigned(
            sampler2d dynTex[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            SampleEnvelope envelope;
            envelope.result.sampled = texture(dynTex[layer][slot], uv);
            envelope.result.loaded = imageLoad(dynImages[layer][slot], pixel);
            envelope.result.texSize = textureSize(dynTex[layer][slot], 0);
            envelope.result.imageSizeValue = imageSize(dynImages[layer][slot]);
            envelope.bias = texture(dynTex[layer][slot], uv) +
                imageLoad(dynImages[layer][slot], pixel);
            return envelope;
        }

        vec4 consumeNested(
            sampler2d dynTex[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            SampleEnvelope envelope = buildNestedAssigned(
                dynTex,
                dynImages,
                layer,
                slot,
                uv,
                pixel
            );
            return envelope.result.sampled + envelope.result.loaded + envelope.bias;
        }

        compute {
            void main() {
                vec2 uv = vec2(0.5, 0.25);
                ivec2 pixel = ivec2(0, 0);
                vec4 value = consumeNested(textureGrid, imageGrid, 1, 2, uv, pixel);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "struct SampleEnvelope" in generated_code
    assert (
        "SampleEnvelope buildNestedAssigned(Sampler2D<float4> dynTex[][3], "
        "RWTexture2D<float4> dynImages[][3], int layer, int slot, "
        "float2 uv, int2 pixel)" in generated_code
    )
    assert "envelope.result.sampled = dynTex[layer][slot].Sample(uv);" in generated_code
    assert "envelope.result.loaded = dynImages[layer][slot][pixel];" in generated_code
    assert (
        "envelope.result.texSize = "
        "cgl_textureSize_sampler2D(dynTex[layer][slot], 0);" in generated_code
    )
    assert (
        "envelope.result.imageSizeValue = "
        "cgl_imageSize_image2D(dynImages[layer][slot]);" in generated_code
    )
    assert (
        "envelope.bias = dynTex[layer][slot].Sample(uv) + "
        "dynImages[layer][slot][pixel];" in generated_code
    )
    assert (
        "SampleEnvelope envelope = buildNestedAssigned("
        "dynTex, dynImages, layer, slot, uv, pixel);" in generated_code
    )
    assert "consumeNested(textureGrid, imageGrid, 1, 2, uv, pixel);" in generated_code
    assert "texture(" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "unsupported Slang" not in generated_code


def test_struct_member_array_resource_assignments_emit_slang_operations():
    code = """
    struct SampleResult {
        vec4 sampled;
        vec4 loaded;
        ivec2 texSize;
        ivec2 imageSizeValue;
    };

    struct SampleEnvelope {
        SampleResult results[3];
        vec4 bias;
    };

    shader Resources {
        sampler2d textureGrid[2][3];
        image2D imageGrid @rgba16f[2][3];

        SampleEnvelope buildArrayEnvelope(
            sampler2d dynTex[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            SampleEnvelope envelope;
            envelope.results[slot].sampled = texture(dynTex[layer][slot], uv);
            envelope.results[slot].loaded = imageLoad(dynImages[layer][slot], pixel);
            envelope.results[slot].texSize = textureSize(dynTex[layer][slot], 0);
            envelope.results[slot].imageSizeValue = imageSize(dynImages[layer][slot]);
            envelope.bias = envelope.results[slot].sampled +
                envelope.results[slot].loaded;
            return envelope;
        }

        vec4 consumeArrayEnvelope(
            sampler2d dynTex[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            SampleEnvelope envelope = buildArrayEnvelope(
                dynTex,
                dynImages,
                layer,
                slot,
                uv,
                pixel
            );
            return envelope.results[slot].sampled +
                envelope.results[slot].loaded + envelope.bias;
        }

        compute {
            void main() {
                vec2 uv = vec2(0.5, 0.25);
                ivec2 pixel = ivec2(0, 0);
                vec4 value = consumeArrayEnvelope(
                    textureGrid,
                    imageGrid,
                    1,
                    2,
                    uv,
                    pixel
                );
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "SampleResult results[3];" in generated_code
    assert (
        "SampleEnvelope buildArrayEnvelope(Sampler2D<float4> dynTex[][3], "
        "RWTexture2D<float4> dynImages[][3], int layer, int slot, "
        "float2 uv, int2 pixel)" in generated_code
    )
    assert (
        "envelope.results[slot].sampled = dynTex[layer][slot].Sample(uv);"
        in generated_code
    )
    assert (
        "envelope.results[slot].loaded = dynImages[layer][slot][pixel];"
        in generated_code
    )
    assert (
        "envelope.results[slot].texSize = "
        "cgl_textureSize_sampler2D(dynTex[layer][slot], 0);" in generated_code
    )
    assert (
        "envelope.results[slot].imageSizeValue = "
        "cgl_imageSize_image2D(dynImages[layer][slot]);" in generated_code
    )
    assert (
        "SampleEnvelope envelope = buildArrayEnvelope("
        "dynTex, dynImages, layer, slot, uv, pixel);" in generated_code
    )
    assert (
        "consumeArrayEnvelope(textureGrid, imageGrid, 1, 2, uv, pixel);"
        in generated_code
    )
    assert "texture(" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "unsupported Slang" not in generated_code


def test_control_flow_struct_resource_assignments_emit_slang_operations():
    code = """
    struct SampleResult {
        vec4 sampled;
        vec4 loaded;
        ivec2 texSize;
        ivec2 imageSizeValue;
    };

    struct SampleEnvelope {
        SampleResult result;
        vec4 accum;
        int chosen;
    };

    shader Resources {
        sampler2d textureGrid[2][3];
        image2D imageGrid @rgba16f[2][3];

        SampleEnvelope fillControlled(
            sampler2d dynTex[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            bool useAlternate,
            vec2 uv,
            ivec2 pixel
        ) {
            SampleEnvelope envelope;
            envelope.accum = vec4(0.0);
            envelope.chosen = slot;

            if (useAlternate) {
                envelope.result.sampled = texture(dynTex[layer][slot], uv);
                envelope.result.loaded = imageLoad(dynImages[layer][slot], pixel);
            } else {
                int fallback = 0;
                envelope.result.sampled = texture(dynTex[fallback][slot], uv);
                envelope.result.loaded = imageLoad(dynImages[fallback][slot], pixel);
            }

            for (int i = 0; i < 3; i++) {
                if (i == 1) {
                    continue;
                }
                envelope.result.texSize = textureSize(dynTex[layer][i], 0);
                envelope.result.imageSizeValue = imageSize(dynImages[layer][i]);
                envelope.accum = envelope.accum + texture(dynTex[layer][i], uv);
                if (i == slot) {
                    break;
                }
            }

            return envelope;
        }

        vec4 consumeControlled(
            sampler2d dynTex[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            SampleEnvelope envelope = fillControlled(
                dynTex,
                dynImages,
                layer,
                slot,
                true,
                uv,
                pixel
            );
            return envelope.result.sampled + envelope.result.loaded +
                envelope.accum;
        }

        compute {
            void main() {
                vec2 uv = vec2(0.5, 0.25);
                ivec2 pixel = ivec2(0, 0);
                vec4 value = consumeControlled(
                    textureGrid,
                    imageGrid,
                    1,
                    2,
                    uv,
                    pixel
                );
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "SampleEnvelope fillControlled(Sampler2D<float4> dynTex[][3], "
        "RWTexture2D<float4> dynImages[][3], int layer, int slot, "
        "bool useAlternate, float2 uv, int2 pixel)" in generated_code
    )
    assert "if (useAlternate)" in generated_code
    assert "else" in generated_code
    assert "for (int i = 0; i < 3; i++)" in generated_code
    assert "continue;" in generated_code
    assert "break;" in generated_code
    assert "envelope.accum = float4(0.0);" in generated_code
    assert "envelope.chosen = slot;" in generated_code
    assert "envelope.result.sampled = dynTex[layer][slot].Sample(uv);" in generated_code
    assert "envelope.result.loaded = dynImages[layer][slot][pixel];" in generated_code
    assert (
        "envelope.result.sampled = dynTex[fallback][slot].Sample(uv);" in generated_code
    )
    assert (
        "envelope.result.loaded = dynImages[fallback][slot][pixel];" in generated_code
    )
    assert (
        "envelope.result.texSize = cgl_textureSize_sampler2D(dynTex[layer][i], 0);"
        in generated_code
    )
    assert (
        "envelope.result.imageSizeValue = cgl_imageSize_image2D(dynImages[layer][i]);"
        in generated_code
    )
    assert (
        "envelope.accum = envelope.accum + dynTex[layer][i].Sample(uv);"
        in generated_code
    )
    assert (
        "fillControlled(dynTex, dynImages, layer, slot, true, uv, pixel);"
        in generated_code
    )
    assert (
        "consumeControlled(textureGrid, imageGrid, 1, 2, uv, pixel);" in generated_code
    )
    assert "texture(" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "unsupported Slang" not in generated_code


def test_struct_parameter_resource_values_round_trip_slang_operations():
    code = """
    struct SampleResult {
        vec4 sampled;
        vec4 loaded;
        ivec2 texSize;
        ivec2 imageSizeValue;
    };

    shader Resources {
        sampler2d textureGrid[2][3];
        image2D imageGrid @rgba16f[2][3];

        SampleResult buildResourceResult(
            sampler2d dynTex[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            SampleResult result;
            result.sampled = texture(dynTex[layer][slot], uv);
            result.loaded = imageLoad(dynImages[layer][slot], pixel);
            result.texSize = textureSize(dynTex[layer][slot], 0);
            result.imageSizeValue = imageSize(dynImages[layer][slot]);
            return result;
        }

        SampleResult adjustResult(SampleResult payload, float weight) {
            payload.sampled = payload.sampled * weight;
            payload.loaded = payload.loaded + payload.sampled;
            return payload;
        }

        vec4 consumeAdjusted(
            sampler2d dynTex[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            SampleResult raw = buildResourceResult(
                dynTex,
                dynImages,
                layer,
                slot,
                uv,
                pixel
            );
            SampleResult adjusted = adjustResult(raw, 0.5);
            return adjusted.sampled + adjusted.loaded;
        }

        compute {
            void main() {
                vec2 uv = vec2(0.5, 0.25);
                ivec2 pixel = ivec2(0, 0);
                vec4 value = consumeAdjusted(textureGrid, imageGrid, 1, 2, uv, pixel);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "SampleResult buildResourceResult(Sampler2D<float4> dynTex[][3], "
        "RWTexture2D<float4> dynImages[][3], int layer, int slot, "
        "float2 uv, int2 pixel)" in generated_code
    )
    assert (
        "SampleResult adjustResult(SampleResult payload, float weight)"
        in generated_code
    )
    assert (
        "float4 consumeAdjusted(Sampler2D<float4> dynTex[][3], "
        "RWTexture2D<float4> dynImages[][3], int layer, int slot, "
        "float2 uv, int2 pixel)" in generated_code
    )
    assert "payload.sampled = payload.sampled * weight;" in generated_code
    assert "payload.loaded = payload.loaded + payload.sampled;" in generated_code
    assert "return payload;" in generated_code
    assert (
        "SampleResult raw = buildResourceResult(dynTex, dynImages, layer, slot, "
        "uv, pixel);" in generated_code
    )
    assert "SampleResult adjusted = adjustResult(raw, 0.5);" in generated_code
    assert "consumeAdjusted(textureGrid, imageGrid, 1, 2, uv, pixel);" in generated_code
    assert "texture(" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "unsupported Slang" not in generated_code


def test_mutable_scalar_and_array_params_emit_slang_updates():
    code = """
    shader MutableParams {
        float bumpScalar(float weight) {
            weight = weight + 1.0;
            return weight * 2.0;
        }

        float bumpArray(float values[3], int slot) {
            values[slot] = values[slot] + 1.0;
            values[0] = values[slot] * 0.5;
            return values[slot] + values[0];
        }

        compute {
            void main() {
                float values[3];
                values[0] = 1.0;
                values[1] = 2.0;
                values[2] = 3.0;
                float a = bumpScalar(0.5);
                float b = bumpArray(values, 1);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float bumpScalar(float weight)" in generated_code
    assert "weight = weight + 1.0;" in generated_code
    assert "return weight * 2.0;" in generated_code
    assert "float bumpArray(float values[3], int slot)" in generated_code
    assert "values[slot] = values[slot] + 1.0;" in generated_code
    assert "values[0] = values[slot] * 0.5;" in generated_code
    assert "return values[slot] + values[0];" in generated_code
    assert "float values[3];" in generated_code
    assert "float a = bumpScalar(0.5);" in generated_code
    assert "float b = bumpArray(values, 1);" in generated_code
    assert "unsupported Slang" not in generated_code


def test_array_literals_emit_slang_brace_initializers():
    code = """
    float globalWeights[4] = {1.0, 2.0};

    float pickScalar(int index) {
        float values[4] = {1.0, 2.0, 3.0, 4.0};
        return values[index];
    }

    vec3 pickColor(int index) {
        vec3 colors[2] = {vec3(1.0, 2.0, 3.0), vec3(4.0, 5.0, 6.0)};
        return colors[index];
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float globalWeights[4] = {1.0, 2.0};" in generated_code
    assert "float values[4] = {1.0, 2.0, 3.0, 4.0};" in generated_code
    assert (
        "float3 colors[2] = {float3(1.0, 2.0, 3.0), " "float3(4.0, 5.0, 6.0)};"
    ) in generated_code
    assert "ArrayLiteralNode" not in generated_code


def test_nested_array_and_struct_member_params_emit_slang_updates():
    code = """
    struct Payload {
        float values[3];
        float bias;
    };

    shader NestedMutableParams {
        float bumpNested(float values[2][3], int row, int col) {
            values[row][col] = values[row][col] + 1.0;
            values[0][col] = values[row][col] * 0.5;
            return values[row][col] + values[0][col];
        }

        float bumpPayload(Payload payload, int slot) {
            payload.values[slot] = payload.values[slot] + payload.bias;
            payload.bias = payload.values[slot] * 0.5;
            return payload.values[slot] + payload.bias;
        }

        compute {
            void main() {
                float grid[2][3];
                grid[0][0] = 1.0;
                grid[1][2] = 3.0;
                Payload payload;
                payload.values[0] = 2.0;
                payload.values[1] = 4.0;
                payload.bias = 0.25;
                float a = bumpNested(grid, 1, 2);
                float b = bumpPayload(payload, 1);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "struct Payload" in generated_code
    assert "float values[3];" in generated_code
    assert "float bumpNested(float values[2][3], int row, int col)" in generated_code
    assert "values[row][col] = values[row][col] + 1.0;" in generated_code
    assert "values[0][col] = values[row][col] * 0.5;" in generated_code
    assert "return values[row][col] + values[0][col];" in generated_code
    assert "float bumpPayload(Payload payload, int slot)" in generated_code
    assert (
        "payload.values[slot] = payload.values[slot] + payload.bias;" in generated_code
    )
    assert "payload.bias = payload.values[slot] * 0.5;" in generated_code
    assert "return payload.values[slot] + payload.bias;" in generated_code
    assert "float grid[2][3];" in generated_code
    assert "float a = bumpNested(grid, 1, 2);" in generated_code
    assert "float b = bumpPayload(payload, 1);" in generated_code
    assert "unsupported Slang" not in generated_code


def test_returned_nested_struct_params_emit_slang_updates():
    code = """
    struct InnerPayload {
        float values[3];
        float bias;
    };

    struct OuterPayload {
        InnerPayload inner;
        float scale;
    };

    shader ReturnedParamPayloads {
        OuterPayload makeOuter(float first, float second, float bias, float scale) {
            OuterPayload outer;
            outer.inner.values[0] = first;
            outer.inner.values[1] = second;
            outer.inner.values[2] = first + second;
            outer.inner.bias = bias;
            outer.scale = scale;
            return outer;
        }

        OuterPayload adjustOuter(OuterPayload payload, int slot) {
            payload.inner.values[slot] = payload.inner.values[slot] + payload.inner.bias;
            payload.inner.values[0] = payload.inner.values[slot] * payload.scale;
            payload.scale = payload.inner.values[0] + payload.inner.bias;
            return payload;
        }

        float consumeAdjustedOuter(int slot) {
            OuterPayload adjusted = adjustOuter(makeOuter(1.0, 2.0, 0.25, 4.0), slot);
            return adjusted.inner.values[slot] + adjusted.inner.values[0] + adjusted.scale;
        }

        compute {
            void main() {
                float value = consumeAdjustedOuter(1);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "OuterPayload adjustOuter(OuterPayload payload, int slot)" in generated_code
    assert (
        "payload.inner.values[slot] = payload.inner.values[slot] + payload.inner.bias;"
        in generated_code
    )
    assert (
        "payload.inner.values[0] = payload.inner.values[slot] * payload.scale;"
        in generated_code
    )
    assert (
        "payload.scale = payload.inner.values[0] + payload.inner.bias;"
        in generated_code
    )
    assert "return payload;" in generated_code
    assert "float consumeAdjustedOuter(int slot)" in generated_code
    assert (
        "OuterPayload adjusted = adjustOuter(makeOuter(1.0, 2.0, 0.25, 4.0), slot);"
        in generated_code
    )
    assert (
        "return adjusted.inner.values[slot] + adjusted.inner.values[0] + "
        "adjusted.scale;" in generated_code
    )
    assert "float value = consumeAdjustedOuter(1);" in generated_code
    assert "unsupported Slang" not in generated_code


def test_conditional_returned_nested_structs_emit_slang_expressions():
    code = """
    struct InnerPayload {
        float values[3];
        float bias;
    };

    struct OuterPayload {
        InnerPayload inner;
        float scale;
    };

    shader ConditionalReturnedPayloads {
        OuterPayload makeOuter(float first, float second, float bias, float scale) {
            OuterPayload outer;
            outer.inner.values[0] = first;
            outer.inner.values[1] = second;
            outer.inner.values[2] = first + second;
            outer.inner.bias = bias;
            outer.scale = scale;
            return outer;
        }

        OuterPayload chooseBranch(bool useSecond) {
            if (useSecond) {
                return makeOuter(3.0, 4.0, 0.5, 5.0);
            }
            return makeOuter(1.0, 2.0, 0.25, 4.0);
        }

        OuterPayload chooseTernary(bool useSecond) {
            return useSecond
                ? makeOuter(3.0, 4.0, 0.5, 5.0)
                : makeOuter(1.0, 2.0, 0.25, 4.0);
        }

        float consumeConditional(int slot, bool useSecond) {
            OuterPayload branchPayload = chooseBranch(useSecond);
            OuterPayload ternaryPayload = chooseTernary(useSecond);
            ternaryPayload.inner.values[slot] =
                ternaryPayload.inner.values[slot] + ternaryPayload.inner.bias;
            branchPayload.scale =
                branchPayload.inner.values[slot] + ternaryPayload.scale;
            return branchPayload.scale + ternaryPayload.inner.values[slot];
        }

        compute {
            void main() {
                float value = consumeConditional(1, true);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "OuterPayload chooseBranch(bool useSecond)" in generated_code
    assert "return makeOuter(3.0, 4.0, 0.5, 5.0);" in generated_code
    assert "return makeOuter(1.0, 2.0, 0.25, 4.0);" in generated_code
    assert "OuterPayload chooseTernary(bool useSecond)" in generated_code
    assert (
        "return (useSecond ? makeOuter(3.0, 4.0, 0.5, 5.0) : "
        "makeOuter(1.0, 2.0, 0.25, 4.0));" in generated_code
    )
    assert "OuterPayload branchPayload = chooseBranch(useSecond);" in generated_code
    assert "OuterPayload ternaryPayload = chooseTernary(useSecond);" in generated_code
    assert (
        "ternaryPayload.inner.values[slot] = "
        "ternaryPayload.inner.values[slot] + ternaryPayload.inner.bias;"
        in generated_code
    )
    assert (
        "branchPayload.scale = branchPayload.inner.values[slot] + "
        "ternaryPayload.scale;" in generated_code
    )
    assert (
        "return branchPayload.scale + ternaryPayload.inner.values[slot];"
        in generated_code
    )
    assert "float value = consumeConditional(1, true);" in generated_code
    assert "unsupported Slang" not in generated_code


def test_temporary_struct_array_member_reads_emit_slang_expressions():
    code = """
    struct Payload {
        float values[3];
        float bias;
    };

    shader TemporaryPayloads {
        Payload makePayload(float first, float second, float bias) {
            Payload payload;
            payload.values[0] = first;
            payload.values[1] = second;
            payload.values[2] = first + second;
            payload.bias = bias;
            return payload;
        }

        float readTemporary(int slot) {
            float dynamicValue = makePayload(1.0, 2.0, 0.25).values[slot];
            float fixedValue = makePayload(3.0, 4.0, 0.5).values[0];
            float biasValue = makePayload(5.0, 6.0, 0.75).bias;
            return dynamicValue + fixedValue + biasValue;
        }

        compute {
            void main() {
                float value = readTemporary(1);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "Payload makePayload(float first, float second, float bias)" in generated_code
    )
    assert "payload.values[2] = first + second;" in generated_code
    assert "float readTemporary(int slot)" in generated_code
    assert (
        "float dynamicValue = makePayload(1.0, 2.0, 0.25).values[slot];"
        in generated_code
    )
    assert "float fixedValue = makePayload(3.0, 4.0, 0.5).values[0];" in generated_code
    assert "float biasValue = makePayload(5.0, 6.0, 0.75).bias;" in generated_code
    assert "return dynamicValue + fixedValue + biasValue;" in generated_code
    assert "float value = readTemporary(1);" in generated_code
    assert "unsupported Slang" not in generated_code


def test_nested_temporary_struct_array_member_reads_emit_slang_expressions():
    code = """
    struct InnerPayload {
        float values[3];
        float bias;
    };

    struct OuterPayload {
        InnerPayload inner;
        float scale;
    };

    shader NestedTemporaryPayloads {
        OuterPayload makeOuter(float first, float second, float bias, float scale) {
            OuterPayload outer;
            outer.inner.values[0] = first;
            outer.inner.values[1] = second;
            outer.inner.values[2] = first + second;
            outer.inner.bias = bias;
            outer.scale = scale;
            return outer;
        }

        float readNestedTemporary(int slot) {
            float dynamicValue = makeOuter(1.0, 2.0, 0.25, 4.0).inner.values[slot];
            float fixedValue = makeOuter(3.0, 4.0, 0.5, 5.0).inner.values[0];
            float biasValue = makeOuter(5.0, 6.0, 0.75, 6.0).inner.bias;
            float scaleValue = makeOuter(7.0, 8.0, 1.0, 9.0).scale;
            return dynamicValue + fixedValue + biasValue + scaleValue;
        }

        compute {
            void main() {
                float value = readNestedTemporary(1);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "struct InnerPayload" in generated_code
    assert "struct OuterPayload" in generated_code
    assert (
        "OuterPayload makeOuter(float first, float second, float bias, float scale)"
        in generated_code
    )
    assert "outer.inner.values[2] = first + second;" in generated_code
    assert "outer.inner.bias = bias;" in generated_code
    assert "outer.scale = scale;" in generated_code
    assert "float readNestedTemporary(int slot)" in generated_code
    assert (
        "float dynamicValue = "
        "makeOuter(1.0, 2.0, 0.25, 4.0).inner.values[slot];" in generated_code
    )
    assert (
        "float fixedValue = makeOuter(3.0, 4.0, 0.5, 5.0).inner.values[0];"
        in generated_code
    )
    assert (
        "float biasValue = makeOuter(5.0, 6.0, 0.75, 6.0).inner.bias;" in generated_code
    )
    assert "float scaleValue = makeOuter(7.0, 8.0, 1.0, 9.0).scale;" in generated_code
    assert (
        "return dynamicValue + fixedValue + biasValue + scaleValue;" in generated_code
    )
    assert "float value = readNestedTemporary(1);" in generated_code
    assert "unsupported Slang" not in generated_code


def test_returned_local_nested_struct_array_writes_emit_slang_expressions():
    code = """
    struct InnerPayload {
        float values[3];
        float bias;
    };

    struct OuterPayload {
        InnerPayload inner;
        float scale;
    };

    shader LocalReturnedPayloads {
        OuterPayload makeOuter(float first, float second, float bias, float scale) {
            OuterPayload outer;
            outer.inner.values[0] = first;
            outer.inner.values[1] = second;
            outer.inner.values[2] = first + second;
            outer.inner.bias = bias;
            outer.scale = scale;
            return outer;
        }

        float mutateReturnedLocal(int slot) {
            OuterPayload outer = makeOuter(1.0, 2.0, 0.25, 4.0);
            outer.inner.values[slot] = outer.inner.values[slot] + outer.inner.bias;
            outer.inner.values[0] = outer.inner.values[slot] * outer.scale;
            outer.scale = outer.inner.values[0] + outer.inner.bias;
            return outer.inner.values[slot] + outer.inner.values[0] + outer.scale;
        }

        compute {
            void main() {
                float value = mutateReturnedLocal(1);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "OuterPayload outer = makeOuter(1.0, 2.0, 0.25, 4.0);" in generated_code
    assert (
        "outer.inner.values[slot] = outer.inner.values[slot] + outer.inner.bias;"
        in generated_code
    )
    assert (
        "outer.inner.values[0] = outer.inner.values[slot] * outer.scale;"
        in generated_code
    )
    assert "outer.scale = outer.inner.values[0] + outer.inner.bias;" in generated_code
    assert (
        "return outer.inner.values[slot] + outer.inner.values[0] + outer.scale;"
        in generated_code
    )
    assert "float value = mutateReturnedLocal(1);" in generated_code
    assert "unsupported Slang" not in generated_code


def test_returned_local_nested_struct_array_writes_in_control_flow_emit_slang():
    code = """
    struct InnerPayload {
        float values[3];
        float bias;
    };

    struct OuterPayload {
        InnerPayload inner;
        float scale;
    };

    shader ControlFlowReturnedPayloads {
        OuterPayload makeOuter(float first, float second, float bias, float scale) {
            OuterPayload outer;
            outer.inner.values[0] = first;
            outer.inner.values[1] = second;
            outer.inner.values[2] = first + second;
            outer.inner.bias = bias;
            outer.scale = scale;
            return outer;
        }

        float mutateReturnedLocalControl(int slot) {
            OuterPayload outer = makeOuter(1.0, 2.0, 0.25, 4.0);
            for (int i = 0; i < 3; i++) {
                outer.inner.values[i] = outer.inner.values[i] + outer.inner.bias;
                if (i == slot) {
                    outer.inner.values[i] = outer.inner.values[i] * outer.scale;
                }
            }
            if (slot > 1) {
                outer.scale = outer.inner.values[slot] + outer.inner.bias;
            } else {
                outer.scale = outer.inner.values[0] - outer.inner.bias;
            }
            return outer.inner.values[slot] + outer.scale;
        }

        compute {
            void main() {
                float value = mutateReturnedLocalControl(1);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float mutateReturnedLocalControl(int slot)" in generated_code
    assert "OuterPayload outer = makeOuter(1.0, 2.0, 0.25, 4.0);" in generated_code
    assert "for (int i = 0; i < 3; i++)" in generated_code
    assert (
        "outer.inner.values[i] = outer.inner.values[i] + outer.inner.bias;"
        in generated_code
    )
    assert "if (i == slot)" in generated_code
    assert (
        "outer.inner.values[i] = outer.inner.values[i] * outer.scale;" in generated_code
    )
    assert "if (slot > 1)" in generated_code
    assert (
        "outer.scale = outer.inner.values[slot] + outer.inner.bias;" in generated_code
    )
    assert "outer.scale = outer.inner.values[0] - outer.inner.bias;" in generated_code
    assert "return outer.inner.values[slot] + outer.scale;" in generated_code
    assert "float value = mutateReturnedLocalControl(1);" in generated_code
    assert "unsupported Slang" not in generated_code


def test_resource_query_builtins_emit_slang_get_dimensions_helpers():
    code = """
    shader Resources {
        sampler2d colorMap;
        sampler2darray layers;
        sampler3d volumeTex;
        sampler2dshadow shadowMap;
        sampler2darrayshadow shadowLayers;
        samplercubeshadow cubeShadow;
        samplercubearrayshadow cubeShadowLayers;
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
                ivec2 shadowSize = textureSize(shadowMap, 0);
                ivec3 shadowLayerSize = textureSize(shadowLayers, 0);
                ivec2 cubeShadowSize = textureSize(cubeShadow, 0);
                ivec3 cubeShadowLayerSize = textureSize(cubeShadowLayers, 0);
                ivec2 msSize = textureSize(msTex, 0);
                ivec3 msLayerSize = textureSize(msLayers, 0);
                int texLevels = textureQueryLevels(colorMap);
                int layerLevels = textureQueryLevels(layers);
                int volumeLevels = textureQueryLevels(volumeTex);
                int shadowLevels = textureQueryLevels(shadowMap);
                int shadowLayerLevels = textureQueryLevels(shadowLayers);
                int cubeShadowLevels = textureQueryLevels(cubeShadow);
                int cubeShadowLayerLevels = textureQueryLevels(cubeShadowLayers);
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
    assert (
        "int2 cgl_textureSize_sampler2DShadow(Sampler2DShadow tex, uint mipLevel)"
        in generated_code
    )
    assert (
        "int3 cgl_textureSize_sampler2DArrayShadow("
        "Sampler2DArrayShadow tex, uint mipLevel)" in generated_code
    )
    assert (
        "int2 cgl_textureSize_samplerCubeShadow(SamplerCubeShadow tex, uint mipLevel)"
        in generated_code
    )
    assert (
        "int3 cgl_textureSize_samplerCubeArrayShadow("
        "SamplerCubeArrayShadow tex, uint mipLevel)" in generated_code
    )
    assert (
        "int cgl_textureQueryLevels_sampler2D(Sampler2D<float4> tex)" in generated_code
    )
    assert "tex.GetDimensions(width, height, levels);" in generated_code
    assert (
        "int cgl_textureQueryLevels_sampler2DArray(Sampler2DArray<float4> tex)"
        in generated_code
    )
    assert "tex.GetDimensions(width, height, elements, levels);" in generated_code
    assert (
        "int cgl_textureQueryLevels_sampler3D(Sampler3D<float4> tex)" in generated_code
    )
    assert "tex.GetDimensions(width, height, depth, levels);" in generated_code
    assert (
        "int cgl_textureQueryLevels_sampler2DShadow(Sampler2DShadow tex)"
        in generated_code
    )
    assert (
        "int cgl_textureQueryLevels_sampler2DArrayShadow(Sampler2DArrayShadow tex)"
        in generated_code
    )
    assert (
        "int cgl_textureQueryLevels_samplerCubeShadow(SamplerCubeShadow tex)"
        in generated_code
    )
    assert (
        "int cgl_textureQueryLevels_samplerCubeArrayShadow("
        "SamplerCubeArrayShadow tex)" in generated_code
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
    assert (
        "int2 shadowSize = cgl_textureSize_sampler2DShadow(shadowMap, 0);"
        in generated_code
    )
    assert (
        "int3 shadowLayerSize = cgl_textureSize_sampler2DArrayShadow("
        "shadowLayers, 0);" in generated_code
    )
    assert (
        "int2 cubeShadowSize = cgl_textureSize_samplerCubeShadow(cubeShadow, 0);"
        in generated_code
    )
    assert (
        "int3 cubeShadowLayerSize = cgl_textureSize_samplerCubeArrayShadow("
        "cubeShadowLayers, 0);" in generated_code
    )
    assert (
        "int texLevels = cgl_textureQueryLevels_sampler2D(colorMap);" in generated_code
    )
    assert (
        "int layerLevels = cgl_textureQueryLevels_sampler2DArray(layers);"
        in generated_code
    )
    assert (
        "int volumeLevels = cgl_textureQueryLevels_sampler3D(volumeTex);"
        in generated_code
    )
    assert (
        "int shadowLevels = cgl_textureQueryLevels_sampler2DShadow(shadowMap);"
        in generated_code
    )
    assert (
        "int shadowLayerLevels = cgl_textureQueryLevels_sampler2DArrayShadow("
        "shadowLayers);" in generated_code
    )
    assert (
        "int cubeShadowLevels = cgl_textureQueryLevels_samplerCubeShadow(cubeShadow);"
        in generated_code
    )
    assert (
        "int cubeShadowLayerLevels = cgl_textureQueryLevels_samplerCubeArrayShadow("
        "cubeShadowLayers);" in generated_code
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
    assert "textureQueryLevels(" not in generated_code


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


def test_explicit_image_formats_emit_slang_storage_element_types():
    code = """
    shader ExplicitImageFormats {
        image2D scalarFloat @r32f;
        image2D rgFloat @rg32f;
        uimage2D rgUnsigned @format(rg32ui);
        image3D rgbaVolume @rgba16f;
        uimage2DMS msCounter @r32ui;
        image2DMSArray msLayers @rgba8;

        float scalarLoad(image2D image @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        vec2 vectorLoad(image2D image @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        uint unsignedLoad(uimage2D image @rg32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        vec4 rgbaLoad(image3D image @format(rgba16f), ivec3 voxel, vec4 value) {
            vec4 oldValue = imageLoad(image, voxel);
            imageStore(image, voxel, oldValue + value);
            return oldValue;
        }

        uint sampleLoad(uimage2DMS image @r32ui, ivec2 pixel, int sampleIndex, uint value) {
            uint oldValue = imageLoad(image, pixel, sampleIndex);
            imageStore(image, pixel, sampleIndex, oldValue + value);
            return oldValue;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "RWTexture2D<float> scalarFloat;" in generated_code
    assert "RWTexture2D<float2> rgFloat;" in generated_code
    assert "RWTexture2D<uint2> rgUnsigned;" in generated_code
    assert "RWTexture3D<float4> rgbaVolume;" in generated_code
    assert "RWTexture2DMS<uint> msCounter;" in generated_code
    assert "RWTexture2DMSArray<float4> msLayers;" in generated_code
    assert (
        "float scalarLoad(RWTexture2D<float2> image, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float2 vectorLoad(RWTexture2D<float2> image, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "uint unsignedLoad(RWTexture2D<uint2> image, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "float4 rgbaLoad(RWTexture3D<float4> image, int3 voxel, float4 value)"
        in generated_code
    )
    assert (
        "uint sampleLoad(RWTexture2DMS<uint> image, int2 pixel, int sampleIndex, uint value)"
        in generated_code
    )
    assert "float oldValue = image[pixel].x;" in generated_code
    assert "image[pixel] = float2(oldValue + value, 0.0);" in generated_code
    assert "float2 oldValue = image[pixel];" in generated_code
    assert "image[pixel] = oldValue + value;" in generated_code
    assert "uint oldValue = image[pixel].x;" in generated_code
    assert "image[pixel] = uint2(oldValue + value, 0u);" in generated_code
    assert "float4 oldValue = image[voxel];" in generated_code
    assert "image[voxel] = oldValue + value;" in generated_code
    assert "uint oldValue = image[pixel, sampleIndex];" in generated_code
    assert "image[pixel, sampleIndex] = oldValue + value;" in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_integer_image_atomics_emit_slang_interlocked_helpers():
    code = """
    shader AtomicImages {
        image2D unsignedCounters @r32ui;
        image2D signedCounters @r32i;
        image3D unsignedVolume @format(r32ui);
        image2DArray signedLayers @format(r32i);

        uint unsignedOps(image2D image @r32ui, ivec2 pixel, uint value) {
            uint added = imageAtomicAdd(image, pixel, value);
            uint minValue = imageAtomicMin(image, pixel, value);
            uint maxValue = imageAtomicMax(image, pixel, value);
            uint andValue = imageAtomicAnd(image, pixel, value);
            uint orValue = imageAtomicOr(image, pixel, value);
            uint xorValue = imageAtomicXor(image, pixel, value);
            uint exchanged = imageAtomicExchange(image, pixel, value);
            return added + minValue + maxValue + andValue + orValue + xorValue + exchanged;
        }

        int signedOps(image2D image @r32i, ivec2 pixel, int value) {
            int minValue = imageAtomicMin(image, pixel, value);
            int exchanged = imageAtomicExchange(image, pixel, value);
            return minValue + exchanged;
        }

        uint swapVolume(image3D image @r32ui, ivec3 voxel, uint expected, uint value) {
            return imageAtomicCompSwap(image, voxel, expected, value);
        }

        int exchangeLayer(image2DArray image @r32i, ivec3 pixelLayer, int value) {
            return imageAtomicExchange(image, pixelLayer, value);
        }

        compute {
            void main() {
                uint a = unsignedOps(unsignedCounters, ivec2(0, 0), 1u);
                int b = signedOps(signedCounters, ivec2(1, 0), -1);
                uint c = swapVolume(unsignedVolume, ivec3(0, 1, 2), 3u, 4u);
                int d = exchangeLayer(signedLayers, ivec3(2, 3, 4), 5);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "RWTexture2D<uint> unsignedCounters;" in generated_code
    assert "RWTexture2D<int> signedCounters;" in generated_code
    assert "RWTexture3D<uint> unsignedVolume;" in generated_code
    assert "RWTexture2DArray<int> signedLayers;" in generated_code
    assert (
        "uint cgl_imageAtomicAdd_uimage2D(RWTexture2D<uint> image, "
        "int2 coord, uint value)" in generated_code
    )
    assert (
        "int cgl_imageAtomicMin_iimage2D(RWTexture2D<int> image, "
        "int2 coord, int value)" in generated_code
    )
    assert (
        "uint cgl_imageAtomicCompSwap_uimage3D(RWTexture3D<uint> image, "
        "int3 coord, uint compareValue, uint value)" in generated_code
    )
    assert (
        "int cgl_imageAtomicExchange_iimage2DArray(RWTexture2DArray<int> image, "
        "int3 coord, int value)" in generated_code
    )
    for intrinsic in [
        "InterlockedAdd",
        "InterlockedMin",
        "InterlockedMax",
        "InterlockedAnd",
        "InterlockedOr",
        "InterlockedXor",
        "InterlockedExchange",
    ]:
        assert f"{intrinsic}(image[coord], value, original);" in generated_code
    assert (
        "InterlockedCompareExchange(image[coord], compareValue, value, original);"
        in generated_code
    )
    assert (
        "uint added = cgl_imageAtomicAdd_uimage2D(image, pixel, value);"
        in generated_code
    )
    assert (
        "uint minValue = cgl_imageAtomicMin_uimage2D(image, pixel, value);"
        in generated_code
    )
    assert (
        "uint maxValue = cgl_imageAtomicMax_uimage2D(image, pixel, value);"
        in generated_code
    )
    assert (
        "uint andValue = cgl_imageAtomicAnd_uimage2D(image, pixel, value);"
        in generated_code
    )
    assert (
        "uint orValue = cgl_imageAtomicOr_uimage2D(image, pixel, value);"
        in generated_code
    )
    assert (
        "uint xorValue = cgl_imageAtomicXor_uimage2D(image, pixel, value);"
        in generated_code
    )
    assert (
        "uint exchanged = cgl_imageAtomicExchange_uimage2D(image, pixel, value);"
        in generated_code
    )
    assert (
        "int minValue = cgl_imageAtomicMin_iimage2D(image, pixel, value);"
        in generated_code
    )
    assert (
        "int exchanged = cgl_imageAtomicExchange_iimage2D(image, pixel, value);"
        in generated_code
    )
    assert (
        "return cgl_imageAtomicCompSwap_uimage3D(image, voxel, expected, value);"
        in generated_code
    )
    assert (
        "return cgl_imageAtomicExchange_iimage2DArray(image, pixelLayer, value);"
        in generated_code
    )
    for operation in [
        "imageAtomicAdd",
        "imageAtomicMin",
        "imageAtomicMax",
        "imageAtomicAnd",
        "imageAtomicOr",
        "imageAtomicXor",
        "imageAtomicExchange",
        "imageAtomicCompSwap",
    ]:
        assert f"{operation}(image" not in generated_code


def test_integer_image_array_atomics_preserve_expression_indices():
    code = """
    shader AtomicImageArrays {
        image2D counterImages @r32ui[(2 + 1) * 2];
        image2DArray layerImages @r32i[(2 + 1) * 2];

        uint addArray(image2D images[(2 + 1) * 2] @r32ui, ivec2 pixel, uint value) {
            return imageAtomicAdd(images[1 + 2], pixel, value);
        }

        int compareLayer(
            image2DArray images[(2 + 1) * 2] @r32i,
            ivec3 pixelLayer,
            int expected,
            int value
        ) {
            return imageAtomicCompSwap(images[1 + 2], pixelLayer, expected, value);
        }

        compute {
            void main() {
                uint a = addArray(counterImages, ivec2(0, 0), 1u);
                int b = compareLayer(layerImages, ivec3(1, 2, 3), 4, 5);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "RWTexture2D<uint> counterImages[((2 + 1) * 2)];" in generated_code
    assert "RWTexture2DArray<int> layerImages[((2 + 1) * 2)];" in generated_code
    assert (
        "uint addArray(RWTexture2D<uint> images[((2 + 1) * 2)], "
        "int2 pixel, uint value)" in generated_code
    )
    assert (
        "int compareLayer(RWTexture2DArray<int> images[((2 + 1) * 2)], "
        "int3 pixelLayer, int expected, int value)" in generated_code
    )
    assert (
        "return cgl_imageAtomicAdd_uimage2D(images[(1 + 2)], pixel, value);"
        in generated_code
    )
    assert (
        "return cgl_imageAtomicCompSwap_iimage2DArray("
        "images[(1 + 2)], pixelLayer, expected, value);" in generated_code
    )
    assert "1 + 2]," not in generated_code
    assert "2 + 1 * 2" not in generated_code
    assert "imageAtomicAdd(images" not in generated_code
    assert "imageAtomicCompSwap(images" not in generated_code


def test_unsupported_image_atomics_emit_slang_diagnostic_stubs():
    code = """
    shader UnsupportedAtomicImages {
        image2DMS msCounters @r32ui;
        image2DMSArray msLayers @r32i;
        image2D vectorCounters @rg32ui;

        uint addMs(image2DMS image @r32ui, ivec2 pixel, int sampleIndex, uint value) {
            return imageAtomicAdd(image, pixel, sampleIndex, value);
        }

        int compareMsLayer(
            image2DMSArray image @r32i,
            ivec3 pixelLayer,
            int sampleIndex,
            int expected,
            int value
        ) {
            return imageAtomicCompSwap(image, pixelLayer, sampleIndex, expected, value);
        }

        uint addVector(image2D image @rg32ui, ivec2 pixel, uint value) {
            return imageAtomicAdd(image, pixel, value);
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "RWTexture2DMS<uint> msCounters;" in generated_code
    assert "RWTexture2DMSArray<int> msLayers;" in generated_code
    assert "RWTexture2D<uint2> vectorCounters;" in generated_code
    assert (
        "return /* unsupported Slang image atomic: imageAtomicAdd "
        "requires scalar int or uint image2D/image3D/image2DArray resource */ 0u;"
        in generated_code
    )
    assert (
        "return /* unsupported Slang image atomic: imageAtomicCompSwap "
        "requires scalar int or uint image2D/image3D/image2DArray resource */ 0;"
        in generated_code
    )
    assert "imageAtomicAdd(image" not in generated_code
    assert "imageAtomicCompSwap(image" not in generated_code
    assert "cgl_imageAtomicAdd_uimage2DMS" not in generated_code
    assert "cgl_imageAtomicCompSwap_iimage2DMSArray" not in generated_code


def test_formatted_image_arrays_preserve_expression_sizes():
    code = """
    shader ExprFormattedImageArrays {
        image2D counters @r32ui[(1 + 1) * 2];
        image2D rgPairs @rg16f[+3];
        image2D afterCounters @r32ui;

        uint touchCounters(image2D images[(1 + 1) * 2] @r32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[2], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        vec2 touchPairs(image2D images[+3] @rg16f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                uint a = touchCounters(counters, ivec2(1, 2), 3u);
                vec2 b = touchPairs(rgPairs, ivec2(2, 3), vec2(0.5));
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "RWTexture2D<uint> counters[((1 + 1) * 2)];" in generated_code
    assert "RWTexture2D<float2> rgPairs[+3];" in generated_code
    assert "RWTexture2D<uint> afterCounters;" in generated_code
    assert (
        "uint touchCounters(RWTexture2D<uint> images[((1 + 1) * 2)], "
        "int2 pixel, uint value)" in generated_code
    )
    assert (
        "float2 touchPairs(RWTexture2D<float2> images[+3], int2 pixel, float2 value)"
        in generated_code
    )
    assert "uint oldValue = images[2][pixel];" in generated_code
    assert "images[1][pixel] = oldValue + value;" in generated_code
    assert "float2 oldValue = images[2][pixel];" in generated_code
    assert "images[1][pixel] = oldValue + value;" in generated_code
    assert "uint a = touchCounters(counters, int2(1, 2), 3u);" in generated_code
    assert "float2 b = touchPairs(rgPairs, int2(2, 3), float2(0.5));" in generated_code
    assert "1 + 1 * 2" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_formatted_image_queries_emit_typed_slang_helpers():
    code = """
    shader FormattedImageQueries {
        image2D rgFloat @rg32f;
        image2DMS msRg @format(rg32f);

        compute {
            void main() {
                ivec2 imageSizeValue = imageSize(rgFloat);
                ivec2 msSizeValue = imageSize(msRg);
                int msSamples = imageSamples(msRg);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "int2 cgl_imageSize_image2D_RWTexture2D_float2(RWTexture2D<float2> tex)"
        in generated_code
    )
    assert (
        "int2 cgl_imageSize_image2DMS_RWTexture2DMS_float2("
        "RWTexture2DMS<float2> tex)" in generated_code
    )
    assert (
        "int cgl_imageSamples_image2DMS_RWTexture2DMS_float2("
        "RWTexture2DMS<float2> tex)" in generated_code
    )
    assert (
        "int2 imageSizeValue = "
        "cgl_imageSize_image2D_RWTexture2D_float2(rgFloat);" in generated_code
    )
    assert (
        "int2 msSizeValue = "
        "cgl_imageSize_image2DMS_RWTexture2DMS_float2(msRg);" in generated_code
    )
    assert (
        "int msSamples = "
        "cgl_imageSamples_image2DMS_RWTexture2DMS_float2(msRg);" in generated_code
    )
    assert "imageSize(" not in generated_code
    assert "imageSamples(" not in generated_code


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


def test_inferred_let_declarations_emit_inferred_slang_types():
    code = """
    shader main {
        compute {
            void main() {
                let scalar = 1.0;
                let flag = true;
                let label = "debug";
                let marker = 'x';
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float scalar = 1.0;" in generated_code
    assert "bool flag = true;" in generated_code
    assert 'string label = "debug";' in generated_code
    assert "char marker = 'x';" in generated_code
    assert "None scalar" not in generated_code
    assert "None flag" not in generated_code


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


def test_binary_expression_precedence_preserves_grouping_in_slang():
    code = """
    shader Precedence {
        compute {
            void main() {
                float a = 1.0;
                float b = 2.0;
                float c = 3.0;
                float grouped = (a + b) * c;
                float divisor = a / (b * c);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float grouped = (a + b) * c;" in generated_code
    assert "float grouped = a + b * c;" not in generated_code
    assert "float divisor = a / (b * c);" in generated_code
    assert "float divisor = a / b * c;" not in generated_code


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
