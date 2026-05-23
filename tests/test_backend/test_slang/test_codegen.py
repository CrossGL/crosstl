from crosstl.backend.slang import SlangCrossGLCodeGen
from crosstl.backend.slang import SlangLexer
from crosstl.backend.slang import SlangParser
from crosstl import translate
import pytest
from typing import List


def generate_code(ast_node):
    """Test the code generator
    Args:
        ast_node: The abstract syntax tree generated from the code
    Returns:
        str: The generated code from the abstract syntax tree
    """
    codegen = SlangCrossGLCodeGen.SlangToCrossGLConverter()
    return codegen.generate(ast_node)


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = SlangLexer(code)
    return lexer.tokenize()


def parse_code(tokens: List):
    """Helper function to parse tokens into an AST."""
    parser = SlangParser(tokens)
    return parser.parse()


def test_struct_codegen():
    code = """
    struct AssembledVertex
    {
    float3	position : POSITION;
    };
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Struct parsing or code generation not implemented.")


def test_if_codegen():
    code = """
    [shader("vertex")]
    VertexStageOutput vertexMain(AssembledVertex assembledVertex){
        VertexStageOutput output;
        output.out_position = assembledVertex.position;
        if (assembledVertex.color.r > 0.5) {
            output.out_position = assembledVertex.color;
        }
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("If statement parsing or code generation not implemented.")


def test_for_codegen():
    code = """
    [shader("vertex")]
    VertexStageOutput vertexMain(AssembledVertex assembledVertex){
        VertexStageOutput output;
        output.out_position = assembledVertex.position;
        for (int i = 0; i < 10; i=i+1) {
            output.out_position += assembledVertex.position;
        }

        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("For loop parsing or code generation not implemented.")


def test_for_initializer_uint_and_bool_declaration_codegen():
    code = """
    void main(){
        for (uint i = 0; i < 4; i = i + 1) {
        }
        for (bool done = false; done == false; done = true) {
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for (uint i = 0; i < 4; i = i + 1) {" in generated_code
    assert "for (bool done = false; done == false; done = true) {" in generated_code


def test_break_continue_statement_codegen():
    code = """
    void main(){
        for (int i = 0; i < 4; i = i + 1) {
            break;
            continue;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "break;" in generated_code
    assert "continue;" in generated_code


def test_switch_case_codegen():
    code = """
    void main() {
        int mode = 0;
        switch (mode) {
            case 0:
                mode = 1;
                break;
            default:
                mode = 2;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "switch (mode) {" in generated_code
    assert "case 0:" in generated_code
    assert "mode = 1;" in generated_code
    assert "break;" in generated_code
    assert "default:" in generated_code
    assert "mode = 2;" in generated_code


def test_for_array_assignment_update_codegen():
    code = """
    void main(){
        int value = 1;
        for (int i = 0; i < 4; items[i] += value) {
        }
        for (int i = 0; i < 4; object.field = value) {
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for (int i = 0; i < 4; items[i] += value) {" in generated_code
    assert "for (int i = 0; i < 4; object.field = value) {" in generated_code


def test_standalone_array_and_member_assignment_targets_codegen():
    code = """
    void main(){
        items[0] = 1;
        values[tid.x] += delta;
        object.field = value;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "items[0] = 1;" in generated_code
    assert "values[tid.x] += delta;" in generated_code
    assert "object.field = value;" in generated_code


def test_logical_not_codegen():
    code = """
    bool negate(bool disabled) {
        return !disabled;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "return !disabled;" in generated_code


def test_binary_expression_precedence_preserves_grouping_codegen():
    code = """
    float grouped(float a, float b, float c) {
        return (a + b) * c;
    }

    float divisor(float a, float b, float c) {
        return a / (b * c);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "return (a + b) * c;" in generated_code
    assert "return a + b * c;" not in generated_code
    assert "return a / (b * c);" in generated_code
    assert "return a / b * c;" not in generated_code


def test_while_codegen():
    code = """
    float countDown(float x) {
        while (x > 0.0) {
            x -= 1.0;
        }
        return x;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)

        assert "float countDown(float x)" in generated_code
        assert "while (x > 0.0)" in generated_code
        assert "x -= 1.0;" in generated_code
        assert "return x;" in generated_code
    except SyntaxError:
        pytest.fail("While loop parsing or code generation not implemented.")


def test_do_while_codegen():
    code = """
    float countDown(float x) {
        do {
            x -= 1.0;
        } while (x > 0.0);
        return x;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)

        assert "float countDown(float x)" in generated_code
        assert "do {" in generated_code
        assert "x -= 1.0;" in generated_code
        assert "} while (x > 0.0);" in generated_code
        assert "return x;" in generated_code
    except SyntaxError:
        pytest.fail("Do-while loop parsing or code generation not implemented.")


def test_top_level_attribute_list_before_shader_function_codegen():
    code = """
    [numthreads(8, 8, 1)]
    [shader("compute")]
    void main(uint3 tid : SV_DispatchThreadID) {
        return;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)

        assert "compute {" in generated_code
        assert (
            "layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;"
            in generated_code
        )
        assert "void main(uvec3 tid @ SV_DispatchThreadID)" in generated_code
        assert "return;" in generated_code
    except SyntaxError:
        pytest.fail(
            "Top-level attribute-list parsing or code generation not implemented."
        )


def test_attribute_list_after_shader_function_codegen():
    code = """
    [shader("compute")]
    [numthreads(8, 8, 1)]
    void main(uint3 tid : SV_DispatchThreadID) {
        return;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)

        assert "compute {" in generated_code
        assert (
            "layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;"
            in generated_code
        )
        assert "void main(uvec3 tid @ SV_DispatchThreadID)" in generated_code
        assert "return;" in generated_code
    except SyntaxError:
        pytest.fail(
            "Post-shader attribute-list parsing or code generation not implemented."
        )


def test_numthreads_roundtrip_to_spirv_local_size(tmp_path):
    code = """
    [numthreads(8, 8, 1)]
    [shader("compute")]
    void main(uint3 tid : SV_DispatchThreadID) {
        return;
    }
    """
    source_path = tmp_path / "compute.slang"
    source_path.write_text(code)

    generated_code = translate(str(source_path), backend="spirv", format_output=False)

    assert "OpExecutionMode" in generated_code
    assert "LocalSize 8 8 1" in generated_code


def test_generic_resource_global_codegen():
    code = """
    Texture2D<float4> albedo;
    SamplerState linearSampler;
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "sampler2D albedo;" in generated_code
    assert "SamplerState linearSampler;" in generated_code


def test_bound_generic_resource_global_codegen():
    code = """
    Texture2D<float4> albedo : register(t0);
    SamplerState linearSampler : register(s0);
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_vars[0].register == "t0"
    assert ast.global_vars[1].register == "s0"
    assert "sampler2D albedo;" in generated_code
    assert "SamplerState linearSampler;" in generated_code


def test_bound_cbuffer_codegen():
    code = """
    cbuffer Camera : register(b0) {
        float4x4 viewProj;
        float4 tint[2];
    };
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.cbuffers[0].register == "b0"
    assert "cbuffer Camera" in generated_code
    assert "mat4 viewProj;" in generated_code
    assert "vec4 tint[2];" in generated_code


def test_global_resource_array_codegen():
    code = """
    StructuredBuffer<float> inputs[2];
    Texture2D<float4> textures[3] : register(t0);
    SamplerState samplers[];
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "StructuredBuffer<float> inputs[2];" in generated_code
    assert "sampler2D textures[3];" in generated_code
    assert "SamplerState samplers[];" in generated_code


def test_texture_method_call_codegen():
    code = """
    Texture2D<float4> albedo;
    SamplerState linearSampler;

    float4 main(float2 uv) {
        return albedo.Sample(linearSampler, uv);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "return albedo.Sample(linearSampler, uv);" in generated_code


def test_standalone_postfix_call_codegen():
    code = """
    void main() {
        getCallback()(1.0);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "getCallback()(1.0);" in generated_code


def test_scalar_and_matrix_top_level_declarations_codegen():
    code = """
    uint addOne(uint x) { return x + 1; }
    bool enabled(bool x) { return x; }
    float4x4 ViewProj;
    int frameIndex;
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "uint addOne(uint x)" in generated_code
    assert "return x + 1;" in generated_code
    assert "bool enabled(bool x)" in generated_code
    assert "return x;" in generated_code
    assert "mat4 ViewProj;" in generated_code
    assert "int frameIndex;" in generated_code


def test_local_matrix_declaration_codegen():
    code = """
    void main() {
        float4x4 transform;
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "mat4 transform;" in generated_code


def test_else_codegen():
    code = """
    [shader("vertex")]
    VertexStageOutput vertexMain(AssembledVertex assembledVertex){
        VertexStageOutput output;
        output.out_position = assembledVertex.position;
        if (assembledVertex.color.r > 0.5) {
            output.out_position = assembledVertex.color;
        }
        else {
            output.out_position = float3(0.0, 0.0, 0.0);
        }
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Else statement parsing or code generation not implemented.")


def test_function_call_codegen():
    code = """
    float4 saturate(float4 color) {
        return color;
    }

    [shader("vertex")]
    VertexStageOutput vertexMain(AssembledVertex assembledVertex){
        VertexStageOutput output;
        output.out_position = assembledVertex.position;
        output.out_position = saturate(assembledVertex.color);
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "vec4 saturate(vec4 color)" in generated_code
        assert (
            "output.out_position = saturate(assembledVertex.color);" in generated_code
        )
        assert "clamp(assembledVertex.color" not in generated_code
    except SyntaxError:
        pytest.fail("Function call parsing or code generation not implemented.")


def test_slang_frac_builtin_lowers_to_crossgl_fract():
    code = """
    void main() {
        float wrapped = frac(1.25);
        float2 wrappedVec = frac(float2(1.25, 2.5));
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float wrapped = fract(1.25);" in generated_code
    assert "vec2 wrappedVec = fract(vec2(1.25, 2.5));" in generated_code
    assert "frac(" not in generated_code


def test_slang_fmod_builtin_lowers_to_crossgl_mod():
    code = """
    void main() {
        float wrapped = fmod(5.0, 2.0);
        float2 wrappedVec = fmod(float2(5.0, 7.0), float2(2.0, 3.0));
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float wrapped = mod(5.0, 2.0);" in generated_code
    assert "vec2 wrappedVec = mod(vec2(5.0, 7.0), vec2(2.0, 3.0));" in generated_code
    assert "fmod(" not in generated_code


def test_slang_lerp_builtin_lowers_to_crossgl_mix():
    code = """
    void main() {
        float blended = lerp(0.0, 1.0, 0.25);
        float2 blendedVec = lerp(float2(0.0, 1.0), float2(1.0, 0.0), 0.5);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float blended = mix(0.0, 1.0, 0.25);" in generated_code
    assert (
        "vec2 blendedVec = mix(vec2(0.0, 1.0), vec2(1.0, 0.0), 0.5);" in generated_code
    )
    assert "lerp(" not in generated_code


def test_slang_rsqrt_builtin_lowers_to_crossgl_inversesqrt():
    code = """
    void main() {
        float inv = rsqrt(x);
        float2 invVec = rsqrt(float2(4.0, 9.0));
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float inv = inversesqrt(x);" in generated_code
    assert "vec2 invVec = inversesqrt(vec2(4.0, 9.0));" in generated_code
    assert "rsqrt(" not in generated_code


def test_slang_saturate_builtin_lowers_to_crossgl_clamp():
    code = """
    void main() {
        float saturated = saturate(1.25);
        float2 saturatedVec = saturate(float2(-1.0, 2.0));
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float saturated = clamp(1.25, 0.0, 1.0);" in generated_code
    assert "vec2 saturatedVec = clamp(vec2(-1.0, 2.0), 0.0, 1.0);" in generated_code
    assert "saturate(" not in generated_code


def test_standalone_function_call_statement_codegen():
    code = """
    void helper(float x) {
        return;
    }

    void main() {
        float x = 1.0;
        helper(x);
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        lines = [line.strip() for line in generated_code.splitlines()]

        assert "helper(x);" in lines
        assert "x;" not in lines
    except SyntaxError:
        pytest.fail(
            "Standalone function call parsing or code generation not implemented."
        )


if __name__ == "__main__":
    pytest.main()
