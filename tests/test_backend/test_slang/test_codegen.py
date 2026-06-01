from typing import List

import pytest

from crosstl import translate
from crosstl.backend.slang import SlangCrossGLCodeGen, SlangLexer, SlangParser


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
    lexer = SlangLexer(code)
    return lexer.tokenize()


def parse_code(tokens: List):
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


def test_import_and_include_paths_codegen():
    code = """
    import MyApp.Shadowing;
    import "dir/file-name.slang";
    __include "scene-helpers";
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "import MyApp.Shadowing;" in generated_code
    assert 'import "dir/file-name.slang";' in generated_code
    assert 'import "scene-helpers";' in generated_code
    assert "dir/file-name.slang;" not in generated_code.replace(
        '"dir/file-name.slang"', ""
    )


def test_struct_array_member_codegen():
    code = """
    struct Cluster {
        float weights[4];
        float4 colors[2][3] : COLOR0;
    };
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float weights[4];" in generated_code
    assert "vec4 colors[2][3]" in generated_code


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


def test_increment_decrement_expression_codegen():
    code = """
    void main(){
        for (int i = 0; i < 4; i++) {
        }
        j--;
        ++k;
        --items[index];
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for (int i = 0; i < 4; i++) {" in generated_code
    assert "j--;" in generated_code
    assert "++k;" in generated_code
    assert "--items[index];" in generated_code


def test_for_initializer_custom_generic_and_qualified_declaration_codegen():
    code = """
    void main(Buffer<float> source){
        for (const uint i = 0; i < 4; i = i + 1) {
        }
        for (Counter cursor = Counter(0); cursor.value < 4; cursor.value += 1) {
        }
        for (Buffer<float> view = source; view.count < 4; view.count += 1) {
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for (uint i = 0; i < 4; i = i + 1) {" in generated_code
    assert (
        "for (Counter cursor = Counter(0); cursor.value < 4; cursor.value += 1) {"
        in generated_code
    )
    assert (
        "for (Buffer<float> view = source; view.count < 4; view.count += 1) {"
        in generated_code
    )


def test_optional_for_clauses_codegen_as_empty_slots():
    code = """
    void main(){
        for (; ; ) {
            break;
        }
        for (int i = 0; ; ) {
            break;
        }
        for (; i < 4; ) {
            break;
        }
        for (; ; i = i + 1) {
            break;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for (; ; ) {" in generated_code
    assert "for (int i = 0; ; ) {" in generated_code
    assert "for (; i < 4; ) {" in generated_code
    assert "for (; ; i = i + 1) {" in generated_code
    assert "None" not in generated_code


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


def test_discard_statement_codegen():
    code = """
    float4 main(float alpha) {
        if (alpha < 0.5) {
            discard;
        }
        return float4(1.0);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "discard;" in generated_code
    assert "VariableNode" not in generated_code


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


def test_switch_default_before_case_preserves_label_order_codegen():
    code = """
    float choose(int mode) {
        switch (mode) {
            default:
                return 9.0;
            case 1:
                return 1.0;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert generated_code.index("default:") < generated_code.index("case 1:")


def test_lambda_expression_codegen():
    code = """
    void main() {
        float folded = fold(values, 0, (int acc, int x) => (acc + x));
        float mapped = map(colors, (float3 color) => { return color; });
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "float folded = fold(values, 0, lambda(int acc, int x, acc + x));"
        in generated_code
    )
    assert (
        "float mapped = map(colors, lambda(vec3 color, { return color; }));"
        in generated_code
    )
    assert "=>" not in generated_code


def test_generic_type_receiver_expression_codegen():
    code = """
    [TorchEntryPoint]
    export __extern_cpp int main() {
        var result = TorchTensor<float>.alloc(Shape(1));
        let count = result.numel();
        let vec = coopVecLoad<4>(input);
        let rs = f.eval<DataTrait0>(1.0);
        return 0;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "var result = TorchTensor<float>.alloc(Shape(1));" in generated_code
    assert "let count = result.numel();" in generated_code
    assert "let vec = coopVecLoad<4>(input);" in generated_code
    assert "let rs = f.eval<DataTrait0>(1.0);" in generated_code
    assert "return 0;" in generated_code


def test_generic_function_declaration_after_name_codegen():
    code = """
    float GetRayT<let RAY_QUERY_FLAGS: uint>(uint rayInlineFlags)
    {
        return 0.0;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float GetRayT(uint rayInlineFlags)" in generated_code
    assert "return 0.0;" in generated_code
    assert "<let" not in generated_code


def test_generic_struct_declaration_after_name_codegen():
    code = """
    struct GenericStruct<T, let N: int>
    {
        T value;
        float weights[N];
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "struct GenericStruct" in generated_code
    assert "T value;" in generated_code
    assert "float weights[N];" in generated_code
    assert "GenericStruct<" not in generated_code


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


def test_qualified_declarations_codegen_lower_qualifiers():
    code = """
    static inline float helper(const float x) {
        return x;
    }

    void main(){
        static const float cached = 1.0;
        constexpr float scale = 2.0;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float helper(float x) {" in generated_code
    assert "float cached = 1.0;" in generated_code
    assert "float scale = 2.0;" in generated_code


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

    bool comparisons(bool a, bool b, bool c, bool d) {
        return a == b && c == d;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "return (a + b) * c;" in generated_code
    assert "return a + b * c;" not in generated_code
    assert "return a / (b * c);" in generated_code
    assert "return a / b * c;" not in generated_code
    assert "return a == b && c == d;" in generated_code
    assert "return (a == b && c) == d;" not in generated_code


def test_binary_bitwise_and_shift_precedence_codegen():
    code = """
    uint combine(uint a, uint b, uint c, uint d) {
        return a | b ^ c & d << 1;
    }

    uint grouped(uint a, uint b) {
        return (a | b) << 1;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "return a | b ^ c & d << 1;" in generated_code
    assert "return (a | b) << 1;" in generated_code
    assert "return a | b << 1;" not in generated_code


def test_compound_bitwise_and_shift_assignment_codegen():
    code = """
    void update(uint value, uint mask) {
        value %= 3;
        value &= mask;
        value |= 1;
        value ^= mask;
        value <<= 1;
        value >>= 2;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "value %= 3;" in generated_code
    assert "value &= mask;" in generated_code
    assert "value |= 1;" in generated_code
    assert "value ^= mask;" in generated_code
    assert "value <<= 1;" in generated_code
    assert "value >>= 2;" in generated_code


def test_assignment_associativity_codegen():
    code = """
    void chain(uint a, uint b, uint c, bool flag) {
        a = b = c;
        a += b = c;
        a = flag ? b : c;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "a = b = c;" in generated_code
    assert "a += b = c;" in generated_code
    assert "a = (flag ? b : c);" in generated_code


def test_assignment_expression_operands_are_parenthesized_codegen():
    code = """
    void mixAssignments(uint a, uint b, uint c, bool flag) {
        uint value = (a = b) + c;
        uint other = c + (a = b);
        uint selected = flag ? (a = b) : c;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "uint value = (a = b) + c;" in generated_code
    assert "uint other = c + (a = b);" in generated_code
    assert "uint selected = (flag ? (a = b) : c);" in generated_code
    assert "uint value = a = b + c;" not in generated_code
    assert "uint other = c + a = b;" not in generated_code


def test_ternary_expression_precedence_preserves_grouping_codegen():
    code = """
    float choose(bool flag, float yes, float no) {
        float selected = flag ? yes : no;
        float shifted = (flag ? yes : no) + 1.0;
        return shifted;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float selected = (flag ? yes : no);" in generated_code
    assert "float shifted = (flag ? yes : no) + 1.0;" in generated_code
    assert "float shifted = flag ? yes : no + 1.0;" not in generated_code


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


def test_modern_slang_compute_attributes_codegen():
    code = """
    [[shader("compute")]]
    [numThreads(4, 2, 1)]
    void main(uint3 tid : SV_DispatchThreadID) {
        return;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "compute {" in generated_code
    assert (
        "layout(local_size_x = 4, local_size_y = 2, local_size_z = 1) in;"
        in generated_code
    )
    assert "void main(uvec3 tid @ SV_DispatchThreadID)" in generated_code


def test_generic_resource_global_codegen():
    code = """
    Texture2D<float4> albedo;
    SamplerState linearSampler;
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "sampler2D albedo;" in generated_code
    assert "sampler linearSampler;" in generated_code


def test_nested_parameter_block_resource_wrapper_codegen():
    code = """
    struct S
    {
        ConstantBuffer<RWStructuredBuffer<int>> cb;
    }

    ParameterBlock<RWStructuredBuffer<int>> rwBuffer;
    ParameterBlock<ConstantBuffer<int>> constBuffer;
    ParameterBlock<ConstantBuffer<RWStructuredBuffer<int>>> nestedBuffer;
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "ConstantBuffer<RWStructuredBuffer<int>> cb;" in generated_code
    assert "ParameterBlock<RWStructuredBuffer<int>> rwBuffer;" in generated_code
    assert "ParameterBlock<ConstantBuffer<int>> constBuffer;" in generated_code
    assert (
        "ParameterBlock<ConstantBuffer<RWStructuredBuffer<int>>> nestedBuffer;"
        in generated_code
    )


def test_line_texture_resource_global_codegen():
    code = """
    Texture1D<float4> line;
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_vars[0].vtype == "Texture1D<float4>"
    assert "sampler1D line;" in generated_code
    assert "Texture1D<float4> line;" not in generated_code


def test_volume_texture_resource_global_codegen():
    code = """
    Texture3D<float4> volume;
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_vars[0].vtype == "Texture3D<float4>"
    assert "sampler3D volume;" in generated_code
    assert "Texture3D<float4> volume;" not in generated_code


def test_array_texture_resource_global_codegen():
    code = """
    Texture2DArray<float4> layers;
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_vars[0].vtype == "Texture2DArray<float4>"
    assert "sampler2DArray layers;" in generated_code
    assert "Texture2DArray<float4> layers;" not in generated_code


def test_cube_array_texture_resource_global_codegen():
    code = """
    TextureCubeArray<float4> probes;
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_vars[0].vtype == "TextureCubeArray<float4>"
    assert "samplerCubeArray probes;" in generated_code
    assert "TextureCubeArray<float4> probes;" not in generated_code


def test_multisample_texture_resource_global_codegen():
    code = """
    Texture2DMS<float4> msTex;
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_vars[0].vtype == "Texture2DMS<float4>"
    assert "sampler2DMS msTex;" in generated_code
    assert "Texture2DMS<float4> msTex;" not in generated_code


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
    assert "sampler linearSampler;" in generated_code


def test_vulkan_binding_attribute_global_resource_codegen():
    code = """
    [[vk::binding(0, 1)]]
    Texture2D<float4> albedo : register(t0);

    [[vk::binding(2)]]
    SamplerState linearSampler;
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "sampler2D albedo @set(1) @binding(0);" in generated_code
    assert "sampler linearSampler @binding(2);" in generated_code


def test_bound_cbuffer_codegen():
    code = """
    cbuffer Camera : register(b0) {
        float4x4 viewProj;
        float4 tint[2];
        float weights[];
        float4x4 transforms[2][3];
    };
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.cbuffers[0].register == "b0"
    assert "cbuffer Camera" in generated_code
    assert "mat4 viewProj;" in generated_code
    assert "vec4 tint[2];" in generated_code
    assert "float weights[];" in generated_code
    assert "mat4 transforms[2][3];" in generated_code


def test_vulkan_attributes_on_cbuffer_codegen():
    code = """
    [[vk::binding(0, 1)]]
    [[vk::push_constant]]
    cbuffer Camera : register(b0) {
        float4x4 viewProj;
    };
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "cbuffer Camera @set(1) @binding(0) @push_constant {" in generated_code
    assert "mat4 viewProj;" in generated_code


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
    assert "sampler samplers[];" in generated_code


def test_local_and_parameter_array_declarator_codegen():
    code = """
    float bump(float values[2], int idx) {
        float local[2];
        float grid[2][3];
        return values[idx];
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float bump(float values[2], int idx)" in generated_code
    assert "float local[2];" in generated_code
    assert "float grid[2][3];" in generated_code
    assert "return values[idx];" in generated_code


def test_local_and_parameter_generic_resource_type_codegen():
    code = """
    float4 sample(Sampler2D<float4> tex, Texture2D<float4> image,
                  SamplerState state, float2 uv) {
        Sampler2D<float4> localTex;
        Texture2D<float4> localImage;
        SamplerState localState;
        return tex.Sample(uv);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "vec4 sample(sampler2D tex, sampler2D image, sampler state, vec2 uv)"
        in generated_code
    )
    assert "sampler2D localTex;" in generated_code
    assert "sampler2D localImage;" in generated_code
    assert "sampler localState;" in generated_code
    assert "return texture(tex, uv);" in generated_code
    assert "Sampler2D<float4>" not in generated_code
    assert "Texture2D<float4>" not in generated_code


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

    assert "sampler2D albedo;" in generated_code
    assert "sampler linearSampler;" in generated_code
    assert "return texture(albedo, linearSampler, uv);" in generated_code
    assert "return albedo.Sample(linearSampler, uv);" not in generated_code


def test_texture_sample_cmp_method_call_codegen():
    code = """
    Texture2D<float> shadowMap;
    SamplerComparisonState cmpSampler;

    float main(float2 uv, float depth) {
        return shadowMap.SampleCmp(cmpSampler, uv, depth);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "sampler2D shadowMap;" in generated_code
    assert "sampler cmpSampler;" in generated_code
    assert "return textureCompare(shadowMap, cmpSampler, uv, depth);" in generated_code
    assert "shadowMap.SampleCmp" not in generated_code


def test_combined_sampler_method_call_codegen():
    code = """
    Sampler2D<float4> colorMap;

    float4 main(float2 uv) {
        return colorMap.Sample(uv);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "sampler2D colorMap;" in generated_code
    assert "return texture(colorMap, uv);" in generated_code
    assert "return colorMap.Sample(uv);" not in generated_code


def test_texture_lod_and_grad_method_call_codegen():
    code = """
    Sampler2D<float4> tex;

    float4 main(float2 uv, float2 ddx, float2 ddy) {
        float4 mip = tex.SampleLevel(uv, 2.0);
        float4 grad = tex.SampleGrad(uv, ddx, ddy);
        return mip + grad;
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "sampler2D tex;" in generated_code
    assert "vec4 mip = textureLod(tex, uv, 2.0);" in generated_code
    assert "vec4 grad = textureGrad(tex, uv, ddx, ddy);" in generated_code
    assert "tex.SampleLevel" not in generated_code
    assert "tex.SampleGrad" not in generated_code


def test_texture_load_method_call_codegen():
    code = """
    Texture2D<float4> albedo;

    float4 main(int2 pixel, int mip) {
        return albedo.Load(int3(pixel, mip));
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "sampler2D albedo;" in generated_code
    assert "return texelFetch(albedo, pixel, mip);" in generated_code
    assert "albedo.Load" not in generated_code


def test_texture_load_method_call_splits_scalar_int3_coordinates_codegen():
    code = """
    Texture2D<float4> albedo;

    float4 main(int mip) {
        return albedo.Load(int3(4, 8, mip));
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "return texelFetch(albedo, ivec2(4, 8), mip);" in generated_code
    assert "texelFetch(albedo, int3(4, 8, mip))" not in generated_code


def test_texture_load_method_call_splits_scalar_uint3_coordinates_codegen():
    code = """
    Texture2D<float4> albedo;

    float4 main(uint mip) {
        return albedo.Load(uint3(pixel.x, pixel.y, mip));
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "return texelFetch(albedo, uvec2(pixel.x, pixel.y), mip);" in generated_code
    assert "texelFetch(albedo, uint3(pixel.x, pixel.y, mip))" not in generated_code


def test_resource_array_sample_method_call_codegen():
    code = """
    Sampler2D<float4> textures[3];

    float4 main(float2 uv) {
        return textures[2].Sample(uv);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "sampler2D textures[3];" in generated_code
    assert "return texture(textures[2], uv);" in generated_code
    assert "textures[2].Sample" not in generated_code


def test_non_resource_sample_method_call_remains_method_call():
    code = """
    Filter filter;

    float4 main(float2 uv) {
        return filter.Sample(uv);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "Filter filter;" in generated_code
    assert "return filter.Sample(uv);" in generated_code
    assert "return texture(filter, uv);" not in generated_code


def test_translate_api_slang_sampler_method_to_crossgl(tmp_path):
    code = """
    Sampler2D<float4> colorMap;

    float4 main(float2 uv) {
        return colorMap.Sample(uv);
    }
    """
    source_path = tmp_path / "sampler.slang"
    source_path.write_text(code, encoding="utf-8")

    generated_code = translate(str(source_path), backend="cgl", format_output=False)

    assert "sampler2D colorMap;" in generated_code
    assert "return texture(colorMap, uv);" in generated_code
    assert "colorMap.Sample(uv)" not in generated_code


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


def test_initialized_top_level_global_codegen():
    code = """
    static const float threshold = 0.5;
    float4 tint = float4(1.0, 0.5, 0.0, 1.0);
    float gain : register(c0) = 1f;
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float threshold = 0.5;" in generated_code
    assert "vec4 tint = vec4(1.0, 0.5, 0.0, 1.0);" in generated_code
    assert "float gain = 1.0;" in generated_code
    assert "gain = 1f" not in generated_code


def test_initializer_list_declaration_codegen():
    code = """
    float weights[2] = {1.0, 2.0};

    void main() {
        float local[2] = {.5f, 1f,};
        float4 colors[1] = {float4(1.0, 0.5, 0.0, 1.0)};
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float weights[2] = {1.0, 2.0};" in generated_code
    assert "float local[2] = {0.5, 1.0};" in generated_code
    assert "vec4 colors[1] = {vec4(1.0, 0.5, 0.0, 1.0)};" in generated_code
    assert "{.5f, 1f" not in generated_code


def test_typed_brace_constructor_codegen():
    code = """
    float4 tint = float4{1.0, .5f, 0.0, 1.0};

    void main() {
        float4 local = float4{0.0, 1.0, 0.0, 1.0};
        float4 colors[1] = {float4{1.0, 0.0, 0.0, 1.0}};
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "vec4 tint = vec4(1.0, 0.5, 0.0, 1.0);" in generated_code
    assert "vec4 local = vec4(0.0, 1.0, 0.0, 1.0);" in generated_code
    assert "vec4 colors[1] = {vec4(1.0, 0.0, 0.0, 1.0)};" in generated_code
    assert "float4{" not in generated_code


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


def test_local_matrix_constructor_declaration_codegen():
    code = """
    void main() {
        float4x4 model = float4x4(1.0);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "mat4 model = mat4(1.0);" in generated_code


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


def test_user_defined_lerp_function_is_not_lowered_to_mix():
    code = """
    float lerp(float a, float b, float t) {
        return a + ((b - a) * t);
    }

    float main() {
        return lerp(0.0, 1.0, 0.25);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float lerp(float a, float b, float t)" in generated_code
    assert "return lerp(0.0, 1.0, 0.25);" in generated_code
    assert "return mix(0.0, 1.0, 0.25);" not in generated_code


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


def test_numeric_literal_codegen_normalizes_crossgl_float_forms():
    code = """
    void main() {
        float exponent = 1e-3f;
        float leading = .5f;
        float trailing = 1.;
        float whole = 1f;
        uint mask = 0xffu;
        uint count = 123u;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float exponent = 0.001;" in generated_code
    assert "float leading = 0.5;" in generated_code
    assert "float trailing = 1.0;" in generated_code
    assert "float whole = 1.0;" in generated_code
    assert "uint mask = 0xffu;" in generated_code
    assert "uint count = 123u;" in generated_code
    assert "1e-3f" not in generated_code
    assert ".5f" not in generated_code
    assert "1f" not in generated_code


def test_generic_struct_member_and_uniform_parameter_codegen_from_official_sample():
    code = """
    struct ImageProcessingOptions
    {
        float3 tintColor;
        float blurRadius;
        bool useLookupTable;
        StructuredBuffer<float4> lookupTable;
    }

    [shader("compute")]
    [numthreads(8, 8)]
    void processImage(
        uint3 threadID : SV_DispatchThreadID,
        uniform Texture2D inputImage,
        uniform RWTexture2D outputImage,
        uniform ImageProcessingOptions options)
    {
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "StructuredBuffer<float4> lookupTable;" in generated_code
    assert "uvec3 threadID @ SV_DispatchThreadID" in generated_code
    assert "sampler2D inputImage" in generated_code
    assert "RWTexture2D outputImage" in generated_code
    assert "ImageProcessingOptions options" in generated_code


def test_export_extern_cpp_function_codegen():
    code = """
    [TorchEntryPoint]
    export __extern_cpp int main()
    {
        return 0;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "int main()" in generated_code
    assert "return 0;" in generated_code
    assert "FunctionNode(" not in generated_code


def test_c_style_scalar_cast_codegen_from_official_select_expr_sample():
    code = """
    int test(int input)
    {
        return input > 1 ? -input : input;
    }

    RWStructuredBuffer<int> outputBuffer;

    [numthreads(4, 1, 1)]
    void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
    {
        outputBuffer[dispatchThreadID.x] = test((int) dispatchThreadID.x);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "outputBuffer[dispatchThreadID.x] = test(int(dispatchThreadID.x));"
        in generated_code
    )


if __name__ == "__main__":
    pytest.main()
