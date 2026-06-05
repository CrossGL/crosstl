from typing import List

import pytest

import crosstl.translator as cgl_translator
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
    cgl_translator.parse(generated_code)


def test_imports_emit_as_parseable_crossgl_unit_preamble():
    code = """
    import common;

    [shader("compute")]
    void main() {
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert generated_code.startswith("import common;\n\nshader main {")
    cgl_translator.parse(generated_code)


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


def test_forward_struct_declaration_codegen_from_generated_conformance_sample():
    # Source: shader-slang/slang@52339028a2aa703271533454c6b9528a534bac31
    # docs/generated/tests/conformance/types-struct/struct-no-body-decl.slang
    code = """
    struct ForwardDeclared;

    struct ForwardDeclared
    {
        int x;
    }

    RWStructuredBuffer<int> output;

    [numthreads(1,1,1)]
    void main()
    {
        ForwardDeclared fd;
        fd.x = 55;
        output[0] = fd.x;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert generated_code.count("struct ForwardDeclared") == 1
    assert "int x;" in generated_code
    assert "ForwardDeclared fd;" in generated_code
    assert "output[0] = fd.x;" in generated_code
    cgl_translator.parse(generated_code)


def test_name_first_property_codegen_from_generated_interface_sample():
    code = """
    struct IntProperty
    {
        int _val;
        property prop : int { get { return _val; } }
    }
    """

    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert "struct IntProperty" in generated_code
    assert "int _val;" in generated_code
    assert "int prop;" in generated_code
    cgl_translator.parse(generated_code)


def test_struct_comma_member_declarators_codegen():
    code = """
    struct Uniforms {
        float screenWidth, screenHeight;
        float focalLength, frameHeight;
    };
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float screenWidth;" in generated_code
    assert "float screenHeight;" in generated_code
    assert "float focalLength;" in generated_code
    assert "float frameHeight;" in generated_code
    assert "screenWidth, screenHeight" not in generated_code


def test_typealias_codegen():
    code = """
    typealias Color = float4;
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "typedef vec4 Color;" in generated_code


def test_visibility_qualified_struct_codegen_from_mlp_training_adam_sample():
    code = """
    public struct AdamState
    {
        internal NFloat mean;
        internal NFloat variance;
        internal int iteration;
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "struct AdamState" in generated_code
    assert "NFloat mean;" in generated_code
    assert "NFloat variance;" in generated_code
    assert "int iteration;" in generated_code
    assert "public" not in generated_code
    assert "internal" not in generated_code


def test_static_const_struct_field_initializer_codegen_from_mlp_training_adam_sample():
    # Source: shader-slang/slang@52339028a2aa703271533454c6b9528a534bac31
    # examples/mlp-training/adam.slang
    code = """
    public struct AdamOptimizer
    {
        public static const NFloat beta1 = 0.9h;
        public static const NFloat epsilon = 1e-7h;
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "static const NFloat beta1 = 0.9;" in generated_code
    assert "static const NFloat epsilon = 0.0000001;" in generated_code
    assert "public" not in generated_code
    cgl_translator.parse(generated_code)


def test_reserved_crossgl_names_are_sanitized_from_reflection_api_sample():
    code = """
    struct RasterVertex
    {
        float2 uv;
    }

    struct Material
    {
        Texture2D<float3> albedoMap;
        SamplerState sampler;
    }

    uniform ParameterBlock<Material> material;

    [shader("fragment")]
    float4 fragmentMain(
        in RasterVertex vertex : R)
        : SV_Target0
    {
        float3 albedo = material.albedoMap.Sample(material.sampler, vertex.uv);
        return float4(albedo, 1);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "sampler sampler_;" in generated_code
    assert "RasterVertex vertex_ @ R" in generated_code
    assert (
        "texture(material.albedoMap, material.sampler_, vertex_.uv)" in generated_code
    )
    assert " material.sampler," not in generated_code
    assert " vertex.uv" not in generated_code
    cgl_translator.parse(generated_code)


def test_reserved_function_name_and_register_space_from_parameter_block_sample():
    code = """
    struct A
    {
        float4 au;
        Texture2D at1;
        SamplerState as;
    }

    [[vk::binding(0, 2)]]
    ParameterBlock<A> a : register(space2);

    float4 use(float4 val) { return val; }
    float4 use(Texture2D t, SamplerState s)
    {
        return t.Sample(s, 0.0);
    }

    float4 main() : SV_Target
    {
        return use(a.au) + use(a.at1, a.as);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "ParameterBlock<A> a @set(2) @binding(0);" in generated_code
    assert "@register(space2)" not in generated_code
    assert "vec4 use_(vec4 val)" in generated_code
    assert "vec4 use_(sampler2D t, sampler s)" in generated_code
    assert "return use_(a.au) + use_(a.at1, a.as_);" in generated_code
    cgl_translator.parse(generated_code)


def test_entry_point_parameter_block_parameter_unwraps_to_struct_codegen():
    # Source: shader-slang/slang tests/bindings/entrypoint-parameter-block.slang
    # at 564ac9f050d6569efd773e2f74e7d067a4e54baa.
    code = """
    struct Params {
        Sampler2D<float4> tex;
    }

    [shader("pixel")]
    float4 main(float2 uv, ParameterBlock<Params> params) {
        return params.tex.Sample(uv);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "vec4 main(vec2 uv, Params params)" in generated_code
    assert "ParameterBlock<Params> params" not in generated_code
    assert "return texture(params.tex, uv);" in generated_code
    cgl_translator.parse(generated_code)


def test_uppercase_hlsl_system_semantics_codegen_from_vireo_fixture():
    # Source style: HenriMichelon/vireo_samples src/shaders/deferred_lighting.frag.slang.
    code = """
    struct VertexOutput {
        float4 position : SV_POSITION;
        float2 uv : TEXCOORD;
    };

    [shader("fragment")]
    float4 fragmentMain(VertexOutput input) : SV_TARGET
    {
        return input.position;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "vec4 position @ Out_Position;" in generated_code
    assert "vec4 fragmentMain(VertexOutput input) @ Out_Color" in generated_code
    assert "@ SV_POSITION" not in generated_code
    assert "@ SV_TARGET" not in generated_code
    cgl_translator.parse(generated_code)


def test_struct_methods_do_not_break_field_codegen():
    code = """
    struct Primitive {
        float4 data0;

        float3 getNormal() {
            return data0.xyz;
        }
    };
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "struct Primitive" in generated_code
    assert "vec4 data0;" in generated_code
    assert "FunctionNode(" not in generated_code


def test_ray_payload_access_semantics_codegen_from_ray_tracing_sample():
    code = """
    [raypayload] struct RayPayload
    {
        float4 color : read(caller) : write(caller, closesthit, miss);
    };
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "vec4 color @ ray_payload_read(caller) "
        "@ ray_payload_write(caller,closesthit,miss);" in generated_code
    )
    assert "@ read(caller):write" not in generated_code
    cgl_translator.parse(generated_code)


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


def test_modern_typed_let_var_declarations_codegen_from_official_docs():
    code = """
    let x : int = 7;
    var y : float = 9.0;

    struct Person
    {
        var age : int;
        float height;
    }

    void main()
    {
        let localCount : int = x;
        var localValue : float;
        var inferred = y;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "int x = 7;" in generated_code
    assert "float y = 9.0;" in generated_code
    assert "int age;" in generated_code
    assert "float height;" in generated_code
    assert "int localCount = x;" in generated_code
    assert "float localValue;" in generated_code
    assert "var inferred = y;" in generated_code
    assert "let x :" not in generated_code
    assert "var y :" not in generated_code
    cgl_translator.parse(generated_code)


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


def test_enum_declarations_codegen_from_gpu_printing_sample():
    code = """
    enum EmptyPrintingOp
    {
    };

    enum PrintingOp
    {
        PrintLine,
        PrintF = 2 + 1,
    };
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "enum EmptyPrintingOp" in generated_code
    assert "enum PrintingOp" in generated_code
    assert "PrintLine," in generated_code
    assert "PrintF = 2 + 1," in generated_code


def test_generic_enum_class_codegen_reparse_from_upstream_bug_test():
    # Reduced from shader-slang/slang tests/bugs/11042-generic-enum-scope-conflict.slang.
    code = """
    enum class GenericEnum<T>
    {
        A,
        B,
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "enum GenericEnum" in generated_code
    assert "GenericEnum<" not in generated_code
    assert "A," in generated_code
    assert "B," in generated_code
    cgl_translator.parse(generated_code)


def test_reverse_codegen_rejects_interface_and_conformance_constructs():
    code = """
    interface IFoo {
        int foo();
    }

    struct MyType : IFoo {
        int value;
    };

    extension MyType : IBar {
        int bar();
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    with pytest.raises(
        NotImplementedError,
        match="interface/conformance constructs",
    ) as exc:
        generate_code(ast)

    message = str(exc.value)
    assert "interface IFoo" in message
    assert "struct MyType : IFoo" in message
    assert "extension MyType : IBar" in message


def test_reverse_codegen_rejects_generic_where_conformance_constraint():
    code = """
    int useFoo<T>(T value) where T : IFoo {
        return value.foo();
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    with pytest.raises(
        NotImplementedError,
        match="function useFoo where T : IFoo",
    ):
        generate_code(ast)


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


def test_multi_index_subscript_expressions_codegen_from_official_operator_sample():
    code = """
    int test(S value, int x, int y)
    {
        value[] = 0;
        value[x] = 1;
        value[x, y] = value[1, 0];
        return value[x, y];
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "value[] = 0;" in generated_code
    assert "value[x] = 1;" in generated_code
    assert "value[x, y] = value[1, 0];" in generated_code
    assert "return value[x, y];" in generated_code
    assert "value[(x, y)]" not in generated_code


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
    assert "sampler2D albedo @register(t0);" in generated_code
    assert "sampler linearSampler @register(s0);" in generated_code


def test_resource_binding_metadata_round_trips_to_crossgl_attributes():
    code = """
    layout(set = 2, binding = 7) uniform sampler2D sourceTexture;
    Texture2D<float4> albedo : register(t3);
    SamplerState linearSampler : register(s4);
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "sampler2D sourceTexture @set(2) @binding(7);" in generated_code
    assert "sampler2D albedo @register(t3);" in generated_code
    assert "sampler linearSampler @register(s4);" in generated_code


def test_standalone_layout_declarations_from_first_slang_shader_docs_codegen():
    code = """
    layout(row_major) uniform;
    layout(row_major) buffer;
    layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

    void main()
    {
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "layout(row_major)" not in generated_code
    assert "layout(local_size_x" not in generated_code
    assert "void main()" in generated_code
    cgl_translator.parse(generated_code)


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

    assert "sampler2D albedo @set(1) @binding(0) @register(t0);" in generated_code
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
    assert "sampler2D textures[3] @register(t0);" in generated_code
    assert "sampler samplers[];" in generated_code


def test_namespace_global_resource_array_codegen_from_current_slang_spirv_test():
    # Source: shader-slang/slang tests/spirv/namespace-texture-array.slang
    # at 8c4e02e4021d73091a4f1d4eba842c0dd986997e.
    code = """
    struct ComputePush
    {
        uint image_id;
    };
    [[vk::push_constant]] ComputePush p;

    namespace test_namespace
    {
        [[vk::binding(0, 0)]] RWTexture2D<float4> textureTable[];
    }

    [shader("compute")]
    [numthreads(8, 8, 1)]
    void main(uint3 pixel_i : SV_DispatchThreadID)
    {
        test_namespace.textureTable[p.image_id][pixel_i.xy] = float4(0,1,0,0);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "ComputePush p @push_constant;" in generated_code
    assert (
        "image2D test_namespace_textureTable[] @set(0) @binding(0);" in generated_code
    )
    assert (
        "test_namespace_textureTable[p.image_id][pixel_i.xy] = "
        "vec4(0, 1, 0, 0);" in generated_code
    )
    assert "test_namespace.textureTable" not in generated_code
    cgl_translator.parse(generated_code)


def test_dotted_namespace_global_resource_codegen_from_namespace_import_sample():
    # Source: shader-slang/slang tests/language-feature/namespaces/namespace-import/m.slang
    # at 52339028a2aa703271533454c6b9528a534bac31.
    code = """
    namespace ns1.ns2
    {
        [[vk::binding(0, 0)]] RWTexture2D<float4> textureTable[];
    }

    [shader("compute")]
    [numthreads(8, 8, 1)]
    void main(uint3 pixel_i : SV_DispatchThreadID)
    {
        ns1.ns2.textureTable[0][pixel_i.xy] = float4(0,1,0,0);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "image2D ns1_ns2_textureTable[] @set(0) @binding(0);" in generated_code
    assert "ns1_ns2_textureTable[0][pixel_i.xy] = vec4(0, 1, 0, 0);" in generated_code
    assert "ns1.ns2.textureTable" not in generated_code
    cgl_translator.parse(generated_code)


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
    assert "float local_[2];" in generated_code
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


def test_parameter_attribute_codegen_from_cuda_format_docs_reparses():
    # Source: shader-slang/slang docs/cuda-target.md at
    # 11e97e77155f454c67dc66daf25c75386d13c378 documents
    # [format(...)] on function parameters.
    code = """
    float2 getValue([format("rg16f")] RWTexture2D<float2> t)
    {
        return t[int2(0, 0)];
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "vec2 getValue(image2D t)" in generated_code
    assert "return t[ivec2(0, 0)];" in generated_code
    assert "[format" not in generated_code
    cgl_translator.parse(generated_code)


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


def test_texture_offset_method_call_codegen():
    code = """
    Texture2D<float4> albedo;
    SamplerState linearSampler;

    float4 main(float2 uv, int2 offset, float bias, float lod, float2 ddx, float2 ddy) {
        float4 offsetColor = albedo.Sample(linearSampler, uv, offset);
        float4 biasColor = albedo.SampleBias(linearSampler, uv, bias, offset);
        float4 lodColor = albedo.SampleLevel(linearSampler, uv, lod, offset);
        float4 gradColor = albedo.SampleGrad(linearSampler, uv, ddx, ddy, offset);
        return offsetColor + biasColor + lodColor + gradColor;
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "sampler2D albedo;" in generated_code
    assert "sampler linearSampler;" in generated_code
    assert (
        "vec4 offsetColor = textureOffset(albedo, linearSampler, uv, offset);"
        in generated_code
    )
    assert (
        "vec4 biasColor = textureOffset(albedo, linearSampler, uv, offset, bias);"
        in generated_code
    )
    assert (
        "vec4 lodColor = textureLodOffset(albedo, linearSampler, uv, lod, offset);"
        in generated_code
    )
    assert (
        "vec4 gradColor = textureGradOffset("
        "albedo, linearSampler, uv, ddx, ddy, offset);" in generated_code
    )
    assert "albedo.Sample(" not in generated_code
    assert "albedo.SampleBias" not in generated_code
    assert "albedo.SampleLevel" not in generated_code
    assert "albedo.SampleGrad" not in generated_code


def test_combined_sampler_offset_method_call_codegen():
    code = """
    Sampler2D<float4> albedo;

    float4 main(float2 uv, int2 offset, float bias, float lod, float2 ddx, float2 ddy) {
        float4 offsetColor = albedo.Sample(uv, offset);
        float4 biasColor = albedo.SampleBias(uv, bias, offset);
        float4 lodColor = albedo.SampleLOD(uv, lod, offset);
        float4 gradColor = albedo.SampleGrad(uv, ddx, ddy, offset);
        return offsetColor + biasColor + lodColor + gradColor;
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "sampler2D albedo;" in generated_code
    assert "vec4 offsetColor = textureOffset(albedo, uv, offset);" in generated_code
    assert "vec4 biasColor = textureOffset(albedo, uv, offset, bias);" in generated_code
    assert (
        "vec4 lodColor = textureLodOffset(albedo, uv, lod, offset);" in generated_code
    )
    assert (
        "vec4 gradColor = textureGradOffset(albedo, uv, ddx, ddy, offset);"
        in generated_code
    )
    assert "albedo.Sample(" not in generated_code
    assert "albedo.SampleBias" not in generated_code
    assert "albedo.SampleLOD" not in generated_code
    assert "albedo.SampleGrad" not in generated_code


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


def test_texture_load_offset_method_call_codegen():
    code = """
    Texture2D<float4> albedo;

    float4 main(int2 pixel, int mip, int2 offset) {
        return albedo.Load(int3(pixel, mip), offset);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "sampler2D albedo;" in generated_code
    assert "return texelFetchOffset(albedo, pixel, mip, offset);" in generated_code
    assert "albedo.Load" not in generated_code


def test_texture2dms_load_sample_index_codegen_from_current_slang_core_module():
    # Source: shader-slang/slang@564ac9f050d6569efd773e2f74e7d067a4e54baa
    # docs/generated/tests/design/cross-cutting/core-module/texture2dms-load.slang
    # https://github.com/shader-slang/slang/blob/564ac9f050d6569efd773e2f74e7d067a4e54baa/docs/generated/tests/design/cross-cutting/core-module/texture2dms-load.slang
    code = """
    Texture2DMS<float4> tex;
    RWStructuredBuffer<float4> outBuf;

    [shader("compute")]
    [numthreads(1, 1, 1)]
    void main(uint3 tid : SV_DispatchThreadID)
    {
        outBuf[tid.x] = tex.Load(int2(int(tid.x), int(tid.y)), 0);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "sampler2DMS tex;" in generated_code
    assert (
        "outBuf[tid.x] = texelFetch(tex, "
        "ivec2(int(tid.x), int(tid.y)), 0);" in generated_code
    )
    assert "texelFetchOffset(tex" not in generated_code
    assert "tex.Load" not in generated_code
    cgl_translator.parse(generated_code)


def test_texture2dms_array_load_sample_index_preserves_array_coord_codegen():
    # Source: shader-slang/slang@564ac9f050d6569efd773e2f74e7d067a4e54baa
    # docs/generated/tests/design/target-pipelines/spirv/texture2dmsarray-image-type.slang
    # https://github.com/shader-slang/slang/blob/564ac9f050d6569efd773e2f74e7d067a4e54baa/docs/generated/tests/design/target-pipelines/spirv/texture2dmsarray-image-type.slang
    code = """
    Texture2DMSArray<float4> tex;
    RWStructuredBuffer<float4> outBuf;

    [shader("compute")]
    [numthreads(1, 1, 1)]
    void main(uint3 tid : SV_DispatchThreadID)
    {
        outBuf[tid.x] = tex.Load(int3(0, 0, 0), 0);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "sampler2DMSArray tex;" in generated_code
    assert "outBuf[tid.x] = texelFetch(tex, ivec3(0, 0, 0), 0);" in generated_code
    assert "texelFetchOffset(tex" not in generated_code
    assert "texelFetch(tex, ivec2(0, 0), 0, 0)" not in generated_code
    assert "tex.Load" not in generated_code
    cgl_translator.parse(generated_code)


def test_structured_buffer_load_method_codegen_from_core_module_docs():
    # Source: shader-slang/slang stdlib docs for StructuredBuffer<T,L>.Load
    # and RWStructuredBuffer<T,L>.Load define the single-index Load overload.
    code = """
    StructuredBuffer<float> values;
    RWStructuredBuffer<float> output;

    [shader("compute")]
    [numthreads(1, 1, 1)]
    void main(uint3 tid : SV_DispatchThreadID) {
        uint i = tid.x;
        float a = values.Load(i);
        float b = output.Load(i);
        output[i] = a + b;
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "StructuredBuffer<float> values;" in generated_code
    assert "RWStructuredBuffer<float> output;" in generated_code
    assert "float a = values[i];" in generated_code
    assert "float b = output[i];" in generated_code
    assert "output[i] = a + b;" in generated_code
    assert ".Load(" not in generated_code
    cgl_translator.parse(generated_code)


def test_buffer_getdimensions_statement_codegen_from_core_module_docs():
    code = """
    StructuredBuffer<float4> lookupTable;
    RWStructuredBuffer<uint> output;
    ByteAddressBuffer rawInput;
    RWByteAddressBuffer rawOutput;

    [shader("compute")]
    [numthreads(1, 1, 1)]
    void main(uint3 tid : SV_DispatchThreadID) {
        uint count;
        uint stride;
        uint byteCount;
        lookupTable.GetDimensions(count, stride);
        output.GetDimensions(count, stride);
        rawInput.GetDimensions(byteCount);
        rawOutput.GetDimensions(byteCount);
        output[tid.x] = count + stride + byteCount;
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "StructuredBuffer<float4> lookupTable;" in generated_code
    assert "RWStructuredBuffer<uint> output;" in generated_code
    assert "ByteAddressBuffer rawInput;" in generated_code
    assert "RWByteAddressBuffer rawOutput;" in generated_code
    assert "buffer_dimensions(lookupTable, count, stride);" in generated_code
    assert "buffer_dimensions(output, count, stride);" in generated_code
    assert "buffer_dimensions(rawInput, byteCount);" in generated_code
    assert "buffer_dimensions(rawOutput, byteCount);" in generated_code
    assert ".GetDimensions(" not in generated_code
    cgl_translator.parse(generated_code)


def test_texture_compare_offset_method_call_codegen():
    code = """
    Texture2D<float> shadowMap;
    SamplerComparisonState cmpSampler;

    float main(float2 uv, float depth, int2 offset, float lod, float2 ddx, float2 ddy) {
        float offsetDepth = shadowMap.SampleCmp(cmpSampler, uv, depth, offset);
        float lodDepth = shadowMap.SampleCmpLevel(cmpSampler, uv, depth, lod, offset);
        float levelZeroDepth = shadowMap.SampleCmpLevelZero(cmpSampler, uv, depth, offset);
        float gradDepth = shadowMap.SampleCmpGrad(cmpSampler, uv, depth, ddx, ddy, offset);
        return offsetDepth + lodDepth + levelZeroDepth + gradDepth;
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "sampler2D shadowMap;" in generated_code
    assert "sampler cmpSampler;" in generated_code
    assert (
        "float offsetDepth = textureCompareOffset("
        "shadowMap, cmpSampler, uv, depth, offset);" in generated_code
    )
    assert (
        "float lodDepth = textureCompareLodOffset("
        "shadowMap, cmpSampler, uv, depth, lod, offset);" in generated_code
    )
    assert (
        "float levelZeroDepth = textureCompareLodOffset("
        "shadowMap, cmpSampler, uv, depth, 0.0, offset);" in generated_code
    )
    assert (
        "float gradDepth = textureCompareGradOffset("
        "shadowMap, cmpSampler, uv, depth, ddx, ddy, offset);" in generated_code
    )
    assert "shadowMap.SampleCmp" not in generated_code


def test_texture_lod_query_method_codegen_from_current_slang_intrinsic_test():
    # Source: shader-slang/slang@564ac9f050d6569efd773e2f74e7d067a4e54baa
    # tests/hlsl-intrinsic/texture/texture-calculate-level-of-detail.slang
    # https://github.com/shader-slang/slang/blob/564ac9f050d6569efd773e2f74e7d067a4e54baa/tests/hlsl-intrinsic/texture/texture-calculate-level-of-detail.slang
    code = """
    Texture2D t;
    SamplerState s;
    SamplerComparisonState sc;

    float fragmentMain()
    {
        float result = 0.0;
        result += t.CalculateLevelOfDetail(s, float2(0, 0));
        result += t.CalculateLevelOfDetailUnclamped(sc, float2(0, 0));
        return result;
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "sampler2D t;" in generated_code
    assert "sampler s;" in generated_code
    assert "sampler sc;" in generated_code
    assert "result += textureQueryLod(t, s, vec2(0, 0)).x;" in generated_code
    assert "result += textureQueryLod(t, sc, vec2(0, 0)).y;" in generated_code
    assert "CalculateLevelOfDetail" not in generated_code
    cgl_translator.parse(generated_code)


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
    assert "float local_[2] = {0.5, 1.0};" in generated_code
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
    assert "vec4 local_ = vec4(0.0, 1.0, 0.0, 1.0);" in generated_code
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


def test_user_defined_mul_function_is_not_lowered_to_binary_operator():
    code = """
    float mul(float a, float b) {
        return a + b;
    }

    float main() {
        return mul(1.0, 2.0);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float mul(float a, float b)" in generated_code
    assert "return mul(1.0, 2.0);" in generated_code
    assert "return (1.0 * 2.0);" not in generated_code


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


def test_slang_rcp_builtin_from_stdlib_reference_lowers_to_reciprocal_expression():
    # Source: shader-slang stdlib rcp reference.
    # URL: https://shader-slang.org/stdlib-reference/global-decls/rcp
    code = """
    void main() {
        float inv = rcp(x + 1.0);
        float2 invVec = rcp(float2(4.0, 8.0));
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float inv = (1.0 / (x + 1.0));" in generated_code
    assert "vec2 invVec = (1.0 / vec2(4.0, 8.0));" in generated_code
    assert "rcp(" not in generated_code
    cgl_translator.parse(generated_code)


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


def test_slang_atan2_builtin_from_stdlib_reference_lowers_to_crossgl_atan():
    # Source: Slang core module reference for atan2.
    # URL: https://docs.shader-slang.org/en/latest/external/core-module-reference/global-decls/atan2.html
    code = """
    void main() {
        float angle = atan2(y, x);
        float2 angleVec = atan2(float2(1.0, 2.0), float2(3.0, 4.0));
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float angle = atan(y, x);" in generated_code
    assert "vec2 angleVec = atan(vec2(1.0, 2.0), vec2(3.0, 4.0));" in generated_code
    assert "atan2(" not in generated_code
    cgl_translator.parse(generated_code)


def test_slang_sincos_builtin_from_stdlib_reference_lowers_to_crossgl_assignments():
    # Source: Slang core module reference for sincos.
    # URL: https://docs.shader-slang.org/en/latest/external/core-module-reference/global-decls/sincos.html
    code = """
    void main() {
        float s;
        float c;
        sincos(1.0, s, c);
        if (c > 0.0) {
            sincos(c, s, c);
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "s = sin(1.0);" in generated_code
    assert "c = cos(1.0);" in generated_code
    assert "        s = sin(c);" in generated_code
    assert "        c = cos(c);" in generated_code
    assert "sincos(" not in generated_code
    cgl_translator.parse(generated_code)


def test_derivative_intrinsics_codegen_from_official_fragment_derivative_tests():
    # Source: shader-slang/slang tests/hlsl-intrinsic/fragment-derivative.slang
    # at 85c65f862c045c929814e1abe6b31828d78030ed exercises ddx/ddy,
    # ddx_fine/ddy_fine, ddx_coarse/ddy_coarse, and fwidth_fine/fwidth_coarse.
    code = """
    float4 derivative(float2 uv)
    {
        float fine = ddx_fine(uv.x);
        float coarse = ddy_coarse(uv.y);
        return float4(ddx(uv.x), ddy(uv.y),
                      fwidth_fine(fine), fwidth_coarse(coarse));
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float fine = dFdxFine(uv.x);" in generated_code
    assert "float coarse = dFdyCoarse(uv.y);" in generated_code
    assert (
        "return vec4(dFdx(uv.x), dFdy(uv.y), "
        "fwidthFine(fine), fwidthCoarse(coarse));" in generated_code
    )
    assert "ddx(" not in generated_code
    assert "ddy(" not in generated_code
    assert "ddx_fine" not in generated_code
    assert "ddy_coarse" not in generated_code
    assert "fwidth_fine" not in generated_code
    assert "fwidth_coarse" not in generated_code
    cgl_translator.parse(generated_code)


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


def test_binary_integer_literals_codegen_from_generated_conformance_sample():
    # Source: shader-slang/slang docs/generated/tests/conformance/
    # expressions-literal/bin-prefix-lowercase.slang at d25453d.
    code = """
    void main() {
        int x = 0b1010;
        int y = 0B1111;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "int x = 0b1010;" in generated_code
    assert "int y = 0B1111;" in generated_code
    assert "b1010" not in generated_code.replace("0b1010", "")
    cgl_translator.parse(generated_code)


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

    assert (
        "layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;"
        in generated_code
    )
    assert "StructuredBuffer<float4> lookupTable;" in generated_code
    assert "uvec3 threadID @ SV_DispatchThreadID" in generated_code
    assert "sampler2D inputImage" in generated_code
    assert "image2D outputImage" in generated_code
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


def test_rwtexture_load_codegen_from_official_texture_capability_sample():
    code = """
    RWStructuredBuffer<int> outputBuffer;
    RWTexture2D<float4> texHandle;

    [shader("fragment")]
    void fragMain()
    {
        int ret = 0;
        ret = int((texHandle.Load(int2(0, 0))).x);
        outputBuffer[0] = 0x12345 + ret;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "image2D texHandle;" in generated_code
    assert "ret = int(imageLoad(texHandle, ivec2(0, 0)).x);" in generated_code
    assert "RWTexture2D<float4> texHandle;" not in generated_code
    assert ".Load(" not in generated_code


def test_parenthesized_expression_swizzle_codegen_from_autodiff_texture_learnmip_sample():
    code = """
    RWTexture2D dstTexture;

    void computeMain(uint2 p)
    {
        var val = float4(1.0);
        float4 color = float4((dstTexture[p] - val).xyz, 1.0);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "vec4 color = vec4((dstTexture[p] - val).xyz, 1.0);" in generated_code
    assert "dstTexture[p] - val.xyz" not in generated_code


def test_parenthesized_unary_swizzle_codegen_preserves_receiver_grouping():
    code = """
    void computeMain(float4 value)
    {
        float x = (-value).x;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float x = (-value).x;" in generated_code
    assert "float x = -value.x;" not in generated_code


def test_pointer_declarator_codegen_from_mlp_training_samples():
    code = """
    public struct FeedForwardLayer<int InputSize, int OutputSize>
    {
        public NFloat* weights;
    }

    void learnGradient(
        uniform MyNetwork* network,
        uniform Atomic<uint32_t>* lossBuffer,
        uniform float2* inputs)
    {
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "NFloat* weights;" in generated_code
    assert "MyNetwork* network" in generated_code
    assert "Atomic<uint32_t>* lossBuffer" in generated_code
    assert "vec2* inputs" in generated_code


def test_official_pointer_address_of_and_arrow_member_codegen():
    # Source: Slang User's Guide, Basic Convenience Features > Pointers (limited).
    code = """
    struct MyType
    {
        int a;
    };

    int test(MyType* pObj)
    {
        MyType* pNext = pObj + 1;
        MyType* pNext2 = &pNext[1];
        return pNext.a + pNext->a + (*pNext2).a + pNext2[0].a;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "MyType* pNext2 = &pNext[1];" in generated_code
    assert "return pNext.a + pNext.a + (*pNext2).a + pNext2[0].a;" in generated_code


if __name__ == "__main__":
    pytest.main()
