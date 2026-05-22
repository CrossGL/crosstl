import pytest
from typing import List
from crosstl.backend.slang import SlangLexer
from crosstl.backend.slang import SlangParser
from crosstl.backend.slang.SlangAst import (
    ArrayAccessNode,
    AssignmentNode,
    BreakNode,
    CallNode,
    CaseNode,
    ContinueNode,
    DoWhileNode,
    ForNode,
    FunctionCallNode,
    MemberAccessNode,
    MethodCallNode,
    ReturnNode,
    SwitchNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)


def parse_code(tokens: List):
    """Test the parser
    Args:
        tokens (List): The list of tokens generated from the lexer

    Returns:
        ASTNode: The abstract syntax tree generated from the code
    """
    parser = SlangParser(tokens)
    return parser.parse()


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = SlangLexer(code)
    return lexer.tokenize()


def find_function(ast, name: str):
    for function in ast.functions:
        if function.name == name:
            return function
    raise AssertionError(f"Function {name} not found")


def test_struct_parsing():
    code = """
    struct AssembledVertex
    {
    float3	position : POSITION;
    };
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_if_parsing():
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
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("if parsing not implemented.")


def test_for_parsing():
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
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("for parsing not implemented.")


def test_for_initializer_parses_uint_and_bool_declarations():
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
    function = find_function(ast, "main")
    uint_loop = function.body[0]
    bool_loop = function.body[1]

    assert isinstance(uint_loop, ForNode)
    assert isinstance(uint_loop.init, AssignmentNode)
    assert uint_loop.init.left.vtype == "uint"
    assert uint_loop.init.left.name == "i"

    assert isinstance(bool_loop, ForNode)
    assert isinstance(bool_loop.init, AssignmentNode)
    assert bool_loop.init.left.vtype == "bool"
    assert bool_loop.init.left.name == "done"


def test_break_continue_statement_parsing():
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
    function = find_function(ast, "main")
    loop = function.body[0]

    assert isinstance(loop, ForNode)
    assert isinstance(loop.body[0], BreakNode)
    assert isinstance(loop.body[1], ContinueNode)


def test_switch_statement_parsing():
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
    function = find_function(ast, "main")
    switch = function.body[1]

    assert isinstance(switch, SwitchNode)
    assert switch.expression.name == "mode"
    assert len(switch.cases) == 1
    assert isinstance(switch.cases[0], CaseNode)
    assert switch.cases[0].value == "0"
    assert isinstance(switch.cases[0].body[0], AssignmentNode)
    assert isinstance(switch.cases[0].body[1], BreakNode)
    assert len(switch.default_case) == 1
    assert isinstance(switch.default_case[0], AssignmentNode)


def test_for_update_parses_array_and_member_assignment_targets():
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
    function = find_function(ast, "main")
    array_loop = function.body[1]
    member_loop = function.body[2]

    assert isinstance(array_loop, ForNode)
    assert isinstance(array_loop.update, AssignmentNode)
    assert array_loop.update.operator == "+="
    assert isinstance(array_loop.update.left, ArrayAccessNode)
    assert array_loop.update.left.array.name == "items"
    assert array_loop.update.left.index.name == "i"
    assert array_loop.update.right.name == "value"

    assert isinstance(member_loop.update, AssignmentNode)
    assert member_loop.update.operator == "="
    assert isinstance(member_loop.update.left, MemberAccessNode)
    assert member_loop.update.left.object.name == "object"
    assert member_loop.update.left.member == "field"
    assert member_loop.update.right.name == "value"


def test_standalone_array_and_member_assignment_targets_parse():
    code = """
    void main(){
        items[0] = 1;
        values[tid.x] += delta;
        object.field = value;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "main")
    item_assign = function.body[0]
    value_assign = function.body[1]
    member_assign = function.body[2]

    assert isinstance(item_assign, AssignmentNode)
    assert item_assign.operator == "="
    assert isinstance(item_assign.left, ArrayAccessNode)
    assert item_assign.left.array.name == "items"
    assert item_assign.left.index == "0"
    assert item_assign.right == "1"

    assert isinstance(value_assign, AssignmentNode)
    assert value_assign.operator == "+="
    assert isinstance(value_assign.left, ArrayAccessNode)
    assert value_assign.left.array.name == "values"
    assert isinstance(value_assign.left.index, MemberAccessNode)
    assert value_assign.left.index.object.name == "tid"
    assert value_assign.left.index.member == "x"
    assert value_assign.right.name == "delta"

    assert isinstance(member_assign, AssignmentNode)
    assert member_assign.operator == "="
    assert isinstance(member_assign.left, MemberAccessNode)
    assert member_assign.left.object.name == "object"
    assert member_assign.left.member == "field"
    assert member_assign.right.name == "value"


def test_logical_not_parsing():
    code = """
    bool negate(bool disabled) {
        return !disabled;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    value = find_function(ast, "negate").body[0].value

    assert isinstance(value, UnaryOpNode)
    assert value.op == "!"
    assert value.operand.name == "disabled"


def test_while_parsing():
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
        function = find_function(ast, "countDown")
        loop = function.body[0]

        assert isinstance(loop, WhileNode)
        assert loop.condition.op == ">"
        assert isinstance(loop.body[0], AssignmentNode)
        assert loop.body[0].operator == "-="
    except SyntaxError:
        pytest.fail("while parsing not implemented.")


def test_do_while_parsing():
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
        function = find_function(ast, "countDown")
        loop = function.body[0]

        assert isinstance(loop, DoWhileNode)
        assert loop.condition.op == ">"
        assert isinstance(loop.body[0], AssignmentNode)
        assert loop.body[0].operator == "-="
    except SyntaxError:
        pytest.fail("do-while parsing not implemented.")


def test_top_level_attribute_list_before_shader_function_parsing():
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
        function = find_function(ast, "main")

        assert function.qualifier == "compute"
        assert function.params[0].vtype.strip() == "uint3"
        assert function.params[0].semantic == "SV_DispatchThreadID"
        assert isinstance(function.body[0], ReturnNode)
        assert function.body[0].value is None
    except SyntaxError:
        pytest.fail("Top-level attribute-list parsing not implemented.")


def test_attribute_list_after_shader_function_parsing():
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
        function = find_function(ast, "main")

        assert function.qualifier == "compute"
        assert function.params[0].vtype.strip() == "uint3"
        assert function.params[0].semantic == "SV_DispatchThreadID"
        assert isinstance(function.body[0], ReturnNode)
        assert function.body[0].value is None
    except SyntaxError:
        pytest.fail("Post-shader attribute-list parsing not implemented.")


def test_generic_resource_global_parsing():
    code = """
    Texture2D<float4> albedo;
    SamplerState linearSampler;
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert [(var.vtype, var.name) for var in ast.global_vars] == [
        ("Texture2D<float4>", "albedo"),
        ("SamplerState", "linearSampler"),
    ]


def test_bound_generic_resource_global_parsing():
    code = """
    Texture2D<float4> albedo : register(t0);
    SamplerState linearSampler : register(s0);
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert len(ast.functions) == 0
    assert [(var.vtype, var.name, var.register) for var in ast.global_vars] == [
        ("Texture2D<float4>", "albedo", "t0"),
        ("SamplerState", "linearSampler", "s0"),
    ]


def test_bound_cbuffer_parsing():
    code = """
    cbuffer Camera : register(b0) {
        float4x4 viewProj;
        float4 tint[2];
    };
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    cbuffer = ast.cbuffers[0]

    assert cbuffer.name == "Camera"
    assert cbuffer.register == "b0"
    assert [(member.vtype, member.name) for member in cbuffer.members] == [
        ("float4x4", "viewProj"),
        ("float4", "tint[2]"),
    ]


def test_global_resource_array_parsing():
    code = """
    StructuredBuffer<float> inputs[2];
    Texture2D<float4> textures[3] : register(t0);
    SamplerState samplers[];
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert [(var.vtype, var.name, var.array_sizes) for var in ast.global_vars] == [
        ("StructuredBuffer<float>", "inputs", ["2"]),
        ("Texture2D<float4>", "textures", ["3"]),
        ("SamplerState", "samplers", [None]),
    ]
    assert ast.global_vars[1].register == "t0"


def test_texture_method_call_parsing():
    code = """
    Texture2D<float4> albedo;
    SamplerState linearSampler;

    float4 main(float2 uv) {
        return albedo.Sample(linearSampler, uv);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "main")
    call = function.body[0].value

    assert isinstance(call, MethodCallNode)
    assert isinstance(call.object, VariableNode)
    assert call.object.name == "albedo"
    assert call.method == "Sample"
    assert len(call.args) == 2
    assert call.args[0].name == "linearSampler"
    assert call.args[1].name == "uv"


def test_standalone_postfix_call_parsing():
    code = """
    void main() {
        getCallback()(1.0);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "main")
    call = function.body[0]

    assert isinstance(call, CallNode)
    assert isinstance(call.callee, FunctionCallNode)
    assert call.callee.name == "getCallback"
    assert call.args == ["1.0"]


def test_scalar_and_matrix_top_level_declarations_parsing():
    code = """
    uint addOne(uint x) { return x + 1; }
    bool enabled(bool x) { return x; }
    float4x4 ViewProj;
    int frameIndex;
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert [(function.return_type, function.name) for function in ast.functions] == [
        ("uint", "addOne"),
        ("bool", "enabled"),
    ]
    assert [(var.vtype, var.name) for var in ast.global_vars] == [
        ("float4x4", "ViewProj"),
        ("int", "frameIndex"),
    ]


def test_else_parsing():
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
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("else parsing not implemented.")


def test_function_call_parsing():
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
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("function call parsing not implemented.")


def test_standalone_function_call_statement_parsing():
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
        main = find_function(ast, "main")
        call = main.body[1]

        assert isinstance(call, FunctionCallNode)
        assert call.name == "helper"
        assert len(call.args) == 1
        assert isinstance(call.args[0], VariableNode)
        assert call.args[0].name == "x"
    except SyntaxError:
        pytest.fail("standalone function call statement parsing not implemented.")


def test_mod_parsing():
    code = """
    [shader("vertex")]
    VertexStageOutput vertexMain(AssembledVertex assembledVertex) {
        VertexStageOutput output;
        int a = 10 % 3;  // Basic modulus
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Modulus operator parsing not implemented")


def test_numeric_literal_parsing():
    code = """
    float f() {
        float a = 1e-3f;
        float b = .5f;
        float c = 1.;
        uint mask = 0xffu;
        uint count = 123u;
        return a + b + c;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    body = find_function(ast, "f").body

    assert [stmt.right for stmt in body[:5]] == [
        "1e-3f",
        ".5f",
        "1.",
        "0xffu",
        "123u",
    ]


if __name__ == "__main__":
    pytest.main()
