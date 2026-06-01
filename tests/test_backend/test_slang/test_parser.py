from typing import List

import pytest

from crosstl.backend.slang import SlangCrossGLCodeGen, SlangLexer, SlangParser
from crosstl.backend.slang.SlangAst import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    BreakNode,
    CallNode,
    CaseNode,
    CastNode,
    ContinueNode,
    DiscardNode,
    DoWhileNode,
    ExtensionNode,
    ForNode,
    FunctionCallNode,
    GenericConstraintNode,
    IfNode,
    InitializerListNode,
    InterfaceNode,
    MemberAccessNode,
    MethodCallNode,
    ReturnNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    VectorConstructorNode,
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


def test_compound_import_parsing():
    code = "import MyApp.Shadowing;\n"

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert [node.module_name for node in ast.imports] == ["MyApp.Shadowing"]


def test_string_import_path_parsing():
    code = 'import "dir/file-name.slang";\n'

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert [node.module_name for node in ast.imports] == ["dir/file-name.slang"]
    assert ast.global_vars == []


def test_module_include_declarations_do_not_parse_as_globals():
    code = """
    module scene;
    __include "scene-helpers";
    implementing scene;
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert ast.modules == ["scene"]
    assert ast.includes == ["scene-helpers"]
    assert ast.implementing_modules == ["scene"]
    assert ast.global_vars == []


def test_struct_array_member_declarator_parsing():
    code = """
    struct Cluster {
        float weights[4];
        float4 colors[2][3] : COLOR0;
    };
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    members = ast.structs[0].members

    assert [
        (member.vtype, member.name, member.array_sizes, member.semantic)
        for member in members
    ] == [
        ("float", "weights", ["4"], None),
        ("float4", "colors", ["2", "3"], "COLOR0"),
    ]


def test_struct_comma_member_declarators_from_ray_tracing_example():
    code = """
    struct Uniforms {
        float screenWidth, screenHeight;
        float focalLength, frameHeight;
    };
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    members = ast.structs[0].members

    assert [(member.vtype, member.name) for member in members] == [
        ("float", "screenWidth"),
        ("float", "screenHeight"),
        ("float", "focalLength"),
        ("float", "frameHeight"),
    ]


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


def test_string_literals_and_braceless_if_parsing():
    code = r"""
    void computeMain(uint tid) {
        if (tid > 0)
            return;
        println("hello from thread", tid);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    body = find_function(ast, "computeMain").body

    assert isinstance(body[0], IfNode)
    assert isinstance(body[0].if_body[0], ReturnNode)
    assert isinstance(body[1], FunctionCallNode)
    assert body[1].args[0] == '"hello from thread"'


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


def test_increment_decrement_expression_parsing():
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
    body = find_function(ast, "main").body
    loop = body[0]

    assert isinstance(loop, ForNode)
    assert isinstance(loop.update, UnaryOpNode)
    assert loop.update.op == "POST_INCREMENT"
    assert loop.update.operand.name == "i"

    assert isinstance(body[1], UnaryOpNode)
    assert body[1].op == "POST_DECREMENT"
    assert body[1].operand.name == "j"

    assert isinstance(body[2], UnaryOpNode)
    assert body[2].op == "PRE_INCREMENT"
    assert body[2].operand.name == "k"

    assert isinstance(body[3], UnaryOpNode)
    assert body[3].op == "PRE_DECREMENT"
    assert isinstance(body[3].operand, ArrayAccessNode)


def test_postfix_update_rejects_call_target():
    code = """
    void main(){
        getValue()++;
    }
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="Invalid postfix update target"):
        parse_code(tokens)


def test_for_initializer_uses_shared_declaration_parsing():
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
    loops = find_function(ast, "main").body

    qualified_loop = loops[0]
    custom_loop = loops[1]
    generic_loop = loops[2]

    assert isinstance(qualified_loop, ForNode)
    assert qualified_loop.init.left.qualifiers == ["const"]
    assert qualified_loop.init.left.vtype == "uint"

    assert isinstance(custom_loop, ForNode)
    assert custom_loop.init.left.vtype == "Counter"
    assert custom_loop.init.left.name == "cursor"

    assert isinstance(generic_loop, ForNode)
    assert generic_loop.init.left.vtype == "Buffer<float>"
    assert generic_loop.init.left.name == "view"


def test_for_optional_clauses_parse_as_empty_slots():
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
    loops = find_function(ast, "main").body

    empty_loop = loops[0]
    init_only_loop = loops[1]
    condition_only_loop = loops[2]
    update_only_loop = loops[3]

    assert isinstance(empty_loop, ForNode)
    assert empty_loop.init is None
    assert empty_loop.condition is None
    assert empty_loop.update is None

    assert isinstance(init_only_loop.init, AssignmentNode)
    assert init_only_loop.condition is None
    assert init_only_loop.update is None

    assert condition_only_loop.init is None
    assert isinstance(condition_only_loop.condition, BinaryOpNode)
    assert condition_only_loop.update is None

    assert update_only_loop.init is None
    assert update_only_loop.condition is None
    assert isinstance(update_only_loop.update, AssignmentNode)


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


def test_discard_statement_parsing():
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
    discard = find_function(ast, "main").body[0].if_body[0]

    assert isinstance(discard, DiscardNode)


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


def test_switch_default_before_case_preserves_order_metadata():
    code = """
    void main() {
        int mode = 0;
        switch (mode) {
            default:
                mode = 2;
            case 0:
                mode = 1;
                break;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    switch = find_function(ast, "main").body[1]

    assert len(switch.cases) == 1
    assert len(switch.default_case) == 1
    assert [case.value for case in switch.ordered_cases] == [None, "0"]
    assert switch.ordered_cases[0].body is switch.default_case


def test_switch_rejects_duplicate_default_labels():
    code = """
    void main() {
        int mode = 0;
        switch (mode) {
            default:
                mode = 2;
            case 0:
                mode = 1;
            default:
                mode = 3;
        }
    }
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="duplicate default"):
        parse_code(tokens)


def test_lambda_expression_parsing():
    code = """
    void main() {
        float folded = fold(values, 0, (int acc, int x) => (acc + x));
        float mapped = map(colors, (float3 color) => { return color; });
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "main")
    folded = function.body[0]
    mapped = function.body[1]

    folded_lambda = folded.right.args[2]
    assert isinstance(folded_lambda, FunctionCallNode)
    assert folded_lambda.name == "lambda"
    assert folded_lambda.args[0].vtype == "int"
    assert folded_lambda.args[0].name == "acc"
    assert folded_lambda.args[1].vtype == "int"
    assert folded_lambda.args[1].name == "x"
    assert isinstance(folded_lambda.args[2], BinaryOpNode)

    mapped_lambda = mapped.right.args[1]
    assert mapped_lambda.name == "lambda"
    assert mapped_lambda.args[0].vtype == "float3"
    assert mapped_lambda.args[0].name == "color"
    assert mapped_lambda.args[1] == "{ return color; }"


def test_generic_type_receiver_expression_parsing():
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
    function = ast.exports[0].item
    result = function.body[0]
    count = function.body[1]
    vector_load = function.body[2]
    generic_method = function.body[3]

    assert isinstance(result, AssignmentNode)
    assert result.left.vtype == "var"
    assert result.left.name == "result"
    assert isinstance(result.right, MethodCallNode)
    assert result.right.object.name == "TorchTensor<float>"
    assert result.right.method == "alloc"
    assert result.right.args[0].name == "Shape"

    assert isinstance(count, AssignmentNode)
    assert count.left.vtype == "let"
    assert count.left.name == "count"
    assert isinstance(count.right, MethodCallNode)
    assert count.right.object.name == "result"
    assert count.right.method == "numel"

    assert isinstance(vector_load.right, FunctionCallNode)
    assert vector_load.right.name == "coopVecLoad<4>"
    assert vector_load.right.args[0].name == "input"

    assert isinstance(generic_method.right, MethodCallNode)
    assert generic_method.right.object.name == "f"
    assert generic_method.right.method == "eval<DataTrait0>"
    assert generic_method.right.args[0] == "1.0"


def test_generic_function_declaration_after_name_parsing():
    code = """
    float GetRayT<let RAY_QUERY_FLAGS: uint>(uint rayInlineFlags)
    {
        return 0.0;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "GetRayT")

    assert function.return_type == "float"
    assert getattr(function, "is_generic", False)
    assert function.generic_parameters == "<letRAY_QUERY_FLAGS:uint>"
    assert function.params[0].vtype == "uint"
    assert function.params[0].name == "rayInlineFlags"
    assert isinstance(function.body[0], ReturnNode)


def test_generic_struct_declaration_after_name_parsing():
    code = """
    struct GenericStruct<T, let N: int>
    {
        T value;
        float weights[N];
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    struct = ast.structs[0]

    assert struct.name == "GenericStruct"
    assert struct.generic_parameters == "<T, letN:int>"
    assert [(member.vtype, member.name) for member in struct.members] == [
        ("T", "value"),
        ("float", "weights"),
    ]
    assert struct.members[1].array_sizes[0].name == "N"


def test_interface_struct_and_extension_conformance_metadata_parsing():
    code = """
    interface IFoo {
        int foo();
    }

    struct MyType : IFoo {
        int value;
    };

    extension MyType : IBar {
        int bar() {
            return 1;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    interface = ast.interfaces[0]
    assert isinstance(interface, InterfaceNode)
    assert interface.name == "IFoo"
    assert len(interface.methods) == 1
    assert interface.methods[0].return_type == "int"
    assert interface.methods[0].name == "foo"
    assert interface.methods[0].is_declaration

    struct = ast.structs[0]
    assert struct.name == "MyType"
    assert struct.conformances == ["IFoo"]

    extension = ast.extensions[0]
    assert isinstance(extension, ExtensionNode)
    assert extension.extended_type == "MyType"
    assert extension.conformances == ["IBar"]
    assert len(extension.methods) == 1
    assert extension.methods[0].name == "bar"
    assert not extension.methods[0].is_declaration


def test_function_generic_where_conformance_constraint_parsing():
    code = """
    int useFoo<T>(T value) where T : IFoo {
        return value.foo();
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "useFoo")

    assert function.generic_parameters == "<T>"
    assert len(function.generic_constraints) == 1
    constraint = function.generic_constraints[0]
    assert isinstance(constraint, GenericConstraintNode)
    assert constraint.parameter == "T"
    assert constraint.constraint_type == "IFoo"


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


def test_logical_and_keeps_equality_operands_grouped():
    code = """
    void main(bool a, bool b, bool c, bool d) {
        bool bothEqual = a == b && c == d;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    assignment = find_function(ast, "main").body[0]
    expression = assignment.right

    assert isinstance(expression, BinaryOpNode)
    assert expression.op == "&&"
    assert isinstance(expression.left, BinaryOpNode)
    assert expression.left.op == "=="
    assert isinstance(expression.right, BinaryOpNode)
    assert expression.right.op == "=="


def test_binary_bitwise_and_shift_precedence_parsing():
    code = """
    uint combine(uint a, uint b, uint c, uint d) {
        uint value = a | b ^ c & d << 1;
        return value;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    expression = find_function(ast, "combine").body[0].right

    assert expression.op == "|"
    assert expression.left.name == "a"
    assert expression.right.op == "^"
    assert expression.right.left.name == "b"
    assert expression.right.right.op == "&"
    assert expression.right.right.left.name == "c"
    assert expression.right.right.right.op == "<<"
    assert expression.right.right.right.left.name == "d"
    assert expression.right.right.right.right == "1"


def test_compound_bitwise_and_shift_assignment_parsing():
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
    body = find_function(ast, "update").body

    assert [stmt.operator for stmt in body] == ["%=", "&=", "|=", "^=", "<<=", ">>="]
    assert all(isinstance(stmt, AssignmentNode) for stmt in body)


def test_assignment_parsing_is_right_associative():
    code = """
    void chain(uint a, uint b, uint c, bool flag) {
        a = b = c;
        a += b = c;
        a = flag ? b : c;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    body = find_function(ast, "chain").body
    simple = body[0]
    compound = body[1]
    ternary_assignment = body[2]

    assert isinstance(simple, AssignmentNode)
    assert simple.operator == "="
    assert simple.left.name == "a"
    assert isinstance(simple.right, AssignmentNode)
    assert simple.right.operator == "="
    assert simple.right.left.name == "b"
    assert simple.right.right.name == "c"

    assert isinstance(compound, AssignmentNode)
    assert compound.operator == "+="
    assert isinstance(compound.right, AssignmentNode)
    assert compound.right.operator == "="

    assert isinstance(ternary_assignment, AssignmentNode)
    assert isinstance(ternary_assignment.right, TernaryOpNode)


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
        assert function.numthreads == ("8", "8", "1")
        assert function.attributes == [
            {"name": "numthreads", "arguments": ["8", "8", "1"]}
        ]
        assert function.params[0].vtype.strip() == "uint3"
        assert function.params[0].semantic == "SV_DispatchThreadID"
        assert isinstance(function.body[0], ReturnNode)
        assert function.body[0].value is None
    except SyntaxError:
        pytest.fail("Top-level attribute-list parsing not implemented.")


def test_extended_shader_stage_attribute_parsing():
    code = """
    [shader("geometry")]
    void gs_main() {
        return;
    }

    [shader("raygeneration")]
    void ray_main() {
        return;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert find_function(ast, "gs_main").qualifier == "geometry"
    assert find_function(ast, "ray_main").qualifier == "raygeneration"
    generated = SlangCrossGLCodeGen.SlangToCrossGLConverter().generate(ast)
    assert "geometry {" in generated
    assert "ray_generation {" in generated


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
        assert function.numthreads == ("8", "8", "1")
        assert function.attributes == [
            {"name": "numthreads", "arguments": ["8", "8", "1"]}
        ]
        assert function.params[0].vtype.strip() == "uint3"
        assert function.params[0].semantic == "SV_DispatchThreadID"
        assert isinstance(function.body[0], ReturnNode)
        assert function.body[0].value is None
    except SyntaxError:
        pytest.fail("Post-shader attribute-list parsing not implemented.")


def test_modern_slang_shader_and_numthreads_attribute_parsing():
    code = """
    [[shader("compute")]]
    [numThreads(4, 2, 1)]
    void main(uint3 tid : SV_DispatchThreadID) {
        return;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "main")

    assert function.qualifier == "compute"
    assert function.numthreads == ("4", "2", "1")
    assert function.attributes == [{"name": "numThreads", "arguments": ["4", "2", "1"]}]


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


def test_nested_parameter_block_resource_wrapper_parsing():
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

    assert ast.structs[0].members[0].vtype == "ConstantBuffer<RWStructuredBuffer<int>>"
    assert [(var.vtype, var.name) for var in ast.global_vars] == [
        ("ParameterBlock<RWStructuredBuffer<int>>", "rwBuffer"),
        ("ParameterBlock<ConstantBuffer<int>>", "constBuffer"),
        (
            "ParameterBlock<ConstantBuffer<RWStructuredBuffer<int>>>",
            "nestedBuffer",
        ),
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


def test_vulkan_binding_attribute_global_resource_parsing():
    code = """
    [[vk::binding(0, 1)]]
    Texture2D<float4> albedo : register(t0);

    [[vk::binding(2)]]
    SamplerState linearSampler;
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    albedo = ast.global_vars[0]
    linear_sampler = ast.global_vars[1]

    assert albedo.attributes == [{"name": "vk::binding", "arguments": ["0", "1"]}]
    assert albedo.register == "t0"
    assert linear_sampler.attributes == [{"name": "vk::binding", "arguments": ["2"]}]
    assert [(var.vtype, var.name) for var in ast.global_vars] == [
        ("Texture2D<float4>", "albedo"),
        ("SamplerState", "linearSampler"),
    ]


def test_bound_cbuffer_parsing():
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
    cbuffer = ast.cbuffers[0]

    assert cbuffer.name == "Camera"
    assert cbuffer.register == "b0"
    assert [
        (member.vtype, member.name, member.array_sizes) for member in cbuffer.members
    ] == [
        ("float4x4", "viewProj", []),
        ("float4", "tint", ["2"]),
        ("float", "weights", [None]),
        ("float4x4", "transforms", ["2", "3"]),
    ]


def test_vulkan_attributes_on_cbuffer_parsing():
    code = """
    [[vk::binding(0, 1)]]
    [[vk::push_constant]]
    cbuffer Camera : register(b0) {
        float4x4 viewProj;
    };
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    cbuffer = ast.cbuffers[0]

    assert cbuffer.name == "Camera"
    assert cbuffer.register == "b0"
    assert cbuffer.attributes == [
        {"name": "vk::binding", "arguments": ["0", "1"]},
        {"name": "vk::push_constant", "arguments": []},
    ]
    assert cbuffer.members[0].name == "viewProj"


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


def test_local_and_parameter_array_declarator_parsing():
    code = """
    float bump(float values[2], int idx) {
        float local[2];
        float grid[2][3];
        return values[idx];
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "bump")

    assert [
        (param.vtype, param.name, param.array_sizes) for param in function.params
    ] == [
        ("float", "values", ["2"]),
        ("int", "idx", []),
    ]
    assert isinstance(function.body[0], VariableNode)
    assert function.body[0].name == "local"
    assert function.body[0].array_sizes == ["2"]
    assert isinstance(function.body[1], VariableNode)
    assert function.body[1].name == "grid"
    assert function.body[1].array_sizes == ["2", "3"]
    assert isinstance(function.body[2].value, ArrayAccessNode)


def test_local_and_parameter_generic_resource_type_parsing():
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
    function = find_function(ast, "sample")

    assert [(param.vtype, param.name) for param in function.params] == [
        ("Sampler2D<float4>", "tex"),
        ("Texture2D<float4>", "image"),
        ("SamplerState", "state"),
        ("float2", "uv"),
    ]
    assert [(stmt.vtype, stmt.name) for stmt in function.body[:3]] == [
        ("Sampler2D<float4>", "localTex"),
        ("Texture2D<float4>", "localImage"),
        ("SamplerState", "localState"),
    ]


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


def test_texture_load_method_parsing():
    code = """
    Texture2D<float4> albedo;

    float4 main(int2 pixel, int mip) {
        return albedo.Load(int3(pixel, mip));
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "main")
    call = function.body[0].value

    assert isinstance(call, MethodCallNode)
    assert isinstance(call.object, VariableNode)
    assert call.object.name == "albedo"
    assert call.method == "Load"
    assert isinstance(call.args[0], FunctionCallNode)
    assert call.args[0].name == "int3"
    assert call.args[0].args[0].name == "pixel"
    assert call.args[0].args[1].name == "mip"


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


def test_initialized_top_level_global_parsing():
    code = """
    static const float threshold = 0.5;
    float4 tint = float4(1.0, 0.5, 0.0, 1.0);
    float gain : register(c0) = 1f;
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    threshold = ast.global_vars[0]
    assert isinstance(threshold, AssignmentNode)
    assert threshold.left.vtype == "float"
    assert threshold.left.name == "threshold"
    assert threshold.left.qualifiers == ["static", "const"]
    assert threshold.right == "0.5"

    tint = ast.global_vars[1]
    assert isinstance(tint, AssignmentNode)
    assert tint.left.vtype == "float4"
    assert tint.left.name == "tint"
    assert isinstance(tint.right, VectorConstructorNode)
    assert tint.right.type_name == "float4"

    gain = ast.global_vars[2]
    assert isinstance(gain, AssignmentNode)
    assert gain.left.register == "c0"
    assert gain.right == "1f"


def test_initializer_list_declaration_parsing():
    code = """
    float weights[2] = {1.0, 2.0};

    void main() {
        float local[2] = {.5f, 1f,};
        float4 colors[1] = {float4(1.0, 0.5, 0.0, 1.0)};
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    global_weights = ast.global_vars[0]
    assert isinstance(global_weights, AssignmentNode)
    assert global_weights.left.name == "weights"
    assert global_weights.left.array_sizes == ["2"]
    assert isinstance(global_weights.right, InitializerListNode)
    assert global_weights.right.elements == ["1.0", "2.0"]

    body = find_function(ast, "main").body
    local = body[0]
    assert isinstance(local.right, InitializerListNode)
    assert local.right.elements == [".5f", "1f"]

    colors = body[1]
    assert isinstance(colors.right, InitializerListNode)
    assert isinstance(colors.right.elements[0], VectorConstructorNode)


def test_typed_brace_constructor_parsing():
    code = """
    float4 tint = float4{1.0, .5f, 0.0, 1.0};

    void main() {
        float4 local = float4{0.0, 1.0, 0.0, 1.0};
        float4 colors[1] = {float4{1.0, 0.0, 0.0, 1.0}};
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    tint = ast.global_vars[0]
    assert isinstance(tint.right, VectorConstructorNode)
    assert tint.right.type_name == "float4"
    assert tint.right.args == ["1.0", ".5f", "0.0", "1.0"]

    body = find_function(ast, "main").body
    assert isinstance(body[0].right, VectorConstructorNode)
    assert body[0].right.type_name == "float4"
    assert isinstance(body[1].right.elements[0], VectorConstructorNode)


def test_local_matrix_declaration_parsing():
    code = """
    void main() {
        float4x4 transform;
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "main")
    declaration = function.body[0]

    assert isinstance(declaration, VariableNode)
    assert declaration.vtype == "float4x4"
    assert declaration.name == "transform"


def test_local_matrix_constructor_declaration_parsing():
    code = """
    void main() {
        float4x4 model = float4x4(1.0);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "main")
    declaration = function.body[0]

    assert isinstance(declaration, AssignmentNode)
    assert declaration.left.vtype == "float4x4"
    assert declaration.left.name == "model"
    assert isinstance(declaration.right, VectorConstructorNode)
    assert declaration.right.type_name == "float4x4"
    assert declaration.right.args == ["1.0"]


def test_qualified_function_parameters_globals_and_locals_parsing():
    code = """
    static Texture2D<float4> sourceTexture;

    static inline float helper(const float x) {
        return x;
    }

    void main() {
        static const float cached = 1.0;
        constexpr float scale = 2.0;
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert ast.global_vars[0].qualifiers == ["static"]

    helper = find_function(ast, "helper")
    assert helper.qualifiers == ["static", "inline"]
    assert helper.params[0].qualifiers == ["const"]

    body = find_function(ast, "main").body
    assert isinstance(body[0], AssignmentNode)
    assert body[0].left.qualifiers == ["static", "const"]
    assert body[0].left.vtype == "float"
    assert isinstance(body[1], AssignmentNode)
    assert body[1].left.qualifiers == ["constexpr"]


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


def test_generic_struct_member_and_uniform_parameters_from_official_sample():
    code = """
    static const uint THREADGROUP_SIZE_X = 8;
    static const uint THREADGROUP_SIZE_Y = THREADGROUP_SIZE_X;

    struct ImageProcessingOptions
    {
        float3 tintColor;
        float blurRadius;
        bool useLookupTable;
        StructuredBuffer<float4> lookupTable;
    }

    [shader("compute")]
    [numthreads(THREADGROUP_SIZE_X, THREADGROUP_SIZE_Y)]
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
    options = ast.structs[0]
    process_image = find_function(ast, "processImage")

    assert options.members[-1].vtype == "StructuredBuffer<float4>"
    assert options.members[-1].name == "lookupTable"
    params = [
        (param.qualifiers, param.vtype, param.name) for param in process_image.params
    ]
    assert params == [
        ([], "uint3", "threadID"),
        (["uniform"], "Texture2D", "inputImage"),
        (["uniform"], "RWTexture2D", "outputImage"),
        (["uniform"], "ImageProcessingOptions", "options"),
    ]


def test_export_extern_cpp_function_parsing():
    code = """
    [TorchEntryPoint]
    export __extern_cpp int main()
    {
        return 0;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    exported = ast.exports[0].item

    assert exported.name == "main"
    assert exported.return_type == "int"
    assert "__extern_cpp" in exported.qualifiers
    assert exported.attributes == [{"name": "TorchEntryPoint", "arguments": []}]
    assert isinstance(exported.body[0], ReturnNode)


def test_c_style_scalar_cast_from_official_select_expr_sample():
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
    compute_main = find_function(ast, "computeMain")
    assignment = compute_main.body[0]
    call = assignment.right
    cast = call.args[0]

    assert isinstance(cast, CastNode)
    assert cast.target_type == "int"
    assert isinstance(cast.expression, MemberAccessNode)
    assert cast.expression.member == "x"


if __name__ == "__main__":
    pytest.main()
