from typing import List

import pytest

from crosstl.backend.Mojo.MojoAst import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    BreakNode,
    CastNode,
    ClassNode,
    ConstantBufferNode,
    ContinueNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    ImportNode,
    MemberAccessNode,
    MethodCallNode,
    RangeForNode,
    ReturnNode,
    SwitchNode,
    TernaryOpNode,
    TupleNode,
    UnaryOpNode,
    VariableDeclarationNode,
    VariableNode,
    VectorConstructorNode,
    WhileNode,
    WithNode,
)
from crosstl.backend.Mojo.MojoLexer import MojoLexer
from crosstl.backend.Mojo.MojoParser import MojoParser


def parse_code(tokens: List):
    """Test the parser
    Args:
        tokens (List): The list of tokens generated from the lexer

    Returns:
        ASTNode: The abstract syntax tree generated from the code
    """
    parser = MojoParser(tokens)
    return parser.parse()


def tokenize_code(code: str) -> List:
    lexer = MojoLexer(code)
    return lexer.tokenize()


def find_function(ast, name: str):
    for node in ast.functions:
        if isinstance(node, FunctionNode) and node.name == name:
            return node
    raise AssertionError(f"Function {name} not found")


def find_class(ast, name: str):
    for node in ast.classes:
        if isinstance(node, ClassNode) and node.name == name:
            return node
    raise AssertionError(f"Class {name} not found")


def find_constant_buffer(ast, name: str):
    for node in ast.constants:
        if isinstance(node, ConstantBufferNode) and node.name == name:
            return node
    raise AssertionError(f"Constant buffer {name} not found")


def test_function_parameter_conventions_parse_from_official_docs():
    code = """
    def get_name_tag(var name: String, out name_tag: NameTag):
        pass
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "get_name_tag")

    assert [
        (param.name, param.vtype, param.parameter_convention)
        for param in function.params
    ] == [
        ("name", "String", "var"),
        ("name_tag", "NameTag", "out"),
    ]


def test_method_self_parameter_convention_without_type_parses():
    code = """
    def __init__(out self, value: Int):
        pass
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "__init__")

    assert [
        (param.name, param.vtype, param.parameter_convention)
        for param in function.params
    ] == [
        ("self", "", "out"),
        ("value", "Int", None),
    ]


def test_function_parameter_separator_markers_parse_from_official_docs():
    code = """
    def kw_only_args(a1: Int, a2: Int, *, double: Bool) -> Int:
        var product = a1 * a2
        if double:
            product *= 2
        return product

    def positional_only_args(a1: Int, a2: Int, /, b1: Int) -> Int:
        return a1 + a2 + b1
    """
    ast = parse_code(tokenize_code(code))
    kw_only = find_function(ast, "kw_only_args")
    positional_only = find_function(ast, "positional_only_args")

    assert [(param.name, param.vtype) for param in kw_only.params] == [
        ("a1", "Int"),
        ("a2", "Int"),
        ("double", "Bool"),
    ]
    assert [(param.name, param.vtype) for param in positional_only.params] == [
        ("a1", "Int"),
        ("a2", "Int"),
        ("b1", "Int"),
    ]


def test_function_optional_argument_default_parses_from_official_docs():
    code = """
    fn my_pow(base: Int, exp: Int = 2) -> Int:
        return base ** exp
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "my_pow")

    assert [(param.name, param.vtype) for param in function.params] == [
        ("base", "Int"),
        ("exp", "Int"),
    ]
    assert function.params[0].default_value is None
    assert function.params[1].default_value == "2"


def test_struct_parsing():
    code = """
    struct VSInput:
        var position: float4
        var color: float2

    struct VSOutput:
        var out_position: float2
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_parenthesized_import_items_and_floor_divide_parsing():
    code = """
    from std.gpu import (
        block_idx,
        thread_idx,
    )

    fn main():
        var threads = max_threads // 2
    """
    ast = parse_code(tokenize_code(code))

    import_node = ast.functions[0]
    function = find_function(ast, "main")
    declaration = function.body[0]

    assert isinstance(import_node, ImportNode)
    assert import_node.items == ["block_idx", "thread_idx"]
    assert isinstance(declaration.value, BinaryOpNode)
    assert declaration.value.op == "//"


def test_multiline_parenthesized_expression_parsing():
    code = """
    fn main():
        var smem_per_tile = (
            max_smem // 4 // 2
        )
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    declaration = function.body[0]

    assert isinstance(declaration.value, BinaryOpNode)
    assert declaration.value.op == "//"


def test_struct_generic_member_parsing():
    code = """
    struct Resources:
        @position
        var transform: Matrix[DType.float32, 4, 4]
        SIMD[DType.float32, 4] tint @color
        samples: InlineArray[SIMD[DType.float32, 4], 2]
    """
    ast = parse_code(tokenize_code(code))
    struct_node = ast.structs[0]

    assert [(member.vtype, member.name) for member in struct_node.members] == [
        ("Matrix[DType.float32, 4, 4]", "transform"),
        ("SIMD[DType.float32, 4]", "tint"),
        ("InlineArray[SIMD[DType.float32, 4], 2]", "samples"),
    ]
    assert [attr.name for attr in struct_node.members[0].attributes] == ["position"]
    assert [attr.name for attr in struct_node.members[1].attributes] == ["color"]


def test_brace_struct_parsing_preserves_generic_members_and_attributes():
    code = """
    @value
    struct Resources {
        @position
        var transform: Matrix[DType.float32, 4, 4]
        SIMD[DType.float32, 4] tint @color;
        samples: InlineArray[SIMD[DType.float32, 4], 2];
    }
    """
    ast = parse_code(tokenize_code(code))
    struct_node = ast.structs[0]

    assert [attr.name for attr in struct_node.attributes] == ["value"]
    assert [(member.vtype, member.name) for member in struct_node.members] == [
        ("Matrix[DType.float32, 4, 4]", "transform"),
        ("SIMD[DType.float32, 4]", "tint"),
        ("InlineArray[SIMD[DType.float32, 4], 2]", "samples"),
    ]
    assert [attr.name for attr in struct_node.members[0].attributes] == ["position"]
    assert [attr.name for attr in struct_node.members[1].attributes] == ["color"]


def test_if_parsing():
    code = """
    fn vertex_main(input: VSInput) -> VSOutput:
        var output: VSOutput
        output.out_position = input.position.xy
        if input.color.x > 0.5:
            output.out_position = input.color
        return output
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("If parsing not implemented.")


def test_bare_identifier_block_condition_parsing():
    code = """
    fn main():
        if ready:
            sink()
        while running:
            tick()
        switch state:
            case active:
                sink()
            default:
                tick()
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")

    if_stmt = function.body[0]
    while_stmt = function.body[1]
    switch_stmt = function.body[2]

    assert isinstance(if_stmt, IfNode)
    assert isinstance(if_stmt.condition, VariableNode)
    assert if_stmt.condition.name == "ready"
    assert isinstance(while_stmt, WhileNode)
    assert isinstance(while_stmt.condition, VariableNode)
    assert while_stmt.condition.name == "running"
    assert isinstance(switch_stmt, SwitchNode)
    assert isinstance(switch_stmt.expression, VariableNode)
    assert switch_stmt.expression.name == "state"
    assert isinstance(switch_stmt.cases[0].condition, VariableNode)
    assert switch_stmt.cases[0].condition.name == "active"
    assert switch_stmt.cases[1].condition is None


def test_for_parsing():
    code = """
    fn main():
        for var i: Int = 0; i < 10; i = i + 1:
            print(i)
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("For parsing not implemented.")


def test_c_style_for_update_parses_array_and_member_assignment_targets():
    code = """
    fn main():
        var value = 1
        for var i: Int = 0; i < 4; items[i] += value:
            pass
        for var i: Int = 0; i < 4; object.field = value:
            pass
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

    assert isinstance(member_loop, ForNode)
    assert isinstance(member_loop.update, AssignmentNode)
    assert member_loop.update.operator == "="
    assert isinstance(member_loop.update.left, MemberAccessNode)
    assert member_loop.update.left.object.name == "object"
    assert member_loop.update.left.member == "field"
    assert member_loop.update.right.name == "value"


def test_for_in_iterable_parsing_preserves_loop_iterable():
    code = """
    fn main():
        for value in values:
            sink(value)
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    loop = ast.functions[0].body[0]

    assert isinstance(loop, RangeForNode)
    assert loop.name == "value"
    assert loop.iterable.name == "values"


def test_for_in_range_parsing_preserves_range_call():
    code = """
    fn main():
        for i in range(4, 0, -1):
            sink(i)
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    loop = ast.functions[0].body[0]

    assert isinstance(loop, RangeForNode)
    assert loop.name == "i"
    assert loop.iterable.name == "range"
    assert loop.iterable.args[:2] == ["4", "0"]
    assert isinstance(loop.iterable.args[2], UnaryOpNode)
    assert loop.iterable.args[2].op == "-"
    assert loop.iterable.args[2].operand == "1"


def test_comptime_for_parsing_preserves_loop_shape():
    code = """
    fn main():
        comptime for value in values:
            sink(value)
        comptime for var i: Int = 0; i < 4; i = i + 1:
            sink(i)
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "main")
    range_loop = function.body[0]
    c_style_loop = function.body[1]

    assert isinstance(range_loop, RangeForNode)
    assert getattr(range_loop, "is_comptime", False)
    assert range_loop.name == "value"
    assert range_loop.iterable.name == "values"

    assert isinstance(c_style_loop, ForNode)
    assert getattr(c_style_loop, "is_comptime", False)
    assert c_style_loop.init.name == "i"
    assert c_style_loop.condition.op == "<"
    assert isinstance(c_style_loop.update, AssignmentNode)


def test_while_parsing():
    code = """
    fn main():
        var i: Int = 0
        while i < 10:
            i = i + 1
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("While parsing not implemented.")


def test_else_parsing():
    code = """
    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        if input.in_position.x > 0.5:
            output.out_color = input.in_position
        else:
            output.out_color = float4(0.0, 0.0, 0.0, 1.0)
        return output
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Else parsing not implemented.")


def test_function_call_parsing():
    code = """
    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        output.out_color = normalize(input.in_position)
        return output
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Function call parsing not implemented.")


def test_method_call_chain_preserves_receiver():
    code = """
    fn main():
        let channel = texture.sample(coord).x
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    expression = function.body[0].initial_value

    assert isinstance(expression, MemberAccessNode)
    assert expression.member == "x"
    assert isinstance(expression.object, MethodCallNode)
    assert expression.object.method == "sample"
    assert expression.object.object.name == "texture"
    assert expression.object.args[0].name == "coord"


def test_parenthesized_method_call_chain_parsing():
    code = """
    fn main():
        let channel = (texture.sample(coord)).x
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    expression = function.body[0].initial_value

    assert isinstance(expression, MemberAccessNode)
    assert expression.member == "x"
    assert isinstance(expression.object, MethodCallNode)
    assert expression.object.method == "sample"
    assert expression.object.object.name == "texture"


def test_as_cast_expression_parsing():
    code = """
    fn main():
        let i: Int = 2
        let x: Float = i as Float
        let y = (i + 1) as Float
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    x_init = function.body[1].initial_value
    y_init = function.body[2].initial_value

    assert isinstance(x_init, CastNode)
    assert x_init.target_type == "Float"
    assert isinstance(x_init.expression, VariableNode)
    assert x_init.expression.name == "i"
    assert isinstance(y_init, CastNode)
    assert y_init.target_type == "Float"
    assert isinstance(y_init.expression, BinaryOpNode)


def test_def_function_parsing():
    code = """
    def helper(x: Float32) -> Float32:
        return x
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "helper")

    assert function.return_type == "Float32"
    assert [(param.vtype, param.name) for param in function.params] == [
        ("Float32", "x")
    ]
    assert isinstance(function.body[0], ReturnNode)


def test_generic_function_signature_parsing():
    code = """
    fn build(
        value: SIMD[DType.float32, 4],
        Matrix[DType.float32, 4, 4] transform @binding(0)
    ) -> Matrix[DType.float32, 4, 4]:
        return transform
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "build")

    assert function.return_type == "Matrix[DType.float32, 4, 4]"
    assert [(param.vtype, param.name) for param in function.params] == [
        ("SIMD[DType.float32, 4]", "value"),
        ("Matrix[DType.float32, 4, 4]", "transform"),
    ]
    assert [attr.name for attr in function.params[1].attributes] == ["binding"]
    assert function.params[1].attributes[0].args == ["0"]


def test_parenthesized_where_clause_with_and_constraints_parsing():
    code = """
    def outer_product_acc[
        dtype: DType
    ](
        res: TileTensor[mut=True, ...],
        lhs: TileTensor,
        rhs: TileTensor,
    ) where (
        type_of(res).flat_rank == 2
        and type_of(lhs).flat_rank == 1
        and type_of(rhs).shape_known
    ):
        pass
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "outer_product_acc")

    assert function.where_clause is not None
    assert "&&" in function.where_clause
    assert "type_of" in function.where_clause
    assert function.body


def test_else_if_parsing():
    code = """
    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        if input.in_position.x > 0.5:
            output.out_color = input.in_position
        elif input.in_position.x == 0.5:
            output.out_color = float4(1.0, 1.0, 1.0, 1.0)
        else:
            output.out_color = float4(0.0, 0.0, 0.0, 1.0)
        return output
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Else-if parsing not implemented.")


def test_assignment_ops_parsing():
    code = """
    fn fragment_main():
        var a: Int = 5
        a += 3
        a -= 2
        a *= 4
        a /= 2
        a %= 3
        a ^= 1
        a |= 2
        a &= 7
        a <<= 1
        a >>= 2
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Assignment operators parsing not implemented.")


def test_bitwise_ops_parsing():
    code = """
    fn main():
        let val: UInt32 = 0x01
        let result = val | 0x02
        let result2 = val & 0x04
        let result3 = val ^ 0x08
        let result4 = ~val
        let result5 = val << 2
        let result6 = val >> 1
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Bitwise operators parsing not implemented.")


def test_logical_ops_parsing():
    code = """
    fn main():
        let val_0: Bool = True
        let val_1 = val_0 and False
        let val_2 = val_0 or False
        let val_3 = not val_0
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        function = find_function(ast, "main")
        and_expr = function.body[1].initial_value
        or_expr = function.body[2].initial_value
        not_expr = function.body[3].initial_value

        assert isinstance(and_expr, BinaryOpNode)
        assert and_expr.op == "&&"
        assert and_expr.right == "false"
        assert isinstance(or_expr, BinaryOpNode)
        assert or_expr.op == "||"
        assert or_expr.right == "false"
        assert isinstance(not_expr, UnaryOpNode)
        assert not_expr.op == "!"
    except SyntaxError:
        pytest.fail("Logical operators parsing not implemented.")


def test_switch_case_parsing():
    code = """
    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        switch input.value:
            case 1:
                output.out_color = float4(1.0, 0.0, 0.0, 1.0)
                break
            case 2:
                output.out_color = float4(0.0, 1.0, 0.0, 1.0)
                break
            default:
                output.out_color = float4(0.0, 0.0, 1.0, 1.0)
                break
        return output
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Switch-case parsing not implemented.")


def test_switch_rejects_duplicate_default_labels():
    code = """
    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        switch input.value:
            default:
                output.out_color = float4(0.0, 0.0, 1.0, 1.0)
            case 1:
                output.out_color = float4(1.0, 0.0, 0.0, 1.0)
            default:
                output.out_color = float4(0.0, 1.0, 0.0, 1.0)
        return output
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="duplicate default"):
        parse_code(tokens)


def test_import_parsing():
    code = """
    import math
    import simd as s
    from tensor import Tensor
    from package.module import A, B as C

    fn main():
        let result = math.sin(0.5)
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        imports = ast.includes

        assert imports[0].module_name == "math"
        assert imports[0].items == []
        assert imports[0].alias is None
        assert imports[1].module_name == "simd"
        assert imports[1].items == []
        assert imports[1].alias == "s"
        assert imports[2].module_name == "tensor"
        assert imports[2].items == ["Tensor"]
        assert imports[2].alias is None
        assert imports[3].module_name == "package.module"
        assert imports[3].items == ["A", "B as C"]
    except SyntaxError:
        pytest.fail("Import parsing not implemented.")


def test_relative_import_path_parsing_from_modular_corpus():
    code = """
    from .add_constant import *
    from .. import PathLike
    from .._linux_x86 import _stat as _stat_linux_x86
    import .warp

    fn main():
        pass
    """
    ast = parse_code(tokenize_code(code))

    assert [import_node.module_name for import_node in ast.includes] == [
        ".add_constant",
        "..",
        ".._linux_x86",
        ".warp",
    ]
    assert ast.includes[0].items == ["*"]
    assert ast.includes[1].items == ["PathLike"]
    assert ast.includes[2].items == ["_stat as _stat_linux_x86"]
    assert ast.includes[3].items == []


def test_hash_prefixed_lines_do_not_affect_import_parsing():
    code = """
    #define ENABLE_MATH 1
    #if ENABLE_MATH
    import math
    from tensor import Tensor
    #endif

    fn main():
        let result = math.sin(0.5)
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    imports = ast.includes

    assert [import_node.module_name for import_node in imports] == ["math", "tensor"]
    assert imports[0].items == []
    assert imports[1].items == ["Tensor"]


def test_data_types_parsing():
    code = """
    fn main():
        var a: Int8 = 10
        var b: Int16 = 20
        var c: Int32 = 30
        var d: Int64 = 40
        var e: UInt8 = 50
        var f: UInt16 = 60
        var g: UInt32 = 70
        var h: UInt64 = 80
        var i: Float16 = 1.5
        var j: Float32 = 2.5
        var k: Float64 = 3.5
        var l: Bool = True
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Data types parsing not implemented.")


def test_vector_types_parsing():
    code = """
    fn main():
        var a: float2 = float2(1.0, 2.0)
        var b: float3 = float3(1.0, 2.0, 3.0)
        var c: float4 = float4(1.0, 2.0, 3.0, 4.0)
        var d: int2 = int2(1, 2)
        var e: int3 = int3(1, 2, 3)
        var f: int4 = int4(1, 2, 3, 4)
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Vector types parsing not implemented.")


def test_generic_constructor_expression_parsing():
    code = """
    fn main():
        let color = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
        let values = InlineArray[Float32, 4](1.0, 2.0, 3.0, 4.0)
        let matrix = Matrix[DType.float64, 2, 2](1.0, 0.0, 0.0, 1.0)
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    color = function.body[0].initial_value
    values = function.body[1].initial_value
    matrix = function.body[2].initial_value

    assert isinstance(color, VectorConstructorNode)
    assert color.type_name == "SIMD[DType.float32, 4]"
    assert color.args == ["1.0", "2.0", "3.0", "4.0"]
    assert isinstance(values, VectorConstructorNode)
    assert values.type_name == "InlineArray[Float32, 4]"
    assert values.args == ["1.0", "2.0", "3.0", "4.0"]
    assert isinstance(matrix, VectorConstructorNode)
    assert matrix.type_name == "Matrix[DType.float64, 2, 2]"
    assert matrix.args == ["1.0", "0.0", "0.0", "1.0"]


def test_nested_generic_type_annotation_parsing():
    code = """
    fn main():
        var values: InlineArray[SIMD[DType.float32, 4], 2]
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")

    assert function.body[0].vtype == "InlineArray[SIMD[DType.float32, 4], 2]"


def test_ternary_operator_parsing():
    code = """
    fn main():
        let x: Float32 = 0.5
        let result = x > 0.0 ? 1.0 : 0.0
        let variable_result = flag ? yes : no
        let native_result = yes if flag else no
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        function = find_function(ast, "main")
        ternary = function.body[2].initial_value
        native_ternary = function.body[3].initial_value

        assert isinstance(ternary, TernaryOpNode)
        assert ternary.condition.name == "flag"
        assert ternary.true_expr.name == "yes"
        assert ternary.false_expr.name == "no"
        assert isinstance(native_ternary, TernaryOpNode)
        assert native_ternary.condition.name == "flag"
        assert native_ternary.true_expr.name == "yes"
        assert native_ternary.false_expr.name == "no"
    except SyntaxError:
        pytest.fail("Ternary operator parsing not implemented.")


def test_at_attributes_attach_to_declarations():
    code = """
    @value
    struct MyStruct:
        @position
        var data: Int32

    @position
    var global_data: Float32
    var bound_texture: Texture2D @binding(0)
    @group(0)
    @binding(1)
    var sampler: SamplerState
    @group(2)
    var combined_texture: Texture2D @binding(3)

    @compute_shader
    fn kernel():
        pass
    """
    ast = parse_code(tokenize_code(code))
    struct_node = ast.functions[0]
    function = find_function(ast, "kernel")
    global_data, bound_texture, sampler, combined_texture = ast.global_variables

    assert [attr.name for attr in struct_node.attributes] == ["value"]
    assert [attr.name for attr in struct_node.members[0].attributes] == ["position"]
    assert [attr.name for attr in global_data.attributes] == ["position"]
    assert [attr.name for attr in bound_texture.attributes] == ["binding"]
    assert bound_texture.attributes[0].args == ["0"]
    assert [attr.name for attr in sampler.attributes] == ["group", "binding"]
    assert sampler.attributes[0].args == ["0"]
    assert sampler.attributes[1].args == ["1"]
    assert [attr.name for attr in combined_texture.attributes] == [
        "group",
        "binding",
    ]
    assert combined_texture.attributes[0].args == ["2"]
    assert combined_texture.attributes[1].args == ["3"]
    assert [attr.name for attr in function.attributes] == ["compute_shader"]


def test_bracket_attributes_attach_to_function_declarations():
    code = """
    [[compute_shader]]
    fn run():
        pass
    """

    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "run")

    assert [attr.name for attr in function.attributes] == ["compute_shader"]


def test_colon_class_parsing_preserves_members_methods_and_attributes():
    code = """
    @storage
    class Accumulator(Base):
        @position
        var value: Int
        Matrix[DType.float32, 4, 4] transform @binding(0)
        samples: InlineArray[SIMD[DType.float32, 4], 2]
        def reset():
            pass
    """
    ast = parse_code(tokenize_code(code))
    class_node = find_class(ast, "Accumulator")

    assert class_node.base_classes == ["Base"]
    assert [attr.name for attr in class_node.attributes] == ["storage"]
    assert len(class_node.members) == 3
    assert class_node.members[0].name == "value"
    assert [attr.name for attr in class_node.members[0].attributes] == ["position"]
    assert [(member.vtype, member.name) for member in class_node.members[1:]] == [
        ("Matrix[DType.float32, 4, 4]", "transform"),
        ("InlineArray[SIMD[DType.float32, 4], 2]", "samples"),
    ]
    assert [attr.name for attr in class_node.members[1].attributes] == ["binding"]
    assert class_node.members[1].attributes[0].args == ["0"]
    assert len(class_node.methods) == 1
    assert class_node.methods[0].name == "reset"


def test_brace_class_parsing_preserves_members_and_methods():
    code = """
    class Accumulator {
        var value: Int
        SIMD[DType.float32, 4] tint @color;
        weights: InlineArray[Float32, 4];
        fn reset():
            pass
    }
    """
    ast = parse_code(tokenize_code(code))
    class_node = find_class(ast, "Accumulator")

    assert len(class_node.members) == 3
    assert class_node.members[0].name == "value"
    assert [(member.vtype, member.name) for member in class_node.members[1:]] == [
        ("SIMD[DType.float32, 4]", "tint"),
        ("InlineArray[Float32, 4]", "weights"),
    ]
    assert [attr.name for attr in class_node.members[1].attributes] == ["color"]
    assert len(class_node.methods) == 1
    assert class_node.methods[0].name == "reset"


def test_constant_buffer_colon_parsing():
    code = """
    @group(0)
    constant Params:
        @binding(0)
        var exposure: Float32 @offset(0)
        count: Int @binding(1)
    """
    ast = parse_code(tokenize_code(code))
    cbuffer = find_constant_buffer(ast, "Params")

    assert [attr.name for attr in cbuffer.attributes] == ["group"]
    assert cbuffer.attributes[0].args == ["0"]
    assert [(member.vtype, member.name) for member in cbuffer.members] == [
        ("Float32", "exposure"),
        ("Int", "count"),
    ]
    assert [attr.name for attr in cbuffer.members[0].attributes] == [
        "binding",
        "offset",
    ]
    assert cbuffer.members[0].attributes[0].args == ["0"]
    assert cbuffer.members[0].attributes[1].args == ["0"]
    assert [attr.name for attr in cbuffer.members[1].attributes] == ["binding"]
    assert cbuffer.members[1].attributes[0].args == ["1"]


def test_constant_buffer_brace_parsing_with_layout_tokens():
    code = """
    constant Params {
        @binding(0)
        Float32 exposure @offset(0);
        Int count @binding(1);
        Matrix[DType.float32, 4, 4] transform @binding(2);
        InlineArray[SIMD[DType.float32, 4], 2] samples;
    }
    """
    ast = parse_code(tokenize_code(code))
    cbuffer = find_constant_buffer(ast, "Params")

    assert [(member.vtype, member.name) for member in cbuffer.members] == [
        ("Float32", "exposure"),
        ("Int", "count"),
        ("Matrix[DType.float32, 4, 4]", "transform"),
        ("InlineArray[SIMD[DType.float32, 4], 2]", "samples"),
    ]
    assert [attr.name for attr in cbuffer.members[0].attributes] == [
        "binding",
        "offset",
    ]
    assert [attr.name for attr in cbuffer.members[1].attributes] == ["binding"]
    assert [attr.name for attr in cbuffer.members[2].attributes] == ["binding"]
    assert cbuffer.members[2].attributes[0].args == ["2"]


def test_array_access_parsing():
    code = """
    fn main():
        var values: StaticTuple[Float32, 4]
        values[0] = 1.0
        values[1] = 2.0
        let result = values[0] + values[1]
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Array access parsing not implemented.")


def test_array_member_access_chain_parsing():
    code = """
    fn main():
        let channel = values[0].x
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    expression = function.body[0].initial_value

    assert isinstance(expression, MemberAccessNode)
    assert expression.member == "x"
    assert isinstance(expression.object, ArrayAccessNode)
    assert expression.object.array.name == "values"
    assert expression.object.index == "0"


def test_gpu_tile_tensor_multi_index_access_parsing():
    code = """
    from std.gpu import thread_idx

    fn tiled_load(tile: TileTensor, matrix: TileTensor):
        tile[thread_idx.y, thread_idx.x] = matrix[
            thread_idx.y,
            thread_idx.x,
        ]
        let value = tile[thread_idx.y, thread_idx.x]
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "tiled_load")
    store = function.body[0]
    load = function.body[1].initial_value

    assert isinstance(store.left, ArrayAccessNode)
    assert isinstance(store.left.index, TupleNode)
    assert [index.member for index in store.left.index.elements] == ["y", "x"]

    assert isinstance(store.right, ArrayAccessNode)
    assert isinstance(store.right.index, TupleNode)
    assert [index.member for index in store.right.index.elements] == ["y", "x"]

    assert isinstance(load, ArrayAccessNode)
    assert isinstance(load.index, TupleNode)


def test_parenthesized_call_indexing_parsing():
    code = """
    fn main():
        let value = (factory())[i]
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    expression = function.body[0].initial_value

    assert isinstance(expression, ArrayAccessNode)
    assert expression.array.name == "factory"
    assert expression.index.name == "i"


def test_member_access_parsing():
    code = """
    struct Vector3:
        var x: Float32
        var y: Float32
        var z: Float32

    fn main():
        var v: Vector3
        v.x = 1.0
        v.y = 2.0
        v.z = 3.0
        let length = sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Member access parsing not implemented.")


def test_function_with_parameters_parsing():
    code = """
    fn add(a: Float32, b: Float32) -> Float32:
        return a + b

    fn multiply(x: Float32, y: Float32, z: Float32) -> Float32:
        return x * y * z

    fn main():
        let result1 = add(1.0, 2.0)
        let result2 = multiply(2.0, 3.0, 4.0)
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Function with parameters parsing not implemented.")


def test_function_parameters_allow_trailing_comma():
    code = """
    def add_10(
        output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ):
        output[0] = a[0] + 10.0
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "add_10")

    assert [param.name for param in function.params] == ["output", "a"]
    assert function.params[0].vtype == "UnsafePointer[Scalar[dtype], MutAnyOrigin]"


def test_comptime_declarations_and_raises_function_parse():
    code = """
    comptime SIZE = 4
    comptime dtype = DType.float32
    comptime THREADS_PER_BLOCK = (3, 3)

    def main() raises:
        comptime BLOCK_SIZE = 16
        var value = BLOCK_SIZE
        with DeviceContext() as ctx:
            value = value + 1
        comptime if value > 0:
            value = value + SIZE
        else:
            raise Error("bad value", value)
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")

    assert [node.name for node in ast.global_variables] == [
        "SIZE",
        "dtype",
        "THREADS_PER_BLOCK",
    ]
    assert all(getattr(node, "is_comptime", False) for node in ast.global_variables)
    assert isinstance(ast.global_variables[2].initial_value, TupleNode)
    assert getattr(function.body[0], "is_comptime", False)
    assert isinstance(function.body[2], WithNode)
    assert function.body[2].alias == "ctx"
    assert isinstance(function.body[3], IfNode)
    assert isinstance(function.body[3].else_body[0], FunctionCallNode)
    assert function.body[3].else_body[0].name == "raise"


def test_alias_declarations_parse_as_comptime_aliases():
    code = """
    alias THREADS_PER_BLOCK = 256
    alias dtype = DType.float32

    def kernel():
        alias LOCAL_BLOCK = THREADS_PER_BLOCK
        var lane = thread_idx.x
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "kernel")

    assert [node.name for node in ast.global_variables] == [
        "THREADS_PER_BLOCK",
        "dtype",
    ]
    assert all(getattr(node, "is_comptime", False) for node in ast.global_variables)
    assert all(getattr(node, "is_alias", False) for node in ast.global_variables)
    assert ast.global_variables[0].initial_value == "256"
    assert isinstance(ast.global_variables[1].initial_value, MemberAccessNode)

    local_alias = function.body[0]
    assert isinstance(local_alias, VariableDeclarationNode)
    assert local_alias.name == "LOCAL_BLOCK"
    assert getattr(local_alias, "is_comptime", False)
    assert getattr(local_alias, "is_alias", False)
    assert not local_alias.is_var


def test_bare_annotated_assignment_with_initializer_parsing():
    code = """
    from std.gpu import global_idx

    def kernel(size: Int):
        idx: UInt = global_idx.x
        limit: Int = size
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "kernel")

    idx_decl = function.body[0]
    limit_decl = function.body[1]

    assert isinstance(idx_decl, VariableDeclarationNode)
    assert idx_decl.name == "idx"
    assert idx_decl.vtype == "UInt"
    assert idx_decl.is_var
    assert isinstance(idx_decl.initial_value, MemberAccessNode)
    assert idx_decl.initial_value.object.name == "global_idx"
    assert idx_decl.initial_value.member == "x"

    assert isinstance(limit_decl, VariableDeclarationNode)
    assert limit_decl.name == "limit"
    assert limit_decl.vtype == "Int"
    assert limit_decl.initial_value.name == "size"


def test_comptime_assert_statement_parse():
    code = """
    def outer_product_acc(res: TileTensor, size: Int):
        comptime assert(type_of(res).flat_rank == 2)
        comptime assert(size > 0, "bad size")
        comptime assert (
            dtype.is_floating_point()
        ), "dtype must be a floating-point type"
    """

    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "outer_product_acc")
    rank_assert = function.body[0]
    size_assert = function.body[1]
    dtype_assert = function.body[2]

    assert isinstance(rank_assert, FunctionCallNode)
    assert getattr(rank_assert, "is_comptime", False)
    assert rank_assert.name == "assert"
    assert isinstance(rank_assert.args[0], BinaryOpNode)
    assert rank_assert.args[0].op == "=="

    assert isinstance(size_assert, FunctionCallNode)
    assert getattr(size_assert, "is_comptime", False)
    assert size_assert.args[1] == '"bad size"'

    assert isinstance(dtype_assert, FunctionCallNode)
    assert getattr(dtype_assert, "is_comptime", False)
    assert dtype_assert.name == "assert"
    assert isinstance(dtype_assert.args[0], MethodCallNode)
    assert dtype_assert.args[0].method == "is_floating_point"
    assert dtype_assert.args[1] == '"dtype must be a floating-point type"'


def test_keyword_style_comptime_assert_statement_parse():
    code = """
    def main(size: Int):
        comptime assert has_accelerator(), "This example requires a supported GPU"
        comptime assert size > 0, "bad size"
    """

    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    gpu_assert = function.body[0]
    size_assert = function.body[1]

    assert isinstance(gpu_assert, FunctionCallNode)
    assert getattr(gpu_assert, "is_comptime", False)
    assert gpu_assert.name == "assert"
    assert isinstance(gpu_assert.args[0], FunctionCallNode)
    assert gpu_assert.args[0].name == "has_accelerator"
    assert gpu_assert.args[1] == '"This example requires a supported GPU"'

    assert isinstance(size_assert, FunctionCallNode)
    assert getattr(size_assert, "is_comptime", False)
    assert isinstance(size_assert.args[0], BinaryOpNode)
    assert size_assert.args[0].op == ">"
    assert size_assert.args[1] == '"bad size"'


def test_nested_decorator_and_generic_function_signature_parse():
    code = """
    @compiler.register("vector_addition")
    struct VectorAddition:
        value: Int

    @staticmethod
    def execute[
        target: StaticString,
    ](
        ctx: DeviceContext,
    ) raises:
        @parameter
        def kernel(length: Int):
            pass
    """
    ast = parse_code(tokenize_code(code))
    struct_node = ast.structs[0]
    function = find_function(ast, "execute")

    assert [attr.name for attr in struct_node.attributes] == ["compiler.register"]
    assert [attr.name for attr in function.attributes] == ["staticmethod"]
    assert [attr.name for attr in function.body[0].attributes] == ["parameter"]


def test_variable_declarations_parsing():
    code = """
    fn main():
        let constant_value: Int = 42
        var mutable_value: Int = 10
        var inferred_type = 3.14
        let computed_value = constant_value + mutable_value
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Variable declarations parsing not implemented.")


def test_nested_expressions_parsing():
    code = """
    fn main():
        let a: Float32 = 1.0
        let b: Float32 = 2.0
        let c: Float32 = 3.0
        let result = ((a + b) * c) / (a - b + c)
        let complex_expr = sin(cos(tan(a))) + sqrt(b * b + c * c)
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Nested expressions parsing not implemented.")


def test_mod_parsing():
    code = """
    fn main():
        let a: Int = 10 % 3  # Basic modulus
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Modulus operator parsing not implemented")


def test_bitwise_not_parsing():
    code = """
    fn main():
        let a: Int = 5
        let b: Int = ~a  # Bitwise NOT
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Bitwise NOT operator parsing not implemented")


def test_string_literals_parsing():
    code = """
    fn main():
        let message: String = "Hello, Mojo!"
        let path: String = "path/to/file.txt"
        print(message)
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("String literals parsing not implemented.")


def test_numeric_literals_parsing():
    code = """
    fn main():
        let decimal: Int = 42
        let hex: Int = 0x2A
        let binary: Int = 0b101010
        let octal: Int = 0o52
        let float_val: Float32 = 3.14159
        let scientific: Float32 = 1.23e-4
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Numeric literals parsing not implemented.")


def test_if_block_stops_at_dedent():
    code = """
    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        if input.value > 0:
            pass
        return output
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "fragment_main")

    assert len(function.body) == 3
    assert isinstance(function.body[1], IfNode)
    assert isinstance(function.body[2], ReturnNode)


def test_nested_if_siblings_stop_at_dedent():
    code = """
    fn main():
        var i: Int = 0
        while i < 10:
            if i == 3:
                continue
            if i == 7:
                break
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    loop = function.body[1]

    assert isinstance(loop, WhileNode)
    assert len(loop.body) == 2
    assert isinstance(loop.body[0], IfNode)
    assert isinstance(loop.body[0].if_body[0], ContinueNode)
    assert isinstance(loop.body[1], IfNode)
    assert isinstance(loop.body[1].if_body[0], BreakNode)


def test_switch_block_stops_at_dedent():
    code = """
    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        switch input.value:
            case 1:
                output.out_color = float4(1.0, 0.0, 0.0, 1.0)
                break
            default:
                output.out_color = float4(0.0, 0.0, 1.0, 1.0)
                break
        return output
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "fragment_main")

    assert len(function.body) == 3
    assert isinstance(function.body[1], SwitchNode)
    assert isinstance(function.body[2], ReturnNode)


@pytest.mark.parametrize(
    "code",
    [
        """
        fn main():
            pass
        stray_token
        """,
        """
        fn main():
            pass
        1 + 2
        """,
        """
        return value
        """,
    ],
)
def test_rejects_unexpected_top_level_tokens(code):
    with pytest.raises(SyntaxError, match="Unexpected top-level token"):
        parse_code(tokenize_code(code))


@pytest.mark.parametrize(
    "code",
    [
        """
        fn main():
            let x = a b
        """,
        """
        fn main():
            return a b
        """,
        """
        fn main():
            a b
        """,
        """
        struct Bad:
            var value: Float32 extra
        """,
    ],
)
def test_rejects_trailing_tokens_after_simple_statement(code):
    with pytest.raises(SyntaxError, match="Expected end of statement"):
        parse_code(tokenize_code(code))


if __name__ == "__main__":
    pytest.main()
