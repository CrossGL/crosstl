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
    ListComprehensionNode,
    ListLiteralNode,
    MemberAccessNode,
    MethodCallNode,
    PassNode,
    RangeForNode,
    ReturnNode,
    SliceNode,
    StructNode,
    SwitchNode,
    TernaryOpNode,
    TraitNode,
    TryExceptNode,
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


def find_struct(ast, name: str):
    for node in ast.structs:
        if isinstance(node, StructNode) and node.name == name:
            return node
    raise AssertionError(f"Struct {name} not found")


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


def test_bracketed_ref_parameter_convention_parse_from_modular_amd_helpers():
    # Reduced from https://github.com/modular/mojo.git commit
    # 7aa053560034c8c5b4f9acb0a5b450e79d2f7c18,
    # max/examples/custom_ops/kernels/amd_helpers.mojo MMATileBuffers.__init__.
    code = """
    struct MMATileBuffers:
        @always_inline
        def __init__(
            out self,
            ref[Self.tensor_origin] tensor: Self.tensor_type,
            warp_idx: Int,
        ):
            pass
    """
    ast = parse_code(tokenize_code(code))
    init_method = find_struct(ast, "MMATileBuffers").methods[0]

    assert [
        (param.name, param.vtype, param.parameter_convention)
        for param in init_method.params
    ] == [
        ("self", "", "out"),
        ("tensor", "Self.tensor_type", "ref[Self.tensor_origin]"),
        ("warp_idx", "Int", None),
    ]


def test_variadic_and_deinit_parameters_parse_from_current_docs():
    code = """
    struct GenericArray[ElementType: Copyable & ImplicitlyDestructible]:
        var size: Int

        def __init__(out self, var *elements: Self.ElementType):
            self.size = len(elements)

        def __del__(deinit self):
            pass

        def __getitem__(self, i: Int) raises -> ref[self] Self.ElementType:
            return self.data[i]
    """
    ast = parse_code(tokenize_code(code))
    struct_node = find_struct(ast, "GenericArray")
    init_method = struct_node.methods[0]
    del_method = struct_node.methods[1]
    getitem_method = struct_node.methods[2]

    assert [
        (
            param.name,
            param.vtype,
            param.parameter_convention,
            getattr(param, "is_variadic", False),
        )
        for param in init_method.params
    ] == [
        ("self", "", "out", False),
        ("elements", "Self.ElementType", "var", True),
    ]
    assert [
        (param.name, param.vtype, param.parameter_convention)
        for param in del_method.params
    ] == [("self", "", "deinit")]
    assert [(param.name, param.vtype) for param in getitem_method.params] == [
        ("self", ""),
        ("i", "Int"),
    ]
    assert getitem_method.return_type == "ref[self] Self.ElementType"


def test_variadic_parameter_without_convention_parse_from_real_world_mojo():
    code = """
    struct Matrix:
        fn __init__(out self, *dims: Int):
            pass
    """
    ast = parse_code(tokenize_code(code))
    init_method = find_struct(ast, "Matrix").methods[0]

    assert [
        (
            param.name,
            param.vtype,
            param.parameter_convention,
            getattr(param, "is_variadic", False),
        )
        for param in init_method.params
    ] == [
        ("self", "", "out", False),
        ("dims", "Int", None, True),
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


def test_struct_base_list_parsing_from_modular_corpus():
    code = """
    @fieldwise_init
    struct NullWriter(Writer):
        var array: InlineArray[Byte, 1024]

        def write_string(mut self, string: StringSlice):
            pass

    struct Complex(
        Boolable,
        Equatable,
        Writable,
    ):
        var re: Float64
        var im: Float64
    """
    ast = parse_code(tokenize_code(code))
    null_writer = find_struct(ast, "NullWriter")
    complex_struct = find_struct(ast, "Complex")

    assert [attr.name for attr in null_writer.attributes] == ["fieldwise_init"]
    assert null_writer.base_classes == ["Writer"]
    assert [(member.vtype, member.name) for member in null_writer.members] == [
        ("InlineArray[Byte, 1024]", "array")
    ]
    assert [method.name for method in null_writer.methods] == ["write_string"]
    assert complex_struct.base_classes == ["Boolable", "Equatable", "Writable"]
    assert [(member.vtype, member.name) for member in complex_struct.members] == [
        ("Float64", "re"),
        ("Float64", "im"),
    ]


def test_struct_generic_parameter_list_parsing_from_modular_corpus():
    code = """
    struct AddConstant[value: Int]:
        var input: Int

    struct SplatList[
        T: ImplicitlyCopyable & ImplicitlyDestructible,
        *,
        fill: T,
        length: Int = 5,
    ]:
        var items: List[Self.T]

    struct Grid[rows: Int, cols: Int](Copyable, Writable):
        var cells: InlineArray[Int, Self.rows * Self.cols]
    """
    ast = parse_code(tokenize_code(code))
    add_constant = find_struct(ast, "AddConstant")
    splat_list = find_struct(ast, "SplatList")
    grid = find_struct(ast, "Grid")

    assert add_constant.generic_parameters == "[value:Int]"
    assert [(member.vtype, member.name) for member in add_constant.members] == [
        ("Int", "input")
    ]

    assert splat_list.generic_parameters == (
        "[T:ImplicitlyCopyable&ImplicitlyDestructible, *, fill:T, " "length:Int=5]"
    )
    assert [(member.vtype, member.name) for member in splat_list.members] == [
        ("List[Self.T]", "items")
    ]

    assert grid.generic_parameters == "[rows:Int, cols:Int]"
    assert grid.base_classes == ["Copyable", "Writable"]
    assert [(member.vtype, member.name) for member in grid.members] == [
        ("InlineArray[Int, Self.rows*Self.cols]", "cells")
    ]


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


def test_tuple_for_target_parsing_from_modular_pmpp_examples():
    code = """
    from std.itertools import product

    fn initialize_image(width: Int, height: Int):
        for row, col in product(range(height), range(width)):
            sink(row, col)
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    loop = find_function(ast, "initialize_image").body[0]

    assert isinstance(loop, RangeForNode)
    assert isinstance(loop.name, TupleNode)
    assert [element.name for element in loop.name.elements] == ["row", "col"]
    assert isinstance(loop.iterable, FunctionCallNode)
    assert loop.iterable.name == "product"


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


def test_mojo_gpu_puzzles_async_shared_memory_copy_call_parsing():
    code = """
    def matmul_idiomatic_tiled[
        rows: Int,
        cols: Int,
        inner: Int,
        dtype: DType = DType.float32,
    ](
        output: TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin],
    ):
        var out_tile = output.tile[MATMUL_BLOCK_DIM_XY, MATMUL_BLOCK_DIM_XY](
            block_idx.y, block_idx.x
        )
        comptime for idx in range(
            (inner + MATMUL_BLOCK_DIM_XY - 1) // MATMUL_BLOCK_DIM_XY
        ):
            copy_dram_to_sram_async[
                thread_layout=load_a_layout,
                num_threads=MATMUL_NUM_THREADS,
                block_dim_count=MATMUL_BLOCK_DIM_COUNT,
            ](a_shared, a_tile)
            async_copy_wait_all()
            barrier()
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "matmul_idiomatic_tiled")
    loop = function.body[1]
    copy_call = loop.body[0]

    assert function.params[0].vtype == (
        "TileTensor[mut=true, dtype, OutLayout, MutAnyOrigin]"
    )
    assert isinstance(loop, RangeForNode)
    assert getattr(loop, "is_comptime", False)
    assert isinstance(copy_call, VectorConstructorNode)
    assert copy_call.type_name == (
        "copy_dram_to_sram_async[thread_layout=load_a_layout, "
        "num_threads=MATMUL_NUM_THREADS, block_dim_count=MATMUL_BLOCK_DIM_COUNT]"
    )
    assert [arg.name for arg in copy_call.args] == ["a_shared", "a_tile"]


def test_modular_top_k_type_of_parameter_type_parsing():
    # Reduced from https://github.com/modular/modular.git commit
    # 7aa053560034c8c5b4f9acb0a5b450e79d2f7c18,
    # max/examples/custom_ops/kernels/top_k.mojo top_k_gpu nested kernel.
    code = """
    def execute():
        @parameter
        def top_k_gpu(
            out_vals: type_of(out_vals_tensor),
            out_idxs: type_of(out_idxs_tensor),
            in_vals: type_of(in_vals_tensor),
        ):
            pass
    """
    ast = parse_code(tokenize_code(code))
    kernel = find_function(ast, "execute").body[0]

    assert isinstance(kernel, FunctionNode)
    assert [attr.name for attr in kernel.attributes] == ["parameter"]
    assert [(param.name, param.vtype) for param in kernel.params] == [
        ("out_vals", "type_of(out_vals_tensor)"),
        ("out_idxs", "type_of(out_idxs_tensor)"),
        ("in_vals", "type_of(in_vals_tensor)"),
    ]


def test_statement_attributes_parse_from_real_world_mojo():
    code = """
    fn main():
        @parameter
        for k in range(n):
            sink(k)
    """
    ast = parse_code(tokenize_code(code))
    loop = find_function(ast, "main").body[0]

    assert isinstance(loop, RangeForNode)
    assert [attr.name for attr in loop.attributes] == ["parameter"]


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


def test_multiline_postfix_chain_parsing_from_modular_custom_ops():
    code = """
    fn fused_attention_cpu():
        var m_1 = (
            LayoutTensor[Q_tile.dtype, Layout(BN, 1), MutAnyOrigin]
            .stack_allocation()
            .fill(Scalar[Q_tile.dtype].MIN)
        )
    """
    ast = parse_code(tokenize_code(code))
    declaration = find_function(ast, "fused_attention_cpu").body[0]
    fill_call = declaration.initial_value

    assert isinstance(fill_call, MethodCallNode)
    assert fill_call.method == "fill"
    assert isinstance(fill_call.object, MethodCallNode)
    assert fill_call.object.method == "stack_allocation"
    assert isinstance(fill_call.object.object, ArrayAccessNode)


def test_multiline_index_expression_parsing_from_modular_pmpp_examples():
    code = """
    fn main():
        A[
            (
                tile_size
                - consumed
            )
            % tile_size
        ] = value
    """
    ast = parse_code(tokenize_code(code))
    store = find_function(ast, "main").body[0]

    assert isinstance(store.left, ArrayAccessNode)
    assert isinstance(store.left.index, BinaryOpNode)
    assert store.left.index.op == "%"


def test_prefixed_string_literal_attribute_parsing_from_modular_kernels():
    code = """
    @__name(t"gemv_kernel_{dtype}")
    def gemv_kernel():
        pass
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "gemv_kernel")

    assert function.attributes[0].name == "__name"
    assert function.attributes[0].args == ['t"gemv_kernel_{dtype}"']


def test_multiline_call_argument_expression_parsing_from_modular_pmpp_examples():
    code = """
    fn main():
        print(
            "Coarsening with contiguous partitioning (COARSE_FACTOR="
            + String(COARSE_FACTOR)
            + ")"
        )
    """
    ast = parse_code(tokenize_code(code))
    print_call = find_function(ast, "main").body[0]

    assert isinstance(print_call, FunctionCallNode)
    assert isinstance(print_call.args[0], BinaryOpNode)
    assert print_call.args[0].op == "+"


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


def test_function_effects_parse_from_modular_and_real_world_mojo():
    code = """
    def exported[M: Int]() abi("C"):
        pass

    fn _sum2[_nelts: Int](j: Int) unified{mut}:
        pass
    """
    ast = parse_code(tokenize_code(code))

    assert find_function(ast, "exported").body
    assert find_function(ast, "_sum2").body


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


def test_identity_expression_parsing_from_modular_kernels():
    code = """
    fn main():
        if elementwise_lambda_fn is None or fallback is not None:
            pass
    """
    ast = parse_code(tokenize_code(code))
    condition = find_function(ast, "main").body[0].condition

    assert isinstance(condition, BinaryOpNode)
    assert condition.op == "||"
    assert condition.left.op == "is"
    assert condition.right.op == "is not"


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


def test_keyword_like_qualified_names_parse_from_modular_kernels():
    code = """
    from linalg.matmul.gpu.sm100_structured.default.matmul_kernels import Kernel

    fn main():
        let value = config.default
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    expression = function.body[0].initial_value

    assert ast.includes[0].module_name == (
        "linalg.matmul.gpu.sm100_structured.default.matmul_kernels"
    )
    assert isinstance(expression, MemberAccessNode)
    assert expression.member == "default"


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


def test_empty_index_access_parse_from_layout_tensor_iterator_docs():
    code = """
    def main():
        var tile = iter[]
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    expression = function.body[0].initial_value

    assert isinstance(expression, ArrayAccessNode)
    assert expression.array.name == "iter"
    assert isinstance(expression.index, TupleNode)
    assert expression.index.elements == []


def test_slice_index_access_parse_from_modular_stdlib_slice_tests():
    # Reduced from https://github.com/modular/modular.git commit
    # 7aa053560034c8c5b4f9acb0a5b450e79d2f7c18,
    # mojo/stdlib/test/builtin/test_slice.mojo test_sliceable/test_slice_stringable.
    code = """
    def main():
        var new_slice = sliceable[1:"hello":4.0]
        var reverse = s[2::-1]
        var open_slice = s[::]
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    new_slice = function.body[0].initial_value
    reverse = function.body[1].initial_value
    open_slice = function.body[2].initial_value

    assert isinstance(new_slice, ArrayAccessNode)
    assert isinstance(new_slice.index, SliceNode)
    assert new_slice.index.start == "1"
    assert new_slice.index.stop == '"hello"'
    assert new_slice.index.step == "4.0"
    assert new_slice.index.has_step

    assert isinstance(reverse.index, SliceNode)
    assert reverse.index.start == "2"
    assert reverse.index.stop is None
    assert isinstance(reverse.index.step, UnaryOpNode)
    assert reverse.index.step.op == "-"
    assert reverse.index.step.operand == "1"
    assert reverse.index.has_step

    assert isinstance(open_slice.index, SliceNode)
    assert open_slice.index.start is None
    assert open_slice.index.stop is None
    assert open_slice.index.step is None
    assert open_slice.index.has_step


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


def test_comptime_expression_prefix_parse_from_modular_gpu_examples():
    code = """
    def main():
        var dev_buf = ctx.enqueue_create_buffer[int_dtype](
            comptime (layout.size())
        )
        for i in range(comptime (tile_layout.size())):
            pass
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")

    buffer_call = function.body[0].initial_value
    assert getattr(buffer_call.args[0], "is_comptime", False)
    assert getattr(function.body[1].iterable.args[0], "is_comptime", False)


def test_function_raises_effect_before_return_type_parse():
    code = """
    fn parse_value(text: String) raises -> Int:
        return 1
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "parse_value")

    assert function.return_type == "Int"
    assert [param.name for param in function.params] == ["text"]
    assert function.params[0].vtype == "String"


def test_function_capturing_raises_effects_parse_from_modular_reduction_example():
    code = """
    def sum_kernel_benchmark(
        mut b: Bencher, input_data: SumKernelBenchmarkParams
    ) capturing raises:
        def kernel_launch_sum(ctx: DeviceContext) raises:
            pass
        b.iter_custom[kernel_launch_sum](ctx)
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "sum_kernel_benchmark")

    assert [param.name for param in function.params] == ["b", "input_data"]
    assert function.params[0].parameter_convention == "mut"
    assert isinstance(function.body[0], FunctionNode)
    assert function.body[0].name == "kernel_launch_sum"


def test_function_capture_list_parse_from_modular_packing_kernel():
    # Reduced from https://github.com/modular/modular.git commit
    # 7aa053560034c8c5b4f9acb0a5b450e79d2f7c18,
    # max/kernels/src/linalg/packing.mojo
    # _pack_matmul_b_shape_func_impl.dispatch_on_kernel_type.
    code = """
    def build_shape_func(b_input: TileTensor):
        var tile_n_k = IndexList[2]()

        @always_inline
        def dispatch_on_kernel_type[kernel_type: Bool]() {mut tile_n_k, b_input}:
            tile_n_k = _get_tile_n_k[b_input.dtype](b_input)

        dispatch_get_kernel_type(dispatch_on_kernel_type)
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "build_shape_func")
    nested = function.body[1]

    assert isinstance(nested, FunctionNode)
    assert nested.name == "dispatch_on_kernel_type"
    assert [attr.name for attr in nested.attributes] == ["always_inline"]
    assert nested.params == []
    assert isinstance(nested.body[0], AssignmentNode)
    assert nested.body[0].left.name == "tile_n_k"


def test_list_literal_argument_parse_from_modular_reduction_example():
    code = """
    def main():
        bench.bench_with_input[Params, kernel](
            BenchId("sum_kernel_benchmark", "gpu"),
            Params(out_ptr, a_ptr),
            [ThroughputMeasure(BenchMetric.bytes, SIZE * size_of[dtype]())],
        )
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    call = function.body[0]

    assert isinstance(call.args[2], ListLiteralNode)
    assert call.args[2].elements
    assert isinstance(call.args[2].elements[0], FunctionCallNode)


def test_list_comprehension_parse_from_current_mojo_docs():
    code = """
    def main():
        var squares = [x * x for x in range(5) if x % 2 == 0]
        var products = [(x, y) for x in range(3) for y in range(2) if x != y]
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    squares = function.body[0].initial_value
    products = function.body[1].initial_value

    assert isinstance(squares, ListComprehensionNode)
    assert isinstance(squares.expression, BinaryOpNode)
    assert [(clause["kind"]) for clause in squares.clauses] == ["for", "if"]
    assert squares.clauses[0]["pattern"] == "x"
    assert isinstance(squares.clauses[0]["iterable"], FunctionCallNode)
    assert isinstance(squares.clauses[1]["condition"], BinaryOpNode)

    assert isinstance(products, ListComprehensionNode)
    assert isinstance(products.expression, TupleNode)
    assert [(clause["kind"]) for clause in products.clauses] == [
        "for",
        "for",
        "if",
    ]


def test_dotted_type_annotation_parse_from_modular_tiled_matmul_example():
    code = """
    def tiled_matmul_kernel(matrix_c: TileTensor):
        var accumulator: matrix_c.ElementType = 0.0
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "tiled_matmul_kernel")
    accumulator = function.body[0]

    assert accumulator.name == "accumulator"
    assert accumulator.vtype == "matrix_c.ElementType"
    assert accumulator.initial_value == "0.0"


def test_adjacent_string_literals_in_call_parse_from_modular_tiled_matmul_example():
    code = """
    def main():
        print(
            "Note: Expected formula is C[i,j] ="
            " (i+1) * 64 * (j+1)"
        )
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    call = function.body[0]

    assert call.args == ['"Note: Expected formula is C[i,j] = (i+1) * 64 * (j+1)"']


def test_identifier_tuple_declaration_and_assignment_parse_from_layout_tensor_docs():
    code = """
    def main():
        var row, col = 0, 1
        row, col = 0, 0
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    declaration = function.body[0]
    assignment = function.body[1]

    assert isinstance(declaration.name, TupleNode)
    assert [element.name for element in declaration.name.elements] == ["row", "col"]
    assert isinstance(declaration.initial_value, TupleNode)
    assert declaration.initial_value.elements == ["0", "1"]
    assert isinstance(assignment.left, TupleNode)
    assert [element.name for element in assignment.left.elements] == ["row", "col"]
    assert isinstance(assignment.right, TupleNode)
    assert assignment.right.elements == ["0", "0"]


def test_multiline_parenthesized_boolean_condition_parse_from_layout_tensor_docs():
    code = """
    def kernel(tensor: LayoutTensor):
        if (
            global_idx.y < tensor.shape[0]()
            and global_idx.x < tensor.shape[1]()
        ):
            tensor[global_idx.y, global_idx.x] = tensor[global_idx.y, global_idx.x] + 1
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "kernel")
    condition = function.body[0].condition

    assert isinstance(condition, BinaryOpNode)
    assert condition.op == "&&"


def test_modular_image_pipeline_blur_chained_bounds_parse():
    # Reduced from modularml/mojo commit
    # 7aa053560034c8c5b4f9acb0a5b450e79d2f7c18,
    # max/examples/custom_ops/kernels/image_pipeline.mojo blur kernel bounds check.
    code = """
    def blur_kernel():
        if 0 <= cur_row < height and 0 <= cur_col < width:
            pix_val_accum += Int(img_in[cur_row, cur_col])
    """
    ast = parse_code(tokenize_code(code))
    condition = find_function(ast, "blur_kernel").body[0].condition
    row_bounds = condition.left
    col_bounds = condition.right

    assert condition.op == "&&"
    assert row_bounds.op == "&&"
    assert row_bounds.left.op == "<="
    assert row_bounds.left.left == "0"
    assert row_bounds.left.right.name == "cur_row"
    assert row_bounds.right.op == "<"
    assert row_bounds.right.left.name == "cur_row"
    assert row_bounds.right.right.name == "height"

    assert col_bounds.op == "&&"
    assert col_bounds.left.op == "<="
    assert col_bounds.left.right.name == "cur_col"
    assert col_bounds.right.op == "<"
    assert col_bounds.right.left.name == "cur_col"
    assert col_bounds.right.right.name == "width"


def test_try_except_parse_from_layout_tensor_gpu_docs():
    code = """
    def main():
        try:
            run_kernel()
        except error:
            print(error)
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    statement = function.body[0]

    assert isinstance(statement, TryExceptNode)
    assert statement.exception_name == "error"
    assert isinstance(statement.try_body[0], FunctionCallNode)
    assert isinstance(statement.except_body[0], FunctionCallNode)


def test_modular_pipeline_schedule_trait_parse():
    # Reduced from modular/modular commit
    # 7aa053560034c8c5b4f9acb0a5b450e79d2f7c18,
    # max/kernels/src/pipeline/compiler.mojo PipelineSchedule trait.
    code = """
    trait PipelineSchedule:
        def config(self) -> PipelineConfig:
            ...

        def build_body(self) -> List[OpDesc]:
            ...
    """
    ast = parse_code(tokenize_code(code))
    trait = ast.traits[0]

    assert isinstance(trait, TraitNode)
    assert trait.name == "PipelineSchedule"
    assert [method.name for method in trait.methods] == ["config", "build_body"]
    assert trait.methods[0].return_type == "PipelineConfig"
    assert trait.methods[1].return_type == "List[OpDesc]"
    assert all(isinstance(method.body[0], PassNode) for method in trait.methods)


def test_modular_tuning_config_trait_base_list_parse():
    # Reduced from modular/modular commit
    # 7aa053560034c8c5b4f9acb0a5b450e79d2f7c18,
    # max/kernels/src/internal_utils/dispatch_utils.mojo TuningConfig trait.
    code = """
    trait TuningConfig(TrivialRegisterPassable, Writable):
        ...
    """
    ast = parse_code(tokenize_code(code))
    trait = ast.traits[0]

    assert trait.name == "TuningConfig"
    assert trait.base_classes == ["TrivialRegisterPassable", "Writable"]
    assert isinstance(trait.members[0], PassNode)


def test_function_local_imports_parse_from_layout_tensor_gpu_docs():
    code = """
    def simd_width_example():
        from std.sys.info import simd_width_of
        from std.gpu.host.compile import get_gpu_target
        comptime simd_width = simd_width_of[DType.float32, get_gpu_target()]
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "simd_width_example")

    assert isinstance(function.body[0], ImportNode)
    assert function.body[0].module_name == "std.sys.info"
    assert function.body[0].items == ["simd_width_of"]
    assert isinstance(function.body[1], ImportNode)
    assert function.body[1].module_name == "std.gpu.host.compile"
    assert function.body[1].items == ["get_gpu_target"]


def test_postfix_transfer_marker_parse_from_life_examples():
    code = """
    def make_grid():
        return Grid(8, 8, glider^)

    def random_grid():
        return grid^

    def xor_value(a: Int, b: Int) -> Int:
        return a ^ b
    """
    ast = parse_code(tokenize_code(code))
    make_grid = find_function(ast, "make_grid")
    random_grid = find_function(ast, "random_grid")
    xor_value = find_function(ast, "xor_value")

    transferred_arg = make_grid.body[0].value.args[2]
    assert transferred_arg.name == "glider"
    assert getattr(transferred_arg, "is_transfer", False)
    assert random_grid.body[0].value.name == "grid"
    assert getattr(random_grid.body[0].value, "is_transfer", False)
    assert isinstance(xor_value.body[0].value, BinaryOpNode)
    assert xor_value.body[0].value.op == "^"


def test_type_member_expression_parse_from_modular_testing_examples():
    code = """
    def inc(n: Int) raises -> Int:
        if n == Int.MAX:
            raise Error("inc overflow")
        return inc(Int.MAX)
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "inc")
    condition = function.body[0].condition
    call_arg = function.body[1].value.args[0]

    assert isinstance(condition.right, MemberAccessNode)
    assert condition.right.object.name == "Int"
    assert condition.right.member == "MAX"
    assert isinstance(call_arg, MemberAccessNode)
    assert call_arg.object.name == "Int"
    assert call_arg.member == "MAX"


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


def test_struct_comptime_member_from_modular_custom_ops_parse():
    code = """
    @fieldwise_init
    struct Tensor[dtype: DType, rank: Int](ImplicitlyCopyable):
        comptime size = product(Self.static_spec.shape_tuple)
        var buffer: DeviceBuffer[Self.dtype]
    """
    ast = parse_code(tokenize_code(code))
    struct_node = find_struct(ast, "Tensor")

    assert struct_node.base_classes == ["ImplicitlyCopyable"]
    assert struct_node.generic_parameters == "[dtype:DType, rank:Int]"
    size_member = struct_node.members[0]
    assert isinstance(size_member, VariableDeclarationNode)
    assert size_member.name == "size"
    assert getattr(size_member, "is_comptime", False)
    assert not size_member.is_var
    assert isinstance(size_member.initial_value, FunctionCallNode)
    assert size_member.initial_value.name == "product"
    assert struct_node.members[1].vtype == "DeviceBuffer[Self.dtype]"


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


def test_keyword_style_runtime_assert_statement_parse_from_modular_packing_kernel():
    # Reduced from modular/modular commit
    # 7aa053560034c8c5b4f9acb0a5b450e79d2f7c18,
    # max/kernels/src/linalg/packing.mojo PackMatrixCols.run.
    code = """
    def run(pack_tile_dim: IndexList[2]):
        assert (
            pack_tile_dim[1] % Self.column_inner_size == 0
        ), "Unimplemented tile pattern."
        assert False, "unreachable"
    """

    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "run")
    shape_assert = function.body[0]
    unreachable_assert = function.body[1]

    assert isinstance(shape_assert, FunctionCallNode)
    assert not getattr(shape_assert, "is_comptime", False)
    assert shape_assert.name == "assert"
    assert isinstance(shape_assert.args[0], BinaryOpNode)
    assert shape_assert.args[0].op == "=="
    assert shape_assert.args[1] == '"Unimplemented tile pattern."'

    assert isinstance(unreachable_assert, FunctionCallNode)
    assert unreachable_assert.args == ["false", '"unreachable"']


def test_comptime_in_expression_and_parameterized_declaration_parse():
    code = """
    def main():
        comptime UInt32Indices[rank: Int] = IndexList[rank, element_type=DType.uint32]
        comptime assert Self.cta_group in (
            1,
            2,
        ), "MmaOpSM100 only supports cta_group 1 or 2"
    """

    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    declaration = function.body[0]
    assertion = function.body[1]

    assert isinstance(declaration, VariableDeclarationNode)
    assert declaration.name == "UInt32Indices[rank:Int]"
    assert getattr(declaration, "is_comptime", False)
    assert isinstance(assertion.args[0], BinaryOpNode)
    assert assertion.args[0].op == "in"
    assert assertion.args[1] == '"MmaOpSM100 only supports cta_group 1 or 2"'


def test_gpu_fundamentals_launch_keyword_tuple_args_parse():
    code = """
    from std.sys import has_accelerator
    from std.gpu.host import DeviceContext
    from std.gpu import block_dim, block_idx, global_idx, thread_idx

    def print_threads():
        print(
            block_idx.x,
            block_idx.y,
            block_idx.z,
            thread_idx.x,
            thread_idx.y,
            thread_idx.z,
            global_idx.x,
            global_idx.y,
            global_idx.z,
            block_dim.x * block_idx.x + thread_idx.x,
            block_dim.y * block_idx.y + thread_idx.y,
            block_dim.z * block_idx.z + thread_idx.z,
            sep="\\t",
        )

    def main() raises:
        comptime if not has_accelerator():
            print("No compatible GPU found")
        else:
            ctx = DeviceContext()
            ctx.enqueue_function[print_threads, print_threads](
                grid_dim=(2, 2, 1),
                block_dim=(4, 4, 2),
            )
            ctx.synchronize()
    """

    ast = parse_code(tokenize_code(code))
    print_threads = find_function(ast, "print_threads")
    main = find_function(ast, "main")

    print_call = print_threads.body[0]
    assert print_call.args[-1].left.name == "sep"
    assert print_call.args[-1].right == '"\\t"'

    assert isinstance(main.body[0], IfNode)
    launch_call = main.body[0].else_body[1]
    assert len(launch_call.args) == 2
    assert [arg.left.name for arg in launch_call.args] == ["grid_dim", "block_dim"]
    assert all(isinstance(arg.right, TupleNode) for arg in launch_call.args)
    assert launch_call.args[0].right.elements == ["2", "2", "1"]
    assert launch_call.args[1].right.elements == ["4", "4", "2"]


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


def test_modular_histogram_nested_gpu_kernel_metadata_parse():
    # Reduced from modularml/mojo commit
    # 7aa053560034c8c5b4f9acb0a5b450e79d2f7c18,
    # max/examples/custom_ops/kernels/histogram.mojo GPU histogram launch.
    code = """
    from std.gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx, thread_idx
    from std.gpu.host import DeviceContext
    from std.gpu.memory import AddressSpace
    from std.memory import stack_allocation
    from std.utils import StaticTuple

    comptime bin_width = Int(UInt8.MAX) + 1

    @compiler.register("histogram")
    struct Histogram:
        @staticmethod
        def execute[target: StaticString](ctx: DeviceContext) raises:
            comptime block_dim = bin_width

            @__llvm_metadata(
                MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_dim))
            )
            def kernel(
                output: UnsafePointer[Int64, MutAnyOrigin],
                input: UnsafePointer[UInt8, MutAnyOrigin],
                n: Int,
            ):
                var tid = global_idx.x
                if tid >= n:
                    return
                var shared_mem = stack_allocation[
                    bin_width, Int64, address_space=AddressSpace.SHARED
                ]()
                shared_mem[thread_idx.x] = 0
                barrier()
                _ = Atomic.fetch_add(output + thread_idx.x, shared_mem[thread_idx.x])

            ctx.enqueue_function[kernel](
                output,
                input,
                n,
                block_dim=block_dim,
                grid_dim=grid_dim,
            )
    """

    ast = parse_code(tokenize_code(code))
    struct_node = find_struct(ast, "Histogram")
    execute = struct_node.methods[0]
    block_dim, kernel, launch = execute.body
    metadata = kernel.attributes[0].args[0]

    assert [attr.name for attr in struct_node.attributes] == ["compiler.register"]
    assert struct_node.attributes[0].args == ['"histogram"']
    assert [attr.name for attr in execute.attributes] == ["staticmethod"]
    assert block_dim.name == "block_dim"
    assert getattr(block_dim, "is_comptime", False)

    assert isinstance(kernel, FunctionNode)
    assert kernel.name == "kernel"
    assert [attr.name for attr in kernel.attributes] == ["__llvm_metadata"]
    assert isinstance(metadata, AssignmentNode)
    assert metadata.left.name == "MAX_THREADS_PER_BLOCK_METADATA"
    assert isinstance(metadata.right, VectorConstructorNode)
    assert metadata.right.type_name == "StaticTuple[Int32, 1]"
    assert [(param.name, param.vtype) for param in kernel.params] == [
        ("output", "UnsafePointer[Int64, MutAnyOrigin]"),
        ("input", "UnsafePointer[UInt8, MutAnyOrigin]"),
        ("n", "Int"),
    ]
    assert isinstance(kernel.body[2].initial_value, VectorConstructorNode)
    assert kernel.body[2].initial_value.type_name == (
        "stack_allocation[bin_width, Int64, address_space=AddressSpace.SHARED]"
    )
    assert [arg.left.name for arg in launch.args[3:]] == ["block_dim", "grid_dim"]


def test_modular_static_default_method_name_parse():
    # Reduced from modularml/mojo commit
    # 7aa053560034c8c5b4f9acb0a5b450e79d2f7c18,
    # max/kernels/src/pipeline/strategies.mojo PipelineStrategy.default.
    code = """
    struct PipelineStrategy:
        var minimal: Bool

        @staticmethod
        def default() -> Self:
            return Self(minimal=false)

        @staticmethod
        def minimal_no_set_prio() -> Self:
            return Self.default()
    """

    ast = parse_code(tokenize_code(code))
    strategy = find_struct(ast, "PipelineStrategy")
    default_method, minimal_method = strategy.methods

    assert default_method.name == "default"
    assert [attr.name for attr in default_method.attributes] == ["staticmethod"]
    assert isinstance(default_method.body[0], ReturnNode)
    assert isinstance(default_method.body[0].value, FunctionCallNode)
    assert default_method.body[0].value.name == "Self"

    default_call = minimal_method.body[0].value
    assert isinstance(default_call, MethodCallNode)
    assert default_call.object.name == "Self"
    assert default_call.method == "default"


def test_modular_vector_addition_nested_gpu_launch_parse():
    # Reduced from modularml/mojo commit
    # 7aa053560034c8c5b4f9acb0a5b450e79d2f7c18,
    # max/examples/custom_ops/kernels/vector_addition.mojo GPU launch helper.
    code = """
    from std.gpu.host import DeviceContext
    from std.math import ceildiv
    from std.gpu import block_dim, block_idx, thread_idx
    from extensibility import InputTensor, ManagedTensorSlice, OutputTensor
    from std.utils.index import IndexList

    def _vector_addition_gpu(
        output: ManagedTensorSlice[mut=True, ...],
        lhs: ManagedTensorSlice[dtype=output.dtype, rank=output.rank, ...],
        rhs: ManagedTensorSlice[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) raises:
        comptime BLOCK_SIZE = 16
        var gpu_ctx = ctx
        var vector_length = output.dim_size(0)

        @parameter
        def vector_addition_gpu_kernel(length: Int):
            var tid = block_dim.x * block_idx.x + thread_idx.x
            if tid < length:
                var idx = IndexList[output.rank](tid)
                var result = lhs.load[1](idx) + rhs.load[1](idx)
                output.store[1](idx, result)

        var num_blocks = ceildiv(vector_length, BLOCK_SIZE)
        gpu_ctx.enqueue_function[vector_addition_gpu_kernel](
            vector_length, grid_dim=num_blocks, block_dim=BLOCK_SIZE
        )
    """

    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "_vector_addition_gpu")
    block_size, gpu_ctx, vector_length, kernel, num_blocks, launch = function.body

    assert [(param.name, param.vtype) for param in function.params] == [
        ("output", "ManagedTensorSlice[mut=true, ...]"),
        ("lhs", "ManagedTensorSlice[dtype=output.dtype, rank=output.rank, ...]"),
        ("rhs", "ManagedTensorSlice[dtype=output.dtype, rank=output.rank, ...]"),
        ("ctx", "DeviceContext"),
    ]
    assert block_size.name == "BLOCK_SIZE"
    assert getattr(block_size, "is_comptime", False)
    assert gpu_ctx.name == "gpu_ctx"
    assert vector_length.name == "vector_length"

    assert isinstance(kernel, FunctionNode)
    assert kernel.name == "vector_addition_gpu_kernel"
    assert [attr.name for attr in kernel.attributes] == ["parameter"]
    assert kernel.params[0].name == "length"
    assert kernel.body[0].name == "tid"
    assert isinstance(kernel.body[0].initial_value, BinaryOpNode)
    assert isinstance(kernel.body[1], IfNode)
    assert kernel.body[1].condition.op == "<"
    store_call = kernel.body[1].if_body[2]
    assert isinstance(store_call.callee, ArrayAccessNode)
    assert store_call.callee.array.member == "store"
    assert store_call.callee.index == "1"

    assert num_blocks.name == "num_blocks"
    assert launch.callee.array.member == "enqueue_function"
    assert launch.callee.index.name == "vector_addition_gpu_kernel"
    assert [arg.left.name for arg in launch.args[1:]] == ["grid_dim", "block_dim"]


def test_modular_mandelbrot_nested_parameter_kernel_parse():
    # Reduced from modular/modular commit
    # 7aa053560034c8c5b4f9acb0a5b450e79d2f7c18,
    # max/examples/custom_ops/kernels/mandelbrot.mojo elementwise kernel.
    code = """
    import compiler
    from std.gpu.host import DeviceContext
    from extensibility import OutputTensor, foreach
    from std.utils.index import IndexList

    @compiler.register("mandelbrot")
    struct Mandelbrot:
        @staticmethod
        def execute[target: StaticString](
            output: OutputTensor, ctx: DeviceContext
        ) raises:
            @parameter
            @always_inline
            def elementwise_mandelbrot[
                width: Int
            ](idx: IndexList[output.rank]) -> SIMD[output.dtype, width]:
                var row = idx[0]
                var iters = SIMD[output.dtype, width](0)
                var in_set_mask = SIMD[DType.bool, width](fill=True)
                for _ in range(max_iterations):
                    if not any(in_set_mask):
                        break
                    iters = in_set_mask.select(iters + 1, iters)
                return iters

            foreach[elementwise_mandelbrot, target=target](output, ctx)
    """
    ast = parse_code(tokenize_code(code))
    execute = find_struct(ast, "Mandelbrot").methods[0]
    kernel = execute.body[0]
    launch = execute.body[1]
    loop = kernel.body[3]

    assert [attr.name for attr in kernel.attributes] == ["parameter", "always_inline"]
    assert kernel.return_type == "SIMD[output.dtype, width]"
    assert [(param.name, param.vtype) for param in kernel.params] == [
        ("idx", "IndexList[output.rank]")
    ]
    assert isinstance(loop, RangeForNode)
    assert loop.name == "_"
    assert loop.iterable.name == "range"
    assert loop.iterable.args[0].name == "max_iterations"
    assert isinstance(loop.body[0], IfNode)
    assert isinstance(loop.body[0].if_body[0], BreakNode)
    assert isinstance(loop.body[1].right, MethodCallNode)
    assert isinstance(launch, VectorConstructorNode)
    assert launch.type_name == "foreach[elementwise_mandelbrot, target=target]"


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


def test_documented_numeric_literal_forms_parse_from_mojo_reference():
    # Reduced from https://mojolang.org/docs/reference/literals/
    code = """
    fn main():
        let grouped = 1_000_000
        let relaxed = 1__000_
        let fraction_only = .5
        let trailing_point = 2.
        let grouped_float = 1_000.000_5
        let exponent_only = 1E10
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")

    assert [declaration.initial_value for declaration in function.body] == [
        "1_000_000",
        "1__000_",
        ".5",
        "2.",
        "1_000.000_5",
        "1E10",
    ]


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
