from typing import List

import pytest

from crosstl.backend.Mojo.MojoAst import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    BreakNode,
    CallNode,
    CastNode,
    ClassNode,
    ConstantBufferNode,
    ContinueNode,
    DictComprehensionNode,
    DictLiteralNode,
    ExtensionNode,
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
    SetComprehensionNode,
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


def test_multiline_header_continuation_dedent_before_body_parses():
    # Reduced from Practical-Mojo-Examples ml_algorithms and snake_game files,
    # where the parameter continuation indentation is deeper than the body.
    code = """
    struct NormResult:
        var data: List[Float64]
        var min_val: Float64
        var max_val: Float64

        fn __init__(out self, var data: List[Float64],
                    min_val: Float64, max_val: Float64):
            self.data = data^
            self.min_val = min_val
            self.max_val = max_val

    fn draw_cell(canvas: PythonObject, x: Int, y: Int,
                 color: String, cell: Int) raises:
        "Draw one grid cell as a filled rectangle."
        var x1 = x * cell
        var y1 = y * cell
    """
    ast = parse_code(tokenize_code(code))
    struct_node = find_struct(ast, "NormResult")
    init_method = struct_node.methods[0]
    draw_cell = find_function(ast, "draw_cell")

    assert len(init_method.body) == 3
    assert all(isinstance(statement, AssignmentNode) for statement in init_method.body)
    assert isinstance(draw_cell.body[0], PassNode)
    assert [statement.name for statement in draw_cell.body[1:]] == ["x1", "y1"]


def test_function_type_parameter_parsing_from_modular_gpu_reduction():
    # Reduced from https://github.com/modular/modular.git commit
    # daa47bb846cc213723a54c51844ea4e923eb5e13,
    # mojo/stdlib/std/algorithm/backend/gpu/reduction.mojo small_reduce_kernel.
    code = """
    def reduce_adapter(
        input_fn: def[dtype: DType, width: Int, rank: Int](
            IndexList[rank]
        ) capturing[_] -> SIMD[dtype, width],
        output_fn: def[dtype: DType, width: SIMDSize, rank: Int](
            IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
        ) capturing[_] -> None,
    ):
        pass
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "reduce_adapter")

    assert [(param.name, param.vtype) for param in function.params] == [
        (
            "input_fn",
            (
                "def[dtype:DType, width:Int, rank:Int](IndexList[rank]) "
                "capturing[_] -> SIMD[dtype, width]"
            ),
        ),
        (
            "output_fn",
            (
                "def[dtype:DType, width:SIMDSize, rank:Int]"
                "(IndexList[rank], StaticTuple[SIMD[dtype, width], "
                "num_reductions]) capturing[_] -> None"
            ),
        ),
    ]


def test_thin_function_type_parameter_parsing_from_official_parameter_docs():
    # Reduced from https://docs.modular.com/mojo/manual/parameters/
    # "Parameters at a glance" documents noncapturing comparator function types
    # with an explicit `thin` effect.
    code = """
    def invoke_compare(
        lhs: Scalar[dtype],
        rhs: Scalar[dtype],
        compare: def(Scalar[dtype], Scalar[dtype]) thin -> Int,
    ) -> Int:
        return compare(lhs, rhs)
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "invoke_compare")

    assert [(param.name, param.vtype) for param in function.params] == [
        ("lhs", "Scalar[dtype]"),
        ("rhs", "Scalar[dtype]"),
        ("compare", "def(Scalar[dtype], Scalar[dtype]) thin -> Int"),
    ]
    assert function.return_type == "Int"


def test_comptime_function_type_value_with_terminal_raises_parse():
    code = """
    struct _Test:
        comptime fn_type = def() thin raises
        var test_fn: Self.fn_type
    """
    ast = parse_code(tokenize_code(code))
    struct_node = find_struct(ast, "_Test")
    fn_type = struct_node.members[0]

    assert isinstance(fn_type, VariableDeclarationNode)
    assert fn_type.initial_value == "def() thin raises"
    assert struct_node.members[1].name == "test_fn"


def test_fn_function_type_alias_parse_from_ksandvik_memset():
    # Reduced from ksandvik/mojo-examples examples/memset.mojo.
    code = """
    alias memset_fn_type = fn (BufferPtrType, ValueType, Int) -> None

    fn measure_time(func: memset_fn_type) -> Int:
        return 0
    """
    ast = parse_code(tokenize_code(code))
    alias = ast.global_variables[0]
    function = find_function(ast, "measure_time")

    assert alias.name == "memset_fn_type"
    assert alias.initial_value == "fn(BufferPtrType, ValueType, Int) -> None"
    assert [(param.name, param.vtype) for param in function.params] == [
        ("func", "memset_fn_type")
    ]


def test_function_type_with_raises_error_type_parse_from_modular_cublaslt():
    # Reduced from Modular max/kernels/src/_cublas/cublaslt.mojo.
    code = """
    fn register_callback(
        callback: def(Result) thin raises Error -> UnsafePointer[Int8, ImmutAnyOrigin]
    ):
        pass
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "register_callback")

    assert function.params[0].vtype == (
        "def(Result) thin raises Error -> UnsafePointer[Int8, ImmutAnyOrigin]"
    )


def test_mlir_region_statement_parse_from_coroutine_stdlib():
    code = """
    def _suspend_async[body: def(AnyCoroutine) capturing -> None]():
        __mlir_region await_body(hdl: __mlir_type.`!co.routine`):
            body(hdl)
            __mlir_op.`co.suspend.end`()
        __mlir_op.`co.suspend`[_region="await_body".value]()
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "_suspend_async")

    assert isinstance(function.body[0], PassNode)
    assert isinstance(function.body[1], CallNode)


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


def test_variadic_keyword_parameter_parse_from_current_docs():
    code = """
    def print_nicely(**kwargs: Int):
        for item in kwargs.items():
            print(item.key, "=", item.value)
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "print_nicely")
    loop = function.body[0]

    assert [
        (
            param.name,
            param.vtype,
            getattr(param, "is_variadic", False),
            getattr(param, "is_variadic_keyword", False),
        )
        for param in function.params
    ] == [("kwargs", "Int", True, True)]
    assert isinstance(loop, RangeForNode)
    assert loop.name == "item"
    assert isinstance(loop.iterable, MethodCallNode)
    assert loop.iterable.object.name == "kwargs"
    assert loop.iterable.method == "items"


def test_heterogeneous_variadic_type_pack_parse_from_current_docs():
    code = """
    def count_many_things[*ArgTypes: Intable](*args: *ArgTypes) -> Int:
        return 0
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "count_many_things")

    assert [(param.name, param.vtype) for param in function.params] == [
        ("args", "*ArgTypes")
    ]
    assert getattr(function.params[0], "is_variadic", False)
    assert not getattr(function.params[0], "is_variadic_keyword", False)
    assert function.return_type == "Int"


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


def test_default_keyword_parameter_name_parse_from_modular_stdlib():
    # Reduced from https://github.com/modular/modular.git commit
    # daa47bb846cc213723a54c51844ea4e923eb5e13,
    # mojo/stdlib/std/memory/unsafe_nullable_pointer.mojo UnsafeNullablePointer.gather.
    code = """
    def gather[
        dtype: DType,
        width: SIMDSize = 1,
    ](
        mask: SIMD[DType.bool, width] = SIMD[DType.bool, width](fill=True),
        default: SIMD[dtype, width] = 0,
    ) -> SIMD[dtype, width]:
        return default
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "gather")

    assert [(param.name, param.vtype) for param in function.params] == [
        ("mask", "SIMD[DType.bool, width]"),
        ("default", "SIMD[dtype, width]"),
    ]
    assert function.params[1].default_value == "0"
    assert function.return_type == "SIMD[dtype, width]"
    assert function.body[0].value.name == "default"


def test_constant_keyword_parameter_name_parse_from_modular_conv():
    # Reduced from /tmp/crossgl-modular
    # max/kernels/src/graph_compiler/builtin_kernels/conv.mojo
    # PadConstant.execute, where `constant` is an argument name.
    code = """
    def execute(
        padding: InputTensor[rank=1, ...],
        constant: Scalar[dtype=dtype],
        ctx: DeviceContext,
    ) raises:
        sink(constant)
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "execute")

    assert [(param.name, param.vtype) for param in function.params] == [
        ("padding", "InputTensor[rank=1, ...]"),
        ("constant", "Scalar[dtype=dtype]"),
        ("ctx", "DeviceContext"),
    ]
    assert function.body[0].args[0].name == "constant"


def test_ref_binding_declaration_parse_from_official_variables_docs():
    # https://docs.modular.com/mojo/manual/variables/#reference-bindings
    code = """
    def bump_item(items: List[Int]):
        ref item_ref = items[1]
        item_ref += 1
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "bump_item")
    ref_binding = function.body[0]
    update = function.body[1]

    assert isinstance(ref_binding, VariableDeclarationNode)
    assert ref_binding.name == "item_ref"
    assert getattr(ref_binding, "binding_convention", None) == "ref"
    assert isinstance(ref_binding.initial_value, ArrayAccessNode)
    assert ref_binding.initial_value.array.name == "items"
    assert ref_binding.initial_value.index == "1"
    assert isinstance(update, AssignmentNode)
    assert update.operator == "+="
    assert update.left.name == "item_ref"


def test_backtick_local_identifier_parse_from_modular_base64_stdlib():
    # Reduced from https://github.com/modular/modular.git commit
    # daa47bb846cc213723a54c51844ea4e923eb5e13,
    # mojo/stdlib/std/base64/_b64encode.mojo _6bit_to_byte.combine.
    code = """
    def combine(shuffled: Bytes, mask: Bytes) -> Bytes:
        var `6bit` = shuffled & mask
        return shift(`6bit`)
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "combine")

    declaration = function.body[0]
    assert declaration.name == "`6bit`"
    assert function.body[1].value.args[0].name == "`6bit`"


def test_backtick_keyword_function_name_parse_from_official_docs():
    # Reduced from https://docs.modular.com/mojo/reference/mojo-function-declarations/
    # "Function names" escaped keyword identifier example.
    code = """
    def `import`() -> Int:
        return 1

    def main() -> Int:
        return `import`()
    """
    ast = parse_code(tokenize_code(code))
    imported = find_function(ast, "`import`")
    main = find_function(ast, "main")

    assert imported.return_type == "Int"
    assert main.body[0].value.name == "`import`"


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


def test_struct_conditional_trait_conformance_parse_from_modular_docs():
    # Reduced from https://docs.modular.com/mojo/reference/mojo-struct-declarations/
    # "Conformance lists" documents conditional `where` clauses, and
    # "Trait conformance" documents `&` trait composition.
    code = """
    @fieldwise_init
    struct Pair[T: Copyable & ImplicitlyDestructible](
        Equatable where conforms_to(T, Equatable),
        Writable where conforms_to(T, Writable) and T.size > 0,
        Copyable & ImplicitlyDestructible,
    ):
        var first: Self.T
        var second: Self.T
    """
    ast = parse_code(tokenize_code(code))
    pair = find_struct(ast, "Pair")

    assert pair.base_classes == [
        "Equatable where conforms_to(T, Equatable)",
        "Writable where conforms_to(T, Writable) && T.size > 0",
        "Copyable & ImplicitlyDestructible",
    ]
    assert [(member.vtype, member.name) for member in pair.members] == [
        ("Self.T", "first"),
        ("Self.T", "second"),
    ]


def test_pass_placeholder_in_struct_and_trait_bodies_parse_from_modular_sources():
    # Reduced from https://github.com/modular/modular.git commit
    # 9ddf207f42fc67a6f33bd7b4ccc94a6a52133c8f,
    # mojo/docs/code/manual/generics/conditional_trait_conformance.mojo Foo
    # and max/kernels/src/structured_kernels/tile_types.mojo TilePayload.
    code = """
    @fieldwise_init
    struct Foo[T: AnyType](Copyable, Writable where conforms_to(T, Writable)):
        pass

    trait TilePayload(TrivialRegisterPassable):
        pass
    """
    ast = parse_code(tokenize_code(code))
    struct_node = find_struct(ast, "Foo")
    trait_node = next(node for node in ast.traits if node.name == "TilePayload")

    assert len(struct_node.members) == 1
    assert isinstance(struct_node.members[0], PassNode)
    assert len(trait_node.members) == 1
    assert isinstance(trait_node.members[0], PassNode)


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


def test_struct_standalone_string_metadata_and_ellipsis_members_parse():
    code = """
    struct cudnnMultiHeadAttnWeightKind_t(TrivialRegisterPassable):
        var _value: Int32
        comptime ATTN_Q_WEIGHTS = Self(0)
        "Input projection weights for 'queries'."

    struct IOSpec[mut: Bool, input: IO](TrivialRegisterPassable):
        ...
        comptime Input = IOSpec[false, IO.Input]()
    """
    ast = parse_code(tokenize_code(code))
    weight_kind = find_struct(ast, "cudnnMultiHeadAttnWeightKind_t")
    io_spec = find_struct(ast, "IOSpec")

    assert isinstance(weight_kind.members[2], PassNode)
    assert isinstance(io_spec.members[0], PassNode)
    assert io_spec.members[1].name == "Input"


def test_attributed_nested_type_declarations_parse_in_type_bodies():
    code = """
    struct Outer:
        @fieldwise_init
        struct Inner:
            var value: Int

        trait Capability:
            ...
    """
    ast = parse_code(tokenize_code(code))
    outer = find_struct(ast, "Outer")

    assert isinstance(outer.members[0], StructNode)
    assert outer.members[0].name == "Inner"
    assert isinstance(outer.members[1], TraitNode)


def test_builtin_type_keyword_struct_names_parse():
    code = """
    struct Bool(Boolable):
        var value: Int1

    struct Int(Intable):
        var value: Int64

    struct String(Writable):
        var value: UnsafePointer[UInt8]
    """
    ast = parse_code(tokenize_code(code))

    assert [node.name for node in ast.structs] == ["Bool", "Int", "String"]


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


def test_multiline_tuple_element_operator_parsing_from_mojo_gpu_puzzles():
    # Reduced from https://github.com/modular/mojo-gpu-puzzles.git commit
    # bb82972c5f55af98ec69a6324383220c0903a8a8,
    # problems/p19/op/attention.mojo transpose_blocks_per_grid.
    code = """
    def configure_attention_grid():
        comptime transpose_blocks_per_grid = (
            (d + TRANSPOSE_BLOCK_DIM_XY - 1) // TRANSPOSE_BLOCK_DIM_XY,
            (seq_len + TRANSPOSE_BLOCK_DIM_XY - 1)
            // TRANSPOSE_BLOCK_DIM_XY,
        )
    """
    ast = parse_code(tokenize_code(code))
    declaration = find_function(ast, "configure_attention_grid").body[0]

    assert isinstance(declaration.initial_value, TupleNode)
    first, second = declaration.initial_value.elements
    assert isinstance(first, BinaryOpNode)
    assert first.op == "//"
    assert isinstance(second, BinaryOpNode)
    assert second.op == "//"


def test_comment_trailing_backslash_before_kernel_specialization_from_mojo_gpu_puzzles():
    # Reduced from https://github.com/modular/mojo-gpu-puzzles.git commit
    # bb82972c5f55af98ec69a6324383220c0903a8a8,
    # solutions/p19/op/attention.mojo attention_orchestration_solution.
    code = """
    def launch_attention():
        var q_2d = q_tensor.reshape(layout_q_2d)
        # Step 2: Transpose K from (seq_len, d) to K^T (d, seq_len)\\
        comptime kernel = transpose_kernel[
            seq_len, d, KTLayout, KLayout, dtype
        ]
        ctx.enqueue_function[kernel](k_t, k_tensor)
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "launch_attention")
    kernel = function.body[1]
    launch = function.body[2]

    assert isinstance(kernel, VariableDeclarationNode)
    assert kernel.name == "kernel"
    assert isinstance(kernel.initial_value, ArrayAccessNode)
    assert isinstance(kernel.initial_value.index, TupleNode)
    assert [arg.name for arg in kernel.initial_value.index.elements] == [
        "seq_len",
        "d",
        "KTLayout",
        "KLayout",
        "dtype",
    ]
    assert isinstance(launch, CallNode)


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


def test_for_in_nested_tuple_target_parse_from_kv_cache_utils():
    code = """
    fn main(prompt_lens: List[Int], cache_lens: List[Int]):
        for batch, (prompt_len, cache_len) in enumerate(zip(prompt_lens, cache_lens)):
            sink(batch, prompt_len, cache_len)
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    loop = ast.functions[0].body[0]

    assert isinstance(loop, RangeForNode)
    assert isinstance(loop.name, TupleNode)
    batch, lengths = loop.name.elements
    assert batch.name == "batch"
    assert isinstance(lengths, TupleNode)
    assert [element.name for element in lengths.elements] == [
        "prompt_len",
        "cache_len",
    ]
    assert isinstance(loop.iterable, FunctionCallNode)


def test_for_in_target_conventions_parse_from_modular_control_flow_docs():
    # Reduced from https://github.com/modular/modular.git commit
    # 04cff5a4cc491ec2bf6850ce99e0253075fc908c,
    # mojo/docs/code/manual/control-flow/for_loop.mojo for-loop examples.
    code = """
    fn main(values: List[Int], capitals: Dict[String, String]):
        for ref value in values:
            value -= 1
        for var state in capitals:
            sink(state)
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "main")
    ref_loop = function.body[0]
    var_loop = function.body[1]

    assert isinstance(ref_loop, RangeForNode)
    assert ref_loop.name == "value"
    assert ref_loop.iterable.name == "values"
    assert getattr(ref_loop, "target_convention", None) == "ref"
    assert isinstance(ref_loop.body[0], AssignmentNode)
    assert ref_loop.body[0].operator == "-="

    assert isinstance(var_loop, RangeForNode)
    assert var_loop.name == "state"
    assert var_loop.iterable.name == "capitals"
    assert getattr(var_loop, "target_convention", None) == "var"


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


def test_floor_divide_assignment_parse_from_mojo_gpu_puzzles():
    # Reduced from https://github.com/modular/mojo-gpu-puzzles.git commit
    # 87de51ac93bea662eba6f09d19e8744e56161027,
    # solutions/p15/p15.mojo axis_sum reduction loop.
    code = """
    def axis_sum():
        var stride = TPB // 2
        while stride > 0:
            stride //= 2
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "axis_sum")
    declaration = function.body[0]
    loop = function.body[1]
    update = loop.body[0]

    assert isinstance(declaration.initial_value, BinaryOpNode)
    assert declaration.initial_value.op == "//"
    assert isinstance(loop, WhileNode)
    assert isinstance(update, AssignmentNode)
    assert update.operator == "//="
    assert update.left.name == "stride"
    assert update.right == "2"


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


def test_comptime_if_elif_survives_multiline_generic_return_parse():
    code = """
    def load[width: Int]():
        comptime if width == 1:
            return call_ld_intrinsic[S]()
        elif width == 2:
            return call_ld_intrinsic[
                _RegisterPackType[S, S]
            ]()
        elif width == 4:
            return call_ld_intrinsic[
                _RegisterPackType[S, S, S, S]
            ]()
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "load")
    branch = function.body[0]

    assert isinstance(branch, IfNode)
    assert getattr(branch, "is_comptime", False)
    assert isinstance(branch.else_body[0], IfNode)
    assert isinstance(branch.else_body[0].else_body[0], IfNode)


def test_multiline_generic_call_return_keeps_following_functions_top_level():
    # Reduced from Modular max/kernels/src/_cublas/cublas.mojo generated wrappers.
    code = """
    def _get_dylib_function[
        func_name: StaticString, result_type: TrivialRegisterPassable
    ]() raises -> result_type:
        return _ffi_get_dylib_function[
            CUDA_CUBLAS_LIBRARY(),
            func_name,
            result_type,
        ]()


    def cublasScopy(handle: cublasHandle_t) raises -> Result:
        return _get_dylib_function[
            "cublasScopy_v2_64",
            def(
                type_of(handle),
            ) thin -> Result,
        ]()(handle)


    def cublasDgemv(handle: cublasHandle_t) raises -> Result:
        return _get_dylib_function[
            "cublasDgemv_v2_64",
            def(
                type_of(handle),
            ) thin -> Result,
        ]()(handle)
    """
    ast = parse_code(tokenize_code(code))
    top_level_functions = [
        node.name for node in ast.functions if isinstance(node, FunctionNode)
    ]

    assert top_level_functions == [
        "_get_dylib_function",
        "cublasScopy",
        "cublasDgemv",
    ]


def test_comptime_if_elif_survives_multiline_call_statement_parse():
    code = """
    def store[width: Int](data: InlineArray[Scalar[dtype], width], tmem_addr: UInt32):
        comptime if width == 1:
            inlined_assembly[asm_str, NoneType, has_side_effect=true](
                data[0],
                tmem_addr,
            )
        elif width == 2:
            inlined_assembly[asm_str, NoneType, has_side_effect=true](
                data[0], data[1],
                tmem_addr,
            )
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "store")
    branch = function.body[0]

    assert isinstance(branch, IfNode)
    assert getattr(branch, "is_comptime", False)
    assert isinstance(branch.else_body[0], IfNode)


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


def test_generic_specialization_ellipsis_parse_from_modular_distributed():
    # Reduced from /tmp/crossgl-modular
    # max/kernels/src/graph_compiler/builtin_kernels/distributed.mojo
    # vendor_ccl.allreduce specialization arguments.
    code = """
    def execute():
        vendor_ccl.allreduce[
            ngpus=num_devices,
            output_lambda=output_lambda[output_index=index, ...],
        ](in_tensors)
    """
    ast = parse_code(tokenize_code(code))
    call = find_function(ast, "execute").body[0]
    specialization_args = call.callee.index.elements
    output_lambda = specialization_args[1].right.index.elements

    assert isinstance(call, CallNode)
    assert call.callee.array.member == "allreduce"
    assert specialization_args[0].left.name == "ngpus"
    assert output_lambda[0].left.name == "output_index"
    assert output_lambda[1].name == "..."


def test_comptime_function_type_expression_parse_from_modular_graph_kernels():
    # Reduced from /tmp/crossgl-modular
    # max/kernels/src/graph_compiler/builtin_kernels/kernels.mojo
    # where a comptime declaration stores a thin function type.
    code = """
    def execute(payload: OpaquePointer):
        comptime _HostFuncTy = def(OpaquePointer[MutAnyOrigin]) thin -> None
    """
    ast = parse_code(tokenize_code(code))
    declaration = find_function(ast, "execute").body[0]

    assert isinstance(declaration, VariableDeclarationNode)
    assert declaration.name == "_HostFuncTy"
    assert declaration.initial_value == (
        "def(OpaquePointer[MutAnyOrigin]) thin -> None"
    )
    assert getattr(declaration, "is_comptime", False)


def test_comptime_assert_parenthesized_or_condition_parse_from_modular_conv():
    # Reduced from /tmp/crossgl-modular
    # max/kernels/src/graph_compiler/builtin_kernels/conv.mojo
    # Conv.execute CUDA rank guard.
    code = """
    def execute():
        comptime assert (input.rank == 4 and filter.rank == 4) or (
            input.rank == 5 and filter.rank == 5
        ), "only rank 4 or 5 tensor is supported on cuda gpu"
    """
    ast = parse_code(tokenize_code(code))
    assertion = find_function(ast, "execute").body[0]

    assert isinstance(assertion, FunctionCallNode)
    assert assertion.name == "assert"
    assert getattr(assertion, "is_comptime", False)
    assert isinstance(assertion.args[0], BinaryOpNode)
    assert assertion.args[0].op == "||"
    assert assertion.args[1] == '"only rank 4 or 5 tensor is supported on cuda gpu"'


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


def test_tuple_assignment_with_index_targets_parse_from_modular_logprobs():
    # Reduced from /tmp/crossgl-modular
    # max/kernels/src/graph_compiler/builtin_kernels/logprobs.mojo
    # FixedHeightMinHeap.swap.
    code = """
    def swap(mut self, a: Int, b: Int) -> None:
        self.k_array[a], self.k_array[b] = self.k_array[b], self.k_array[a]
    """
    ast = parse_code(tokenize_code(code))
    assignment = find_function(ast, "swap").body[0]

    assert isinstance(assignment, AssignmentNode)
    assert isinstance(assignment.left, TupleNode)
    assert isinstance(assignment.right, TupleNode)
    assert [target.array.member for target in assignment.left.elements] == [
        "k_array",
        "k_array",
    ]
    assert [target.index.name for target in assignment.left.elements] == ["a", "b"]
    assert [value.index.name for value in assignment.right.elements] == ["b", "a"]


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


def test_where_clause_with_sibling_parenthesized_constraints_parse():
    code = """
    def move_from(
        self: UnsafePointer[T, _],
        src: UnsafePointer[T, _],
    ) where (type_of(self).mut) && (type_of(src).mut):
        pass
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "move_from")

    assert function.where_clause is not None
    assert "&&" in function.where_clause
    assert function.where_clause.count("type_of") == 2
    assert function.body


def test_unparenthesized_where_clause_constraints_parsing_from_current_docs():
    # Reduced from https://mojolang.org/docs/manual/metaprogramming/constraints/
    # and https://mojolang.org/docs/reference/function-declarations/.
    code = """
    def pow2[n: Int]() -> Int where n >= 0:
        return 1

    def compare[T: AnyType](x: T, y: T) -> Int32 where conforms_to(T, Comparable):
        return 0
    """
    ast = parse_code(tokenize_code(code))
    pow2 = find_function(ast, "pow2")
    compare = find_function(ast, "compare")

    assert pow2.where_clause == "n >= 0"
    assert "conforms_to" in compare.where_clause
    assert "Comparable" in compare.where_clause
    assert isinstance(pow2.body[0], ReturnNode)
    assert isinstance(compare.body[0], ReturnNode)


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


def test_typed_raises_effect_parses_from_official_function_docs():
    # Reduced from https://mojolang.org/docs/reference/function-declarations/
    # "Function effects" documents an optional error type after `raises`.
    code = """
    def parse_strict(text: String) raises ValueError -> Int:
        pass

    def use_validator(
        validator: def(String) raises ValueError -> Int,
        text: String,
    ) raises ValueError -> Int:
        return validator(text)
    """
    ast = parse_code(tokenize_code(code))
    parse_strict = find_function(ast, "parse_strict")
    use_validator = find_function(ast, "use_validator")

    assert parse_strict.return_type == "Int"
    assert use_validator.return_type == "Int"
    assert [(param.name, param.vtype) for param in use_validator.params] == [
        ("validator", "def(String) raises ValueError -> Int"),
        ("text", "String"),
    ]


def test_async_nested_function_parse_from_modular_builtin_kernels():
    # Source: https://github.com/modular/modular
    # Commit: daa47bb846cc213723a54c51844ea4e923eb5e13
    # Path: max/kernels/src/graph_compiler/builtin_kernels/kernels.mojo
    # Lines: 2207-2215
    # dispatch_async_tasks_to_devices wrapper.
    code = """
    def dispatch_devices():
        @always_inline
        @parameter
        async def wrapper[index: Int]() -> None:
            try:
                func[index]()
            except e:
                errors[index] = e^
    """
    ast = parse_code(tokenize_code(code))
    wrapper = find_function(ast, "dispatch_devices").body[0]
    try_statement = wrapper.body[0]

    assert isinstance(wrapper, FunctionNode)
    assert wrapper.name == "wrapper"
    assert wrapper.return_type == "None"
    assert getattr(wrapper, "is_async", False)
    assert [attr.name for attr in wrapper.attributes] == [
        "always_inline",
        "parameter",
    ]
    assert isinstance(try_statement, TryExceptNode)
    assert try_statement.exception_name == "e"
    assert isinstance(try_statement.try_body[0], CallNode)
    assert isinstance(try_statement.except_body[0], AssignmentNode)


def test_await_expressions_parse_from_modular_runtime_async_tests():
    # Reduced from https://github.com/modular/modular.git commit
    # daa47bb846cc213723a54c51844ea4e923eb5e13,
    # mojo/stdlib/test/runtime/test_asyncrt.mojo test_runtime_task and
    # test_runtime_taskgroup await return expressions.
    code = """
    def runtime_task():
        @parameter
        async def add_two(a: Int, b: Int) -> Int:
            return await create_task(add[1](a)) + await create_task(add[2](b))

        async def run_as_group() -> Int:
            var t0 = create_task(return_value[1]())
            var t1 = create_task(return_value[2]())
            return await t0 + await t1
    """
    ast = parse_code(tokenize_code(code))
    runtime_task = find_function(ast, "runtime_task")
    add_two = runtime_task.body[0]
    run_as_group = runtime_task.body[1]
    add_return = add_two.body[0]
    group_return = run_as_group.body[2]

    assert getattr(add_two, "is_async", False)
    assert getattr(run_as_group, "is_async", False)
    assert isinstance(add_return, ReturnNode)
    assert isinstance(add_return.value, BinaryOpNode)
    assert isinstance(add_return.value.left, UnaryOpNode)
    assert isinstance(add_return.value.right, UnaryOpNode)
    assert add_return.value.left.op == "await"
    assert add_return.value.right.op == "await"
    assert isinstance(add_return.value.left.operand, FunctionCallNode)
    assert add_return.value.left.operand.name == "create_task"
    assert isinstance(group_return.value.left, UnaryOpNode)
    assert isinstance(group_return.value.right, UnaryOpNode)
    assert group_return.value.left.op == "await"
    assert group_return.value.right.op == "await"
    assert isinstance(group_return.value.left.operand, VariableNode)
    assert group_return.value.left.operand.name == "t0"


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
        a **= 3
        a /= 2
        a %= 3
        a @= matrix
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


def test_inline_if_generic_type_argument_parse_from_modular_comm():
    # Reduced from Modular max/kernels/src/comm/allreduce.mojo and
    # max/kernels/src/comm/reducescatter.mojo parameter type arguments.
    code = """
    fn reduce[
        dtype: DType,
        in_layout: TensorLayout,
        use_multimem: Bool = False,
        ngpus: Int,
    ](
        src_tensors: InlineArray[
            TileTensor[dtype, in_layout, ImmutAnyOrigin],
            1 if use_multimem else ngpus,
        ],
    ):
        pass
    """
    ast = parse_code(tokenize_code(code))
    param = find_function(ast, "reduce").params[0]

    assert param.vtype == (
        "InlineArray[TileTensor[dtype, in_layout, ImmutAnyOrigin], "
        "1 if use_multimem else ngpus]"
    )


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


def test_multiline_parenthesized_inline_if_parsing_from_mojo_gpu_puzzles():
    # Reduced from https://github.com/modular/mojo-gpu-puzzles.git commit
    # 87de51ac93bea662eba6f09d19e8744e56161027,
    # problems/p31/p31.mojo sophisticated_kernel cached_correction.
    code = """
    def sophisticated_kernel():
        var cached_correction = (
            shared_cache[local_i + 3072] if local_i
            < 1024 else series_correction
        )
    """
    ast = parse_code(tokenize_code(code))
    declaration = find_function(ast, "sophisticated_kernel").body[0]
    inline_if = declaration.initial_value

    assert isinstance(inline_if, TernaryOpNode)
    assert isinstance(inline_if.condition, BinaryOpNode)
    assert inline_if.condition.op == "<"
    assert inline_if.condition.left.name == "local_i"
    assert inline_if.condition.right == "1024"
    assert isinstance(inline_if.true_expr, ArrayAccessNode)
    assert inline_if.true_expr.array.name == "shared_cache"
    assert inline_if.false_expr.name == "series_correction"


def test_multiline_inline_if_else_chain_parse_from_modular_dtype_helpers():
    # Reduced from Modular stdlib DType/unsafe helpers with fmt-off inline-if chains.
    code = """
    def _llvm_bitwidth(dtype: DType) -> Int:
        return (
            1 if dtype == DType._uint1 else
            2 if dtype == DType._uint2 else
            4 if dtype == DType._uint4 else
            8
        )
    """
    ast = parse_code(tokenize_code(code))
    result = find_function(ast, "_llvm_bitwidth").body[0].value

    assert isinstance(result, TernaryOpNode)
    assert result.true_expr == "1"
    assert isinstance(result.false_expr, TernaryOpNode)
    assert result.false_expr.true_expr == "2"
    assert isinstance(result.false_expr.false_expr, TernaryOpNode)
    assert result.false_expr.false_expr.true_expr == "4"
    assert result.false_expr.false_expr.false_expr == "8"


def test_multiline_parenthesized_operator_rhs_parse_from_modular_conv():
    code = """
    def get_input_idx(self, q: Int, r: Int, s: Int):
        return (
            Index(n, do, ho, wo) * self.stride
            +
            (Index(q, r, s) * self.dilation)
            -
            self.padding
        )
    """
    ast = parse_code(tokenize_code(code))
    result = find_function(ast, "get_input_idx").body[0].value

    assert isinstance(result, BinaryOpNode)
    assert result.op == "-"
    assert isinstance(result.left, BinaryOpNode)
    assert result.left.op == "+"


def test_multiline_parenthesized_string_concat_parse_from_modular_asserts():
    # Reduced from Modular layout/matmul comptime assert diagnostic strings.
    code = """
    def main():
        comptime assert value == expected, (
            "Expected value "
            "to match "
            + String(expected)
            + ", got "
            + String(value)
        )
    """
    ast = parse_code(tokenize_code(code))
    assertion = find_function(ast, "main").body[0]
    message = assertion.args[1]

    assert isinstance(message, BinaryOpNode)
    assert message.op == "+"
    assert '"Expected value to match "' in repr(message)


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


def test_keyword_slice_index_parse_from_modular_stdlib():
    # Reduced from Modular string/path stdlib byte-indexed slices.
    code = """
    def main():
        var head = path_str[byte=i:]
        var prefix = String(e)[byte=:expected_msg.byte_length()]
        var suffix = self[byte=:-suffix.byte_length()]
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    head = function.body[0].initial_value
    prefix = function.body[1].initial_value
    suffix = function.body[2].initial_value

    assert isinstance(head.index, AssignmentNode)
    assert head.index.left.name == "byte"
    assert isinstance(head.index.right, SliceNode)
    assert head.index.right.start.name == "i"
    assert head.index.right.stop is None

    assert isinstance(prefix.index.right, SliceNode)
    assert prefix.index.right.start is None
    assert isinstance(prefix.index.right.stop, MethodCallNode)

    assert isinstance(suffix.index.right, SliceNode)
    assert suffix.index.right.start is None
    assert isinstance(suffix.index.right.stop, UnaryOpNode)


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


def test_multiple_with_context_managers_parse_from_mojo_gpu_puzzles():
    # Reduced from https://github.com/modular/mojo-gpu-puzzles.git commit
    # 87de51ac93bea662eba6f09d19e8744e56161027,
    # solutions/p02/p02.mojo main host-buffer initialization.
    code = """
    def main():
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            a_host[0] = Scalar[dtype](0)
            b_host[0] = Scalar[dtype](1)
    """
    ast = parse_code(tokenize_code(code))
    with_node = find_function(ast, "main").body[0]

    assert isinstance(with_node, WithNode)
    assert with_node.alias == "a_host"
    assert len(with_node.contexts) == 2
    assert [alias for _, alias in with_node.contexts] == ["a_host", "b_host"]
    assert [context.method for context, _ in with_node.contexts] == [
        "map_to_host",
        "map_to_host",
    ]
    assert [context.object.name for context, _ in with_node.contexts] == ["a", "b"]


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


def test_dict_display_and_comprehension_parse_from_current_mojo_docs():
    # Reduced from https://mojolang.org/docs/reference/expressions/
    # Version 1.0.0b1, "Collection displays" dictionaries and comprehensions.
    code = """
    def main():
        var empty: Dict[String, Int] = {}
        var ages = {"Alice": 30, "Bob": 25}
        var dict_squares = {x: x * x for x in range(3)}
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    empty = function.body[0].initial_value
    ages = function.body[1].initial_value
    dict_squares = function.body[2].initial_value

    assert isinstance(empty, DictLiteralNode)
    assert empty.entries == []

    assert isinstance(ages, DictLiteralNode)
    assert ages.entries == [('"Alice"', "30"), ('"Bob"', "25")]

    assert isinstance(dict_squares, DictComprehensionNode)
    assert dict_squares.key.name == "x"
    assert isinstance(dict_squares.value, BinaryOpNode)
    assert [(clause["kind"]) for clause in dict_squares.clauses] == ["for"]
    assert dict_squares.clauses[0]["pattern"] == "x"
    assert isinstance(dict_squares.clauses[0]["iterable"], FunctionCallNode)


def test_set_comprehension_parse_from_current_mojo_docs():
    # Reduced from https://github.com/modular/modular/blob/04cff5a4cc491ec2bf6850ce99e0253075fc908c/mojo/docs/reference/expressions.mdx,
    # "Set comprehensions" in the Mojo expression reference.
    code = """
    def main():
        var fibs = {fib(x) for x in range(6)}
    """
    ast = parse_code(tokenize_code(code))
    fibs = find_function(ast, "main").body[0].initial_value

    assert isinstance(fibs, SetComprehensionNode)
    assert isinstance(fibs.expression, FunctionCallNode)
    assert [(clause["kind"]) for clause in fibs.clauses] == ["for"]
    assert fibs.clauses[0]["pattern"] == "x"
    assert isinstance(fibs.clauses[0]["iterable"], FunctionCallNode)


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


def test_mlir_backtick_type_parse_from_modular_gpu_globals():
    # Reduced from https://github.com/modular/modular.git commit
    # daa47bb846cc213723a54c51844ea4e923eb5e13,
    # mojo/stdlib/std/gpu/globals.mojo _resolve_max_threads_per_block_metadata.
    code = """
    def _resolve_max_threads_per_block_metadata() -> __mlir_type.`!kgen.string`:
        return "nvvm.maxntid".value
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "_resolve_max_threads_per_block_metadata")
    returned = function.body[0].value

    assert function.return_type == "__mlir_type.`!kgen.string`"
    assert isinstance(returned, MemberAccessNode)
    assert returned.object == '"nvvm.maxntid"'
    assert returned.member == "value"


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


def test_prefixed_adjacent_string_literals_in_call_parse_from_modular_trace_gemv():
    # Reduced from https://github.com/modular/modular.git commit
    # 9ddf207f42fc67a6f33bd7b4ccc94a6a52133c8f,
    # max/kernels/benchmarks/gpu/nn/trace_gemv_partial_norm.mojo.
    code = """
    def main():
        print(
            t"{b},"
            t"{Int(trace_host[base + 0])},"
            t"{Int(trace_host[base + 1])}"
        )
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    call = function.body[0]

    assert call.args == [
        't"{b},{Int(trace_host[base + 0])},{Int(trace_host[base + 1])}"'
    ]


def test_adjacent_string_after_binary_string_parse_from_modular_topk_gpu_fi():
    # Reduced from https://github.com/modular/modular.git commit
    # 9ddf207f42fc67a6f33bd7b4ccc94a6a52133c8f,
    # max/kernels/test/gpu/nn/test_topk_gpu_fi.mojo.
    code = """
    def main():
        raise Error(
            "Sampled index "
            + String(idx)
            + " is NOT in the top-K set! This indicates a bug in the"
            " sampling kernel."
        )
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    error_call = function.body[0].args[0]
    message = error_call.args[0]

    assert isinstance(error_call, FunctionCallNode)
    assert isinstance(message, BinaryOpNode)
    assert (
        message.right
        == '" is NOT in the top-K set! This indicates a bug in the sampling kernel."'
    )


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


def test_bool_literal_tokens_can_be_local_binding_names_parse():
    code = """
    def main():
        var false = false
        var true, flag = true, false
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    false_decl = function.body[0]
    tuple_decl = function.body[1]

    assert false_decl.name == "false"
    assert false_decl.initial_value == "false"
    assert isinstance(tuple_decl.name, TupleNode)
    assert [element.name for element in tuple_decl.name.elements] == ["true", "flag"]
    assert isinstance(tuple_decl.initial_value, TupleNode)
    assert tuple_decl.initial_value.elements == ["true", "false"]


def test_list_literal_elements_continue_binary_expression_across_newline_parse():
    code = """
    def main():
        var values = [
            "Warmup Total: "
            + String(count),
            List("123456789012345".as_bytes())
            + [0xED],
        ]
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    values = function.body[0].initial_value

    assert isinstance(values, ListLiteralNode)
    assert len(values.elements) == 2
    assert all(isinstance(element, BinaryOpNode) for element in values.elements)
    assert values.elements[0].op == "+"
    assert values.elements[1].op == "+"


def test_tuple_assignment_with_mixed_targets_and_ref_binding_parse():
    code = """
    def main():
        curr_index, res.data[i] = divmod(curr_index, IntType(shape.get[i]()))
        count, ref chars = mapping.value()
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    index_assignment = function.body[0]
    ref_assignment = function.body[1]

    assert isinstance(index_assignment, AssignmentNode)
    assert isinstance(index_assignment.left, TupleNode)
    index_target = index_assignment.left.elements[1]
    assert index_assignment.left.elements[0].name == "curr_index"
    assert isinstance(index_target, ArrayAccessNode)
    assert isinstance(index_target.array, MemberAccessNode)
    assert index_target.array.object.name == "res"
    assert index_target.array.member == "data"
    assert index_target.index.name == "i"

    assert isinstance(ref_assignment, AssignmentNode)
    assert isinstance(ref_assignment.left, TupleNode)
    count_target, chars_target = ref_assignment.left.elements
    assert count_target.name == "count"
    assert chars_target.name == "chars"
    assert getattr(chars_target, "target_convention", None) == "ref"


def test_empty_tuple_expression_parse_from_variadic_and_unsafe_union():
    code = """
    def main():
        var empty = ()
        var u = UnsafeUnion[Int32, Float32](unsafe_uninitialized=())
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    empty = function.body[0].initial_value
    unsafe_union = function.body[1].initial_value

    assert isinstance(empty, TupleNode)
    assert empty.elements == []
    assert isinstance(unsafe_union, VectorConstructorNode)
    assert isinstance(unsafe_union.args[0].right, TupleNode)
    assert unsafe_union.args[0].right.elements == []


def test_binding_convention_assignment_targets_parse_from_mojo_reference():
    code = """
    def main():
        ref a = var b = var c = "Hello"
        (var list) = values
        (var codepoint), (ref expected_utf8) = elements
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    ref_binding = function.body[0]
    list_assignment = function.body[1]
    tuple_assignment = function.body[2]

    assert isinstance(ref_binding, VariableDeclarationNode)
    assert getattr(ref_binding, "binding_convention", None) == "ref"
    assert isinstance(ref_binding.initial_value, AssignmentNode)
    assert ref_binding.initial_value.left.name == "b"
    assert getattr(ref_binding.initial_value.left, "target_convention", None) == "var"
    nested_assignment = ref_binding.initial_value.right
    assert isinstance(nested_assignment, AssignmentNode)
    assert nested_assignment.left.name == "c"
    assert getattr(nested_assignment.left, "target_convention", None) == "var"

    assert isinstance(list_assignment, AssignmentNode)
    assert list_assignment.left.name == "list"
    assert getattr(list_assignment.left, "target_convention", None) == "var"

    assert isinstance(tuple_assignment, AssignmentNode)
    assert isinstance(tuple_assignment.left, TupleNode)
    codepoint, expected_utf8 = tuple_assignment.left.elements
    assert codepoint.name == "codepoint"
    assert getattr(codepoint, "target_convention", None) == "var"
    assert expected_utf8.name == "expected_utf8"
    assert getattr(expected_utf8, "target_convention", None) == "ref"


def test_var_default_parameter_parse_from_stdlib_collections():
    code = """
    struct Counter:
        def pop(mut self, value: Self.V, var default: Int) -> Int:
            return self._data.pop(value, default)
    """
    ast = parse_code(tokenize_code(code))
    struct = find_struct(ast, "Counter")
    method = struct.methods[0]
    default_param = method.params[2]

    assert default_param.name == "default"
    assert default_param.vtype == "Int"
    assert getattr(default_param, "parameter_convention", None) == "var"


def test_typed_member_assignment_parse_from_platform_stat_structs():
    code = """
    struct Stat:
        def __init__(out self):
            self.unused: InlineArray[Int64, 2] = [0, 0]
    """
    ast = parse_code(tokenize_code(code))
    struct = find_struct(ast, "Stat")
    assignment = struct.methods[0].body[0]

    assert isinstance(assignment, AssignmentNode)
    assert isinstance(assignment.left, MemberAccessNode)
    assert assignment.left.object.name == "self"
    assert assignment.left.member == "unused"
    assert getattr(assignment, "target_type", None) == "InlineArray[Int64, 2]"
    assert isinstance(assignment.right, ListLiteralNode)
    assert assignment.right.elements == ["0", "0"]


def test_parenthesized_tuple_var_declaration_parse_from_modular_sm90_matmul():
    # Reduced from https://github.com/modular/modular.git commit
    # 04cff5a4cc491ec2bf6850ce99e0253075fc908c,
    # max/kernels/src/linalg/matmul/gpu/sm90/matmul_kernels.mojo lines 895-902.
    code = """
    def kernel():
        var (
            warp_group_idx,
            warp_group_thread_idx,
            rank_m,
            rank_n,
            warp_id,
            lane_predicate,
        ) = Self.common_kernel_init()
    """
    ast = parse_code(tokenize_code(code))
    declaration = find_function(ast, "kernel").body[0]

    assert isinstance(declaration.name, TupleNode)
    assert [element.name for element in declaration.name.elements] == [
        "warp_group_idx",
        "warp_group_thread_idx",
        "rank_m",
        "rank_n",
        "warp_id",
        "lane_predicate",
    ]
    assert isinstance(declaration.initial_value, MethodCallNode)
    assert declaration.initial_value.object.name == "Self"
    assert declaration.initial_value.method == "common_kernel_init"


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


def test_for_else_parse_from_official_compound_statement_reference():
    # Reduced from https://docs.modular.com/mojo/reference/mojo-compound-statements/
    # "Loops and else clauses" documents that a for-else block skips the
    # else body when break exits the loop.
    code = """
    def search(items: List[Int], target: Int):
        var found = False
        for item in items:
            if item == target:
                found = True
                break
        else:
            print("not found")
    """
    ast = parse_code(tokenize_code(code))
    loop = find_function(ast, "search").body[1]

    assert isinstance(loop, RangeForNode)
    assert loop.name == "item"
    assert loop.iterable.name == "items"
    assert isinstance(loop.body[0], IfNode)
    assert isinstance(loop.body[0].if_body[1], BreakNode)
    assert loop.else_body[0].name == "print"


def test_while_else_parse_from_official_control_flow_docs():
    # Reduced from https://docs.modular.com/mojo/manual/control-flow/
    # "The while statement" documents while loops with else clauses.
    code = """
    def main():
        var n = 5
        while n < 4:
            print(n)
            n += 1
        else:
            print("Loop completed")
    """
    ast = parse_code(tokenize_code(code))
    loop = find_function(ast, "main").body[1]

    assert isinstance(loop, WhileNode)
    assert isinstance(loop.condition, BinaryOpNode)
    assert len(loop.body) == 2
    assert loop.else_body[0].name == "print"


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


def test_try_except_name_after_multiline_specialized_call_parse():
    code = """
    def main():
        try:
            bench_bf16[
                num_heads=32,
                head_dim=256,
            ](m, ctx)
        except e:
            print("bench failed:", e)
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    statement = function.body[0]

    assert isinstance(statement, TryExceptNode)
    assert statement.exception_name == "e"
    assert isinstance(statement.try_body[0], VectorConstructorNode)


def test_try_except_else_finally_parse_from_official_error_docs():
    # Reduced from the full try syntax documented at:
    # https://docs.modular.com/mojo/manual/errors/
    code = """
    def main():
        try:
            run_kernel()
        except error:
            recover(error)
        else:
            mark_success()
        finally:
            cleanup()
    """
    ast = parse_code(tokenize_code(code))
    statement = find_function(ast, "main").body[0]

    assert isinstance(statement, TryExceptNode)
    assert statement.exception_name == "error"
    assert statement.try_body[0].name == "run_kernel"
    assert statement.except_body[0].name == "recover"
    assert statement.else_body[0].name == "mark_success"
    assert statement.finally_body[0].name == "cleanup"


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


def test_extension_block_parse_from_modular_gpu_kernels():
    # Reduced from Modular SM90 matmul and RDNA attention __extension blocks.
    code = """
    __extension HopperMatmulSM90Kernel:
        @staticmethod
        @always_inline
        def run_persistent[
            tile_shape: IndexList[2],
        ](problem_shape: IndexList[3]) -> Int:
            return 1

    __extension AttentionRDNA:
        def mha_decode(mut self, num_partitions: Int):
            pass
    """
    ast = parse_code(tokenize_code(code))

    assert len(ast.extensions) == 2
    extension = ast.extensions[0]
    assert isinstance(extension, ExtensionNode)
    assert extension.name == "HopperMatmulSM90Kernel"
    assert [method.name for method in extension.methods] == ["run_persistent"]
    assert extension.methods[0].return_type == "Int"
    assert [attr.name for attr in extension.methods[0].attributes] == [
        "staticmethod",
        "always_inline",
    ]
    assert ast.extensions[1].methods[0].params[0].name == "self"


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


def test_postfix_transfer_marker_before_member_call_parse_from_modular_primitives():
    # Reduced from https://github.com/modular/modular.git commit
    # 9ddf207f42fc67a6f33bd7b4ccc94a6a52133c8f,
    # max/kernels/src/graph_compiler/builtin_primitives/primitives.mojo.
    code = """
    def make_buffer(buf: ByteBuffer, shape: Shape) -> MutByteBuffer:
        return MutByteBuffer(buf^.take_ptr(), shape)

    def xor_value(a: Int, b: Int) -> Int:
        return a ^ b
    """
    ast = parse_code(tokenize_code(code))
    make_buffer = find_function(ast, "make_buffer")
    xor_value = find_function(ast, "xor_value")

    transferred_call = make_buffer.body[0].value.args[0]
    assert isinstance(transferred_call, MethodCallNode)
    assert transferred_call.method == "take_ptr"
    assert transferred_call.object.name == "buf"
    assert getattr(transferred_call.object, "is_transfer", False)
    assert isinstance(xor_value.body[0].value, BinaryOpNode)
    assert xor_value.body[0].value.op == "^"


def test_return_tuple_parse_from_modular_comma_bucket():
    # Reduced from Modular max/kernels examples that previously left the comma
    # for statement termination after parsing only the first return expression.
    code = """
    def map_fn() -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        return lower_bound, upper_bound

    def take_results(deinit self) -> Tuple[Int, InlineArray[Int, Self.num_allocs]]:
        return self.pool_size, self.offsets
    """
    ast = parse_code(tokenize_code(code))
    map_fn = find_function(ast, "map_fn")
    take_results = find_function(ast, "take_results")

    map_return = map_fn.body[0].value
    assert isinstance(map_return, TupleNode)
    assert [element.name for element in map_return.elements] == [
        "lower_bound",
        "upper_bound",
    ]

    results_return = take_results.body[0].value
    assert isinstance(results_return, TupleNode)
    assert [element.member for element in results_return.elements] == [
        "pool_size",
        "offsets",
    ]


def test_comptime_tuple_declaration_parse_from_modular_grouped_matmul():
    # Reduced from Modular max/kernels/src/linalg/grouped_matmul.mojo.
    code = """
    def writeback():
        comptime dst_m_offset, dst_n_offset = divmod(dst_idx, N)
    """
    ast = parse_code(tokenize_code(code))
    declaration = find_function(ast, "writeback").body[0]

    assert isinstance(declaration, VariableDeclarationNode)
    assert getattr(declaration, "is_comptime", False)
    assert isinstance(declaration.name, TupleNode)
    assert [target.name for target in declaration.name.elements] == [
        "dst_m_offset",
        "dst_n_offset",
    ]
    assert isinstance(declaration.initial_value, FunctionCallNode)
    assert declaration.initial_value.name == "divmod"


def test_parenthesized_return_type_parse_from_modular_shared_memory():
    # Reduced from Modular max/kernels/src/linalg/structuring.mojo and
    # max/kernels/src/linalg/matmul/gpu/sm90/matmul_kernels.mojo.
    code = """
    def ptr() -> (
        UnsafePointer[
            Int8, MutExternalOrigin, address_space=AddressSpace.SHARED
        ]
    ):
        pass

    def common_kernel_init() -> (
        Tuple[
            Int,
            Int,
            Bool,
        ]
    ):
        pass
    """
    ast = parse_code(tokenize_code(code))
    ptr = find_function(ast, "ptr")
    common_kernel_init = find_function(ast, "common_kernel_init")

    assert ptr.return_type == (
        "UnsafePointer[Int8, MutExternalOrigin, " "address_space=AddressSpace.SHARED]"
    )
    assert common_kernel_init.return_type == "Tuple[Int, Int, Bool]"


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


def test_double_bracket_specialized_call_parse_from_vendor_blas_tests():
    code = """
    def main() raises:
        if has_amd_gpu_accelerator():
            test_matmul[[DType.float8_e4m3fnuz, DType.bfloat16]]()
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    call = function.body[0].if_body[0]

    assert isinstance(call, CallNode)
    assert isinstance(call.callee, ArrayAccessNode)
    assert call.callee.array.name == "test_matmul"
    assert isinstance(call.callee.index, TupleNode)
    assert [element.member for element in call.callee.index.elements] == [
        "float8_e4m3fnuz",
        "bfloat16",
    ]


def test_switch_keyword_can_be_value_name_parse_from_tile_tests():
    code = """
    def main():
        def print_wrapper[tile_size: Int, switch: Bool](offset: Int):
            print(offset, tile_size, switch)
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    nested = function.body[0]
    call = nested.body[0]

    assert isinstance(nested, FunctionNode)
    assert nested.name == "print_wrapper"
    assert isinstance(call, FunctionCallNode)
    assert call.args[2].name == "switch"


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


def test_trait_comptime_associated_type_intersection_parse():
    code = """
    trait Boxable:
        comptime Associated: Writable & Copyable & ImplicitlyDestructible

        def unbox(self) -> Self.Associated:
            ...
    """
    ast = parse_code(tokenize_code(code))
    trait = ast.traits[0]
    associated = trait.members[0]

    assert isinstance(associated, VariableDeclarationNode)
    assert associated.name == "Associated"
    assert associated.vtype == "Writable&Copyable&ImplicitlyDestructible"
    assert getattr(associated, "is_comptime", False)


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


def test_backtick_comptime_names_parse_from_official_gpu_notebook_example():
    # Reduced from https://docs.modular.com/mojo/tools/notebooks/
    # "Example: Hello writing" uses escaped emoji comptime constants in a GPU
    # kernel and host-side fill.
    code = """
    comptime `✅`: Int32 = 1
    comptime `❌`: Int32 = 0

    def kernel(value: UnsafePointer[Scalar[DType.int32], MutAnyOrigin]):
        value[0] = `✅`

    def main():
        out.enqueue_fill(`❌`)
    """

    ast = parse_code(tokenize_code(code))
    success, failure = ast.global_variables
    kernel = find_function(ast, "kernel")
    main = find_function(ast, "main")

    assert success.name == "`✅`"
    assert success.vtype == "Int32"
    assert success.initial_value == "1"
    assert getattr(success, "is_comptime", False)

    assert failure.name == "`❌`"
    assert failure.vtype == "Int32"
    assert failure.initial_value == "0"
    assert getattr(failure, "is_comptime", False)

    assert kernel.body[0].right.name == "`✅`"
    assert main.body[0].args[0].name == "`❌`"


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


def test_not_in_membership_condition_parse_from_mojo_gpu_puzzles_dispatch():
    # Reduced from https://github.com/modular/mojo-gpu-puzzles.git commit
    # 87de51ac93bea662eba6f09d19e8744e56161027,
    # problems/p13/p13.mojo main command-dispatch guard.
    code = """
    def main():
        if len(argv()) != 2 or argv()[1] not in [
            "--simple",
            "--block-boundary",
        ]:
            pass
    """

    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    condition = function.body[0].condition

    assert isinstance(condition, BinaryOpNode)
    assert condition.op == "||"
    assert isinstance(condition.right, BinaryOpNode)
    assert condition.right.op == "not in"
    assert isinstance(condition.right.right, ListLiteralNode)
    assert condition.right.right.elements == ['"--simple"', '"--block-boundary"']


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


def test_matrix_multiplication_operator_parsing_from_official_docs():
    # Reduced from https://docs.modular.com/mojo/manual/operators/
    # "Matrix multiplication" documents @ as Mojo's matrix multiplication operator.
    code = """
    fn main(lhs: Matrix[DType.float32, 4, 4], rhs: Matrix[DType.float32, 4, 4]):
        var result = lhs @ rhs
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    declaration = function.body[0]

    assert isinstance(declaration.initial_value, BinaryOpNode)
    assert declaration.initial_value.op == "@"
    assert declaration.initial_value.left.name == "lhs"
    assert declaration.initial_value.right.name == "rhs"
    assert declaration.attributes == []


def test_walrus_assignment_expression_parse_from_official_docs():
    # Reduced from https://docs.modular.com/mojo/manual/operators/
    # "Walrus operator" assignment-expression example.
    code = """
    def main():
        var name = ""
        while (name := input("Name or 'quit': ")) != "quit":
            print("Hello,", name)
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    condition = function.body[1].condition

    assert isinstance(condition, BinaryOpNode)
    assert condition.op == "!="
    assignment = condition.left
    assert isinstance(assignment, AssignmentNode)
    assert assignment.operator == ":="
    assert assignment.left.name == "name"
    assert assignment.right.name == "input"
    assert assignment.right.args == ["\"Name or 'quit': \""]


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


def test_triple_quoted_string_literal_parsing_from_mojo_reference():
    # Reduced from https://mojolang.org/docs/reference/literals/
    code = '''
    fn main():
        let message = """Multi-line
string"""
        print(message)
    '''
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    declaration = function.body[0]

    assert isinstance(declaration, VariableDeclarationNode)
    assert declaration.name == "message"
    assert declaration.initial_value == '"Multi-line\\nstring"'


def test_triple_quoted_call_argument_parse_from_modular_fast_div():
    code = '''
    fn main():
        assert_equal(
            """div: 33
mprime: 4034666248
""",
            String(value),
        )
    '''
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    call = function.body[0]

    assert isinstance(call, FunctionCallNode)
    assert call.args[0] == '"div: 33\\nmprime: 4034666248\\n"'
    assert isinstance(call.args[1], VectorConstructorNode)
    assert call.args[1].type_name == "String"


def test_soft_infix_line_continuations_parse_from_practical_examples():
    # Reduced from ahmetax/Practical-Mojo-Examples v0.26.2 examples that
    # split long expressions after and before infix operators.
    code = """
    fn fmt_time(ns: UInt) -> String:
        return String(ns // 1_000_000) + "." +
               String((ns % 1_000_000) // 10_000) + " ms"

    fn accumulate(state: RunState, a: Float32):
        var xbi = (state.xb.data + 1).load[width=4](0)
            + a * (state.value_cache.data + 1).load[width=4](0)
    """
    ast = parse_code(tokenize_code(code))
    fmt_time = find_function(ast, "fmt_time")
    accumulate = find_function(ast, "accumulate")

    assert isinstance(fmt_time.body[0], ReturnNode)
    assert isinstance(fmt_time.body[0].value, BinaryOpNode)
    declaration = accumulate.body[0]
    assert isinstance(declaration, VariableDeclarationNode)
    assert isinstance(declaration.initial_value, BinaryOpNode)


def test_braced_type_initializer_suffix_parse_from_nbody_example():
    # Reduced from ksandvik/mojo-examples examples/nbody.mojo Planet.__init__.
    code = """
    struct Planet:
        var pos: SIMD[DType.float64, 4]
        var velocity: SIMD[DType.float64, 4]
        var mass: Float64

        fn __init__(
            pos: SIMD[DType.float64, 4],
            velocity: SIMD[DType.float64, 4],
            mass: Float64,
        ) -> Self:
            return Self {
                pos: pos,
                velocity: velocity,
                mass: mass,
            }
    """
    ast = parse_code(tokenize_code(code))
    init_method = find_struct(ast, "Planet").methods[0]
    value = init_method.body[0].value

    assert isinstance(value, FunctionCallNode)
    assert value.name == "Self"
    assert [arg.left.name for arg in value.args] == ["pos", "velocity", "mass"]


def test_nested_prefixed_tstring_parse_from_modular_format_tests():
    code = r"""
    fn main():
        var tstring = t"hello \t{x}, {rt"world \t{x}"}"
        var greeting = String(t"Hello, {t"dear {name}"}!")
    """
    ast = parse_code(tokenize_code(code))
    function = find_function(ast, "main")
    tstring = function.body[0]
    greeting = function.body[1]

    assert isinstance(tstring, VariableDeclarationNode)
    assert tstring.initial_value == r't"hello \\t{x}, {rt\"world \\t{x}\"}"'
    assert isinstance(greeting.initial_value, VectorConstructorNode)


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
