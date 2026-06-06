from typing import List

import pytest

from crosstl.backend.Mojo import MojoCrossGLCodeGen
from crosstl.backend.Mojo.MojoLexer import MojoLexer
from crosstl.backend.Mojo.MojoParser import MojoParser
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser


def generate_code(ast_node):
    """Test the code generator
    Args:
        ast_node: The abstract syntax tree generated from the code
    Returns:
        str: The generated code from the abstract syntax tree
    """
    codegen = MojoCrossGLCodeGen.MojoToCrossGLConverter()
    return codegen.generate(ast_node)


def tokenize_code(code: str) -> List:
    lexer = MojoLexer(code)
    return lexer.tokenize()


def parse_code(tokens: List):
    parser = MojoParser(tokens)
    return parser.parse()


def parse_crossgl(code: str):
    return CrossGLParser(CrossGLLexer(code).tokens).parse()


def test_struct_codegen():
    code = """
    struct VSInput:
        var position: float4
        var color: float2

    struct VSOutput:
        var out_position: float2

    fn vertex_main(input: VSInput) -> VSOutput:
        var output: VSOutput
        output.out_position = input.position.xy
        return output

    struct PSInput:
        var in_position: float2

    struct PSOutput:
        var out_color: float4

    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        output.out_color = float4(input.in_position, 0.0, 1.0)
        return output
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "struct VSInput" in generated_code
        assert "struct VSOutput" in generated_code
    except SyntaxError:
        pytest.fail("Struct parsing or code generation not implemented.")


def test_struct_generic_member_codegen():
    code = """
    struct Resources:
        @position
        var transform: Matrix[DType.float32, 4, 4]
        SIMD[DType.float32, 4] tint @color
        samples: InlineArray[SIMD[DType.float32, 4], 2]
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "mat4 transform @ Position;" in generated_code
    assert "vec4 tint @ Color;" in generated_code
    assert "InlineArray[SIMD[DType.float32, 4], 2] samples;" in generated_code


def test_struct_method_codegen_preserves_method_body():
    code = """
    struct VectorAddition:
        @staticmethod
        def execute(ctx: DeviceContext) raises:
            raise Error("No known target")
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert "struct VectorAddition" in generated_code
    assert "void execute(DeviceContext ctx)" in generated_code
    assert 'raise(Error("No known target"));' in generated_code


def test_function_parameter_convention_codegen_drops_mojo_conventions():
    code = """
    def incr(a: Int, out b: Int):
        pass

    def __init__(out self, value: Int):
        pass
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert "void incr(int a, int b)" in generated_code
    assert "void __init__(int value)" in generated_code
    assert "out b" not in generated_code
    assert "self" not in generated_code


def test_bracketed_ref_parameter_convention_codegen_from_modular_amd_helpers():
    # Reduced from https://github.com/modular/mojo.git commit
    # 7aa053560034c8c5b4f9acb0a5b450e79d2f7c18,
    # max/examples/custom_ops/kernels/amd_helpers.mojo MMATileBuffers.__init__.
    code = """
    def init_tile_buffer(
        ref[Self.tensor_origin] tensor: Self.tensor_type,
        warp_idx: Int,
    ):
        pass
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert "void init_tile_buffer(Self.tensor_type tensor, int warp_idx)" in (
        generated_code
    )
    assert "ref[Self.tensor_origin]" not in generated_code


def test_function_type_parameter_codegen_from_modular_gpu_reduction():
    # Reduced from https://github.com/modular/modular.git commit
    # daa47bb846cc213723a54c51844ea4e923eb5e13,
    # mojo/stdlib/std/algorithm/backend/gpu/reduction.mojo small_reduce_kernel.
    code = """
    def reduce_adapter(
        input_fn: def[dtype: DType, width: Int, rank: Int](
            IndexList[rank]
        ) capturing[_] -> SIMD[dtype, width],
    ):
        pass
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert (
        "void reduce_adapter(def[dtype:DType, width:Int, rank:Int]"
        "(IndexList[rank]) capturing[_] -> SIMD[dtype, width] input_fn)"
        in generated_code
    )
    assert "Unhandled expression" not in generated_code


def test_thin_function_type_parameter_codegen_from_official_parameter_docs():
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
    generated_code = generate_code(ast)

    assert (
        "int invoke_compare(Scalar[dtype] lhs, Scalar[dtype] rhs, "
        "def(Scalar[dtype], Scalar[dtype]) thin -> Int compare)"
    ) in generated_code
    assert "return compare(lhs, rhs);" in generated_code
    assert "Unhandled expression" not in generated_code


def test_variadic_and_reference_parameter_codegen_from_current_docs():
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
    generated_code = generate_code(ast)

    assert "void __init__(Self.ElementType elements)" in generated_code
    assert "void __del__()" in generated_code
    assert "Self.ElementType __getitem__(int i)" in generated_code
    assert "var *elements" not in generated_code
    assert "deinit self" not in generated_code
    assert "ref[self]" not in generated_code


def test_variadic_keyword_parameter_codegen_from_current_docs():
    code = """
    def print_nicely(**kwargs: Int):
        for item in kwargs.items():
            print(item.key, "=", item.value)
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert "void print_nicely(int kwargs)" in generated_code
    assert "for item in kwargs.items()" in generated_code
    assert "**kwargs" not in generated_code
    assert "Unhandled" not in generated_code


def test_function_parameter_separator_markers_codegen_drops_markers():
    code = """
    def kw_only_args(a1: Int, a2: Int, *, double: Bool) -> Int:
        return a1 + a2

    def positional_only_args(a1: Int, a2: Int, /, b1: Int) -> Int:
        return a1 + a2 + b1
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert "int kw_only_args(int a1, int a2, bool double)" in generated_code
    assert "int positional_only_args(int a1, int a2, int b1)" in generated_code
    assert "*, double" not in generated_code
    assert "/, b1" not in generated_code


def test_function_optional_argument_default_codegen_preserves_default():
    code = """
    fn my_pow(base: Int, exp: Int = 2) -> Int:
        return base ** exp
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert "int my_pow(int base, int exp = 2)" in generated_code
    assert "return (base ** exp);" in generated_code


def test_default_keyword_parameter_name_codegen_from_modular_stdlib():
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
    generated_code = generate_code(ast)

    assert (
        "SIMD[dtype, width] gather("
        "SIMD[DType.bool, width] mask = SIMD[DType.bool, width](fill = true), "
        "SIMD[dtype, width] default = 0)"
    ) in generated_code
    assert "return default;" in generated_code
    assert "Unhandled expression" not in generated_code


def test_ref_binding_declaration_codegen_from_official_variables_docs():
    # https://docs.modular.com/mojo/manual/variables/#reference-bindings
    code = """
    def bump_item(items: List[Int]):
        ref item_ref = items[1]
        item_ref += 1
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert "void bump_item(List[Int] items)" in generated_code
    assert "var item_ref = items[1];" in generated_code
    assert "item_ref += 1;" in generated_code
    assert "ref item_ref" not in generated_code
    assert "Unhandled" not in generated_code


def test_backtick_local_identifier_codegen_from_modular_base64_stdlib_reparses_crossgl():
    # Reduced from https://github.com/modular/modular.git commit
    # daa47bb846cc213723a54c51844ea4e923eb5e13,
    # mojo/stdlib/std/base64/_b64encode.mojo _6bit_to_byte.combine.
    code = """
    def combine(shuffled: Bytes, mask: Bytes) -> Bytes:
        var `6bit` = shuffled & mask
        return shift(`6bit`)
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert "var _6bit = (shuffled & mask);" in generated_code
    assert "return shift(_6bit);" in generated_code
    assert "`" not in generated_code
    parse_crossgl(generated_code)


def test_backtick_keyword_function_name_codegen_from_official_docs_reparses_crossgl():
    # Reduced from https://docs.modular.com/mojo/reference/mojo-function-declarations/
    # "Function names" escaped keyword identifier example.
    code = """
    def `import`() -> Int:
        return 1

    def main() -> Int:
        return `import`()
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)
    reparsed = parse_crossgl(generated_code)

    assert "int import_()" in generated_code
    assert "return import_();" in generated_code
    assert "int import()" not in generated_code
    assert "`" not in generated_code
    assert reparsed.functions[0].name == "import_"
    assert reparsed.functions[1].body.statements[0].value.function.name == "import_"


def test_comptime_expression_prefix_codegen_drops_mojo_marker():
    code = """
    def main():
        var dev_buf = ctx.enqueue_create_buffer[int_dtype](
            comptime (layout.size())
        )
        for i in range(comptime (tile_layout.size())):
            pass
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert "comptime" not in generated_code
    assert "layout.size()" in generated_code
    assert "tile_layout.size()" in generated_code


def test_single_quoted_string_literal_codegen():
    code = """
    fn message() -> String:
        let status: String = 'done'
        return status
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert "String status = 'done';" in generated_code
    assert "return status;" in generated_code


def test_documented_numeric_literal_forms_codegen_from_mojo_reference():
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
    generated_code = generate_code(ast)

    assert "let grouped = 1_000_000;" in generated_code
    assert "let relaxed = 1__000_;" in generated_code
    assert "let fraction_only = .5;" in generated_code
    assert "let trailing_point = 2.;" in generated_code
    assert "let grouped_float = 1_000.000_5;" in generated_code
    assert "let exponent_only = 1E10;" in generated_code


def test_function_capturing_raises_effects_codegen_are_dropped():
    code = """
    def sum_kernel_benchmark(
        mut b: Bencher, input_data: SumKernelBenchmarkParams
    ) capturing raises:
        pass
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert (
        "void sum_kernel_benchmark(Bencher b, SumKernelBenchmarkParams input_data)"
        in generated_code
    )
    assert "capturing" not in generated_code
    assert "raises" not in generated_code


def test_async_function_codegen_from_modular_builtin_kernels_reparses_crossgl():
    # Source: https://github.com/modular/modular
    # Commit: daa47bb846cc213723a54c51844ea4e923eb5e13
    # Path: max/kernels/src/graph_compiler/builtin_kernels/kernels.mojo
    # Lines: 2207-2215
    # dispatch_async_tasks_to_devices wrapper.
    code = """
    @always_inline
    @parameter
    async def wrapper[index: Int]() -> None:
        return
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)
    reparsed = parse_crossgl(generated_code)

    assert getattr(ast.functions[0], "is_async", False)
    assert "void wrapper() @ always_inline @ parameter" in generated_code
    assert "async" not in generated_code
    assert reparsed.functions[0].name == "wrapper"
    assert [attr.name for attr in reparsed.functions[0].attributes] == [
        "always_inline",
        "parameter",
    ]


def test_await_expression_codegen_from_modular_runtime_async_tests():
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
    generated_code = generate_code(ast)

    assert "int add_two(int a, int b) @ parameter" in generated_code
    assert "int run_as_group()" in generated_code
    assert "return (create_task(add[1](a)) + create_task(add[2](b)));" in generated_code
    assert "return (t0 + t1);" in generated_code
    assert "await" not in generated_code
    assert "Unhandled expression" not in generated_code


def test_function_capture_list_codegen_from_modular_packing_kernel():
    # Reduced from https://github.com/modular/modular.git commit
    # 7aa053560034c8c5b4f9acb0a5b450e79d2f7c18,
    # max/kernels/src/linalg/packing.mojo
    # _pack_b_ndbuffer_impl.dispatch_on_kernel_type.
    code = """
    def pack_dispatch(output_buffer: TileTensor, b_input: TileTensor):
        @always_inline
        def dispatch_on_kernel_type[kernel_type: Bool]() {output_buffer, b_input}:
            pack_b(output_buffer, b_input)

        dispatch_get_kernel_type(dispatch_on_kernel_type)
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert "void pack_dispatch(TileTensor output_buffer, TileTensor b_input)" in (
        generated_code
    )
    assert "void dispatch_on_kernel_type() @ always_inline" in generated_code
    assert "pack_b(output_buffer, b_input);" in generated_code
    assert "dispatch_get_kernel_type(dispatch_on_kernel_type);" in generated_code
    assert "{output_buffer, b_input}" not in generated_code


def test_list_literal_argument_codegen_from_modular_reduction_example():
    code = """
    def main():
        bench.bench_with_input[Params, kernel](
            BenchId("sum_kernel_benchmark", "gpu"),
            Params(out_ptr, a_ptr),
            [ThroughputMeasure(BenchMetric.bytes, SIZE * size_of[dtype]())],
        )
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert (
        "[ThroughputMeasure(BenchMetric.bytes, (SIZE * size_of[dtype]()))]"
        in generated_code
    )


def test_list_comprehension_codegen_from_current_mojo_docs():
    code = """
    def main():
        var squares = [x * x for x in range(5) if x % 2 == 0]
        var products = [(x, y) for x in range(3) for y in range(2) if x != y]
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert "[(x * x) for x in range(5) if ((x % 2) == 0)]" in generated_code
    assert "[(x, y) for x in range(3) for y in range(2) if (x != y)]" in generated_code
    assert "Unhandled expression: ListComprehensionNode" not in generated_code


def test_dict_display_and_comprehension_codegen_from_current_mojo_docs():
    # Reduced from https://mojolang.org/docs/reference/expressions/
    # Version 1.0.0b1, "Collection displays" dictionaries and comprehensions.
    code = """
    def main():
        var empty: Dict[String, Int] = {}
        var ages = {"Alice": 30, "Bob": 25}
        var dict_squares = {x: x * x for x in range(3)}
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert "Dict[String, Int] empty = {};" in generated_code
    assert 'var ages = {"Alice": 30, "Bob": 25};' in generated_code
    assert "var dict_squares = {x: (x * x) for x in range(3)};" in generated_code
    assert "Unhandled expression: DictLiteralNode" not in generated_code
    assert "Unhandled expression: DictComprehensionNode" not in generated_code


def test_braced_set_and_initializer_list_codegen_from_current_mojo_docs():
    # Reduced from https://mojolang.org/docs/reference/expressions/
    # Version 1.0.0b1, "Collection displays" sets and initializer lists.
    code = """
    def main():
        var primes = {2, 3, 5}
        var point: Point = {x=1.0, y=2.0}
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert "var primes = {2, 3, 5};" in generated_code
    assert "Point point = {x = 1.0, y = 2.0};" in generated_code
    assert "Unhandled expression: BracedLiteralNode" not in generated_code


def test_dotted_type_annotation_codegen_from_modular_tiled_matmul_example():
    code = """
    def tiled_matmul_kernel(matrix_c: TileTensor):
        var accumulator: matrix_c.ElementType = 0.0
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert "matrix_c.ElementType accumulator = 0.0;" in generated_code


def test_mlir_backtick_type_codegen_from_modular_gpu_globals_reparses_crossgl():
    # Reduced from https://github.com/modular/modular.git commit
    # daa47bb846cc213723a54c51844ea4e923eb5e13,
    # mojo/stdlib/std/gpu/globals.mojo _resolve_max_threads_per_block_metadata.
    code = """
    def _resolve_max_threads_per_block_metadata() -> __mlir_type.`!kgen.string`:
        return "nvvm.maxntid".value
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert "String _resolve_max_threads_per_block_metadata()" in generated_code
    assert 'return "nvvm.maxntid".value;' in generated_code
    assert "__mlir_type" not in generated_code
    assert "`" not in generated_code
    parse_crossgl(generated_code)


def test_adjacent_string_literals_in_call_codegen_from_modular_tiled_matmul_example():
    code = """
    def main():
        print(
            "Note: Expected formula is C[i,j] ="
            " (i+1) * 64 * (j+1)"
        )
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert (
        'print("Note: Expected formula is C[i,j] = (i+1) * 64 * (j+1)");'
        in generated_code
    )


def test_identifier_tuple_declaration_and_assignment_codegen_from_layout_tensor_docs():
    code = """
    def main():
        var row, col = 0, 1
        row, col = 0, 0
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert "var (row, col) = (0, 1);" in generated_code
    assert "(row, col) = (0, 0);" in generated_code


def test_multiline_parenthesized_boolean_condition_codegen_from_layout_tensor_docs():
    code = """
    def kernel(tensor: LayoutTensor):
        if (
            global_idx.y < tensor.shape[0]()
            and global_idx.x < tensor.shape[1]()
        ):
            tensor[global_idx.y, global_idx.x] = tensor[global_idx.y, global_idx.x] + 1
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert (
        "if (((global_idx.y < tensor.shape[0]()) && "
        "(global_idx.x < tensor.shape[1]())))"
    ) in generated_code


def test_matrix_multiplication_operator_codegen_from_official_docs_reparses_crossgl():
    # Reduced from https://docs.modular.com/mojo/manual/operators/
    # "Matrix multiplication" documents @ as Mojo's matrix multiplication operator.
    code = """
    fn main(lhs: Matrix[DType.float32, 4, 4], rhs: Matrix[DType.float32, 4, 4]):
        var result = lhs @ rhs
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert "var result = (lhs * rhs);" in generated_code
    assert " @ rhs" not in generated_code
    parse_crossgl(generated_code)


def test_walrus_assignment_expression_codegen_from_official_docs():
    # Reduced from https://docs.modular.com/mojo/manual/operators/
    # "Walrus operator" assignment-expression example.
    code = """
    def main():
        var name = ""
        while (name := input("Name or 'quit': ")) != "quit":
            print("Hello,", name)
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert 'while (((name = input("Name or \'quit\': ")) != "quit"))' in (
        generated_code
    )
    assert 'print("Hello,", name);' in generated_code
    assert ":=" not in generated_code
    assert "Unhandled expression" not in generated_code
    parse_crossgl(generated_code)


def test_modular_image_pipeline_blur_chained_bounds_codegen():
    # Reduced from modularml/mojo commit
    # 7aa053560034c8c5b4f9acb0a5b450e79d2f7c18,
    # max/examples/custom_ops/kernels/image_pipeline.mojo blur kernel bounds check.
    code = """
    def blur_kernel():
        if 0 <= cur_row < height and 0 <= cur_col < width:
            pix_val_accum += Int(img_in[cur_row, cur_col])
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert (
        "if ((((0 <= cur_row) && (cur_row < height)) && "
        "((0 <= cur_col) && (cur_col < width))))"
    ) in generated_code
    assert "((0 <= cur_row) < height)" not in generated_code
    assert "((0 <= cur_col) < width)" not in generated_code
    assert "pix_val_accum += int(img_in[cur_row, cur_col]);" in generated_code


def test_empty_index_access_codegen_from_layout_tensor_iterator_docs():
    code = """
    def main():
        var tile = iter[]
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert "var tile = iter[];" in generated_code


def test_slice_index_access_codegen_from_modular_stdlib_slice_tests():
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
    generated_code = generate_code(ast)

    assert 'var new_slice = sliceable[1:"hello":4.0];' in generated_code
    assert "var reverse = s[2::(-1)];" in generated_code
    assert "var open_slice = s[::];" in generated_code


def test_try_except_codegen_from_layout_tensor_gpu_docs():
    code = """
    def main():
        try:
            run_kernel()
        except error:
            print(error)
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert "try {" in generated_code
    assert "run_kernel();" in generated_code
    assert "catch (error)" in generated_code
    assert "print(error);" in generated_code


def test_try_except_else_finally_codegen_from_official_error_docs():
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
    generated_code = generate_code(ast)

    assert "try {" in generated_code
    assert "run_kernel();" in generated_code
    assert "catch (error)" in generated_code
    assert "recover(error);" in generated_code
    assert "} else {" in generated_code
    assert "mark_success();" in generated_code
    assert "} finally {" in generated_code
    assert "cleanup();" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_modular_trait_codegen_omits_abstract_contract_methods():
    code = """
    trait PipelineSchedule:
        def config(self) -> PipelineConfig:
            ...

    fn helper() -> Int:
        return 1
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert "int helper()" in generated_code
    assert "return 1;" in generated_code
    assert "PipelineSchedule" not in generated_code
    assert "config(" not in generated_code
    assert "..." not in generated_code


def test_multiple_with_context_managers_codegen_from_mojo_gpu_puzzles():
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
    generated_code = generate_code(ast)

    assert (
        "{ // with a.map_to_host() as a_host, b.map_to_host() as b_host"
        in generated_code
    )
    assert "a_host[0] = Scalar[dtype](0);" in generated_code
    assert "b_host[0] = Scalar[dtype](1);" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_function_local_imports_codegen_from_layout_tensor_gpu_docs():
    code = """
    def simd_width_example():
        from std.sys.info import simd_width_of
        from std.gpu.host.compile import get_gpu_target
        comptime simd_width = simd_width_of[DType.float32, get_gpu_target()]
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert "// from std.sys.info import simd_width_of" in generated_code
    assert "// from std.gpu.host.compile import get_gpu_target" in generated_code
    assert (
        "let simd_width = simd_width_of[DType.float32, get_gpu_target()];"
        in generated_code
    )


def test_postfix_transfer_marker_codegen_from_life_examples():
    code = """
    def make_grid():
        return Grid(8, 8, glider^)

    def random_grid():
        return grid^
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert "return Grid(8, 8, glider);" in generated_code
    assert "return grid;" in generated_code
    assert "glider^" not in generated_code
    assert "grid^" not in generated_code


def test_gpu_fundamentals_launch_keyword_tuple_args_codegen():
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
    generated_code = generate_code(ast)

    assert (
        "// from std.gpu import block_dim, block_idx, global_idx, thread_idx"
        in generated_code
    )
    assert 'sep = "\\t"' in generated_code
    assert "if ((!has_accelerator()))" in generated_code
    assert (
        "ctx.enqueue_function[print_threads, print_threads]"
        "(grid_dim = (2, 2, 1), block_dim = (4, 4, 2));"
    ) in generated_code
    assert "ctx.synchronize();" in generated_code
    assert "comptime" not in generated_code


def test_backtick_comptime_names_codegen_from_official_gpu_notebook_example():
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
    generated_code = generate_code(ast)

    assert "int u2705 = 1;" in generated_code
    assert "int u274c = 0;" in generated_code
    assert "value[0] = u2705;" in generated_code
    assert "out.enqueue_fill(u274c);" in generated_code
    assert "metadata" not in generated_code
    assert "`" not in generated_code
    assert "Unhandled expression" not in generated_code


def test_not_in_membership_condition_codegen_from_mojo_gpu_puzzles_dispatch():
    # Reduced from https://github.com/modular/mojo-gpu-puzzles.git commit
    # 87de51ac93bea662eba6f09d19e8744e56161027,
    # problems/p13/p13.mojo main command-dispatch guard.
    code = """
    def main():
        if len(argv()) != 2 or argv()[1] not in [
            "--simple",
            "--block-boundary",
        ]:
            raise Error("Expected mode")
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert (
        'if (((len(argv()) != 2) || (!(argv()[1] in ["--simple", "--block-boundary"]))))'
        in generated_code
    )
    assert " not in " not in generated_code
    assert "Unhandled expression" not in generated_code


def test_mojo_gpu_intro_vector_addition_host_buffer_codegen():
    # Reduced from modular/modular commit
    # 7aa053560034c8c5b4f9acb0a5b450e79d2f7c18,
    # mojo/examples/gpu-intro/vector_addition.mojo host setup and kernel launch.
    code = """
    from std.math import ceildiv
    from std.gpu.host import DeviceContext

    comptime float_dtype = DType.float32
    comptime vector_size = 1000
    comptime block_size = 256
    comptime num_blocks = ceildiv(vector_size, block_size)

    def main() raises:
        ctx = DeviceContext()
        lhs_host_buffer = ctx.enqueue_create_host_buffer[float_dtype](vector_size)
        lhs_device_buffer = ctx.enqueue_create_buffer[float_dtype](vector_size)
        ctx.enqueue_copy(dst_buf=lhs_device_buffer, src_buf=lhs_host_buffer)
        ctx.enqueue_function[vector_addition](
            lhs_tensor,
            rhs_tensor,
            result_tensor,
            grid_dim=num_blocks,
            block_dim=block_size,
        )
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert "// from std.math import ceildiv" in generated_code
    assert "let float_dtype = DType.float32;" in generated_code
    assert (
        "lhs_host_buffer = " "ctx.enqueue_create_host_buffer[float_dtype](vector_size);"
    ) in generated_code
    assert (
        "ctx.enqueue_copy(dst_buf = lhs_device_buffer, " "src_buf = lhs_host_buffer);"
    ) in generated_code
    assert (
        "ctx.enqueue_function[vector_addition]"
        "(lhs_tensor, rhs_tensor, result_tensor, grid_dim = num_blocks, "
        "block_dim = block_size);"
    ) in generated_code


def test_modular_histogram_nested_gpu_kernel_metadata_codegen():
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
    generated_code = generate_code(ast)

    assert "// from std.gpu import MAX_THREADS_PER_BLOCK_METADATA" in generated_code
    assert "let bin_width = (int(UInt8.MAX) + 1);" in generated_code
    assert "void execute(DeviceContext ctx) @ staticmethod" in generated_code
    assert (
        "void kernel(UnsafePointer[Int64, MutAnyOrigin] output, "
        "UnsafePointer[UInt8, MutAnyOrigin] input, int n) "
        "@ __llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA = "
        "StaticTuple[Int32, 1](Int32(block_dim)))"
    ) in generated_code
    assert (
        "stack_allocation[bin_width, Int64, "
        "address_space=AddressSpace.SHARED]()" in generated_code
    )
    assert "shared_mem[thread_idx.x] = 0;" in generated_code
    assert "barrier();" in generated_code
    assert (
        "Atomic.fetch_add((output + thread_idx.x), shared_mem[thread_idx.x]);"
        in generated_code
    )
    assert (
        "ctx.enqueue_function[kernel](output, input, n, block_dim = block_dim, "
        "grid_dim = grid_dim);"
    ) in generated_code
    assert "Unhandled statement type: FunctionNode" not in generated_code


def test_modular_vector_addition_nested_gpu_launch_codegen():
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
    generated_code = generate_code(ast)

    assert "// from std.gpu import block_dim, block_idx, thread_idx" in generated_code
    assert (
        "void _vector_addition_gpu(ManagedTensorSlice[mut=true, ...] output, "
        "ManagedTensorSlice[dtype=output.dtype, rank=output.rank, ...] lhs, "
        "ManagedTensorSlice[dtype=output.dtype, rank=output.rank, ...] rhs, "
        "DeviceContext ctx)"
    ) in generated_code
    assert "let BLOCK_SIZE = 16;" in generated_code
    assert "var vector_length = output.dim_size(0);" in generated_code
    assert "void vector_addition_gpu_kernel(int length) @ parameter" in generated_code
    assert "var tid = ((block_dim.x * block_idx.x) + thread_idx.x);" in generated_code
    assert "var idx = IndexList[output.rank](tid);" in generated_code
    assert "output.store[1](idx, result);" in generated_code
    assert (
        "gpu_ctx.enqueue_function[vector_addition_gpu_kernel]"
        "(vector_length, grid_dim = num_blocks, block_dim = BLOCK_SIZE);"
    ) in generated_code
    assert "Unhandled statement type: FunctionNode" not in generated_code


def test_modular_mandelbrot_nested_parameter_kernel_codegen():
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
    generated_code = generate_code(ast)

    assert "// from std.gpu.host import DeviceContext" in generated_code
    assert "void execute(OutputTensor output, DeviceContext ctx) @ staticmethod" in (
        generated_code
    )
    assert (
        "SIMD[output.dtype, width] elementwise_mandelbrot"
        "(IndexList[output.rank] idx) @ parameter @ always_inline"
    ) in generated_code
    assert "var in_set_mask = SIMD[DType.bool, width](fill = true);" in generated_code
    assert "for (int _ = 0; _ < max_iterations; _++)" in generated_code
    assert "if ((!any(in_set_mask)))" in generated_code
    assert "break;" in generated_code
    assert "iters = in_set_mask.select((iters + 1), iters);" in generated_code
    assert "foreach[elementwise_mandelbrot, target=target](output, ctx);" in (
        generated_code
    )
    assert "Unhandled statement type: FunctionNode" not in generated_code


def test_keyword_style_runtime_assert_codegen_from_modular_packing_kernel():
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
    generated_code = generate_code(ast)

    assert (
        "assert(((pack_tile_dim[1] % Self.column_inner_size) == 0), "
        '"Unimplemented tile pattern.");'
    ) in generated_code
    assert 'assert(false, "unreachable");' in generated_code
    assert "comptime" not in generated_code


def test_mojo_gpu_puzzles_async_shared_memory_copy_call_codegen():
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
    generated_code = generate_code(ast)

    assert (
        "copy_dram_to_sram_async[thread_layout=load_a_layout, "
        "num_threads=MATMUL_NUM_THREADS, block_dim_count=MATMUL_BLOCK_DIM_COUNT]"
        "(a_shared, a_tile);"
    ) in generated_code
    assert "async_copy_wait_all();" in generated_code
    assert "barrier();" in generated_code
    assert "Unhandled statement type: VectorConstructorNode" not in generated_code


def test_floor_divide_assignment_codegen_from_mojo_gpu_puzzles_reparses_crossgl():
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
    generated_code = generate_code(ast)

    assert "var stride = (TPB / 2);" in generated_code
    assert "stride /= 2;" in generated_code
    assert "//=" not in generated_code
    assert "TPB // 2" not in generated_code
    parse_crossgl(generated_code)


def test_modular_top_k_type_of_parameter_type_codegen():
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
    generated_code = generate_code(ast)

    assert (
        "void top_k_gpu(type_of(out_vals_tensor) out_vals, "
        "type_of(out_idxs_tensor) out_idxs, type_of(in_vals_tensor) in_vals) "
        "@ parameter"
    ) in generated_code


def test_type_member_expression_codegen_from_modular_testing_examples():
    code = """
    def inc(n: Int) raises -> Int:
        if n == Int.MAX:
            raise Error("inc overflow")
        return inc(Int.MAX)
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert "if ((n == Int.MAX))" in generated_code
    assert 'raise(Error("inc overflow"));' in generated_code
    assert "return inc(Int.MAX);" in generated_code


def test_brace_struct_codegen_preserves_generic_members_and_attributes():
    code = """
    struct Resources {
        @position
        var transform: Matrix[DType.float32, 4, 4]
        SIMD[DType.float32, 4] tint @color;
        samples: InlineArray[SIMD[DType.float32, 4], 2];
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "struct Resources" in generated_code
    assert "mat4 transform @ Position;" in generated_code
    assert "vec4 tint @ Color;" in generated_code
    assert "InlineArray[SIMD[DType.float32, 4], 2] samples;" in generated_code


def test_if_codegen():
    code = """
    struct VSInput:
        var position: float4
        var color: float4

    struct VSOutput:
        var out_position: float4

    fn vertex_main(input: VSInput) -> VSOutput:
        var output: VSOutput
        output.out_position = input.position
        if input.color.x > 0.5:
            output.out_position = input.color
        return output

    struct PSInput:
        var in_position: float4

    struct PSOutput:
        var out_color: float4

    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        output.out_color = input.in_position
        if input.in_position.x > 0.5:
            output.out_color = float4(1.0, 1.0, 1.0, 1.0)
        return output
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "if " in generated_code
    except SyntaxError:
        pytest.fail("If statement parsing or code generation not implemented.")


def test_bare_identifier_block_condition_codegen():
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
        let result = flag ? yes : no
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "if (ready)" in generated_code
    assert "while (running)" in generated_code
    assert "switch (state)" in generated_code
    assert "case active:" in generated_code
    assert "(flag ? yes : no)" in generated_code


def test_identity_expression_codegen_from_modular_kernels():
    code = """
    fn main():
        if elementwise_lambda_fn is None or fallback is not None:
            pass
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert (
        "if (((elementwise_lambda_fn == None) || (fallback != None)))" in generated_code
    )
    assert " is " not in generated_code


def test_for_codegen():
    code = """
    struct VSInput:
        var position: float4
        var color: float4

    struct VSOutput:
        var out_position: float4

    fn vertex_main(input: VSInput) -> VSOutput:
        var output: VSOutput
        output.out_position = input.position
        for var i: Int = 0; i < 10; i = i + 1:
            output.out_position = input.color
        return output

    struct PSInput:
        var in_position: float4

    struct PSOutput:
        var out_color: float4

    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        output.out_color = input.in_position
        for var i: Int = 0; i < 10; i = i + 1:
            output.out_color = float4(1.0, 1.0, 1.0, 1.0)
        return output
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "for " in generated_code or "while " in generated_code
    except SyntaxError:
        pytest.fail("For loop parsing or code generation not implemented.")


def test_c_style_for_codegen_preserves_initializer():
    code = """
    fn main():
        for var i: Int = 0; i < 4; i = i + 1:
            sink(i)
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for (int i = 0; (i < 4); i = (i + 1))" in generated_code


def test_c_style_for_array_assignment_update_codegen():
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
    generated_code = generate_code(ast)

    assert "for (int i = 0; (i < 4); items[i] += value)" in generated_code
    assert "for (int i = 0; (i < 4); object.field = value)" in generated_code


def test_for_in_iterable_codegen_preserves_iterable_loop():
    code = """
    fn main():
        for value in values:
            sink(value)
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for value in values {" in generated_code


def test_tuple_for_target_codegen_from_modular_pmpp_examples():
    code = """
    fn initialize_image(width: Int, height: Int):
        for row, col in product(range(height), range(width)):
            sink(row, col)
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for (row, col) in product(range(height), range(width)) {" in generated_code
    assert "sink(row, col);" in generated_code


def test_for_in_range_codegen_lowers_range_call():
    code = """
    fn main():
        for i in range(4):
            sink(i)
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for (int i = 0; i < 4; i++)" in generated_code
    assert "range(4)" not in generated_code


def test_for_in_descending_range_codegen_uses_greater_than_condition():
    code = """
    fn main():
        for i in range(4, 0, -1):
            sink(i)
        for j in range(4, 0, step):
            sink(j)
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for (int i = 4; i > 0; i += (-1))" in generated_code
    assert (
        "for (int j = 4; ((step > 0) ? (j < 0) : (j > 0)); j += step)" in generated_code
    )


def test_comptime_for_codegen_preserves_loop_output():
    code = """
    fn main():
        comptime for value in values:
            sink(value)
        comptime for var i: Int = 0; i < 4; i = i + 1:
            sink(i)
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for value in values {" in generated_code
    assert "for (int i = 0; (i < 4); i = (i + 1))" in generated_code


def test_for_in_dynamic_step_range_codegen_uses_step_sign_condition():
    code = """
    fn main(step: Int):
        for j in range(4, 0, step):
            sink(j)
        for k in range(0, 4, step):
            sink(k)
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "for (int j = 4; ((step > 0) ? (j < 0) : (j > 0)); j += step)" in generated_code
    )
    assert (
        "for (int k = 0; ((step > 0) ? (k < 4) : (k > 4)); k += step)" in generated_code
    )


def test_while_codegen():
    code = """
    struct VSInput:
        var position: float4
        var color: float4

    struct VSOutput:
        var out_position: float4

    fn vertex_main(input: VSInput) -> VSOutput:
        var output: VSOutput
        output.out_position = input.position
        var i: Int = 0
        while i < 10:
            output.out_position = input.color
            i = i + 1
        return output

    struct PSInput:
        var in_position: float4

    struct PSOutput:
        var out_color: float4

    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        output.out_color = input.in_position
        var i: Int = 0
        while i < 10:
            output.out_color = float4(1.0, 1.0, 1.0, 1.0)
            i = i + 1
        return output
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "while " in generated_code
    except SyntaxError:
        pytest.fail("While loop parsing or code generation not implemented.")


def test_else_codegen():
    code = """
    struct VSInput:
        var position: float4
        var color: float4

    struct VSOutput:
        var out_position: float4

    fn vertex_main(input: VSInput) -> VSOutput:
        var output: VSOutput
        output.out_position = input.position
        if input.color.x > 0.5:
            output.out_position = input.color
        else:
            output.out_position = float4(0.0, 0.0, 0.0, 1.0)
        return output

    struct PSInput:
        var in_position: float4

    struct PSOutput:
        var out_color: float4

    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        output.out_color = input.in_position
        if input.in_position.x > 0.5:
            output.out_color = float4(1.0, 1.0, 1.0, 1.0)
        else:
            output.out_color = float4(0.0, 0.0, 0.0, 1.0)
        return output
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "} else {" in generated_code or "else" in generated_code
    except SyntaxError:
        pytest.fail("Else statement parsing or code generation not implemented.")


def test_function_call_codegen():
    code = """
    struct VSInput:
        var position: float4
        var color: float4

    struct VSOutput:
        var out_position: float4

    fn add(a: float4, b: float4) -> float4:
        return a + b

    fn vertex_main(input: VSInput) -> VSOutput:
        var output: VSOutput
        output.out_position = input.position
        let result = add(input.position, input.color)
        return output

    struct PSInput:
        var in_position: float4

    struct PSOutput:
        var out_color: float4

    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        output.out_color = input.in_position
        let result = add(input.in_position, float4(1.0, 1.0, 1.0, 1.0))
        return output
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "add(" in generated_code
    except SyntaxError:
        pytest.fail("Function call parsing or code generation not implemented.")


def test_multiline_function_call_codegen():
    code = """
    fn main():
        let result = add(
            1.0,
            2.0
        )
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let result = add(1.0, 2.0);" in generated_code


def test_implicit_assignment_declaration_codegen_from_current_variables_docs():
    # Reduced from https://mojolang.org/docs/manual/variables/
    code = """
    def main():
        name = "Sam"
        user_id = 0
        user_id = user_id + 1
        _ = sink(user_id)
    """
    ast = parse_code(tokenize_code(code))
    generated_code = generate_code(ast)

    assert 'var name = "Sam";' in generated_code
    assert "var user_id = 0;" in generated_code
    assert "user_id = (user_id + 1);" in generated_code
    assert "_ = sink(user_id);" in generated_code
    assert "var user_id = (user_id + 1);" not in generated_code
    assert "var _ =" not in generated_code


def test_multiline_expression_layout_codegen_from_modular_examples():
    code = """
    fn main():
        var m_1 = (
            LayoutTensor[Q_tile.dtype, Layout(BN, 1), MutAnyOrigin]
            .stack_allocation()
            .fill(Scalar[Q_tile.dtype].MIN)
        )
        print(
            (
                "Launching shared memory sum reduction kernel (Fig 10.10) with 1"
                " block and"
            ),
            BLOCK_DIM,
            "threads",
        )
        print(
            "Coarsening with contiguous partitioning (COARSE_FACTOR="
            + String(COARSE_FACTOR)
            + ")"
        )
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "LayoutTensor[Q_tile.dtype, Layout(BN, 1), "
        "MutAnyOrigin].stack_allocation().fill(Scalar[Q_tile.dtype].MIN)"
    ) in generated_code
    assert (
        'print("Launching shared memory sum reduction kernel (Fig 10.10) '
        'with 1 block and", BLOCK_DIM, "threads");'
    ) in generated_code
    assert (
        'print((("Coarsening with contiguous partitioning (COARSE_FACTOR=" '
        '+ String(COARSE_FACTOR)) + ")"));'
    ) in generated_code


def test_generic_function_signature_codegen():
    code = """
    fn build(
        value: SIMD[DType.float32, 4],
        Matrix[DType.float32, 4, 4] transform @binding(0)
    ) -> Matrix[DType.float32, 4, 4]:
        return transform
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "mat4 build(vec4 value, mat4 transform @ binding(0))" in generated_code
    assert "return transform;" in generated_code


def test_where_clause_constraints_are_accepted_and_not_emitted_codegen():
    code = """
    fn constrained[BM: Int, BN: Int]() -> Int where (
        BM == 32
        and BN == 32
    ):
        return 1

    def pow2[n: Int]() -> Int where n >= 0:
        return 1
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "int constrained()" in generated_code
    assert "int pow2()" in generated_code
    assert "return 1;" in generated_code
    assert "where" not in generated_code


def test_method_call_chain_codegen_preserves_receiver():
    code = """
    fn main():
        let channel = texture.sample(coord).x
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let channel = texture.sample(coord).x;" in generated_code


def test_as_cast_expression_codegen():
    code = """
    fn main():
        let i: Int = 2
        let x: Float = i as Float
        let y = (i + 1) as Float
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "int i = 2;" in generated_code
    assert "float x = float(i);" in generated_code
    assert "let y = float((i + 1));" in generated_code
    assert " as " not in generated_code
    assert "Unhandled expression" not in generated_code


def test_parenthesized_method_call_chain_codegen():
    code = """
    fn main():
        let channel = (texture.sample(coord)).x
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let channel = texture.sample(coord).x;" in generated_code


def test_def_function_codegen_preserves_parameter_name():
    code = """
    def helper(x: Float32) -> Float32:
        return x
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "float helper(float x)" in generated_code
        assert "return x;" in generated_code
    except SyntaxError:
        pytest.fail("Def function parsing or code generation not implemented.")


def test_else_if_codegen():
    code = """
    struct VSInput:
        var position: float4
        var color: float4

    struct VSOutput:
        var out_position: float4

    fn vertex_main(input: VSInput) -> VSOutput:
        var output: VSOutput
        output.out_position = input.position
        if input.color.x > 0.5:
            output.out_position = input.color
        else:
            output.out_position = float4(0.0, 0.0, 0.0, 1.0)
        return output

    struct PSInput:
        var in_position: float4

    struct PSOutput:
        var out_color: float4

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
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "} else if (" in generated_code or "else if" in generated_code
    except SyntaxError:
        pytest.fail("Else-if statement parsing or code generation not implemented.")


def test_assignment_ops_codegen():
    code = """
    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        output.out_color = float4(0.0, 0.0, 0.0, 1.0)

        if input.in_position.x > 0.5:
            output.out_color += input.in_position

        if input.in_position.x < 0.5:
            output.out_color -= float4(0.1, 0.1, 0.1, 0.1)

        if input.in_position.y > 0.5:
            output.out_color *= 2.0

        if input.in_position.z > 0.5:
            output.out_color /= 2.0

        return output
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "+=" in generated_code or "-=" in generated_code
    except SyntaxError:
        pytest.fail("Assignment ops parsing or code generation not implemented.")


def test_assignment_expression_operands_are_parenthesized_codegen():
    code = """
    fn main():
        var value: Int = (a = b) + c
        var other: Int = c + (a = b)
        var selected: Int = flag ? (a = b) : c
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "int value = ((a = b) + c);" in generated_code
    assert "int other = (c + (a = b));" in generated_code
    assert "int selected = (flag ? (a = b) : c);" in generated_code
    assert "int value = (a = b + c);" not in generated_code
    assert "int other = (c + a = b);" not in generated_code


def test_bitwise_ops_codegen():
    code = """
    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        output.out_color = float4(0.0, 0.0, 0.0, 1.0)
        let val: UInt32 = 0x01
        if val | 0x02:
            # Test case for bitwise OR
            pass
        let filterA: UInt32 = 0b0001  # First filter
        let filterB: UInt32 = 0b1000  # Second filter

        # Merge both filters
        let combinedFilter = filterA | filterB  # combinedFilter becomes 0b1001
        return output
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "|" in generated_code
    except SyntaxError:
        pytest.fail("Bitwise ops parsing or code generation not implemented.")


def test_modulo_codegen():
    code = """
    fn main():
        let a: Int = 10 % 3
        let b: Float32 = 10.0 % 3.0
        let wrapped = fmod(5.0, 2.0)
        let wrapped_math = math.fmod(5.0, 2.0)
        var c: Int = 10
        c %= 3
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)

        assert "int a = (10 % 3);" in generated_code
        assert "float b = (10.0 % 3.0);" in generated_code
        assert "let wrapped = mod(5.0, 2.0);" in generated_code
        assert "let wrapped_math = mod(5.0, 2.0);" in generated_code
        assert "int c = 10;" in generated_code
        assert "c %= 3;" in generated_code
        assert "fmod(" not in generated_code
        assert "math.fmod(" not in generated_code
    except SyntaxError:
        pytest.fail("Modulo parsing or code generation not implemented.")


def test_mojo_generated_builtin_names_lower_to_crossgl():
    code = """
    fn main():
        let x: Float32 = 4.0
        let a = lerp(0.0, 1.0, 0.25)
        let b = math.lerp(0.0, 1.0, 0.75)
        let c = power(2.0, 3.0)
        let d = rsqrt(x)
        let e = math.rsqrt(x)
        let f = dot_product(lhs, rhs)
        let g = cross_product(lhs, rhs)
        let h = magnitude(lhs)
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let a = mix(0.0, 1.0, 0.25);" in generated_code
    assert "let b = mix(0.0, 1.0, 0.75);" in generated_code
    assert "let c = pow(2.0, 3.0);" in generated_code
    assert "let d = inversesqrt(x);" in generated_code
    assert "let e = inversesqrt(x);" in generated_code
    assert "let f = dot(lhs, rhs);" in generated_code
    assert "let g = cross(lhs, rhs);" in generated_code
    assert "let h = length(lhs);" in generated_code
    assert "lerp(" not in generated_code
    assert "math.lerp(" not in generated_code
    assert "power(" not in generated_code
    assert "rsqrt(" not in generated_code
    assert "math.rsqrt(" not in generated_code
    assert "dot_product(" not in generated_code
    assert "cross_product(" not in generated_code
    assert "magnitude(" not in generated_code


def test_mojo_import_alias_math_and_simd_builtins_lower_to_crossgl():
    code = """
    import math as m
    import simd as s

    fn main():
        let x: Float32 = 4.0
        let bounded = m.clamp(m.sqrt(x), m.min(low, high), m.max(low, high))
        let magnitude = m.abs(x)
        let projected = s.dot(lhs, rhs)
        let perpendicular = s.cross(lhs, rhs)
        let normal = s.normalize(lhs)
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let bounded = clamp(sqrt(x), min(low, high), max(low, high));" in (
        generated_code
    )
    assert "let magnitude = abs(x);" in generated_code
    assert "let projected = dot(lhs, rhs);" in generated_code
    assert "let perpendicular = cross(lhs, rhs);" in generated_code
    assert "let normal = normalize(lhs);" in generated_code
    assert "m.clamp(" not in generated_code
    assert "m.sqrt(" not in generated_code
    assert "s.dot(" not in generated_code
    assert "s.cross(" not in generated_code
    assert "s.normalize(" not in generated_code

    parse_crossgl(generated_code)


def test_mojo_from_import_alias_math_and_simd_builtins_lower_to_crossgl():
    code = """
    from math import fmod as wrap
    from simd import dot as dot2, normalize as normalized

    fn main():
        let wrapped = wrap(5.0, 2.0)
        let projected = dot2(lhs, rhs)
        let normal = normalized(lhs)
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let wrapped = mod(5.0, 2.0);" in generated_code
    assert "let projected = dot(lhs, rhs);" in generated_code
    assert "let normal = normalize(lhs);" in generated_code
    assert "wrap(" not in generated_code
    assert "dot2(" not in generated_code
    assert "normalized(" not in generated_code

    parse_crossgl(generated_code)


def test_user_defined_from_import_alias_call_does_not_lower_to_builtin():
    code = """
    from simd import dot as project

    fn project(lhs: Float32, rhs: Float32) -> Float32:
        return lhs + rhs

    fn main():
        let result = project(lhs, rhs)
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float project(float lhs, float rhs)" in generated_code
    assert "let result = project(lhs, rhs);" in generated_code
    assert "let result = dot(lhs, rhs);" not in generated_code

    parse_crossgl(generated_code)


def test_user_defined_lerp_call_does_not_lower_to_mix():
    code = """
    fn lerp(x: Float32) -> Float32:
        return x

    fn main():
        let y = lerp(1.0)
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float lerp(float x)" in generated_code
    assert "let y = lerp(1.0);" in generated_code
    assert "let y = mix(1.0);" not in generated_code


def test_comptime_assert_statement_codegen():
    code = """
    def outer_product_acc(res: TileTensor, size: Int):
        comptime assert(type_of(res).flat_rank == 2)
        comptime assert(size > 0, "bad size")
        comptime assert (
            dtype.is_floating_point()
        ), "dtype must be a floating-point type"
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "assert((type_of(res).flat_rank == 2));" in generated_code
    assert 'assert((size > 0), "bad size");' in generated_code
    assert (
        'assert(dtype.is_floating_point(), "dtype must be a floating-point type");'
        in generated_code
    )


def test_alias_declaration_codegen_matches_comptime_declarations():
    code = """
    alias THREADS_PER_BLOCK = 256

    def kernel():
        alias LOCAL_BLOCK = THREADS_PER_BLOCK
        for i in range(LOCAL_BLOCK):
            sink(i)
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let THREADS_PER_BLOCK = 256;" in generated_code
    assert "let LOCAL_BLOCK = THREADS_PER_BLOCK;" in generated_code
    assert "for (int i = 0; i < LOCAL_BLOCK; i++)" in generated_code


def test_struct_comptime_member_codegen_skips_metadata_field():
    code = """
    @fieldwise_init
    struct Tensor[dtype: DType, rank: Int](ImplicitlyCopyable):
        comptime size = product(Self.static_spec.shape_tuple)
        var buffer: DeviceBuffer[Self.dtype]
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "struct Tensor" in generated_code
    assert "DeviceBuffer[Self.dtype] buffer;" in generated_code
    assert "void size;" not in generated_code
    assert "let size" not in generated_code


def test_user_defined_lerp_matching_arity_does_not_lower_to_mix():
    code = """
    fn lerp(a: Float32, b: Float32, t: Float32) -> Float32:
        return a

    fn main():
        let y = lerp(0.0, 1.0, 0.25)
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float lerp(float a, float b, float t)" in generated_code
    assert "let y = lerp(0.0, 1.0, 0.25);" in generated_code
    assert "let y = mix(0.0, 1.0, 0.25);" not in generated_code


def test_class_method_lerp_does_not_shadow_builtin_lerp_arity():
    code = """
    class Mixer:
        fn lerp(x: Float32) -> Float32:
            return x

    fn main():
        let y = lerp(0.0, 1.0, 0.25)
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float lerp(float x)" in generated_code
    assert "let y = mix(0.0, 1.0, 0.25);" in generated_code
    assert "let y = lerp(0.0, 1.0, 0.25);" not in generated_code


def test_mojo_method_call_on_math_parameter_does_not_lower_to_builtin():
    code = """
    fn main(math: Mixer):
        let blended = math.lerp(0.0, 1.0, 0.25)
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "void main(Mixer math)" in generated_code
    assert "let blended = math.lerp(0.0, 1.0, 0.25);" in generated_code
    assert "let blended = mix(0.0, 1.0, 0.25);" not in generated_code


def test_mojo_method_call_on_global_math_value_does_not_lower_to_builtin():
    code = """
    var math: Mixer

    fn main():
        let blended = math.lerp(0.0, 1.0, 0.25)
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "Mixer math;" in generated_code
    assert "let blended = math.lerp(0.0, 1.0, 0.25);" in generated_code
    assert "let blended = mix(0.0, 1.0, 0.25);" not in generated_code


def test_logical_ops_codegen():
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
        generated_code = generate_code(ast)
        print(generated_code)
        assert "bool val_0 = true;" in generated_code
        assert "let val_1 = (val_0 && false);" in generated_code
        assert "let val_2 = (val_0 || false);" in generated_code
        assert "let val_3 = (!val_0);" in generated_code
        assert "\nand;" not in generated_code
        assert "\nor;" not in generated_code
    except SyntaxError:
        pytest.fail("Logical ops parsing or code generation not implemented.")


def test_switch_case_codegen():
    code = """
    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        switch input.in_position.x:
            case 0.0:
                output.out_color = float4(1.0, 0.0, 0.0, 1.0)
                break
            case 0.5:
                output.out_color = float4(0.0, 1.0, 0.0, 1.0)
                break
            default:
                output.out_color = float4(0.0, 0.0, 1.0, 1.0)
                break
        return output
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "switch " in generated_code
        assert "case " in generated_code
    except SyntaxError:
        pytest.fail("Switch case parsing or code generation not implemented.")


def test_pass_statement_codegen_is_noop():
    code = """
    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        if input.value > 0:
            pass
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "if " in generated_code
        assert "pass;" not in generated_code
        assert "Unhandled statement type: PassNode" not in generated_code
    except SyntaxError:
        pytest.fail("Pass statement code generation not implemented.")


def test_if_dedent_return_codegen():
    code = """
    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        if input.value > 0:
            pass
        return output
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert (
            "if ((input.value > 0)) {\n            }\n            return output;"
            in generated_code
        )
    except SyntaxError:
        pytest.fail("If dedent code generation not implemented.")


def test_break_continue_statement_codegen():
    code = """
    fn main():
        var i: Int = 0
        while i < 10:
            i = i + 1
            if i == 3:
                continue
            if i == 7:
                break
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "continue;" in generated_code
        assert "break;" in generated_code
        assert "continue;\n            }\n            if ((i == 7))" in generated_code
        assert "Unhandled statement type: ContinueNode" not in generated_code
        assert "Unhandled statement type: BreakNode" not in generated_code
    except SyntaxError:
        pytest.fail("Break/continue code generation not implemented.")


def test_switch_explicit_break_not_duplicated_codegen():
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
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert generated_code.count("break;") == 2
        assert "Unhandled statement type: BreakNode" not in generated_code
    except SyntaxError:
        pytest.fail("Switch explicit break code generation not implemented.")


def test_switch_fallthrough_codegen_does_not_insert_break():
    code = """
    fn main():
        switch value:
            case 0:
                hit_zero()
            case 1:
                hit_one()
                break
            default:
                hit_default()
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    case_zero_block = generated_code.split("case 0:", 1)[1].split("case 1:", 1)[0]
    default_block = generated_code.split("default:", 1)[1].split("}", 1)[0]

    assert "break;" not in case_zero_block
    assert "break;" not in default_block
    assert generated_code.count("break;") == 1
    assert "Unhandled statement type" not in generated_code


def test_typed_local_declaration_codegen_preserves_type():
    code = """
    fn main():
        var x: Float32
        let y: Int = 1
        var transform: Matrix[DType.float32, 4, 4]
        var values: InlineArray[Float32, 4]
        let inferred = y
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float x;" in generated_code
    assert "int y = 1;" in generated_code
    assert "mat4 transform;" in generated_code
    assert "InlineArray[Float32, 4] values;" in generated_code
    assert "let inferred = y;" in generated_code
    assert "var transform;" not in generated_code


def test_bare_annotated_assignment_codegen_preserves_type_and_initializer():
    code = """
    def kernel(size: Int):
        idx: UInt = global_idx.x
        limit: Int = size
        if idx < UInt(limit):
            process(idx)
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "uint idx = global_idx.x;" in generated_code
    assert "int limit = size;" in generated_code
    assert "var idx" not in generated_code
    assert "let limit" not in generated_code


def test_double_dtype_codegen():
    code = """
    struct VSInput:
        var position: Float64
        var color: Float64

    fn vertex_main(input: VSInput) -> VSInput:
        var output: VSInput
        output.position = input.position * 2.0
        output.color = input.color / 2.0
        return output
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "Float64" in generated_code or "double" in generated_code
    except SyntaxError:
        pytest.fail("Double data type parsing or code generation not implemented.")


def test_vector_constructor_codegen():
    code = """
    fn vertex_main() -> float4:
        let uv = float2(0.5, 0.5)
        let color = float4(uv.x, uv.y, 0.0, 1.0)
        return color
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "float2(" in generated_code or "vec2(" in generated_code
        assert "float4(" in generated_code or "vec4(" in generated_code
    except SyntaxError:
        pytest.fail("Vector constructor parsing or code generation not implemented.")


def test_generic_simd_constructor_codegen():
    code = """
    fn main():
        let color = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let color = vec4(1.0, 2.0, 3.0, 4.0);" in generated_code


def test_float64_simd_constructor_from_modular_docs_reparses_crossgl():
    code = """
    fn reduce_example() -> SIMD[DType.float64, 4]:
        var data = SIMD[DType.float64, 4](10.5, 20.3, 30.1, 40.7)
        return data
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "dvec4 reduce_example()" in generated_code
    assert "var data = dvec4(10.5, 20.3, 30.1, 40.7);" in generated_code
    assert "SIMD[DType.float64, 4]" not in generated_code
    parse_crossgl(generated_code)


def test_generic_inline_array_constructor_codegen():
    code = """
    fn main():
        let values = InlineArray[Float32, 4](1.0, 2.0, 3.0, 4.0)
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let values = InlineArray[Float32, 4](1.0, 2.0, 3.0, 4.0);" in generated_code


def test_generic_matrix_constructor_codegen():
    code = """
    fn main():
        let transform = Matrix[DType.float32, 3, 4](
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 0.0
        )
        let precise = Matrix[DType.float64, 2, 2](1.0, 0.0, 0.0, 1.0)
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let transform = mat3x4(" in generated_code
    assert "let precise = dmat2(1.0, 0.0, 0.0, 1.0);" in generated_code
    assert "Matrix[DType" not in generated_code


def test_callable_array_element_codegen_remains_array_call():
    code = """
    fn main():
        let value = callbacks[i]()
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let value = callbacks[i]();" in generated_code


def test_array_access_codegen():
    code = """
    fn vertex_main() -> float4:
        var values: StaticTuple[Float32, 4]
        values[0] = 1.0
        values[1] = 2.0
        values[2] = 3.0
        values[3] = 4.0
        return float4(values[0], values[1], values[2], values[3])
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "[0]" in generated_code
        assert "[1]" in generated_code
    except SyntaxError:
        pytest.fail("Array access parsing or code generation not implemented.")


def test_array_member_access_chain_codegen():
    code = """
    fn main():
        let channel = values[0].x
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let channel = values[0].x;" in generated_code


def test_gpu_tile_tensor_multi_index_access_codegen():
    code = """
    from std.gpu import thread_idx

    fn tiled_load(tile: TileTensor, matrix: TileTensor):
        tile[thread_idx.y, thread_idx.x] = matrix[
            thread_idx.y,
            thread_idx.x,
        ]
        let value = tile[thread_idx.y, thread_idx.x]
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "tile[thread_idx.y, thread_idx.x] = matrix[thread_idx.y, thread_idx.x];"
        in generated_code
    )
    assert "let value = tile[thread_idx.y, thread_idx.x];" in generated_code


def test_parenthesized_call_indexing_codegen():
    code = """
    fn main():
        let value = (factory())[i]
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let value = factory()[i];" in generated_code


def test_ternary_operator_codegen():
    code = """
    fn vertex_main() -> float4:
        let x: Float32 = 0.5
        let result = x > 0.0 ? 1.0 : 0.0
        let native_result = 1.0 if x > 0.0 else 0.0
        return float4(result, 0.0, 0.0, 1.0)
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "?" in generated_code or "if" in generated_code
        assert "let native_result = ((x > 0.0) ? 1.0 : 0.0);" in generated_code
    except SyntaxError:
        pytest.fail("Ternary operator parsing or code generation not implemented.")


def test_multiline_parenthesized_inline_if_codegen_from_mojo_gpu_puzzles():
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
    generated_code = generate_code(ast)

    assert (
        "var cached_correction = ((local_i < 1024) ? "
        "shared_cache[(local_i + 3072)] : series_correction);"
    ) in generated_code
    assert "Unhandled expression" not in generated_code


def test_import_codegen():
    code = """
    import math
    import simd as s
    from tensor import Tensor

    fn vertex_main() -> float4:
        let result = math.sin(0.5)
        let wrapped = math.fract(1.25)
        let logged = math.log(4.0)
        let raised = math.exp(1.0)
        return float4(result, wrapped, logged, raised)
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "// import math" in generated_code
        assert "// import simd as s" in generated_code
        assert "// from tensor import Tensor" in generated_code
        assert "let result = sin(0.5);" in generated_code
        assert "let wrapped = fract(1.25);" in generated_code
        assert "let logged = log(4.0);" in generated_code
        assert "let raised = exp(1.0);" in generated_code
        assert "math.fract(" not in generated_code
        assert "math.log(" not in generated_code
        assert "math.exp(" not in generated_code
    except SyntaxError:
        pytest.fail("Import parsing or code generation not implemented.")


def test_global_variable_codegen_preserves_typed_globals():
    code = """
    var exposure: Float32
    let sample_count: Int = 4
    var transform: Matrix[DType.float32, 4, 4]
    let scale = 1.0
    @position
    var global_data: Float32
    var bound_texture: Texture2D @binding(0)
    @group(0)
    @binding(1)
    var sampler: SamplerState
    @group(2)
    var combined_texture: Texture2D @binding(3)

    fn main():
        return
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "// Global Variables" in generated_code
    assert "float exposure;" in generated_code
    assert "int sample_count = 4;" in generated_code
    assert "mat4 transform;" in generated_code
    assert "let scale = 1.0;" in generated_code
    assert "float global_data @ Position;" in generated_code
    assert "sampler2D bound_texture @ binding(0);" in generated_code
    assert "SamplerState sampler @ group(0) @ binding(1);" in generated_code
    assert "sampler2D combined_texture @ group(2) @ binding(3);" in generated_code
    assert generated_code.index("float exposure;") < generated_code.index("void main")


def test_explicit_none_return_codegen_maps_to_void():
    code = """
    fn helper() -> None:
        return

    fn main() -> None:
        helper()
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "void helper()" in generated_code
    assert "void main()" in generated_code
    assert "helper();" in generated_code
    assert "None helper" not in generated_code
    assert "None main" not in generated_code


def test_return_none_literal_codegen_from_generated_mojo_example_reparses_crossgl():
    # Reduced from examples/output/mojo/gpu_computing/MatrixMultiplication.mojo,
    # matmul_batched's early exit.
    code = """
    fn matmul_batched(done: Bool) -> None:
        if done:
            return None
        return
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "void matmul_batched(bool done)" in generated_code
    assert "return;\n" in generated_code
    assert "return None;" not in generated_code
    assert "None matmul_batched" not in generated_code
    parse_crossgl(generated_code)


def test_at_attribute_codegen():
    code = """
    struct MyStruct:
        @position
        var data: Int32

    @compute_shader
    fn kernel():
        pass
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "int data @ Position;" in generated_code
        assert "// Compute Shader" in generated_code
        assert "compute {" in generated_code
        assert "void kernel() {" in generated_code
        assert "kernel()@ compute_shader" not in generated_code
    except SyntaxError:
        pytest.fail("Attribute parsing or code generation not implemented.")


def test_bracket_attribute_codegen_uses_shader_stage():
    code = """
    [[compute_shader]]
    fn run():
        pass
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "// Compute Shader" in generated_code
    assert "compute {" in generated_code
    assert "void run() {" in generated_code
    assert "run()@ compute_shader" not in generated_code


def test_class_codegen_uses_struct_shape_and_method_functions():
    code = """
    class Accumulator:
        @position
        var value: Int
        Matrix[DType.float32, 4, 4] transform @binding(0)
        SIMD[DType.float32, 4] tint @color
        fn reset():
            pass
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "struct Accumulator" in generated_code
        assert "int value @ Position;" in generated_code
        assert "mat4 transform @ binding(0);" in generated_code
        assert "vec4 tint @ Color;" in generated_code
        assert "void reset() {" in generated_code
    except SyntaxError:
        pytest.fail("Class parsing or code generation not implemented.")


def test_constant_buffer_codegen_from_colon_body():
    code = """
    @group(0)
    constant Params:
        @binding(0)
        var exposure: Float32 @offset(0)
        count: Int @binding(1)
        Matrix[DType.float32, 4, 4] transform @binding(2)
        SIMD[DType.float32, 4] tint @binding(3)
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "cbuffer Params @ group(0)" in generated_code
        assert "float exposure @ binding(0) @ offset(0);" in generated_code
        assert "int count @ binding(1);" in generated_code
        assert "mat4 transform @ binding(2);" in generated_code
        assert "vec4 tint @ binding(3);" in generated_code
    except SyntaxError:
        pytest.fail("Constant buffer parsing or code generation not implemented.")


if __name__ == "__main__":
    pytest.main()
