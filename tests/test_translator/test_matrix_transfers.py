"""SIMD-group matrix transfers, fills and the loops MLX's chunked kernels use.

MLX's ``gated_delta_fused_chunk`` declares ``simdgroup_float8x8`` tiles, some
in arrays and several per declaration, fills them with
``make_filled_simdgroup_matrix``, moves them with ``simdgroup_load`` and
``simdgroup_store`` (transposed and not), multiplies them with
``simdgroup_multiply``, and wraps the products in two loops OpenGL and
DirectX software subgroups must prove uniform: the constant
``for (int step = 1; (1 << step) < C; step++)`` and the chunk loop
``int t = 0; for (; t + C <= T; t += C)`` whose counter decides the trailing
``if (t < T)``. Software lowering moves each lane's two fragment elements
through the registered ``tile_4x4_row_pair`` coordinates, so transfers need no
cross-lane exchange.
"""

import os
import struct
import sys
from fractions import Fraction

import pytest

from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.project import (
    ProjectConfig,
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    translate_project,
)
from crosstl.translator import parse
from crosstl.translator.ast import (
    AssignmentNode,
    BinaryOpNode,
    IdentifierNode,
    LiteralNode,
    PrimitiveType,
    UnaryOpNode,
)
from crosstl.translator.codegen.array_utils import _UnsignedLiteralInt
from crosstl.translator.codegen.directx_codegen import (
    DirectXCooperativeMatrixUnsupportedError,
    HLSLCodeGen,
)
from crosstl.translator.codegen.GLSL_codegen import (
    GLSLCodeGen,
    OpenGLCooperativeMatrixError,
)
from crosstl.translator.codegen.loop_induction import (
    _checked_static_value,
    init_less_counter_loops,
    offset_counter_loop_terminates,
    static_loop_trip_count,
)
from crosstl.translator.codegen.metal_codegen import (
    MetalCodeGen,
    UnsupportedMetalFeatureError,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from tests.runtime_helpers import _validate
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile

REQUIRE_ENV = "CROSTL_REQUIRE_MATRIX_TRANSFER_RUNTIME"
TARGETS = ("opengl", "directx")
ERRORS = {
    "opengl": "project.translate.opengl-software-subgroup-invalid",
    "directx": "project.translate.directx-software-subgroup-invalid",
}
MATRIX = (
    "CooperativeMatrix<float, 8, 8, subgroup, unspecified, unspecified, "
    "metal_thread_elements, 32, 2, metal_thread_elements_reference_view, "
    "tile_4x4_row_pair, unit_lane_coordinates>"
)
INT_MAX = (1 << 31) - 1
GUARD = -4096.0


def _int(value):
    return LiteralNode(value, PrimitiveType("int"))


def _name(name):
    return IdentifierNode(name)


def _binary(left, operator, right):
    return BinaryOpNode(left, operator, right)


def tokenize_code(code):
    return Lexer(code).tokens


# --- loop_induction: checked constant evaluation -----------------------------


@pytest.mark.parametrize(
    "expression, expected",
    [
        (_binary(_int(-7), "/", _int(2)), -3),
        (_binary(_int(-7), "%", _int(2)), -1),
        (_binary(_int(7), "%", _int(-2)), 1),
        (_binary(_int(1), "<<", _int(30)), 1 << 30),
        (_binary(_int(1), "<<", _int(31)), None),
        (_binary(_int(-1), "<<", _int(2)), None),
        (_binary(_int(1), "<<", _int(-1)), None),
        (_binary(_int(INT_MAX), "+", _int(1)), None),
        (_binary(_int(-INT_MAX - 1), "/", _int(-1)), None),
        (_binary(_int(5), "/", _int(0)), None),
        (_binary(_int(1 << 16), "*", _int(1 << 15)), None),
        (_binary(_int(6), "&", _int(3)), 2),
        (_binary(_int(6), "^", _int(3)), 5),
        (_binary(_binary(_int(1), "<<", _int(3)), "<", _int(8)), 0),
        (_binary(_int(2), "<=", _name("C")), 1),
        (_binary(_int(0), "&&", _binary(_int(1), "/", _int(0))), None),
        (UnaryOpNode("~", _int(0)), -1),
        (UnaryOpNode("!", _int(3)), 0),
        (_name("missing"), None),
        (LiteralNode(1.5, PrimitiveType("float")), None),
        (LiteralNode(_UnsignedLiteralInt(3), PrimitiveType("uint")), None),
        # The parser keeps ``8u`` as an int value with a uint literal type.
        (LiteralNode(3, PrimitiveType("uint")), None),
        (_binary(_int(-2), "<", LiteralNode(8, PrimitiveType("uint"))), None),
        (LiteralNode(3, PrimitiveType("int64_t")), None),
        (LiteralNode(True, PrimitiveType("bool")), 1),
    ],
)
def test_checked_static_values_follow_cpp_and_reject_undefined_behavior(
    expression, expected
):
    assert _checked_static_value(expression, {"C": 8}) == expected


def _step_loop(condition, update=None, start=1, type_name="int", constants=None):
    update = update or UnaryOpNode("POST_INCREMENT", _name("step"))
    return static_loop_trip_count(
        type_name,
        _int(start),
        condition,
        update,
        "step",
        {"C": 8} if constants is None else constants,
    )


def test_static_trip_counts_run_mlx_power_of_two_loops():
    condition = _binary(_binary(_int(1), "<<", _name("step")), "<", _name("C"))
    # MLX's Neumann-series inverse squares P for C = 8: steps 1 and 2.
    assert _step_loop(condition) == 2
    for chunk, trips in ((1, 0), (2, 0), (3, 1), (16, 3), (64, 5)):
        assert _step_loop(condition, constants={"C": chunk}) == trips


@pytest.mark.parametrize(
    "arguments",
    [
        # 1 << 31 is undefined, so the loop never provably exits.
        dict(condition=_binary(_binary(_int(1), "<<", _name("step")), ">", _int(0))),
        # The counter itself wraps.
        dict(
            condition=_binary(_name("step"), ">", _int(0)),
            update=AssignmentNode(_name("step"), _int(2), "*="),
        ),
        # Too many trips to evaluate.
        dict(condition=_binary(_name("step"), "<", _int(5000))),
        # A short counter cannot reach 40000.
        dict(condition=_binary(_name("step"), "<", _int(40000)), type_name="short"),
        # Unsigned counters, other types and runtime names are not proven.
        dict(condition=_binary(_name("step"), "<", _int(4)), type_name="uint"),
        dict(condition=_binary(_name("step"), "<", _int(4)), type_name="long"),
        dict(condition=_binary(_name("step"), "<", _int(4)), type_name="float"),
        dict(condition=_binary(_name("step"), "<", _name("T"))),
        dict(
            condition=_binary(_name("step"), "<", _int(4)),
            update=AssignmentNode(_name("other"), _int(1), "+="),
        ),
        dict(
            condition=_binary(_name("step"), "<", _int(4)),
            update=AssignmentNode(_name("step"), _name("T"), "+="),
        ),
        dict(
            condition=_binary(_name("step"), "<", _int(4)),
            update=AssignmentNode(_name("step"), _int(1), "/="),
        ),
        dict(
            condition=_binary(_name("step"), "<", _int(64)),
            update=AssignmentNode(_name("step"), _int(31), "<<="),
        ),
    ],
)
def test_static_trip_counts_reject_unbounded_or_runtime_loops(arguments):
    assert _step_loop(**arguments) is None


def test_static_trip_counts_support_descending_and_shifting_updates():
    shift = AssignmentNode(_name("step"), _int(1), "<<=")
    assert _step_loop(_binary(_name("step"), "<", _int(64)), shift) == 6
    halve = AssignmentNode(_name("step"), _int(1), ">>=")
    assert _step_loop(_binary(_name("step"), ">", _int(0)), halve, start=64) == 7
    down = UnaryOpNode("PRE_DECREMENT", _name("step"))
    assert _step_loop(_binary(_name("step"), ">=", _int(0)), down, start=3) == 4
    # The loop's own name shadows a constant of the same name.
    assert (
        _step_loop(_binary(_name("step"), "<", _int(3)), constants={"step": 100}) == 2
    )


# --- loop_induction: offset counter termination ------------------------------


def _bits(type_name):
    return 16 if "short" in type_name else 32


def _range(type_name):
    unsigned = type_name.startswith("u")
    low = 0 if unsigned else -(1 << (_bits(type_name) - 1))
    return low, (1 << (_bits(type_name) - (not unsigned))) - 1


def _taken(value, offset, stop, operator):
    return value + offset < stop or operator == "<=" and value + offset == stop


def _simulated_offset_loop(type_name, start, offset, stops, step, operator):
    """Run the loop one trip at a time; only feasible for 16-bit counters."""
    low, high = _range(type_name)
    if step <= 0 or offset < 0 or operator not in {"<", "<="}:
        return False
    for stop in stops:
        if not low <= stop <= high:
            return False
        value = start
        while True:
            if not low <= value <= high or not low <= value + offset <= high:
                return False
            if not _taken(value, offset, stop, operator):
                break
            value += step
    return True


def _searched_offset_loop(type_name, start, offset, stops, step, operator):
    """Find the terminating trip by searching the condition itself.

    The counter only grows, so every visited value lies between the first and
    the terminating one; the trip that ends the loop is the least ``k`` whose
    condition fails, found by galloping then bisecting on ``_taken``.
    """
    low, high = _range(type_name)
    if step <= 0 or offset < 0 or operator not in {"<", "<="}:
        return False
    for stop in stops:
        if not low <= stop <= high:
            return False
        upper = 1
        while _taken(start + upper * step, offset, stop, operator):
            upper *= 2
        lower = 0
        while lower < upper:
            middle = (lower + upper) // 2
            if _taken(start + middle * step, offset, stop, operator):
                lower = middle + 1
            else:
                upper = middle
        final = start + lower * step
        for value in (start, start + offset, final, final + offset):
            if not low <= value <= high:
                return False
    return True


@pytest.mark.parametrize("type_name", ["short", "ushort", "int", "uint"])
@pytest.mark.parametrize("operator", ["<", "<="])
def test_offset_counter_termination_matches_simulation(type_name, operator):
    low, high = _range(type_name)
    accepted = rejected = 0
    for start in (low, 0, 5, high - 9, high):
        for offset in (0, 1, 8, 9):
            for step in (1, 8, 9):
                for first in (low, 0, 3, high - 17, high - 8, high):
                    stops = list(range(first, min(high, first + 3) + 1))
                    expected = _searched_offset_loop(
                        type_name, start, offset, stops, step, operator
                    )
                    if _bits(type_name) == 16:
                        # Check the search against one-trip-at-a-time runs.
                        assert expected == _simulated_offset_loop(
                            type_name, start, offset, stops, step, operator
                        )
                    actual = offset_counter_loop_terminates(
                        type_name,
                        start,
                        offset,
                        [min(stops), max(stops)],
                        step,
                        operator,
                    )
                    assert actual == expected, (
                        type_name,
                        operator,
                        start,
                        offset,
                        step,
                        first,
                    )
                    accepted += expected
                    rejected += not expected
    assert accepted > 50 and rejected > 50


@pytest.mark.parametrize(
    "arguments, expected",
    [
        # MLX's chunk loop: T up to INT_MAX - 8 keeps t + 8 representable.
        (("int", 0, 8, [0, INT_MAX - 8], 8, "<="), True),
        (("int", 0, 8, [0, INT_MAX - 7], 8, "<="), False),
        (("int", 0, 8, [0, INT_MAX], 8, "<"), False),
        (("int", 0, 8, [-5, 7], 8, "<="), True),
        (("int", 0, -8, [0, 64], 8, "<="), False),
        (("int", 0, 8, [0, 64], 0, "<="), False),
        (("int", 0, 8, [0, 64], -8, "<="), False),
        (("int", 0, 8, [0, 64], 8, ">"), False),
        (("int", 0, 8, [], 8, "<="), False),
        (("int", True, 8, [0, 64], 8, "<="), False),
        (("int", 0, 8, [0, 64.0], 8, "<="), False),
        (
            ("uint", 0, 8, [_UnsignedLiteralInt(0), _UnsignedLiteralInt(64)], 8, "<"),
            True,
        ),
        (
            ("int", -8, 8, [_UnsignedLiteralInt(0), _UnsignedLiteralInt(64)], 8, "<"),
            False,
        ),
        (("long", 0, 8, [0, 64], 8, "<="), False),
        (("int2", 0, 8, [0, 64], 8, "<="), False),
        (("not a type", 0, 8, [0, 64], 8, "<="), False),
    ],
)
def test_offset_counter_termination_contract(arguments, expected):
    assert offset_counter_loop_terminates(*arguments) is expected


# --- loop_induction: discovering initializer-free counter loops --------------

COUNTER_SHADER = """shader Counter {{
    {helpers}
    compute {{
        void main(StructuredBuffer<int> limits @register(t0),
                  RWStructuredBuffer<int> results @register(u1)) {{
            int total = 0;
            {before}
            int t = 0;
            {between}
            for (; t + 8 <= limits[0]; t += 8) {{
                total += t;
            }}
            {after}
            results[0] = total + t;
        }}
    }}
}}"""


def _counter_loops(writes=None, *, helpers="", before="", between="", after=""):
    ast = parse(
        COUNTER_SHADER.format(
            helpers=helpers, before=before, between=between, after=after
        )
    )
    return init_less_counter_loops(_entry(ast), GLSLCodeGen().walk_ast, writes)


def _entry(ast):
    ((_stage, function, _config),) = GLSLCodeGen().glsl_software_subgroup_stage_entries(
        ast
    )
    return function


def test_counter_loop_discovery_finds_the_mlx_chunk_loop():
    ((name, declaration, loop),) = _counter_loops().values()
    assert name == "t" and declaration.name == "t" and loop.init is None
    assert _counter_loops(after="results[1] = t;") != {}
    assert _counter_loops(after="if (t < limits[0]) { total += 1; }") != {}


@pytest.mark.parametrize(
    "fields",
    [
        dict(after="t += 1;"),
        dict(after="t = 3;"),
        dict(after="t++;"),
        dict(between="--t;"),
        dict(before="if (total > 0) { return; }"),
        dict(after="int t = 4;"),
        dict(between="int& alias = t;"),
        dict(after="int* pointer = &t;"),
        dict(
            helpers="void bump(inout int value) { value += 1; }",
            after="bump(t);",
        ),
    ],
)
def test_counter_loop_discovery_rejects_other_writes_and_aliases(fields):
    assert _counter_loops(**fields) == {}


def test_counter_loop_discovery_uses_resolved_call_effects():
    helpers = "int read(int value) { return value; }"
    # Without resolved effects, any call that reads the counter may write it.
    assert _counter_loops(helpers=helpers, after="total += read(t);") == {}
    assert _counter_loops(
        lambda node: set(), helpers=helpers, after="total += read(t);"
    )
    assert (
        _counter_loops(lambda node: {"t"}, helpers=helpers, after="total += read(t);")
        == {}
    )


def test_counter_loop_discovery_requires_an_initializer_free_outer_loop():
    source = COUNTER_SHADER.format(helpers="", before="", between="", after="")
    for old, new in [
        ("for (; t + 8", "for (t = 0; t + 8"),
        ("int t = 0;", "int t;"),
        ("t += 8)", "t -= 8)"),
    ]:
        ast = parse(source.replace(old, new))
        walk = GLSLCodeGen().walk_ast
        assert init_less_counter_loops(_entry(ast), walk) == {}, old


# --- Metal preprocessor: library matrix typedefs -----------------------------


@pytest.mark.parametrize(
    "source, expected",
    [
        ("simdgroup_float8x8 a;", "simdgroup_matrix<float, 8, 8> a;"),
        ("simdgroup_half8x8 a[2];", "simdgroup_matrix<half, 8, 8> a[2];"),
        ("metal::simdgroup_bfloat8x8 a;", "metal::simdgroup_matrix<bfloat, 8, 8> a;"),
        ("metal :: simdgroup_float8x8 a;", "metal :: simdgroup_matrix<float, 8, 8> a;"),
        ("other::simdgroup_float8x8 a;", "other::simdgroup_float8x8 a;"),
        ("my_simdgroup_float8x8 a;", "my_simdgroup_float8x8 a;"),
        ("simdgroup_float8x8x a;", "simdgroup_float8x8x a;"),
        ("// simdgroup_float8x8 a;", "// simdgroup_float8x8 a;"),
        (
            'const char* s = "simdgroup_float8x8";',
            'const char* s = "simdgroup_float8x8";',
        ),
        (
            "struct simdgroup_float8x8 { float x; }; simdgroup_float8x8 a;",
            "struct simdgroup_float8x8 { float x; }; simdgroup_float8x8 a;",
        ),
        (
            "typedef float4 simdgroup_half8x8; simdgroup_half8x8 a; simdgroup_float8x8 b;",
            "typedef float4 simdgroup_half8x8; simdgroup_half8x8 a; "
            "simdgroup_matrix<float, 8, 8> b;",
        ),
        (
            "using simdgroup_float8x8 = float; simdgroup_float8x8 a;",
            "using simdgroup_float8x8 = float; simdgroup_float8x8 a;",
        ),
    ],
)
def test_library_matrix_typedefs_expand_unless_the_source_declares_them(
    source, expected
):
    canonical = MetalPreprocessor()._canonicalize_metal_library_matrix_typedefs(source)
    assert canonical == expected


def _declares(code, receiver):
    preprocessor = MetalPreprocessor()
    cursor = code.index(">") + 1
    while code[cursor].isspace():
        cursor += 1
    position = code.rindex(receiver)
    return preprocessor._builtin_declarator_list_declares(
        code, cursor, receiver, position
    )


@pytest.mark.parametrize(
    "code, receiver, expected",
    [
        ("simdgroup_matrix<float, 8, 8> A, B; B", "B", True),
        ("simdgroup_matrix<float, 8, 8> A[2], B[3][4], C; C", "C", True),
        ("simdgroup_matrix<float, 8, 8> A = D, B; B", "B", True),
        ("simdgroup_matrix<float, 8, 8> A = f(x, y), B; B", "B", True),
        ("simdgroup_matrix<float, 8, 8> A = g<int, 2>(x), B; B", "B", True),
        ("simdgroup_matrix<float, 8, 8> A(1), B{2}, C; C", "C", True),
        ("simdgroup_matrix<float, 8, 8> A, &B = A; B", "B", True),
        ("simdgroup_matrix<float, 8, 8> A; B", "B", False),
        ("simdgroup_matrix<float, 8, 8> f(int a, B", "B", False),
        ("simdgroup_matrix<float, 8, 8> A = (x, B", "B", False),
        ("simdgroup_matrix<float, 8, 8> A, 3; B", "B", False),
    ],
)
def test_builtin_declarator_lists_name_every_declared_receiver(
    code, receiver, expected
):
    assert _declares(code, receiver) is expected


# --- Metal frontend, CrossGL parser and Metal codegen ------------------------

METAL_KERNEL = """#include <metal_stdlib>
using namespace metal;

kernel void matrix_transfers(
    const device float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant int& T [[buffer(2)]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  simdgroup_float8x8 tiles[2];
  simdgroup_float8x8 A, B, C;
  const device float* in = src + sg * 128;
  device float* out = dst + 4 + sg * 320;
  simdgroup_load(A, in, 8);
  simdgroup_load(tiles[0], in + 64, 8, ulong2(0, 0), true);
  B = make_filled_simdgroup_matrix<float, 8>(0.5f);
  B.thread_elements()[0] = float(int(lane % 3) - 1);
  simdgroup_multiply(C, A, tiles[0]);
  for (int step = 1; (1 << step) < 8; step++) {
    simdgroup_multiply_accumulate(C, B, A, C);
  }
  int t = 0;
  for (; t + 8 <= T; t += 8) {
    simdgroup_multiply(tiles[1], C, B);
    C = tiles[1];
  }
  if (t < T) {
    simdgroup_multiply_accumulate(C, A, A, C);
  }
  simdgroup_store(C, out, 8);
  simdgroup_store(C, out + 64, 8, ulong2(0, 0), true);
  simdgroup_store(B, out + 128, 16);
  simdgroup_store(tiles[0], out + 256, 8);
}
"""
MAPPED = (
    "CooperativeMatrix<float,8,8,subgroup,unspecified,unspecified,"
    "metal_thread_elements,32,2,metal_thread_elements_reference_view,"
    "tile_4x4_row_pair,unit_lane_coordinates>"
)


def _crossgl(source=METAL_KERNEL, **options):
    ast = MetalParser(MetalLexer(source).tokenize()).parse()
    return MetalToCrossGLConverter(**options).generate(ast)


def test_metal_frontend_lowers_typedefs_fills_and_products():
    crossgl = _crossgl(
        cooperative_matrix_fragment_mapping="tile_4x4_row_pair",
        cooperative_matrix_fragment_mapping_provenance="unit_lane_coordinates",
    )
    assert "simdgroup_float8x8" not in crossgl
    assert f"{MAPPED}[2] tiles;" in crossgl
    for name in "ABC":
        assert f"{MAPPED} {name};" in crossgl
    assert "B = cooperative_matrix_fill(float(0.5f));" in crossgl
    assert "C = cooperative_matrix_multiply(A, tiles[0]);" in crossgl
    assert "tiles[1] = cooperative_matrix_multiply(C, B);" in crossgl
    # ``in`` and ``out`` are CrossGL keywords, so the pointers are renamed.
    assert "cooperative_matrix_load(tiles[0], in_ + 64, 8, u64vec2(0, 0), true);" in (
        crossgl
    )
    assert "cooperative_matrix_store(out_ + 128, B, 16);" in crossgl
    assert "simdgroup_multiply" not in crossgl.replace(
        "cooperative_matrix_multiply", ""
    )


def test_metal_frontend_keeps_user_functions_with_library_names():
    source = """#include <metal_stdlib>
using namespace metal;
float simdgroup_multiply(float a, float b, float c) { return a * b + c; }
kernel void user_call(device float* out [[buffer(0)]],
                      uint lane [[thread_position_in_grid]]) {
  out[lane] = simdgroup_multiply(float(lane), 2.0f, 1.0f);
}
"""
    crossgl = _crossgl(source)
    assert "cooperative_matrix_multiply" not in crossgl
    assert "simdgroup_multiply(float(lane), 2.0f, 1.0f)" in crossgl


def test_metal_round_trip_restores_library_fills(tmp_path):
    crossgl = _crossgl(
        cooperative_matrix_fragment_mapping="tile_4x4_row_pair",
        cooperative_matrix_fragment_mapping_provenance="unit_lane_coordinates",
    )
    metal = MetalCodeGen().generate(parse(crossgl))
    assert "B = make_filled_simdgroup_matrix<float, 8, 8>(float(0.5));" in metal
    assert "simdgroup_load(tiles[0], in_ + 64, 8, ulong2(0, 0), true);" in metal
    assert "simdgroup_store(B, out_ + 128, 16);" in metal
    assert "C = (A * tiles[0]);" in metal
    _compile(metal, "metal", tmp_path, metal_compile_flags=("-fno-fast-math",))


PARSER_SHADER = f"""shader Fills {{
    {MATRIX} filled(float value) {{
        return cooperative_matrix_fill(value);
    }}
    compute {{
        @numthreads(32, 1, 1)
        void main(RWStructuredBuffer<float> results @register(u0),
                  uint lane @gl_SubgroupInvocationID) {{
            {MATRIX} declared = cooperative_matrix_fill(float(lane));
            {MATRIX} assigned;
            assigned = cooperative_matrix_fill(2.0);
            {MATRIX}[2] tiles;
            tiles[0] = filled(3.0);
            tiles[1] = declared;
            {MATRIX} product = cooperative_matrix_multiply(tiles[0], tiles[1]);
            BODY
            results[lane] = cooperative_matrix_element(product, 0)
                + cooperative_matrix_element(assigned, 1);
        }}
    }}
}}"""


def test_parser_types_fills_and_matrix_array_elements_from_context():
    metal = MetalCodeGen().generate(parse(PARSER_SHADER.replace("BODY", "")))
    assert metal.count("make_filled_simdgroup_matrix<float, 8, 8>(") == 3
    assert "(tiles[0] * tiles[1])" in metal


def test_parser_rejects_fills_with_other_arities():
    source = PARSER_SHADER.replace("float(lane)", "float(lane), 1.0")
    with pytest.raises(SyntaxError, match="fill expects 1 argument, got 2"):
        Parser(tokenize_code(source), strict_function_bodies=True).parse()
    # Permissive parses keep the call for the target's diagnostic.
    with pytest.raises(UnsupportedMetalFeatureError, match="fill requires one value"):
        MetalCodeGen().generate(parse(source))


def test_backends_reject_fills_without_a_matrix_context():
    tree = parse(PARSER_SHADER.replace("BODY", "cooperative_matrix_fill(1.0);"))
    with pytest.raises(UnsupportedMetalFeatureError, match="fill requires"):
        MetalCodeGen().generate(tree)
    with pytest.raises(OpenGLCooperativeMatrixError) as gl_error:
        _gl().generate_stage(tree, "compute")
    assert gl_error.value.reason == "missing-result-contract"
    with pytest.raises(DirectXCooperativeMatrixUnsupportedError) as dx_error:
        _dx().generate_stage(tree, "compute")
    assert "fill" in str(dx_error.value)


# --- OpenGL and DirectX software lowering ------------------------------------

TRANSFER_SHADER = f"""shader Transfers {{
    compute {{
        @numthreads(32, 4, 1)
        void main(StructuredBuffer<float> src @register(t0),
                  RWStructuredBuffer<float> dst @register(u1),
                  StructuredBuffer<int> words @register(t2),
                  uint invocation @gl_LocalInvocationIndex) {{
            {MATRIX} A;
            {MATRIX}[2] tiles;
            uint stride = 8u;
            int counter = 0;
            bool flag = invocation > 3u;
            BODY
        }}
    }}
}}"""
TRANSFERS = """
            A = cooperative_matrix_fill(float(counter++));
            cooperative_matrix_load(A, src, 8);
            cooperative_matrix_load(tiles[0], src + 64, stride, u64vec2(0, 0), true);
            tiles[1] = cooperative_matrix_multiply(A, tiles[0]);
            cooperative_matrix_store(dst, tiles[1], 16);
            cooperative_matrix_store(dst + 128, A, 8, u64vec2(0u, 0u), false);
"""


def _gl(**options):
    return GLSLCodeGen(
        cooperative_matrix_software_lowering=True, software_subgroup_width=32, **options
    )


def _dx(**options):
    return HLSLCodeGen(
        cooperative_matrix_software_lowering=True,
        software_subgroup_width=32,
        relative_wave_shuffle_out_of_range="self",
        **options,
    )


def _dx_compile(generated, tmp_path):
    from tests.test_translator.test_directx_float_atomics import _compile as compile_

    compile_(generated, tmp_path, profile="cs_6_2")


def test_opengl_transfers_address_each_lane_fragment(tmp_path):
    generated = _gl().generate_stage(
        parse(TRANSFER_SHADER.replace("BODY", TRANSFERS)), "compute"
    )
    helper = "crossgl_cooperative_matrix_fragment_float_8x8_32x2_tile_4x4_row_pair"
    assert f"{helper}_fill(float(float((counter++))))" in generated
    assert generated.count("counter++") == 1
    assert f"uint {helper}_fragment_offset(" in generated
    assert "uint lane = gl_LocalInvocationIndex % 32u;" in generated
    assert "uint row = ((lane >> 2u) & 4u) + ((lane >> 1u) & 3u);" in generated
    assert (
        "uint column = ((lane >> 2u) & 2u) * 2u + (lane & 1u) * 2u + element;"
        in generated
    )
    assert f"src[(64 + int({helper}_fragment_offset(1u, uint(stride), true)))]" in (
        generated
    )
    assert f"dst[int({helper}_fragment_offset(0u, uint(16), false))] = " in generated
    assert "cooperative_matrix_load" not in generated
    assert "cooperative_matrix_store" not in generated
    _compile(generated, "opengl", tmp_path)


def test_directx_transfers_address_each_lane_fragment(tmp_path):
    generated = _dx().generate_stage(
        parse(TRANSFER_SHADER.replace("BODY", TRANSFERS)), "compute"
    )
    assert generated.count("counter++") == 1
    assert "__crossgl_software_matrix_fragment_offset(uint lane, uint element" in (
        generated
    )
    assert "return transpose ? column * stride + row : row * stride + column;" in (
        generated
    )
    assert "cooperative_matrix_load" not in generated
    assert "cooperative_matrix_store" not in generated
    _dx_compile(generated, tmp_path)


@pytest.mark.parametrize(
    "body, reason",
    [
        (
            "cooperative_matrix_load(A, src, 8, u64vec2(0, 0), flag);",
            "dynamic-transpose",
        ),
        (
            "cooperative_matrix_load(A, src, 8, u64vec2(1, 0), true);",
            "nonzero-matrix-origin",
        ),
        (
            "cooperative_matrix_store(dst, A, 8, u64vec2(0, 2));",
            "nonzero-matrix-origin",
        ),
        ("cooperative_matrix_load(A, src + counter++, 8);", "impure-transfer-operand"),
        ("cooperative_matrix_load(A, src, stride++);", "impure-transfer-operand"),
        (
            "cooperative_matrix_load(tiles[counter++], src, 8);",
            "impure-transfer-operand",
        ),
        ("cooperative_matrix_load(A, words, 8);", "unsupported-transfer-pointer"),
        ("cooperative_matrix_store(words, A, 8);", "unsupported-transfer-pointer"),
    ],
)
@pytest.mark.parametrize("target", TARGETS)
def test_transfers_fail_closed_on_unproven_operands(target, body, reason):
    tree = parse(TRANSFER_SHADER.replace("BODY", body))
    if target == "opengl":
        with pytest.raises(OpenGLCooperativeMatrixError) as error:
            _gl().generate_stage(tree, "compute")
    else:
        with pytest.raises(DirectXCooperativeMatrixUnsupportedError) as error:
            _dx().generate_stage(tree, "compute")
    assert error.value.reason == reason


def test_opengl_store_requires_a_writable_buffer():
    tree = parse(
        TRANSFER_SHADER.replace("BODY", "cooperative_matrix_store(src, A, 8);")
    )
    with pytest.raises(OpenGLCooperativeMatrixError) as error:
        _gl().generate_stage(tree, "compute")
    assert error.value.reason == "read-only-transfer-pointer"


def test_directx_store_requires_a_writable_buffer():
    tree = parse(
        TRANSFER_SHADER.replace("BODY", "cooperative_matrix_store(src, A, 8);")
    )
    with pytest.raises(Exception, match="(?i)writ|read-only|RWStructuredBuffer"):
        _dx().generate_stage(tree, "compute")


@pytest.mark.parametrize("target", TARGETS)
def test_transfers_require_the_registered_row_pair_mapping(target):
    source = TRANSFER_SHADER.replace(
        "BODY", "cooperative_matrix_load(A, src, 8);"
    ).replace("tile_4x4_row_pair", "tile_8x4_column_pair")
    with pytest.raises(Exception, match="(?i)mapping|contract|fragment"):
        tree = parse(source)
        (_gl() if target == "opengl" else _dx()).generate_stage(tree, "compute")


def test_opengl_transfers_require_exact_32_lane_subgroups():
    tree = parse(TRANSFER_SHADER.replace("BODY", "cooperative_matrix_load(A, src, 8);"))
    generator = GLSLCodeGen(cooperative_matrix_software_lowering=True)
    with pytest.raises(OpenGLCooperativeMatrixError) as error:
        generator.generate_stage(tree, "compute")
    assert error.value.reason == "exact-subgroup-width-required"


def test_directx_transfers_require_software_subgroups():
    tree = parse(TRANSFER_SHADER.replace("BODY", "cooperative_matrix_load(A, src, 8);"))
    with pytest.raises(DirectXCooperativeMatrixUnsupportedError) as error:
        HLSLCodeGen(cooperative_matrix_software_lowering=True).generate_stage(
            tree, "compute"
        )
    assert error.value.operation == "load"
    assert error.value.reason == "exact-subgroup-width-required"


@pytest.mark.parametrize("target", TARGETS)
def test_transfer_only_kernels_need_complete_subgroups(target):
    body = "cooperative_matrix_load(A, src, 8); cooperative_matrix_store(dst, A, 8);"
    tree = parse(
        TRANSFER_SHADER.replace("BODY", body).replace(
            "@numthreads(32, 4, 1)", "@numthreads(16, 3, 1)"
        )
    )
    generator = _gl() if target == "opengl" else _dx()
    with pytest.raises(Exception, match="complete logical subgroups") as error:
        generator.generate_stage(tree, "compute")
    assert "incomplete-matrix-subgroup" in repr(vars(error.value))


@pytest.mark.parametrize("target", TARGETS)
def test_transfers_in_helpers_without_products_get_the_lane(tmp_path, target):
    source = TRANSFER_SHADER.replace(
        "compute {",
        f"""void copy(StructuredBuffer<float> src, RWStructuredBuffer<float> dst) {{
        {MATRIX} tile;
        cooperative_matrix_load(tile, src, 8);
        cooperative_matrix_store(dst, tile, 8);
    }}
    compute {{""",
    ).replace("BODY", "copy(src, dst);")
    tree = parse(source)
    if target == "opengl":
        generated = _gl().generate_stage(tree, "compute")
        assert "gl_LocalInvocationIndex % 32u" in generated
        _compile(generated, "opengl", tmp_path)
    else:
        generated = _dx().generate_stage(tree, "compute")
        assert "static uint __crossgl_software_subgroup_invocation;" in generated
        assert "__crossgl_software_subgroup_invocation = uint(" in generated
        _dx_compile(generated, tmp_path)


# --- Project translation: the static and chunk loops -------------------------

CONDITION = dict(
    entryPoint="matrix_transfers", parameter="T", minimum=0, maximum=INT_MAX - 8
)


def _translate(root, target, source=METAL_KERNEL, conditions=(CONDITION,)):
    root.mkdir(parents=True, exist_ok=True)
    (root / "transfers.metal").write_text(source)
    options = {
        "preserve_resource_origins": True,
        "cooperative_matrix_fragment_mapping": "tile_4x4_row_pair",
        "cooperative_matrix_fragment_mapping_provenance": "unit_lane_coordinates",
    }
    if target != "metal":
        target_options = {
            "cooperative_matrix_software_lowering": True,
            "software_subgroup_width": 32,
        }
        if target == "directx":
            target_options["relative_wave_shuffle_out_of_range"] = "self"
        options["target_options"] = {target: target_options}
    report = translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("transfers.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=(32, 4, 1),
            metadata_preconditions=conditions,
            source_options={"metal": options},
        ),
        format_output=False,
    )
    return report


@pytest.mark.parametrize("target", ("metal", *TARGETS))
def test_chunked_kernel_translates_and_validates(tmp_path, target):
    report = _translate(tmp_path, target).to_json()
    assert report["summary"]["translatedCount"] == 1, report["diagnostics"]
    (artifact,) = report["artifacts"]
    text = (tmp_path / artifact["path"]).read_text()
    # Both loops survive as written; only their matrix products are lowered.
    assert "for (int step = 1; " in text
    assert "for (; " in text and "t += 8" in text
    if target != "metal":
        assert "simdgroup_" not in text
    work = tmp_path / "validation"
    work.mkdir()
    if target == "metal" or os.environ.get("CROSTL_SKIP_NATIVE_VALIDATORS") != "1":
        if target == "metal":
            _compile(text, "metal", work, metal_compile_flags=("-fno-fast-math",))
        elif target == "opengl":
            _compile(text, "opengl", work)
        else:
            _dx_compile(text, work)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "conditions",
    [
        (),
        ({**CONDITION, "maximum": INT_MAX - 7},),
        ({**CONDITION, "maximum": INT_MAX},),
        ({**CONDITION, "minimum": -(1 << 31)},),
    ],
)
def test_chunk_loop_needs_a_bound_that_cannot_overflow(tmp_path, target, conditions):
    report = _translate(tmp_path, target, conditions=conditions).to_json()
    if conditions and conditions[0]["minimum"] < 0:
        # Negative bounds take zero trips; only the maximum can overflow.
        assert report["summary"]["translatedCount"] == 1, report["diagnostics"]
        return
    assert report["summary"]["translatedCount"] == 0, report
    codes = {item["code"] for item in report["diagnostics"]}
    assert ERRORS[target] in codes, codes


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "variant",
    [
        "lane-start",
        "constant-start",
        "lane-bound",
        "lane-stride",
        "written-in-body",
        "written-after",
        "exit-in-body",
        "reference-alias",
        "mutating-call",
        "step-overflow",
        "lane-step-bound",
        "written-step",
        "volatile-counter",
    ],
)
def test_divergent_or_unproven_loops_are_rejected(tmp_path, target, variant):
    rewrites = {
        "lane-start": [("int t = 0;", "int t = int(lane % 2);")],
        "constant-start": [("int t = 0;", "const int zero = 0; int t = zero;")],
        "lane-bound": [("t + 8 <= T;", "t + 8 <= T - int(lane % 2);")],
        "lane-stride": [("t += 8)", "t += 8 + int(lane % 2))")],
        "written-in-body": [("C = tiles[1];", "C = tiles[1]; t += int(lane % 2);")],
        "written-after": [("if (t < T) {", "t += int(lane % 2); if (t < T) {")],
        "exit-in-body": [("C = tiles[1];", "C = tiles[1]; if (lane == 3) { break; }")],
        "reference-alias": [
            ("int t = 0;", "int t = 0; thread int& alias = t;"),
            ("if (t < T) {", "alias += int(lane % 2); if (t < T) {"),
        ],
        "mutating-call": [
            (
                "kernel void",
                "static void bump(thread int& value, uint lane) "
                "{ value += int(lane % 2); }\nkernel void",
            ),
            ("if (t < T) {", "bump(t, lane); if (t < T) {"),
        ],
        "step-overflow": [("(1 << step) < 8", "(1 << step) > 0")],
        "lane-step-bound": [("(1 << step) < 8", "(1 << step) < 8 + int(lane % 2)")],
        "written-step": [
            (
                "simdgroup_multiply_accumulate(C, B, A, C);\n  }",
                "simdgroup_multiply_accumulate(C, B, A, C);\n    step += int(lane % 2);\n  }",
            )
        ],
        "volatile-counter": [("int t = 0;", "volatile int t = 0;")],
    }
    source = METAL_KERNEL
    for old, new in rewrites[variant]:
        assert old in source, old
        source = source.replace(old, new)
    report = _translate(tmp_path, target, source=source).to_json()
    assert report["summary"]["translatedCount"] == 0, (variant, report)
    codes = {item["code"] for item in report["diagnostics"]}
    rejections = {ERRORS[target], "project.translate.failed"}
    if variant == "reference-alias" and target == "opengl":
        # GLSL has no private references, so the alias fails even earlier.
        rejections.add("project.translate.opengl-private-pointer-unsupported")
    assert codes & rejections, codes


# --- Native execution ---------------------------------------------------------


def _fragment_coordinate(lane, element):
    row = ((lane >> 2) & 4) + ((lane >> 1) & 3)
    column = ((lane >> 2) & 2) * 2 + (lane & 1) * 2 + element
    return row, column


def _product(left, right):
    return [
        [sum(left[i][k] * right[k][j] for k in range(8)) for j in range(8)]
        for i in range(8)
    ]


def _add(left, right):
    return [[left[i][j] + right[i][j] for j in range(8)] for i in range(8)]


def _inputs():
    signs = (-2, -1, 1, 2)
    return [
        float(signs[(index * 3 + index // 5 + index // 128) % 4])
        for index in range(512)
    ]


def _expected(src, steps):
    """The kernel's result for every SIMD-group in exact rational arithmetic."""
    dst = [GUARD] * (8 + 4 * 320)
    for group in range(4):
        values = [Fraction(value) for value in src[group * 128 : group * 128 + 128]]
        a = [[values[row * 8 + column] for column in range(8)] for row in range(8)]
        tile = [
            [values[64 + column * 8 + row] for column in range(8)] for row in range(8)
        ]
        b = [[Fraction(1, 2)] * 8 for _ in range(8)]
        for lane in range(32):
            row, column = _fragment_coordinate(lane, 0)
            b[row][column] = Fraction(lane % 3 - 1)
        c = _product(a, tile)
        for _ in range(2):
            c = _add(_product(b, a), c)
        t = 0
        while t + 8 <= steps:
            c = _product(c, b)
            t += 8
        if t < steps:
            c = _add(_product(a, a), c)
        for matrix in (a, tile, b, c):
            # Every product, sum and partial sum is a small dyadic rational, so
            # each summation order rounds to the same binary32 value.
            assert all(
                abs(x) * 4 < (1 << 20) and (x * 4).denominator == 1
                for row in matrix
                for x in row
            )
        base = 4 + group * 320
        for row in range(8):
            for column in range(8):
                dst[base + row * 8 + column] = float(c[row][column])
                dst[base + 64 + column * 8 + row] = float(c[row][column])
                dst[base + 128 + row * 16 + column] = float(b[row][column])
                dst[base + 256 + row * 8 + column] = float(tile[row][column])
    return dst


def _package(root, target):
    report = _translate(root, target)
    report.write_json(root / "report.json")
    assert report.to_json()["summary"]["failedCount"] == 0, report.to_json()
    manifest = build_runtime_artifact_manifest(root / "report.json")
    assert manifest["success"], manifest
    import json

    (root / "artifacts.json").write_text(json.dumps(manifest))
    package = root / "package"
    assert build_runtime_package(root / "artifacts.json", package)["success"]
    loader = build_runtime_loader_manifest(package / "runtime-package.json")
    assert loader["success"] and len(loader["loadUnits"]) == 1, loader
    descriptor = build_native_loader_abi_descriptor(
        loader, load_unit_id=loader["loadUnits"][0]["id"]
    )
    return descriptor, package


def test_expected_results_are_exact_in_binary32():
    src = _inputs()
    for steps in (0, 3, 8, 13):
        for value in _expected(src, steps):
            assert struct.unpack("<f", struct.pack("<f", value))[0] == value


@pytest.mark.parametrize("steps", [0, 3, 8, 13])
def test_chunked_kernel_executes_natively(tmp_path, steps):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native matrix transfer checks")
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    descriptor, package = _package(tmp_path, target)
    src = _inputs()
    size = 8 + 4 * 320
    inputs = _bound_values(
        descriptor,
        {
            "src": dict(dtype="float32", shape=[len(src)], values=src),
            "dst": dict(dtype="float32", shape=[size], values=[GUARD] * size),
            "T": dict(dtype="int32", shape=[1], values=[steps]),
        },
    )
    outputs = _bound_values(
        descriptor,
        {"dst": dict(dtype="float32", shape=[size], values=_expected(src, steps))},
    )
    names = {
        binding["scalarLayout"].get("memberName", binding["name"]): binding["name"]
        for binding in descriptor["bindings"]
    }
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [1, 1, 1], "workgroupSize": [32, 4, 1]},
        observe_inputs=[names["T"]],
    )
    work = tmp_path / "native"
    work.mkdir()
    _execute(
        request,
        outputs,
        work,
        validate=_validate,
        original_source=METAL_KERNEL,
        original_entry="matrix_transfers",
    )


def test_ci_requires_matrix_transfer_execution():
    from pathlib import Path

    from tests.ci_helpers import assert_paths_covered
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate collective helper arguments"
    )
    assert "tests/test_translator/test_matrix_transfers.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "continue-on-error" not in step
    for event in ("push", "pull_request"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_matrix_transfers.py",
        )
