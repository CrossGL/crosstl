"""Calls of program functions in compile-time initializers are evaluated at translation time.

MSL and HLSL constants may call ``constexpr`` functions of the program (``constant constexpr int C = twice(base);``).
GLSL and WGSL constant expressions call built-ins only, so the GLSL, HLSL and SPIR-V generators refused such a
constant, and the MSL, Slang, WGSL and CUDA generators printed the call ahead of its function.
"""

import struct
import textwrap

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import StaticAssertionFailedError
from crosstl.translator.semantics.constant_calls import fold_constant_calls

TARGETS = ("metal", "directx", "opengl", "vulkan", "wgsl", "cuda", "hip", "slang")


def _program(declarations):
    source = textwrap.dedent("""
        shader main {
        %s
            compute {
                RWStructuredBuffer<int> out_ @buffer(0);
                @numthreads(1, 1, 1)
                void main(uint3 gid @gl_GlobalInvocationID) {
                    out_[gid.x] = 1;
                }
            }
        }
        """) % textwrap.indent(textwrap.dedent(declarations), "    ")
    return Parser(Lexer(source).get_tokens()).parse()


def _initializer(program, name):
    for node in program.global_variables:
        if getattr(node, "name", None) == name:
            return node.initial_value
    raise AssertionError(f"no global {name}")


def _value(node):
    """The value a folded initializer spells, or ``None`` when it still holds a call of a program function."""
    if isinstance(node, A.LiteralNode):
        return node.value
    if isinstance(node, A.UnaryOpNode) and node.operator == "-":
        return -_value(node.operand)
    if isinstance(node, A.BinaryOpNode) and node.operator == "-":
        return _value(node.left) - _value(node.right)
    if isinstance(node, A.FunctionCallNode) and node.function.name.endswith("_t"):
        return (node.function.name, _value(node.arguments[0]))
    return None


def _folded(declarations, name="C"):
    program = _program(declarations)
    fold_constant_calls(program)
    return _value(_initializer(program, name))


def _float32(number):
    return struct.unpack("<f", struct.pack("<f", number))[0]


@pytest.mark.parametrize(
    "declarations, expected",
    [
        (
            """
            constant int base = 3;
            int twice(int x) { return 2 * x; }
            constant int C = twice(base);
            """,
            6,
        ),
        (
            """
            int triangle(int n) {
                int sum = 0;
                for (int i = 1; i <= n; i++) { sum += i; }
                return sum;
            }
            constant int C = triangle(10);
            """,
            55,
        ),
        (
            """
            int fib(int n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }
            constant int C = fib(12);
            """,
            144,
        ),
        (
            """
            int odd_sum_below(int n) {
                int sum = 0;
                int i = 0;
                while (true) {
                    i = i + 1;
                    if (i >= n) { break; }
                    if (i % 2 == 0) { continue; }
                    sum += i;
                }
                return sum;
            }
            constant int C = odd_sum_below(10);
            """,
            25,
        ),
        (
            """
            int count_down(int n) {
                int steps = 0;
                do { n--; ++steps; } while (n > 0);
                return steps;
            }
            constant int C = count_down(0);
            """,
            1,
        ),
        (
            """
            int pick(int x) {
                if (x < 0) { return -1; } else if (x == 0) { return 0; } else { return 1; }
            }
            constant int C = pick(-5) + pick(0) * 10 + pick(9) * 100;
            """,
            None,  # The sum around the calls is left to the generators; each call folds.
        ),
    ],
)
def test_program_functions_are_evaluated(declarations, expected):
    program = _program(declarations)
    assert fold_constant_calls(program) >= 1
    initializer = _initializer(program, "C")
    if expected is None:
        assert not any(
            isinstance(node, A.FunctionCallNode)
            for node in (initializer.left, initializer.right)
        )
    else:
        assert _value(initializer) == expected


@pytest.mark.parametrize(
    "body, expected",
    [
        # C++ truncates the quotient toward zero; the remainder takes the dividend's sign.
        ("return -7 / 2;", -3),
        ("return -7 % 2;", -1),
        ("return 7 % -2;", 1),
        ("return (1 << 30) + ((1 << 30) - 1);", 2**31 - 1),
        ("return -8 >> 1;", -4),
        ("return ~5;", -6),
        ("return int(-2.75);", -2),
        ("return int(true) + int(3u);", 4),
        ("int x; x = 4; x *= 3; x -= 2; x <<= 1; return x;", 20),
        ("return min(3, -4) + max(3, -4) * 10 + clamp(12, 0, 9) * 100;", 926),
        ("return abs(-6);", 6),
        ("return -2147483647 - 1;", -(2**31)),
        ("return 5 > 3 && 2 > 1 ? 1 : 0;", 1),
        ("return false && 1 / 0 == 0 ? 1 : 2;", 2),
    ],
)
def test_signed_integer_semantics(body, expected):
    assert _folded(f"int f() {{ {body} }}\nconstant int C = f();") == expected


@pytest.mark.parametrize(
    "body, expected",
    [
        # Unsigned arithmetic wraps modulo 2**32 ([basic.fundamental]/4).
        ("return 0u - 1u;", 2**32 - 1),
        ("return 4294967295u * 2u;", 2**32 - 2),
        ("return 1u << 31u;", 2**31),
        # int and uint operands convert to uint ([expr]/11), so -1 becomes 2**32 - 1.
        ("return -1 + 0u;", 2**32 - 1),
        ("return -1 < 0u ? 1u : 2u;", 2),
        # The type of ?: is the common type of both branches, even the one not taken.
        ("return true ? -1 : 0u;", 2**32 - 1),
        ("return uint(-1);", 2**32 - 1),
    ],
)
def test_unsigned_integer_semantics(body, expected):
    assert _folded(f"uint f() {{ {body} }}\nconstant uint C = f();") == expected


def test_float_arithmetic_rounds_every_operation_to_binary32():
    accumulated = _folded("""
        float tenths() {
            float sum = 0.0;
            for (int i = 0; i < 10; i++) { sum += 0.1; }
            return sum;
        }
        constant float C = tenths();
        """)
    expected = 0.0
    for _ in range(10):
        expected = _float32(expected + _float32(0.1))
    assert accumulated == expected == _float32(1.0000001)
    assert _folded("float f() { return 1.0 / 3.0; }\nconstant float C = f();") == (
        _float32(1.0 / 3.0)
    )
    assert _folded(
        "float f() { return float(16777217); }\nconstant float C = f();"
    ) == (16777216.0)
    negative_zero = _folded("float f() { return -0.0; }\nconstant float C = f();")
    assert negative_zero == 0.0 and str(negative_zero) == "-0.0"


def test_narrow_and_wide_results_keep_their_type():
    assert _folded(
        "uint8_t f() { return uint8_t(300); }\nconstant uint8_t C = f();"
    ) == (
        "uint8_t",
        44,
    )
    assert _folded(
        "int64_t f() { return int64_t(-5); }\nconstant int64_t C = f();"
    ) == (
        "int64_t",
        -5,
    )


def test_overloads_resolve_by_argument_type():
    program = _program("""
        int f(int x) { return x + 1; }
        float f(float x) { return x * 2.0; }
        constant int C = f(2);
        constant float D = f(2.5);
        """)
    assert fold_constant_calls(program) == 2
    assert _value(_initializer(program, "C")) == 3
    assert _value(_initializer(program, "D")) == 5.0


def test_stage_constants_call_stage_and_shader_functions():
    source = textwrap.dedent("""
        shader main {
            int twice(int x) { return 2 * x; }
            compute {
                constant int BASE = 4;
                int plus_one(int x) { return x + 1; }
                constant int C = plus_one(twice(BASE));
                RWStructuredBuffer<int> out_ @buffer(0);
                @numthreads(1, 1, 1)
                void main(uint3 gid @gl_GlobalInvocationID) { out_[gid.x] = C; }
            }
        }
        """)
    program = Parser(Lexer(source).get_tokens()).parse()
    assert fold_constant_calls(program) == 1
    stage = next(iter(program.stages.values()))
    folded = next(v for v in stage.local_variables if v.name == "C")
    assert _value(folded.initial_value) == 9


@pytest.mark.parametrize(
    "declarations",
    [
        # Signed overflow and division by zero are not constant expressions ([expr.const]/2.6).
        "int f() { return 2147483647 + 1; }\nconstant int C = f();",
        "int f(int d) { return 1 / d; }\nconstant int C = f(0);",
        "int f() { return 1 << 32; }\nconstant int C = f();",
        "int f() { return -1 << 1; }\nconstant int C = f();",
        "int f() { return int(3.0e9); }\nconstant int C = f();",
        "float f() { return 1.0e38 * 10.0; }\nconstant float C = f();",
        "float f() { return 1.0 / 0.0; }\nconstant float C = f();",
        # Narrow arithmetic promotes in C++ but not in GLSL or WGSL.
        "uint8_t f(uint8_t x) { return x + x; }\nconstant uint8_t C = f(uint8_t(200));",
        # A specialization constant takes the pipeline's value.
        "constant int N @constant_id(0) = 4;\nint f() { return N; }\nconstant int C = f();",
        "int f() { int x; return x; }\nconstant int C = f();",
        "int f() { for (;;) {} return 0; }\nconstant int C = f();",
        "int f(int n) { return f(n + 1); }\nconstant int C = f(0);",
        "void g(out int x) { x = 1; }\nint f() { int x = 0; g(x); return x; }\nconstant int C = f();",
        "int max(int a, int b) { return a; }\nconstant int C = max(1, 2);",
        "half f() { return half(1.0); }\nconstant half C = f();",
        "int f() { return 1; }\nint f(int x = 2) { return x; }\nconstant int C = f();",
    ],
)
def test_calls_the_evaluator_cannot_compute_stay(declarations):
    program = _program(declarations)
    fold_constant_calls(program)
    assert isinstance(_initializer(program, "C"), A.FunctionCallNode)


def test_runtime_values_are_not_read():
    program = _program("""
        uniform int scale;
        int f() { return scale; }
        constant int C = f();
        """)
    fold_constant_calls(program)
    assert isinstance(_initializer(program, "C"), A.FunctionCallNode)


def test_static_assertions_see_the_values(tmp_path):
    source = """
        shader main {
            int twice(int x) { return 2 * x; }
            static_assert(twice(3) == %s, "twice");
            compute {
                RWStructuredBuffer<int> out_ @buffer(0);
                @numthreads(1, 1, 1)
                void main(uint3 gid @gl_GlobalInvocationID) { out_[gid.x] = 1; }
            }
        }
        """
    holds, fails = tmp_path / "holds.cgl", tmp_path / "fails.cgl"
    holds.write_text(source % "6")
    fails.write_text(source % "7")
    for target in ("opengl", "metal"):
        crosstl.translate(str(holds), backend=target)
        with pytest.raises(StaticAssertionFailedError):
            crosstl.translate(str(fails), backend=target)


METAL_SOURCE = """
#include <metal_stdlib>
using namespace metal;
constant constexpr int base = 3;
constexpr int twice(int x) { return 2 * x; }
constexpr uint mask(uint bits) {
    uint m = 0u;
    for (uint i = 0u; i < bits; ++i) { m = (m << 1u) | 1u; }
    return m;
}
constant constexpr int C = twice(base);
constant constexpr uint M = mask(12u);
constant constexpr float H = 1.0f / float(twice(2));
kernel void k(device uint* out [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    out[id] = uint(C) + M + uint(H * 4.0f) + uint(twice(int(id)));
}
"""


@pytest.mark.parametrize("target", TARGETS)
def test_metal_constexpr_calls_reach_every_target(tmp_path, target):
    source = tmp_path / "constexpr.metal"
    source.write_text(METAL_SOURCE)
    crossgl = tmp_path / "constexpr.cgl"
    crossgl.write_text(crosstl.translate(str(source), backend="cgl"))
    output = crosstl.translate(str(crossgl), backend=target)
    assert "4095" in output
    if target != "vulkan":
        # The run-time call stays a call.
        assert "twice(" in output
        assert "twice(base)" not in output and "mask(12" not in output
