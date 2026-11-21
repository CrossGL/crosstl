"""CrossGL's integer combinatorics functions become CrossGL helpers on every target.

``gcd``, ``lcm``, ``factorial``, ``comb`` and ``perm`` are Mojo ``std.math`` functions with no counterpart in any
GPU language, so ``crosstl/translator/semantics/integer_math.py`` replaces each call with a helper written in
unsigned CrossGL arithmetic. ``tests/conformance/cases/x_integer_combinatorics`` executes the helpers against
Python's ``math`` functions; these tests pin the rewriting and its refusals.
"""

from __future__ import annotations

import math
import re

import pytest

import crosstl
from crosstl.translator.semantics.builtin_function_names import BUILTIN_FUNCTION_NAMES
from crosstl.translator.semantics.integer_math import (
    FUNCTIONS,
    PREFIX,
    IntegerMathError,
    helper_program_source,
)

TARGETS = ("opengl", "vulkan", "directx", "metal", "cuda", "hip", "mojo", "slang")


def _program(body, functions=""):
    return f"""
shader IntegerMath {{
    {functions}
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* iin;
        buffer int* outp;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            int a = iin[i];
            int b = iin[i + 1u];
            {body}
        }}
    }}
}}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "integer_math.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


_EVERY_FORM = """
int64_t wide = gcd(int64_t(a), int64_t(b)) + lcm(int64_t(a), int64_t(b)) + comb(int64_t(a), int64_t(b))
    + perm(int64_t(a)) + perm(int64_t(a), int64_t(b)) + factorial(int64_t(b));
uint u = gcd(uint(a), 6u) + lcm(uint(a), uint(b)) + comb(uint(a), 3u) + perm(uint(a), 2u) + factorial(uint(b));
ivec3 v = gcd(ivec3(a, b, 7), 4);
int16_t s = lcm(int16_t(a), int16_t(b));
outp[i] = int(wide) + int(u) + v.x + int(s) + comb(a, b) + perm(a);
"""


def test_every_name_is_a_crossgl_builtin():
    assert set(FUNCTIONS) <= BUILTIN_FUNCTION_NAMES


@pytest.mark.parametrize("backend", TARGETS)
def test_calls_become_helpers(tmp_path, backend):
    generated = _translate(tmp_path, _program(_EVERY_FORM), backend)
    if backend == "vulkan":
        # SPIR-V names no function; the helpers are functions the entry point calls.
        assert generated.count("OpFunctionCall") >= 10
        return
    for name in FUNCTIONS:
        # No call of the CrossGL function itself survives: every target lacks it or takes other operand types.
        assert not re.search(rf"(?<![\w.]){name}\s*\(", generated), name
    assert f"{PREFIX}gcd_i64(" in generated
    assert f"{PREFIX}gcd_u32(" in generated
    assert f"{PREFIX}factorial_u64(" in generated
    assert f"{PREFIX}gcd_i32_v3(" in generated


def test_narrow_operands_compute_in_32_bits_and_convert_back(tmp_path):
    generated = _translate(
        tmp_path,
        _program("int16_t s = lcm(int16_t(a), int16_t(b)); outp[i] = int(s);"),
        "metal",
    )
    assert re.search(rf"short\(\s*{PREFIX}lcm_i32\(int\(", generated)


def test_one_operand_perm_is_factorial(tmp_path):
    generated = _translate(tmp_path, _program("outp[i] = perm(a);"), "opengl")
    assert f"{PREFIX}factorial_i32(a)" in generated
    assert f"{PREFIX}perm_" not in generated


def test_mixed_operands_take_the_common_type(tmp_path):
    generated = _translate(
        tmp_path, _program("outp[i] = int(gcd(int64_t(a), b));"), "opengl"
    )
    assert re.search(rf"{PREFIX}gcd_i64\(int64_t\(a\), int64_t\(b\)\)", generated)


def test_a_program_function_of_the_same_name_is_called(tmp_path):
    functions = "int gcd(int x, int y) { return x + y; }"
    generated = _translate(
        tmp_path, _program("outp[i] = gcd(a, b);", functions), "opengl"
    )
    assert PREFIX not in generated
    assert "gcd(a, b)" in generated


def test_floating_point_operands_are_refused(tmp_path):
    with pytest.raises(IntegerMathError, match="integer operands"):
        _translate(tmp_path, _program("outp[i] = int(gcd(1.5, float(a)));"), "metal")


def test_another_number_of_operands_is_refused(tmp_path):
    with pytest.raises(IntegerMathError, match="takes 2 operands, not 3"):
        _translate(tmp_path, _program("outp[i] = gcd(a, b, a);"), "directx")


def test_the_reserved_prefix_is_refused(tmp_path):
    functions = f"int {PREFIX}gcd_i32(int x, int y) {{ return x; }}"
    with pytest.raises(IntegerMathError, match="reserved"):
        _translate(tmp_path, _program("outp[i] = gcd(a, b);", functions), "cuda")


def test_helpers_are_added_once_with_their_dependencies(tmp_path):
    generated = _translate(
        tmp_path, _program("outp[i] = comb(a, b) + comb(b, a) + lcm(a, b);"), "opengl"
    )
    # Each definition appears once (GLSL also writes a prototype ahead of the definitions).
    assert generated.count(f"uint {PREFIX}gcd_u32(uint a, uint b) {{") == 1
    assert generated.count(f"int {PREFIX}comb_i32(int n, int k) {{") == 1
    assert f"{PREFIX}magnitude_i32(" in generated


def test_helper_loops_are_bounded():
    """Every loop of a helper either is Euclid's (which shrinks its divisor) or counts to a constant bound."""
    source = helper_program_source(
        [
            f"{PREFIX}{name}_{kind}{bits}"
            for name in FUNCTIONS
            for kind in "iu"
            for bits in (32, 64)
        ]
    )
    loops = re.findall(r"(for|while) \(([^;{]*;)?([^;{]*)", source)
    assert loops
    for keyword, _, condition in loops:
        if keyword == "while":
            assert (
                condition.strip() == "y != uint(0))"
                or condition.strip() == "y != uint64_t(0))"
            )
        else:
            assert re.match(r"i <=? (21|34)u && ", condition.strip()), condition


@pytest.mark.parametrize(
    ("n", "bound"),
    [(64, 20), (32, 12)],
)
def test_the_bounds_cover_every_result_that_fits(n, bound):
    """21 factors and 34 comb steps reach every factorial, permutation count and binomial that fits 64 bits."""
    assert math.factorial(bound) < 2**n <= math.factorial(bound + 1)
    assert bound + 1 <= 21
    largest_comb_steps = max(
        k
        for k in range(1, 70)
        if any(math.comb(m, k) < 2**n for m in range(2 * k, 2 * k + 2))
    )
    assert largest_comb_steps < 34
