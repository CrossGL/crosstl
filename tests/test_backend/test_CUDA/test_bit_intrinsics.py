"""CUDA and HIP ``__clz``, ``__clzll``, ``__byte_perm`` and ``__funnelshift_*`` import as CrossGL bit operations.

The CUDA importer refused ``__byte_perm`` with any selector but ``0x0123`` and every ``__funnelshift_*``, and both
importers wrote ``__clz`` as ``countLeadingZeros``, which CrossGL does not declare. HIP's ``__byte_perm`` read all
four selector bits, so a nibble from 8 to 15 gave 0 where HIP and CUDA read byte ``nibble & 7``. Each test parses the
CrossGL the importer writes and evaluates it with C integer semantics, against a model of the CUDA Math API
definitions; the evaluator also fails on any shift by the operand width or more, which C++, GLSL and SPIR-V leave
undefined.
"""

from __future__ import annotations

import random
import re
import shutil
import subprocess
from typing import NamedTuple

import pytest

import crosstl
from crosstl.backend.CUDA.bit_intrinsics import format_bit_intrinsic
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

M32 = (1 << 32) - 1


def _clz32(x):
    return 32 - (x & M32).bit_length()


def _clz64(w):
    return 64 - (w & ((1 << 64) - 1)).bit_length()


def _byte_perm(x, y, s):
    value = (y << 32) | x
    return sum(
        ((value >> (8 * ((s >> (4 * n)) & 7))) & 0xFF) << (8 * n) for n in range(4)
    )


def _funnel_left(lo, hi, shift):
    return ((((hi << 32) | lo) << shift) >> 32) & M32


def _funnel_right(lo, hi, shift):
    return (((hi << 32) | lo) >> shift) & M32


# CUDA Math API, "Integer Intrinsics": each call over x, y, s (unsigned int), sx = (int)x and w (long long, y:x).
CASES = {
    "__clz(sx)": lambda x, y, s: _clz32(x),
    "__clz(sx >> 3)": lambda x, y, s: _clz32((_signed(x) >> 3) & M32),
    "__clzll(w)": lambda x, y, s: _clz64((y << 32) | x),
    "__byte_perm(x, y, s)": _byte_perm,
    "__byte_perm(x, y, 0x3210u)": lambda x, y, s: x,
    "__byte_perm(x, y, 0x0123)": lambda x, y, s: _byte_perm(x, y, 0x0123),
    "__byte_perm(x, y, 0x89AB)": lambda x, y, s: _byte_perm(x, y, 0x89AB),
    "__byte_perm(x, y, 0x7546)": lambda x, y, s: _byte_perm(x, y, 0x7546),
    "__funnelshift_l(x, y, s)": lambda x, y, s: _funnel_left(x, y, s & 31),
    "__funnelshift_lc(x, y, s)": lambda x, y, s: _funnel_left(x, y, min(s, 32)),
    "__funnelshift_r(x, y, s)": lambda x, y, s: _funnel_right(x, y, s & 31),
    "__funnelshift_rc(x, y, s)": lambda x, y, s: _funnel_right(x, y, min(s, 32)),
    "__funnelshift_l(x, y, 0u)": lambda x, y, s: y,
    "__funnelshift_l(x, y, 7u)": lambda x, y, s: _funnel_left(x, y, 7),
    "__funnelshift_lc(x, y, 40u)": lambda x, y, s: x,
    "__funnelshift_r(x, y, 39u)": lambda x, y, s: _funnel_right(x, y, 7),
    "__funnelshift_rc(x, y, 32u)": lambda x, y, s: y,
    # An operand with side effects is read once, in the 64-bit form.
    "__funnelshift_l(x, y, same(s))": lambda x, y, s: _funnel_left(x, y, s & 31),
    "__funnelshift_lc(x, y, same(s))": lambda x, y, s: _funnel_left(x, y, min(s, 32)),
    "__funnelshift_r(same(x), y, s)": lambda x, y, s: _funnel_right(x, y, s & 31),
    "__funnelshift_rc(x, same(y), s)": lambda x, y, s: _funnel_right(x, y, min(s, 32)),
    "__funnelshift_r(x, same(y), 0u)": lambda x, y, s: x,
}

KERNEL = """
__device__ unsigned int same(unsigned int v) {{ return v; }}

extern "C" __global__ void probe(unsigned int* uin, unsigned int* outp) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int x = uin[i];
    unsigned int y = uin[i + 1u];
    unsigned int s = uin[i + 2u];
    int sx = (int)x;
    long long w = (long long)(((unsigned long long)y << 32) | (unsigned long long)x);
{body}
}}
"""


def _signed(x):
    return x - (1 << 32) if x & 0x80000000 else x


def _inputs():
    edges = (0, 1, 0x80000000, M32, 0x12345678)
    shifts = (0, 1, 7, 31, 32, 33, 40, 0x7777, 0x8888, M32, 0x89AB3210)
    cases = [(x, y, s) for x in edges for y in (0, M32, 0x9ABCDEF0) for s in shifts]
    rng = random.Random(4821)
    for _ in range(200):
        shift = rng.choice((rng.getrandbits(32), rng.randrange(0, 70)))
        cases.append((rng.getrandbits(32), rng.choice((0, rng.getrandbits(32))), shift))
    return cases


INPUTS = _inputs()


class Value(NamedTuple):
    value: int
    bits: int
    signed: bool


def _wrap(value, bits, signed):
    value &= (1 << bits) - 1
    if signed and value >> (bits - 1):
        value -= 1 << bits
    return Value(value, bits, signed)


def _common(left, right):
    """C's usual arithmetic conversions of two integer operands (C++ [expr.arith.conv])."""
    bits = max(left.bits, right.bits)
    if left.bits != right.bits:
        signed = (left if left.bits > right.bits else right).signed
    else:
        signed = left.signed and right.signed
    return _wrap(left.value, bits, signed), _wrap(right.value, bits, signed)


_COMPARISONS = {
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "!=": lambda a, b: a != b,
    "==": lambda a, b: a == b,
}
_ARITHMETIC = {
    "&": lambda a, b: a & b,
    "|": lambda a, b: a | b,
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
}


def _evaluate(node, env):
    kind = type(node).__name__
    if kind == "LiteralNode":
        return _wrap(node.value, 32, node.literal_type.name == "int")
    if kind == "IdentifierNode":
        return env[node.name]
    if kind == "TernaryOpNode":
        # Both branches are evaluated, as SPIR-V OpSelect does, so neither may shift out of range.
        condition = _evaluate(node.condition, env)
        taken = _evaluate(node.true_expr, env)
        other = _evaluate(node.false_expr, env)
        return taken if condition.value else other
    if kind == "BinaryOpNode":
        left, right = _evaluate(node.left, env), _evaluate(node.right, env)
        if node.operator in ("<<", ">>"):
            assert 0 <= right.value < left.bits, f"shift by {right.value}"
            shifted = (
                left.value << right.value
                if node.operator == "<<"
                else left.value >> right.value
            )
            return _wrap(shifted, left.bits, left.signed)
        left, right = _common(left, right)
        if node.operator in _COMPARISONS:
            return Value(
                int(_COMPARISONS[node.operator](left.value, right.value)), 1, False
            )
        result = _ARITHMETIC[node.operator](left.value, right.value)
        return _wrap(result, left.bits, left.signed)
    if kind == "FunctionCallNode":
        name = node.function.name
        args = [_evaluate(argument, env) for argument in node.arguments]
        if name == "u32":
            return _wrap(args[0].value, 32, False)
        if name == "u64":
            return _wrap(args[0].value, 64, False)
        if name == "findMSB":
            assert args[0].bits == 32 and not args[0].signed
            return _wrap(args[0].value.bit_length() - 1, 32, True)
        if name == "min":
            left, right = _common(*args)
            return left if left.value <= right.value else right
        if name == "same":
            return args[0]
    raise AssertionError(f"unexpected CrossGL node {kind}")


def _parse_expression(text):
    program = Parser(Lexer(f"fn f() -> u32 {{ return {text}; }}").tokens).parse()
    return program.functions[0].body.statements[0].value


def _check(expression, model):
    node = _parse_expression(expression)
    for x, y, s in INPUTS:
        env = {
            "x": Value(x, 32, False),
            "y": Value(y, 32, False),
            "s": Value(s, 32, False),
            "sx": _wrap(x, 32, True),
            "w": _wrap((y << 32) | x, 64, True),
        }
        got = _evaluate(node, env).value & M32
        want = model(x, y, s) & M32
        assert got == want, f"{expression} at x={x:#x} y={y:#x} s={s:#x}"


def _import(tmp_path, language, calls):
    body = "\n".join(
        f"    outp[{index}u * i] = (unsigned int){call};"
        for index, call in enumerate(calls)
    )
    source = KERNEL.format(body=body)
    if language == "hip":
        source = "#include <hip/hip_runtime.h>\n" + source.replace('extern "C" ', "")
    path = tmp_path / ("probe.cu" if language == "cuda" else "probe.hip")
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="cgl", format_output=False)


def _assigned_expressions(crossgl, count):
    expressions = {}
    for line in crossgl.splitlines():
        match = re.match(r"\s*outp\[\((\d+)u \* i\)\] = (.*);$", line)
        if match:
            expressions[int(match.group(1))] = match.group(2)
    assert sorted(expressions) == list(range(count)), crossgl
    return [expressions[index] for index in range(count)]


@pytest.mark.parametrize("language", ["cuda", "hip"])
def test_the_imported_intrinsics_compute_what_cuda_defines(tmp_path, language):
    calls = list(CASES)
    crossgl = _import(tmp_path, language, calls)
    for call, expression in zip(calls, _assigned_expressions(crossgl, len(calls))):
        _check(expression, CASES[call])
    assert "countLeadingZeros" not in crossgl
    assert not re.search(r"__(clz|byte_perm|funnelshift)", crossgl)


def test_cuda_and_hip_import_the_same_crossgl(tmp_path):
    calls = list(CASES)
    cuda = _assigned_expressions(_import(tmp_path, "cuda", calls), len(calls))
    hip = _assigned_expressions(_import(tmp_path, "hip", calls), len(calls))
    assert cuda == hip


def test_operands_without_side_effects_stay_in_32_bit_words(tmp_path):
    calls = [call for call in CASES if "same(" not in call and "clzll" not in call]
    crossgl = _import(tmp_path, "cuda", calls)
    for expression in _assigned_expressions(crossgl, len(calls)):
        assert "u64" not in expression, expression


@pytest.mark.parametrize("language", ["cuda", "hip"])
@pytest.mark.parametrize(
    "call, operand",
    [
        ("__clzll(w++)", "x"),
        ("__byte_perm(uin[i++], y, s)", "x"),
        ("__byte_perm(x, same(y), 0x3210u)", "y"),
        ("__byte_perm(x, y, same(s))", "s"),
    ],
)
def test_an_operand_read_twice_must_have_no_side_effects(
    tmp_path, language, call, operand
):
    with pytest.raises(Exception) as refusal:
        _import(tmp_path, language, [call])
    assert type(refusal.value).__name__ in {"CudaImportRefused", "HipImportRefused"}
    assert f"reads its operand {operand} more than once" in str(refusal.value)


class _Converter:
    def __init__(self):
        self.refusals = []

    def refuse_import(self, reason, placeholder=""):
        self.refusals.append(reason)
        return placeholder


def test_other_calls_are_left_to_the_importer():
    converter = _Converter()
    assert format_bit_intrinsic(converter, "__popc", ["x"], ["x"], "CUDA") is None
    assert (
        format_bit_intrinsic(converter, "__clz", ["x", "y"], ["x", "y"], "CUDA") is None
    )
    assert format_bit_intrinsic(converter, "::__clz", ["x"], ["x"], "CUDA") == (
        "(31 - findMSB(u32(x)))"
    )
    assert converter.refusals == []


@pytest.mark.parametrize("language", ["cuda", "hip"])
def test_fdivide_is_a_division(tmp_path, language):
    source = (
        'extern "C" __global__ void probe(float* fin, float* outp) {\n'
        "    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    double a = (double)fin[i];\n"
        "    double b = (double)fin[i + 1u];\n"
        "    outp[i] = (float)fdivide(a, b) + fdividef(fin[i], fin[i + 1u]);\n"
        "}\n"
    )
    if language == "hip":
        source = "#include <hip/hip_runtime.h>\n" + source.replace('extern "C" ', "")
    path = tmp_path / ("probe.cu" if language == "cuda" else "probe.hip")
    path.write_text(source, encoding="utf-8")
    crossgl = crosstl.translate(str(path), backend="cgl", format_output=False)
    assert "(a / b)" in crossgl
    assert "(fin[i] / fin[(i + 1u)])" in crossgl
    assert "fdivide" not in crossgl


@pytest.mark.parametrize(
    "language, tool, arguments",
    [
        ("cuda", "nvcc", ["-std=c++17"]),
        ("hip", "hipcc", ["--offload-arch=gfx1100", "-std=c++17"]),
    ],
)
def test_the_round_trip_compiles(tmp_path, language, tool, arguments):
    compiler = shutil.which(tool)
    if compiler is None:
        pytest.skip(f"{tool} is not installed")
    calls = list(CASES)
    body = "\n".join(
        f"    outp[{index}u * i] = (unsigned int){call};"
        for index, call in enumerate(calls)
    )
    source = KERNEL.format(body=body)
    if language == "hip":
        source = "#include <hip/hip_runtime.h>\n" + source.replace('extern "C" ', "")
    native = tmp_path / ("probe.cu" if language == "cuda" else "probe.hip")
    native.write_text(source, encoding="utf-8")
    generated = tmp_path / ("round_trip.cu" if language == "cuda" else "round_trip.hip")
    generated.write_text(
        crosstl.translate(str(native), backend=language), encoding="utf-8"
    )
    result = subprocess.run(
        [
            compiler,
            *arguments,
            "-c",
            str(generated),
            "-o",
            str(tmp_path / "round_trip.o"),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
