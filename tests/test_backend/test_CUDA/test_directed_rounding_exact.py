"""The CrossGL functions behind CUDA's directed-rounding intrinsics compute the exact IEEE 754 result.

The functions the importers emit are parsed with the CrossGL parser and run by a strict interpreter of their integer
subset (crossgl_integer_interpreter.py), then compared bit for bit with an independent reference that rounds the exact
rational result once (ieee_reference.py). The inputs are every pair of special values (signed zeros, subnormals, the
largest finite value, infinities, NaN), cancellations, products that the addend nearly cancels, and seeded random bit
patterns weighted toward the subnormal and overflow ranges.
"""

from __future__ import annotations

import math
import random
import struct
from fractions import Fraction
from functools import lru_cache
from itertools import product

import pytest

from crosstl.backend.CUDA.directed_rounding_helpers import DirectedRoundingLibrary

from . import ieee_reference as reference
from .crossgl_integer_interpreter import Interpreter, InterpreterError

MODES = ("rd", "ru", "rz")
ARITHMETIC = {
    "fadd": ("f32", reference.add),
    "fsub": ("f32", reference.sub),
    "fmul": ("f32", reference.mul),
    "fdiv": ("f32", reference.div),
    "frcp": ("f32", reference.rcp),
    "fsqrt": ("f32", reference.sqrt),
    "fmaf": ("f32", reference.fma),
    "fmaf_ieee": ("f32", reference.fma),
    "dadd": ("f64", reference.add),
    "dsub": ("f64", reference.sub),
    "dmul": ("f64", reference.mul),
    "ddiv": ("f64", reference.div),
    "drcp": ("f64", reference.rcp),
    "dsqrt": ("f64", reference.sqrt),
    "fma": ("f64", reference.fma),
}
ARITY = {
    "frcp": 1,
    "fsqrt": 1,
    "drcp": 1,
    "dsqrt": 1,
    "fmaf": 3,
    "fmaf_ieee": 3,
    "fma": 3,
}
CONVERSIONS = {
    "int2float": ("i32", "f32"),
    "uint2float": ("u32", "f32"),
    "ll2float": ("i64", "f32"),
    "ull2float": ("u64", "f32"),
    "ll2double": ("i64", "f64"),
    "ull2double": ("u64", "f64"),
}


@lru_cache(maxsize=None)
def library():
    functions = DirectedRoundingLibrary()
    for stem in list(ARITHMETIC) + list(CONVERSIONS) + ["double2float"]:
        for mode in MODES:
            name = f"__{stem}_{mode}"
            functions.call(name, ["x"] * functions.arity(name))
    return functions, Interpreter(functions.render())


def run(stem, mode, *arguments):
    functions, interpreter = library()
    return interpreter.call(functions.wrappers[f"__{stem}_{mode}"], *arguments)


def bits_of(fmt):
    width, precision, exponent_bits = reference.FORMATS[fmt]
    sign = 1 << (width - 1)
    infinity = ((1 << exponent_bits) - 1) << (precision - 1)
    one = ((1 << (exponent_bits - 1)) - 1) << (precision - 1)
    return width, precision, exponent_bits, sign, infinity, one


def specials(fmt, small=False):
    width, precision, _bits, sign, infinity, one = bits_of(fmt)
    values = [0, 1, one, infinity - 1, infinity, infinity | (1 << (precision - 2))]
    if not small:
        values += [3, 1 << (precision - 1), one + 1, (1 << (precision - 1)) - 1]
    return values + [value | sign for value in values]


def random_bits(fmt, rng):
    width, precision, exponent_bits, _sign, _inf, _one = bits_of(fmt)
    if rng.random() < 0.4:
        return rng.getrandbits(width)
    bias = (1 << (exponent_bits - 1)) - 1
    field = rng.choice(
        [
            rng.randrange(bias - 30, bias + 30),
            rng.randrange(0, 40),
            rng.randrange((1 << exponent_bits) - 40, (1 << exponent_bits) - 1),
        ]
    )
    mantissa = rng.getrandbits(precision - 1)
    if rng.random() < 0.25:
        mantissa &= ~((1 << rng.randrange(precision - 1)) - 1)
    return (rng.getrandbits(1) << (width - 1)) | (field << (precision - 1)) | mantissa


def arithmetic_cases(stem, fmt, rng, count=40):
    arity = ARITY.get(stem, 2)
    sign = bits_of(fmt)[3]
    cases = list(product(specials(fmt, small=arity == 3), repeat=arity))
    cases += [tuple(random_bits(fmt, rng) for _ in range(arity)) for _ in range(count)]
    if stem[1:] in ("add", "sub"):
        for _ in range(count // 2):
            a = random_bits(fmt, rng)
            partner = (a ^ sign) if stem[1:] == "add" else a
            cases.append(
                (a, (partner + rng.choice([-2, -1, 0, 1, 2])) % (1 << bits_of(fmt)[0]))
            )
    if arity == 3:
        for _ in range(count // 2):
            a, b = random_bits(fmt, rng), random_bits(fmt, rng)
            c = reference.mul(fmt, a, b, "rz") ^ sign
            cases.append((a, b, (c + rng.choice([-1, 0, 1])) % (1 << bits_of(fmt)[0])))
    return cases


def assert_same(fmt, got, want, case):
    assert (
        got == want
    ), f"{[hex(v) for v in case]}: got {got:#x}, want {want:#x} ({fmt})"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("stem", list(ARITHMETIC))
def test_arithmetic_is_exact(stem, mode):
    fmt, operation = ARITHMETIC[stem]
    rng = random.Random(f"{stem}-{mode}")
    for case in arithmetic_cases(stem, fmt, rng):
        result_type, got = run(stem, mode, *[(fmt, value) for value in case])
        assert result_type == fmt
        assert_same(fmt, got, operation(fmt, *case, mode), case)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("stem", list(CONVERSIONS))
def test_integer_conversions_are_exact(stem, mode):
    source, target = CONVERSIONS[stem]
    width = 32 if source in ("i32", "u32") else 64
    signed = source.startswith("i")
    low, high = (
        (-(1 << (width - 1)), (1 << (width - 1)) - 1)
        if signed
        else (0, (1 << width) - 1)
    )
    rng = random.Random(f"{stem}-{mode}")
    values = {
        low,
        high,
        0,
        1,
        -1 % (high + 1) if not signed else -1,
        (1 << 24) + 1,
        (1 << 53) + 1,
        (1 << 25) - 1,
    }
    values |= {rng.randint(low, high) >> rng.randrange(width) for _ in range(60)}
    for value in sorted(v for v in values if low <= v <= high):
        result_type, got = run(stem, mode, (source, value))
        assert result_type == target
        assert_same(target, got, reference.from_integer(target, value, mode), [value])


@pytest.mark.parametrize("mode", MODES)
def test_double_to_float_is_exact(mode):
    rng = random.Random(f"narrow-{mode}")
    cases = specials("f64") + [random_bits("f64", rng) for _ in range(60)]
    for _ in range(
        80
    ):  # around the float overflow threshold, the subnormal range and below it
        field = rng.choice(
            [
                rng.randrange(1023 - 160, 1023 - 120),
                rng.randrange(1023 + 125, 1023 + 130),
            ]
        )
        cases.append((rng.getrandbits(1) << 63) | (field << 52) | rng.getrandbits(52))
    for case in cases:
        result_type, got = run("double2float", mode, ("f64", case))
        assert result_type == "f32"
        assert_same("f32", got, reference.narrow(case, mode), [case])


@pytest.mark.parametrize("fmt", ["f32", "f64"])
def test_shifting_right_keeps_a_sticky_bit(fmt):
    functions, interpreter = library()
    width = 32 if fmt == "f32" else 64
    limb = "u32" if fmt == "f32" else "u64"
    name = functions.helper_name("jam", fmt)
    rng = random.Random(f"jam-{fmt}")
    shifts = sorted(
        {0, 1, width - 1, width, width + 1, 2 * width - 1, 2 * width, 2 * width + 7}
        | {rng.randrange(2 * width) for _ in range(12)}
    )
    for d in shifts:
        for _ in range(12):
            hi, lo = rng.getrandbits(width) >> rng.randrange(width), rng.getrandbits(
                width
            ) >> rng.randrange(width)
            wide = (hi << width) | lo
            shifted = wide >> d if d < 2 * width else 0
            if wide & ((1 << min(d, 2 * width)) - 1):
                shifted |= 1
            arguments = [(limb, hi), (limb, lo), ("i32", d), (limb, 0), (limb, 0)]
            _result, outputs = interpreter.call_with_outputs(name, *arguments)
            assert (outputs["rhi"], outputs["rlo"]) == (
                shifted >> width,
                shifted & ((1 << width) - 1),
            ), (hi, lo, d)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fmt", ["f32", "f64"])
def test_rounding_an_exact_significand(fmt, mode):
    functions, interpreter = library()
    width = 32 if fmt == "f32" else 64
    limb = "u32" if fmt == "f32" else "u64"
    name = functions.helper_name("round", fmt)
    rng = random.Random(f"round-{fmt}-{mode}")
    emin = -126 if fmt == "f32" else -1022
    emax = 127 if fmt == "f32" else 1023
    for _ in range(80):
        wide = rng.getrandbits(2 * width) >> rng.randrange(2 * width - 1) or 1
        lead = wide.bit_length() - 1
        target = rng.choice(
            [
                rng.randrange(emin - 60, emin + 5),
                rng.randrange(-20, 20),
                rng.randrange(emax - 3, emax + 3),
            ]
        )
        exponent = target - lead
        negative = rng.random() < 0.5
        value = Fraction(wide) * Fraction(2) ** exponent
        mode_value = MODES.index(mode)
        arguments = [
            ("bool", negative),
            ("i32", exponent),
            (limb, wide >> width),
            (limb, wide & ((1 << width) - 1)),
            ("i32", mode_value),
        ]
        result_type, got = interpreter.call(name, *arguments)
        assert result_type == fmt
        want = reference._Format(fmt).round(-value if negative else value, mode)
        assert_same(fmt, got, want, [wide, exponent])


def test_the_reference_brackets_the_nearest_result():
    rng = random.Random(5)
    for _ in range(400):
        a, b = (random_bits("f64", rng) for _ in range(2))
        x, y = (struct.unpack("<d", struct.pack("<Q", v))[0] for v in (a, b))
        if not all(map(math.isfinite, (x, y))) or y == 0:
            continue
        for operation, nearest in (
            (reference.add, x + y),
            (reference.mul, x * y),
            (reference.div, x / y),
        ):
            if not math.isfinite(nearest) or nearest == 0:
                continue
            down, up, zero = (
                struct.unpack("<d", struct.pack("<Q", operation("f64", a, b, m)))[0]
                for m in MODES
            )
            assert down <= nearest <= up and nearest in (down, up)
            assert zero == (down if nearest > 0 else up)
            assert down == up or math.nextafter(down, math.inf) == up
        if x > 0:
            down, up = (
                struct.unpack("<d", struct.pack("<Q", reference.sqrt("f64", a, m)))[0]
                for m in ("rd", "ru")
            )
            assert down <= math.sqrt(x) <= up and (
                down == up or math.nextafter(down, math.inf) == up
            )


def test_the_interpreter_rejects_what_targets_treat_differently():
    interpreter = Interpreter(
        "u32 wide(u32 x) { return x << 32u; }\n"
        "u32 mixed(u32 x, i32 y) { return x + y; }\n"
        "i32 overflow(i32 x) { return x + 1; }\n"
    )
    for name, argument in (("wide", ("u32", 1)), ("overflow", ("i32", 2**31 - 1))):
        with pytest.raises(InterpreterError):
            interpreter.call(name, argument)
    with pytest.raises(InterpreterError):
        interpreter.call("mixed", ("u32", 1), ("i32", 1))
