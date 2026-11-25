"""Evaluate the portable math helpers' CrossGL text over binary32 operands and measure it against numpy.

``crosstl/translator/semantics/portable_math.py`` defines ``sinpi``, ``cospi``, ``fdim``, ``nextafter``, ``ilogb``
and ``copysign`` for targets that lack them. This module evaluates the parsed helpers themselves, one numpy lane per
operand, with the integer evaluator of :mod:`.libm_exhaustive` extended by what those helpers use besides integer
arithmetic: ``int`` as int32, correctly rounded binary32 ``+``, ``-`` and ``*`` without contraction, binary32
comparisons, ``floor`` and the conversions between ``float``, ``int`` and ``uint``. The references are numpy's binary32
``nextafter``, ``copysign`` and ``frexp`` and an exact binary64 reduction for the pi functions.

``python -m tests.conformance.portable_math_eval`` evaluates ``sinpi`` and ``cospi`` on every binary32 operand and
prints the largest error; the unit tests evaluate every helper on a stratified sample.
"""

from __future__ import annotations

import argparse
import functools
import sys

import numpy as np

from crosstl.translator import ast as A
from crosstl.translator.semantics import portable_math

from .libm_exhaustive import _COMPARISONS, Evaluator, Unmodelled, sample_bits

U32, I32, F32 = np.uint32, np.int32, np.float32
_ARITHMETIC = {"+": np.add, "-": np.subtract, "*": np.multiply}
_BITWISE = {"&": np.bitwise_and, "|": np.bitwise_or, "^": np.bitwise_xor}


class PortableEvaluator(Evaluator):
    """:class:`Evaluator` with ``int`` and binary32 arithmetic."""

    def call(self, name, arguments, mask):
        if name in ("int", "uint", "float") and name not in self.functions:
            (value,) = arguments
            return self.convert(name, np.asarray(value), mask)
        function = self.functions.get(name)
        if function is not None and not mask.any():
            # No lane calls it: the select or branch around the call discards the value.
            dtype = {"int": I32, "uint": U32, "float": F32}[
                getattr(function.return_type, "name", None)
            ]
            return np.zeros(mask.shape, dtype)
        return super().call(name, arguments, mask)

    @staticmethod
    def convert(name, value, mask):
        dtype = {"int": I32, "uint": U32, "float": F32}[name]
        if value.dtype == F32 and dtype != F32:
            active = np.broadcast_to(value, mask.shape)[mask]
            if np.any(~np.isfinite(active)) or np.any(np.abs(active) >= 2.0**31):
                raise Unmodelled(f"{name}() of a binary32 value outside the int range")
            # Lanes outside the mask hold anything; they convert as zero.
            value = np.where(
                np.isfinite(value) & (np.abs(value) < 2.0**31), value, F32(0)
            )
        if value.dtype in (U32, I32) and dtype in (U32, I32):
            return value.view(dtype)
        return value.astype(dtype)

    def builtin(self, name, arguments):
        if name == "floor":
            (value,) = arguments
            return np.floor(self.need(value, F32))
        return super().builtin(name, arguments)

    @staticmethod
    def typed(value, type_node):
        if getattr(type_node, "name", None) == "int":
            return Evaluator.need(value, I32)
        return Evaluator.typed(value, type_node)

    def eval(self, node, mask, scope):
        if isinstance(node, A.UnaryOpNode) and node.operator == "-":
            operand = np.asarray(self.eval(node.operand, mask, scope))
            if operand.dtype not in (F32, I32):
                raise Unmodelled("negation of a uint")
            return -operand
        return super().eval(node, mask, scope)

    @staticmethod
    def literal(node):
        name = getattr(node.literal_type, "name", None)
        if (
            name == "int"
            and isinstance(node.value, int)
            and -(2**31) <= node.value < 2**31
        ):
            return I32(node.value)
        if name in ("float", "double") and isinstance(node.value, (int, float)):
            return F32(node.value)
        return Evaluator.literal(node)

    def binary(self, operator, left, right, mask):
        left, right = np.asarray(left), np.asarray(right)
        kinds = {left.dtype, right.dtype}
        if kinds not in ({np.dtype(F32)}, {np.dtype(I32)}):
            return super().binary(operator, left, right, mask)
        if operator in _COMPARISONS:
            return _COMPARISONS[operator](left, right)
        function = _ARITHMETIC.get(operator) or (
            _BITWISE.get(operator) if left.dtype == I32 else None
        )
        if function is None:
            raise Unmodelled(
                f"the operator {operator} on {left.dtype.name} is not modelled"
            )
        if left.dtype == I32:
            wide = function(left.astype(np.int64), right.astype(np.int64))
            if np.any((wide < -(2**31)) | (wide >= 2**31)):
                raise Unmodelled(f"a lane overflows int with {operator}")
            return wide.astype(I32)
        return function(left, right, dtype=F32)


@functools.lru_cache(maxsize=None)
def evaluator():
    from crosstl.translator.source_registry import (
        SOURCE_REGISTRY,
        register_default_sources,
    )

    register_default_sources()
    program = SOURCE_REGISTRY.get("cgl").parse(
        portable_math.helper_program_source(list(portable_math.FUNCTIONS))
    )
    return PortableEvaluator(program.functions)


def evaluate(name, *operands):
    """The binary32 helper of ``name`` on numpy binary32 operand arrays."""
    operands = [np.asarray(operand, F32) for operand in operands]
    mask = np.ones(operands[0].shape, bool)
    with np.errstate(over="ignore", invalid="ignore"):
        return evaluator().call(portable_math.PREFIX + name, operands, mask)


def pi_reference(x, sine):
    """sinpi or cospi of binary32 ``x`` in binary64: ``x - n/2`` is exact for ``n = round(2x)``."""
    x = np.asarray(x, F32).astype(np.float64)
    with np.errstate(invalid="ignore"):
        nearest = np.where(np.isfinite(x), np.round(2.0 * x), 0.0)
        t = x - nearest / 2.0
        quadrant = np.mod(nearest, 4.0).astype(np.int64)
        s, c = np.sin(np.pi * t), np.cos(np.pi * t)
        if sine:
            value = np.where(quadrant % 2 == 0, s, c)
            value = np.where(quadrant >= 2, -value, value)
            value = np.where((t == 0) & (quadrant % 2 == 0), np.copysign(0.0, x), value)
        else:
            value = np.where(quadrant % 2 == 0, c, s)
            value = np.where((quadrant == 1) | (quadrant == 2), -value, value)
            value = np.where((t == 0) & (quadrant % 2 == 1), 0.0, value)
    return np.where(np.isfinite(x), value, np.nan)


def ulp_error(result, reference):
    """The error of binary32 ``result`` in ulps of the binary64 ``reference`` (subnormal ulps below 2^-126)."""
    exponent = np.floor(np.log2(np.maximum(np.abs(reference), 2.0**-126)))
    return np.abs(result.astype(np.float64) - reference) / np.exp2(exponent - 23)


def pi_check(name, bits):
    """The largest ulp error of ``name`` (``sinpi`` or ``cospi``) on the patterns ``bits``, and whether the zeros,
    signs and NaNs are exact."""
    x = np.asarray(bits, U32).view(F32)
    result = evaluate(name, x)
    reference = pi_reference(x, name == "sinpi")
    nan = np.isnan(reference)
    exact = bool(np.all(np.isnan(result[nan])))
    zero = reference == 0
    exact &= bool(np.array_equal(np.signbit(result[zero]), np.signbit(reference[zero])))
    exact &= bool(np.all(result[zero] == 0))
    error = np.zeros(x.shape)
    error[~nan] = ulp_error(result[~nan], reference[~nan])
    worst = int(np.argmax(error))
    return float(error[worst]), int(np.asarray(bits)[worst]), exact


def stratified_bits(per_exponent=64, seed=0):
    return sample_bits(per_exponent, seed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-bits", type=int, default=24)
    args = parser.parse_args(argv)
    step = 1 << args.chunk_bits
    for name in ("sinpi", "cospi"):
        worst, at, exact = 0.0, 0, True
        for start in range(0, 1 << 32, step):
            bits = np.arange(start, start + step, dtype=np.uint64).astype(U32)
            error, pattern, ok = pi_check(name, bits)
            exact &= ok
            if error > worst:
                worst, at = error, pattern
        print(
            f"{name}: {worst:.4f} ulp at 0x{at:08x}; zeros, signs and NaNs exact: {exact}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
