"""The exhaustive libm checker: its integer model, its refusals, and the certificates it records."""

import numpy as np
import pytest

from crosstl.translator.semantics import rule_table
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources

from . import libm_exhaustive
from .libm_exhaustive import Evaluator, Unmodelled


def _functions(text):
    register_default_sources()
    return SOURCE_REGISTRY.get("cgl").parse("shader T {\n" + text + "\n}\n").functions


def _run(text, name, *arguments):
    mask = np.ones(np.asarray(arguments[0]).shape, bool)
    return Evaluator(_functions(text)).call(name, list(arguments), mask)


def test_unsigned_arithmetic_wraps_and_branches_are_per_lane():
    text = """
uint f(uint a) {
    uint b = a * 0x9e3779b9u + 7u;
    if ((b >> 31u) == 0u) { return 0u - b; }
    return b ^ 0xffffffffu;
}
"""
    a = np.array([0, 1, 0xFFFFFFFF, 12345], np.uint32)
    b = (a.astype(np.uint64) * 0x9E3779B9 + 7) % (1 << 32)
    want = np.where(b >> 31 == 0, (-b) % (1 << 32), b ^ 0xFFFFFFFF)
    assert np.array_equal(_run(text, "f", a), want.astype(np.uint32))


def test_a_shift_by_32_in_an_executed_lane_is_undefined():
    text = """
uint f(uint a) {
    if (a > 4u) { return a << a; }
    return a;
}
"""
    assert np.array_equal(_run(text, "f", np.array([1, 3], np.uint32)), [1, 3])
    with pytest.raises(Unmodelled, match="32 or more"):
        _run(text, "f", np.array([1, 40], np.uint32))


def test_floating_point_arithmetic_is_outside_the_base():
    text = "float f(float a) { return a * a; }"
    with pytest.raises(Unmodelled):
        _run(text, "f", np.array([1.5], np.float32))


def test_a_function_that_can_end_without_returning_is_refused():
    text = "uint f(uint a) { if (a == 0u) { return 1u; } }"
    with pytest.raises(Unmodelled, match="without a return"):
        _run(text, "f", np.array([0, 1], np.uint32))


def test_the_sample_covers_every_exponent_and_both_signs():
    bits = libm_exhaustive.sample_bits(4)
    exponents = set(((bits >> 23) & 0xFF).tolist())
    assert exponents == set(range(256))
    assert {int(b >> 31) for b in bits} == {0, 1}


@pytest.mark.parametrize("intrinsic", sorted(rule_table.LIBM_INTRINSICS))
def test_each_certificate_proves_the_current_helper_text(intrinsic):
    certificate = libm_exhaustive.load_proofs()[intrinsic]
    assert certificate["helperSHA256"] == libm_exhaustive.helper_sha256(
        intrinsic
    ), "the helper text changed: run python -m tests.conformance.libm_exhaustive"
    assert certificate["operands"] == 1 << 32
    assert certificate["maxUlp"] <= rule_table.LIBM_ULP_BOUND
