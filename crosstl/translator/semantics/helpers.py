"""Reserved helper functions for polyfills that use an operand more than once (ADR 0001, section 4).

A template in ``rule_table.POLYFILLS`` names each operand exactly once, so it can be substituted in place. A polyfill
that needs an operand twice, such as the zero test and the division of the ``udiv_zero`` family, is a helper
function instead: the operation becomes a call, which evaluates each operand once, and the lowering pass injects the
helper into the program. Vector operations call a per-lane lift of the scalar helper. A helper whose body calls
further reserved functions (``Helper.support``, such as the libm family's integer evaluation) brings them along:
they are injected once per program, ahead of every helper.
"""

from __future__ import annotations

import functools

from .hub_types import iter_functions
from .rewrite import call, clone
from .rule_table import HELPERS

HELPER_MARK = "cgl_rule_helper"
_VECTOR_PREFIX = {"uint": "uvec", "int": "ivec", "float": "vec"}
_LANE_NAMES = "xyzw"


def helper_name(intrinsic: str, lanes: int) -> str:
    base = HELPERS[intrinsic].name
    return base if lanes == 1 else f"{base}{lanes}"


def canonical_intrinsic(intrinsic: str) -> str:
    """The first intrinsic whose polyfill is ``intrinsic``'s helper, so intrinsics sharing one are defined once."""
    helper = HELPERS[intrinsic]
    return next(name for name, other in HELPERS.items() if other is helper)


def helper_source(intrinsic: str, lanes: int) -> str:
    """The CrossGL definition of the helper for ``lanes`` lanes."""
    helper = HELPERS[intrinsic]
    parameters = helper.parameter_types
    names = [f"x{index}" for index in range(len(parameters))]
    if lanes == 1:
        signature = ", ".join(f"{t} {n}" for t, n in zip(parameters, names))
        return (
            f"{helper.result_type} {helper.name}({signature}) "
            f"{{ return {helper.body}; }}"
        )

    def vector(scalar):
        return f"{_VECTOR_PREFIX[scalar]}{lanes}"

    signature = ", ".join(f"{vector(t)} {n}" for t, n in zip(parameters, names))
    per_lane = ", ".join(
        f"{helper.name}({', '.join(f'{n}.{lane}' for n in names)})"
        for lane in _LANE_NAMES[:lanes]
    )
    result = vector(helper.result_type)
    return (
        f"{result} {helper_name(intrinsic, lanes)}({signature}) "
        f"{{ return {result}({per_lane}); }}"
    )


def helper_program_source(intrinsic: str) -> str:
    """A CrossGL program defining the scalar helper of ``intrinsic`` and every function it calls."""
    support = HELPERS[intrinsic].support or ""
    return (
        "shader __cgl_rule_helpers {\n"
        + (support.strip() + "\n" if support else "")
        + helper_source(intrinsic, 1)
        + "\n}\n"
    )


def _parse(text: str):
    from ..source_registry import SOURCE_REGISTRY, register_default_sources

    register_default_sources()
    return SOURCE_REGISTRY.get("cgl").parse(text).functions


@functools.lru_cache(maxsize=None)
def _parsed(intrinsic: str, lanes: int):
    text = (
        "shader __cgl_rule_helpers {\n    " + helper_source(intrinsic, lanes) + "\n}\n"
    )
    (function,) = _parse(text)
    return function


@functools.lru_cache(maxsize=None)
def _parsed_support(intrinsic: str):
    support = HELPERS[intrinsic].support
    if not support:
        return ()
    functions = tuple(_parse("shader __cgl_rule_helpers {\n" + support + "\n}\n"))
    if (
        tuple(function.name for function in functions)
        != HELPERS[intrinsic].support_names
    ):
        # An authoring error in the rule table, not a property of the program: an internal invariant.
        raise AssertionError(
            f"the support functions of {intrinsic} must all have reserved names"
        )
    return functions


def helper_function(intrinsic: str, lanes: int):
    """A fresh copy of the helper's ``FunctionNode``, marked as a rule helper."""
    function = clone(_parsed(intrinsic, lanes))
    function.annotations[HELPER_MARK] = intrinsic
    return function


def support_functions(intrinsic: str):
    """Fresh copies of the functions the helper of ``intrinsic`` calls, marked as rule helpers."""
    functions = []
    for function in _parsed_support(intrinsic):
        copy = clone(function)
        copy.annotations[HELPER_MARK] = intrinsic
        functions.append(copy)
    return functions


def helper_body(intrinsic: str):
    """The parsed return expression of the scalar helper; the SMT check evaluates exactly this tree."""
    (statement,) = _parsed(intrinsic, 1).body.statements
    return statement.value


def is_helper(function) -> bool:
    annotations = getattr(function, "annotations", None) or {}
    return bool(annotations.get(HELPER_MARK))


def float_to_integer_helper_names() -> frozenset:
    """The helpers that convert binary32 to a 32-bit integer (``cgl_rule_sat_ftoi``, ``cgl_rule_sat_ftou2``, ...).

    A generator that guards ``float(int(x))`` against folding to ``trunc(x)`` treats a call of one of them as the
    ``int(x)`` it replaces. The ``cgl_rule_`` prefix is reserved, so no program function has these names.
    """
    return frozenset(
        helper_name(intrinsic, lanes)
        for intrinsic in ("__cgl_sat_ftoi", "__cgl_sat_ftou")
        for lanes in (1, 2, 3, 4)
    )


class HelperSet:
    """The helpers one program needs; ``inject`` adds them, scalar before vector, ahead of the program's functions."""

    def __init__(self, program, target, error):
        self.program = program
        self.target = target
        self.error = error
        self.required: set[tuple[str, int]] = set()
        self.existing = {
            function.name: function for function in iter_functions(program)
        }

    def call(self, intrinsic, lanes, operands):
        intrinsic = canonical_intrinsic(intrinsic)
        if lanes not in (1, 2, 3, 4):
            raise self.error(
                f"{intrinsic} on {lanes} lanes has no helper for {self.target}",
                target=self.target,
            )
        for needed in sorted({(intrinsic, 1), (intrinsic, lanes)}):
            name = helper_name(*needed)
            self._reserve(name, intrinsic)
            if name not in self.existing:
                self.required.add(needed)
        for name in HELPERS[intrinsic].support_names:
            self._reserve(name, intrinsic)
        return call(helper_name(intrinsic, lanes), operands)

    def _reserve(self, name, intrinsic):
        known = self.existing.get(name)
        if known is not None and not is_helper(known):
            raise self.error(
                f"the program defines {name}, a name reserved for the rule helper of {intrinsic}",
                target=self.target,
            )

    def inject(self):
        if not self.required:
            return
        functions = getattr(self.program, "functions", None)
        if not isinstance(functions, list):
            raise self.error(
                f"the program has no function list to receive the rule helpers for {self.target}",
                target=self.target,
            )
        support = {}
        for intrinsic, _ in sorted(self.required):
            for function in support_functions(intrinsic):
                if function.name not in self.existing:
                    support.setdefault(function.name, function)
        functions[0:0] = list(support.values()) + [
            helper_function(intrinsic, lanes)
            for intrinsic, lanes in sorted(self.required)
        ]
        self.existing.update(support)
        self.required.clear()
