"""Reserved helper functions for polyfills that use an operand more than once (ADR 0001, section 4).

A template in ``rule_table.POLYFILLS`` names each operand exactly once, so it can be substituted in place. A polyfill
that needs an operand twice, such as the zero test and the division of the ``udiv_zero`` family, is a helper
function instead: the operation becomes a call, which evaluates each operand once, and the lowering pass injects the
helper into the program. Vector operations call a per-lane lift of the scalar helper.
"""

from __future__ import annotations

import functools

from .hub_types import iter_functions
from .rewrite import call, clone
from .rule_table import HELPERS

HELPER_MARK = "cgl_rule_helper"
_VECTOR_PREFIX = {"uint": "uvec", "int": "ivec"}
_LANE_NAMES = "xyzw"


def helper_name(intrinsic: str, lanes: int) -> str:
    base = HELPERS[intrinsic].name
    return base if lanes == 1 else f"{base}{lanes}"


def helper_source(intrinsic: str, lanes: int) -> str:
    """The CrossGL definition of the helper for ``lanes`` lanes."""
    helper = HELPERS[intrinsic]
    scalar = helper.scalar
    if lanes == 1:
        return (
            f"{scalar} {helper.name}({scalar} x0, {scalar} x1) "
            f"{{ return {helper.body}; }}"
        )
    vector = f"{_VECTOR_PREFIX[scalar]}{lanes}"
    per_lane = ", ".join(
        f"{helper.name}(x0.{lane}, x1.{lane})" for lane in _LANE_NAMES[:lanes]
    )
    return (
        f"{vector} {helper_name(intrinsic, lanes)}({vector} x0, {vector} x1) "
        f"{{ return {vector}({per_lane}); }}"
    )


@functools.lru_cache(maxsize=None)
def _parsed(intrinsic: str, lanes: int):
    from ..source_registry import SOURCE_REGISTRY, register_default_sources

    register_default_sources()
    text = (
        "shader __cgl_rule_helpers {\n    " + helper_source(intrinsic, lanes) + "\n}\n"
    )
    program = SOURCE_REGISTRY.get("cgl").parse(text)
    (function,) = program.functions
    return function


def helper_function(intrinsic: str, lanes: int):
    """A fresh copy of the helper's ``FunctionNode``, marked as a rule helper."""
    function = clone(_parsed(intrinsic, lanes))
    function.annotations[HELPER_MARK] = intrinsic
    return function


def helper_body(intrinsic: str):
    """The parsed return expression of the scalar helper; the SMT check evaluates exactly this tree."""
    (statement,) = _parsed(intrinsic, 1).body.statements
    return statement.value


def is_helper(function) -> bool:
    annotations = getattr(function, "annotations", None) or {}
    return bool(annotations.get(HELPER_MARK))


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
        if lanes not in (1, 2, 3, 4):
            raise self.error(
                f"{intrinsic} on {lanes} lanes has no helper for {self.target}",
                target=self.target,
            )
        for needed in sorted({(intrinsic, 1), (intrinsic, lanes)}):
            name = helper_name(*needed)
            known = self.existing.get(name)
            if known is None:
                self.required.add(needed)
            elif not is_helper(known):
                raise self.error(
                    f"the program defines {name}, a name reserved for the rule helper of {intrinsic}",
                    target=self.target,
                )
        return call(helper_name(intrinsic, lanes), operands)

    def inject(self):
        if not self.required:
            return
        functions = getattr(self.program, "functions", None)
        if not isinstance(functions, list):
            raise self.error(
                f"the program has no function list to receive the rule helpers for {self.target}",
                target=self.target,
            )
        functions[0:0] = [
            helper_function(intrinsic, lanes)
            for intrinsic, lanes in sorted(self.required)
        ]
        self.required.clear()
