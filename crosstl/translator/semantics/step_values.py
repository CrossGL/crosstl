"""Signed increments and decrements used as values, for targets that polyfill wrapping arithmetic (ADR 0001).

A statement ``x++`` becomes ``x = asint(asuint(x) + 1u)``. An increment used as a value, as in ``f(x++)`` or
``for (int i = 0; i++ < n;)``, cannot become an assignment in place, because the CrossGL emitters do not handle
assignment expressions reliably. It becomes a call of a reserved helper function instead. The helper takes the operand
as an ``inout`` parameter, updates it with the target's wrapping polyfill and returns the value the source operator
yields: the old value for a postfix operator, the new one for a prefix operator. An ``inout`` argument is an lvalue
that is copied in and out, so the operand is evaluated once and each emitter's handling of ``inout`` parameters
applies unchanged.

Only the targets in ``STEP_HELPER_TARGETS`` emit CrossGL ``inout`` parameters with write-back. The CUDA, HIP, Mojo
and SPIR-V emitters pass them by value, so an increment used as a value is still refused for those targets.
"""

from __future__ import annotations

import functools
import re

from .helpers import HELPER_MARK, is_helper
from .hub_types import iter_functions
from .rewrite import call, clone
from .rule_table import POLYFILLS

STEP_HELPER_PREFIX = "cgl_rule_wrap_"
STEP_HELPER_TARGETS = {
    "metal": (
        "MSL thread references, with write-back temporaries for other address spaces"
    ),
    "directx": "HLSL inout parameters, copied in and out",
    "slang": "Slang inout parameters, copied in and out",
}
_STEP_NAMES = {
    ("__cgl_wrap_add", True): "postinc",
    ("__cgl_wrap_add", False): "preinc",
    ("__cgl_wrap_sub", True): "postdec",
    ("__cgl_wrap_sub", False): "predec",
}


def step_helper_name(intrinsic: str, postfix: bool, lanes: int) -> str:
    base = STEP_HELPER_PREFIX + _STEP_NAMES[(intrinsic, postfix)]
    return base if lanes == 1 else f"{base}{lanes}"


def step_helper_source(intrinsic: str, postfix: bool, lanes: int) -> str:
    """The CrossGL definition of the helper for ``x++``/``x--`` (``postfix``) or ``++x``/``--x`` on ``lanes`` lanes."""
    value_type = "int" if lanes == 1 else f"ivec{lanes}"
    name = step_helper_name(intrinsic, postfix, lanes)
    # The polyfill template is written over the operands x0 and x1; the step adds or subtracts one.
    update = re.sub(r"\basuint\(x1\)", "1u", POLYFILLS[intrinsic])
    update = re.sub(r"\bx1\b", "1", update)
    if postfix:
        body = f"{value_type} x1 = x0; x0 = {update}; return x1;"
    else:
        body = f"x0 = {update}; return x0;"
    return f"{value_type} {name}(inout {value_type} x0) {{ {body} }}"


@functools.lru_cache(maxsize=None)
def _parsed(intrinsic: str, postfix: bool, lanes: int):
    from ..source_registry import SOURCE_REGISTRY, register_default_sources

    register_default_sources()
    text = (
        "shader __cgl_rule_helpers {\n    "
        + step_helper_source(intrinsic, postfix, lanes)
        + "\n}\n"
    )
    (function,) = SOURCE_REGISTRY.get("cgl").parse(text).functions
    return function


def step_helper_function(intrinsic: str, postfix: bool, lanes: int):
    """A fresh copy of the helper's ``FunctionNode``, marked as a rule helper."""
    function = clone(_parsed(intrinsic, postfix, lanes))
    function.annotations[HELPER_MARK] = intrinsic
    return function


class StepValueHelpers:
    """The step helpers one program needs; ``inject`` adds them ahead of the program's functions."""

    def __init__(self, program, target, error):
        self.program = program
        self.target = target
        self.error = error
        self.required: set[tuple[str, bool, int]] = set()
        self.existing = {
            function.name: function for function in iter_functions(program)
        }

    def call(self, intrinsic, node, result_type):
        """The helper call that replaces the step ``node`` used as a value."""
        if self.target not in STEP_HELPER_TARGETS:
            raise self.error(
                f"the signed {node.op} used as a value cannot be rewritten with wrapping arithmetic for "
                f"{self.target}, whose code generator passes inout parameters by value; write it as a "
                "separate statement",
                target=self.target,
            )
        if intrinsic not in POLYFILLS or (intrinsic, True) not in _STEP_NAMES:
            raise self.error(
                f"{intrinsic} has no step helper for {self.target}", target=self.target
            )
        lanes = getattr(result_type, "lanes", None)
        if lanes not in (1, 2, 3, 4):
            raise self.error(
                f"the signed {node.op} used as a value has no step helper for {lanes} lanes on {self.target}",
                target=self.target,
            )
        postfix = bool(getattr(node, "is_postfix", False))
        name = step_helper_name(intrinsic, postfix, lanes)
        known = self.existing.get(name)
        if known is None:
            self.required.add((intrinsic, postfix, lanes))
        elif not is_helper(known):
            raise self.error(
                f"the program defines {name}, a name reserved for the rule helper of {node.op}",
                target=self.target,
            )
        replacement = call(name, [node.operand])
        replacement.source_location = getattr(node, "source_location", None)
        return replacement

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
            step_helper_function(intrinsic, postfix, lanes)
            for intrinsic, postfix, lanes in sorted(self.required)
        ]
        self.required.clear()
