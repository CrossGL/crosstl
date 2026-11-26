"""Calls of the exponential, logarithmic, inverse trigonometric and hyperbolic functions on binary64 values become
calls of the helpers.

GLSL declares ``exp``, ``exp2``, ``log``, ``log2``, ``pow``, ``asin``, ``acos``, ``atan``, ``sinh``, ``cosh``,
``tanh``, ``asinh``, ``acosh`` and ``atanh`` for ``genFType`` only (GLSL 4.60 sections 8.1 and 8.2), the matching
GLSL.std.450 instructions take 16- and 32-bit floating-point operands only, and HLSL's take ``float`` (DXC converts
a ``double`` operand to ``float``). A CrossGL program that calls them on ``double`` values, as CUDA, HIP and OpenCL C
programs do, therefore had no correct form on the OpenGL, Vulkan and DirectX targets: GLSL and HLSL output computed
in binary32 or failed to compile, and SPIR-V output was refused. :func:`lower_binary64_math` rewrites each such call
on those targets into a call of the binary64 helper (:mod:`.exp_log`, :mod:`.power`, :mod:`.inverse_trig`,
:mod:`.hyperbolic`), which it adds to the program once, ahead of the program's functions, with the support functions
it calls (:mod:`.bits`, :mod:`.roots`). A vector operand calls a per-component lift of the helper, and GLSL's
two-operand ``atan(y, x)`` is ``atan2(y, x)``.

The helpers are ``precise`` (no contraction or reassociation: GLSL ``precise``, HLSL ``precise``, SPIR-V
``NoContraction``) and use only binary64 addition, subtraction, multiplication, division, comparison and
conversion, 32-bit integer arithmetic and the exact word access of ``packDouble2x32``/``unpackDouble2x32``. Their
error bound assumes correctly rounded binary64 operations with subnormals.

Targets whose math library has binary64 versions (CUDA, HIP, OpenCL C, Slang, Rust, Mojo) keep the call, and targets
without ``double`` (Metal, WGSL, WebGL) refuse the program elsewhere. A call of a function the program defines
itself under one of these names is left alone, and a program that defines a function with a helper's reserved name
is refused with :class:`Binary64MathError`.
"""

from __future__ import annotations

import functools
import re

from ... import ast as A
from ...diagnostics import CODE_PREFIX, TranslationDiagnostic
from ..helpers import HELPER_MARK, is_helper
from ..hub_types import HubType, call_name, iter_functions
from ..rewrite import call, clone
from ..scoped_walk import ScopedWalker
from . import (
    bits,
    exp_log,
    extended,
    hyperbolic,
    inverse_trig,
    power,
    roots,
    rounding,
)

__all__ = [
    "BINARY64_TARGETS",
    "Binary64MathError",
    "FUNCTIONS",
    "helper_program_source",
    "lower_binary64_math",
]

# Targets whose built-in exponential, logarithmic, inverse trigonometric and hyperbolic functions take binary32
# operands at most.
BINARY64_TARGETS = frozenset({"opengl", "vulkan", "directx"})

# CrossGL built-in -> (helper, arity).
FUNCTIONS = {
    "exp": (exp_log.EXP, 1),
    "log": (exp_log.LOG, 1),
    "pow": (power.POW, 2),
    "exp2": (power.EXP2, 1),
    "log2": (power.LOG2, 1),
    "asin": (inverse_trig.ASIN, 1),
    "acos": (inverse_trig.ACOS, 1),
    "atan": (inverse_trig.ATAN, 1),
    "atan2": (inverse_trig.ATAN2, 2),
    "sinh": (hyperbolic.SINH, 1),
    "cosh": (hyperbolic.COSH, 1),
    "tanh": (hyperbolic.TANH, 1),
    "asinh": (hyperbolic.ASINH, 1),
    "acosh": (hyperbolic.ACOSH, 1),
    "atanh": (hyperbolic.ATANH, 1),
    # The round_away rule family's hub intrinsic (rule_table.py), which lower.py keeps for binary64 operands.
    "__cgl_round_away": (rounding.ROUND_AWAY, 1),
}
# Built-ins with a second form of another arity: (name, arity) -> the FUNCTIONS entry it computes.
_ALTERNATE_FORMS = {("atan", 2): "atan2"}
_LANE_NAMES = "xyzw"
_MARK = "libm_binary64"


class Binary64MathError(TranslationDiagnostic):
    """A binary64 math call the helpers cannot replace."""

    code = CODE_PREFIX + "binary64-math-unsupported"


@functools.lru_cache(maxsize=None)
def _definitions():
    """Every helper and support function, in dependency order: name -> CrossGL definition."""
    return dict(
        bits.support_sources()
        + exp_log.sources()
        + power.sources()
        + roots.sources()
        + extended.sources()
        + inverse_trig.sources()
        + hyperbolic.sources()
        + rounding.sources()
    )


def _dependencies(name):
    """The reserved functions the definition of ``name`` calls, each before its callers, then ``name``."""
    definitions = _definitions()
    order = list(definitions)
    needed = set()

    def visit(current):
        if current in needed:
            return
        needed.add(current)
        text = definitions[current]
        for other in order:
            if other != current and re.search(rf"\b{re.escape(other)}\(", text):
                visit(other)

    visit(name)
    return [other for other in order if other in needed]


def _lift_name(helper, lanes):
    return helper if lanes == 1 else f"{helper}_v{lanes}"


def _lift_source(helper, arity, lanes):
    vector = f"dvec{lanes}"
    parameters = ", ".join(f"{vector} x{index}" for index in range(arity))
    per_lane = ", ".join(
        f"{helper}({', '.join(f'x{index}.{lane}' for index in range(arity))})"
        for lane in _LANE_NAMES[:lanes]
    )
    return (
        f"{vector} {_lift_name(helper, lanes)}({parameters}) {{\n"
        f"    return {vector}({per_lane});\n}}"
    )


def helper_program_source(names) -> str:
    """A CrossGL program defining the helpers ``names`` and every function they call."""
    definitions = _definitions()
    needed = []
    for name in names:
        for dependency in _dependencies(name):
            if dependency not in needed:
                needed.append(dependency)
    ordered = [name for name in definitions if name in needed]
    return (
        "shader __cgl_rule_helpers {\n"
        + "\n\n".join(definitions[name] for name in ordered)
        + "\n}\n"
    )


@functools.lru_cache(maxsize=None)
def _parsed(key):
    """The parsed ``FunctionNode`` of a definition (a name) or of a lift (``(helper, arity, lanes)``)."""
    from ...source_registry import SOURCE_REGISTRY, register_default_sources

    register_default_sources()
    if isinstance(key, tuple):
        text = _lift_source(*key)
    else:
        text = _definitions()[key]
    program = SOURCE_REGISTRY.get("cgl").parse(
        "shader __cgl_rule_helpers {\n" + text + "\n}\n"
    )
    (function,) = program.functions
    return function


class _Calls(ScopedWalker):
    """Rewrites binary64 calls in place and records the helpers they need."""

    def __init__(self, program, target, user_names):
        super().__init__(program)
        self.target = target
        self.user_names = user_names
        self.needed = []
        self.count = 0

    def visit(self, node):
        if not isinstance(node, A.FunctionCallNode):
            return
        name = call_name(node)
        if name not in FUNCTIONS or name in self.user_names:
            return
        arguments = list(node.arguments or [])
        helper, arity = FUNCTIONS[_ALTERNATE_FORMS.get((name, len(arguments)), name)]
        if len(arguments) != arity:
            return
        types = [self.scope.infer(argument) for argument in arguments]
        if not any(_is_binary64(value) for value in types):
            return
        lanes = max(value.lanes for value in types if value is not None)
        operands = [
            _as_binary64(argument, value, lanes)
            for argument, value in zip(arguments, types)
        ]
        if lanes not in (1, 2, 3, 4):
            raise Binary64MathError(
                f"{name} on {lanes} binary64 components has no helper for {self.target}",
                target=self.target,
            )
        for key in [*_dependencies(helper), (helper, arity, lanes)]:
            if key == (helper, arity, 1):
                continue
            if key not in self.needed:
                self.needed.append(key)
        callee = A.IdentifierNode(_lift_name(helper, lanes))
        node.function = node.name = callee
        node.arguments = node.args = operands
        node.annotations["cgl_rule_type"] = HubType("f", 64, lanes)
        self.count += 1


def _is_binary64(value):
    return value is not None and value.kind == "f" and value.bits == 64


def _as_binary64(argument, value, lanes):
    """``argument`` as a binary64 operand of ``lanes`` components.

    A binary64 scalar of a vector call is splat, and an operand of another type is converted, which is exact for
    every binary16, binary32 and 32-bit integer value. A scalar literal keeps its spelling, which the generators
    convert to the parameter's type, and an operand of unknown type is left to them as well.
    """
    if value is None or (isinstance(argument, A.LiteralNode) and lanes == 1):
        return argument
    if _is_binary64(value) and value.lanes == lanes:
        return argument
    if value.lanes not in (1, lanes):
        return argument
    return call("double" if lanes == 1 else f"dvec{lanes}", [argument])


def _order_key(key, order):
    """Definitions in dependency order, then the vector lifts."""
    if isinstance(key, tuple):
        return (1, order.index(key[0]), key[2])
    return (0, order.index(key), 0)


def lower_binary64_math(program, target) -> int:
    """Rewrite the binary64 calls of the FUNCTIONS built-ins for ``target``; returns their count."""
    target = str(target or "").lower()
    if target not in BINARY64_TARGETS:
        return 0
    functions = list(iter_functions(program))
    existing = {function.name: function for function in functions}
    for function in functions:
        name = getattr(function, "name", None) or ""
        if name.startswith(bits.PREFIX) and not is_helper(function):
            raise Binary64MathError(
                f"the program defines {name}, a name reserved for the binary64 math helpers",
                target=target,
            )
    user_names = {name for name in existing if name in FUNCTIONS}
    walker = _Calls(program, target, user_names)
    walker.run_program()
    if not walker.count:
        return 0
    program_functions = getattr(program, "functions", None)
    if not isinstance(program_functions, list):
        raise Binary64MathError(
            f"the program has no function list to receive the binary64 math helpers for {target}",
            target=target,
        )
    order = list(_definitions())
    added = []
    for key in sorted(walker.needed, key=lambda key: _order_key(key, order)):
        name = _lift_name(key[0], key[2]) if isinstance(key, tuple) else key
        if name in existing:
            continue
        function = clone(_parsed(key))
        function.annotations[HELPER_MARK] = _MARK
        added.append(function)
        existing[name] = function
    program_functions[0:0] = added
    return walker.count
