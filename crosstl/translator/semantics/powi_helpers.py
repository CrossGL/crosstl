"""CrossGL ``powi``: a floating-point base raised to an integer power, for the targets that lack the function.

``powi(x, n)`` raises ``x`` to the integer power ``n`` with the C rules for an integral exponent (C17 F.10.4.4): the
result has the base's precision, ``powi(x, 0)`` is 1 for every ``x``, a negative base keeps its sign for an odd
exponent, and zero to a negative power is an infinity of the base's sign for an odd exponent and ``+inf`` otherwise.
HIP declares the function (``float powif(float, int)``, ``double powi(double, int)``) and the HIP and CUDA generators
write it themselves (:mod:`~crosstl.translator.codegen.cuda_integer_powers`); ``powif(x, n)`` is ``powi`` of a
``float`` base. CrossGL defines ``powi`` on 16-, 32- and 64-bit floating-point scalars and vectors, component by
component; an exponent of another integer type converts to ``int``, and a scalar operand of a vector call is
splat.

No other target declares an integer power, and the ``pow`` of GLSL, HLSL, MSL, WGSL and SPIR-V (GLSL.std.450
``Pow``) is undefined for a negative or zero base. :func:`lower_powi_calls` therefore rewrites every call into a
call of a CrossGL helper for every target but CrossGL, CUDA and HIP, and adds the helpers the program needs once,
ahead of its functions. The helper multiplies by repeated squaring, as LLVM's ``llvm.powi`` and compiler-rt's
``__powisf2`` do, over at most 32 bits of the exponent's magnitude, so the loop is bounded, and takes the reciprocal
for a negative exponent. The result is exact whenever every intermediate product is, and otherwise rounds once per
multiplication; a result whose positive power overflows while the reciprocal is subnormal becomes zero. A binary16
base computes in binary32 and the result rounds to binary16.

A call of a function the program defines itself under one of these names is left alone. A program that defines a
function with a helper's reserved name is refused with :class:`PowiUnsupportedError`, as is a call whose base is
not floating-point, whose exponent is not an integer, whose vector operands differ in width, or whose operand
types cannot be determined.
"""

from __future__ import annotations

import functools
import re

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .helpers import HELPER_MARK, is_helper
from .hub_types import HubType, call_name, iter_functions
from .rewrite import call, clone
from .scoped_walk import ScopedWalker

__all__ = [
    "FUNCTIONS",
    "KEEPING_TARGETS",
    "PREFIX",
    "PowiUnsupportedError",
    "helper_program_source",
    "lower_powi_calls",
]

PREFIX = "cgl_rule_powi_"
# CrossGL function -> the precision its base computes in (``None``: the base's own).
FUNCTIONS = {"powi": None, "powif": 32}
# Targets that write the call themselves: CrossGL as it stands, CUDA and HIP through their generators.
KEEPING_TARGETS = frozenset({"", "cgl", "crossgl", "cuda", "hip"})
_MARK = "powi_helpers"
_LANE_NAMES = "xyzw"
_FLOAT_NAMES = {16: "half", 32: "float", 64: "double"}
_VECTOR_PREFIXES = {16: "f16vec", 32: "vec", 64: "dvec"}
_EXPONENT = HubType("i", 32)

# Repeated squaring of ``x`` over the bits of ``|n|``, computed without a signed overflow; ``T`` is the base type.
_SQUARING = """T {name}(T x, int n) {{
    uint m = n < 0 ? uint(-(n + 1)) + 1u : uint(n);
    T r = T(1);
    T a = x;
    for (int i = 0; i < 32; i++) {{
        if ((m & 1u) != 0u) {{
            r = r * a;
        }}
        m = m >> 1u;
        if (m == 0u) {{
            break;
        }}
        a = a * a;
    }}
    if (n < 0) {{
        return T(1) / r;
    }}
    return r;
}}"""


class PowiUnsupportedError(TranslationDiagnostic):
    """A ``powi`` call that CrossGL cannot compute for the target."""

    code = CODE_PREFIX + "powi-unsupported"


def helper_name(bits, lanes=1):
    """The reserved name of the helper for a ``bits``-bit base of ``lanes`` lanes."""
    name = f"{PREFIX}f{bits}"
    return name if lanes == 1 else f"{name}_v{lanes}"


def _scalar_source(bits):
    name = helper_name(bits)
    if bits == 16:
        return (
            f"half {name}(half x, int n) {{\n"
            f"    return half({helper_name(32)}(float(x), n));\n}}"
        )
    return re.sub(r"\bT\b", _FLOAT_NAMES[bits], _SQUARING.format(name=name))


def _lift_source(bits, lanes):
    vector = f"{_VECTOR_PREFIXES[bits]}{lanes}"
    per_lane = ", ".join(
        f"{helper_name(bits)}(x.{lane}, n.{lane})" for lane in _LANE_NAMES[:lanes]
    )
    return (
        f"{vector} {helper_name(bits, lanes)}({vector} x, ivec{lanes} n) {{\n"
        f"    return {vector}({per_lane});\n}}"
    )


def _source(key):
    bits, lanes = key
    return _scalar_source(bits) if lanes == 1 else _lift_source(bits, lanes)


def _dependencies(bits, lanes):
    """The helper keys a ``(bits, lanes)`` call needs, each before its callers."""
    keys = [(32, 1)] if bits == 16 else []
    keys.append((bits, 1))
    if lanes > 1:
        keys.append((bits, lanes))
    return keys


def _order_key(key):
    bits, lanes = key
    return (lanes > 1, {32: 0, 64: 1, 16: 2}[bits], lanes)


def helper_program_source(keys) -> str:
    """A CrossGL program defining the helpers of the ``(bits, lanes)`` calls ``keys`` and every helper they call."""
    needed = []
    for bits, lanes in keys:
        for key in _dependencies(bits, lanes):
            if key not in needed:
                needed.append(key)
    return (
        "shader __cgl_rule_helpers {\n"
        + "\n\n".join(_source(key) for key in sorted(needed, key=_order_key))
        + "\n}\n"
    )


@functools.lru_cache(maxsize=None)
def _parsed(key):
    """The parsed ``FunctionNode`` of the helper ``key``."""
    from ..source_registry import SOURCE_REGISTRY, register_default_sources

    register_default_sources()
    program = SOURCE_REGISTRY.get("cgl").parse(
        "shader __cgl_rule_helpers {\n" + _source(key) + "\n}\n"
    )
    (function,) = program.functions
    return function


def _spelling(bits, lanes):
    if lanes == 1:
        return _FLOAT_NAMES[bits]
    return f"{_VECTOR_PREFIXES[bits]}{lanes}"


class _Calls(ScopedWalker):
    """Rewrites the calls in place and records the helpers they need."""

    def __init__(self, program, target, user_names):
        super().__init__(program)
        self.target = target
        self.user_names = user_names
        self.needed = []
        self.count = 0

    def _refuse(self, reason):
        raise PowiUnsupportedError(reason, target=self.target)

    def _operand_types(self, name, arguments):
        base, exponent = (self.scope.infer(argument) for argument in arguments)
        if base is None or exponent is None:
            self._refuse(
                f"the operand types of {name} cannot be determined for {self.target}"
            )
        if base.kind != "f" or base.bits not in _FLOAT_NAMES:
            self._refuse(
                f"{name} raises a 16-, 32- or 64-bit floating-point base; this call has a "
                f"{base.bits}-bit {'boolean' if base.kind == 'b' else 'integer'} one"
                if base.kind != "f"
                else f"{name} has no helper for a {base.bits}-bit floating-point base"
            )
        if exponent.kind not in ("i", "u"):
            self._refuse(
                f"{name} takes an integer exponent; this call has a "
                f"{'floating-point' if exponent.kind == 'f' else 'boolean'} one"
            )
        if base.lanes > 1 and exponent.lanes > 1 and base.lanes != exponent.lanes:
            self._refuse(
                f"{name} of a {base.lanes}-lane base and a {exponent.lanes}-lane exponent has no lane pairing"
            )
        lanes = max(base.lanes, exponent.lanes)
        if lanes not in (1, 2, 3, 4):
            self._refuse(f"{name} on {lanes} lanes has no helper for {self.target}")
        return base, exponent, lanes

    def visit(self, node):
        if not isinstance(node, A.FunctionCallNode):
            return
        name = call_name(node)
        if name not in FUNCTIONS or name in self.user_names:
            return
        arguments = list(node.arguments or [])
        if len(arguments) != 2:
            self._refuse(f"{name} takes 2 operands, not {len(arguments)}")
        base, exponent, lanes = self._operand_types(name, arguments)
        bits = FUNCTIONS[name] or base.bits
        for key in _dependencies(bits, lanes):
            if key not in self.needed:
                self.needed.append(key)
        base_type = HubType("f", bits, lanes)
        exponent_type = _EXPONENT.with_lanes(lanes)
        operands = [
            _converted(arguments[0], base, base_type),
            _converted(arguments[1], exponent, exponent_type),
        ]
        node.function = node.name = A.IdentifierNode(helper_name(bits, lanes))
        node.arguments = node.args = operands
        node.annotations["cgl_rule_type"] = base_type
        self.count += 1


def _converted(argument, value, wanted):
    """``argument`` as an operand of the ``wanted`` type: converted or splat when its type differs."""
    if value == wanted:
        return argument
    if value.lanes == 1 and wanted.lanes > 1 and value.scalar != wanted.scalar:
        # Convert the scalar first, then splat it, so no target reads a vector constructor as a conversion.
        argument = call(
            _spelling(wanted.bits, 1) if wanted.kind == "f" else "int", [argument]
        )
    spelling = (
        _spelling(wanted.bits, wanted.lanes)
        if wanted.kind == "f"
        else ("int" if wanted.lanes == 1 else f"ivec{wanted.lanes}")
    )
    return call(spelling, [argument])


def lower_powi_calls(program, target) -> int:
    """Rewrite ``powi`` and ``powif`` calls into calls of CrossGL helpers for ``target``; returns their count."""
    target = str(target or "").lower()
    if target in KEEPING_TARGETS:
        return 0
    functions = list(iter_functions(program))
    existing = {getattr(function, "name", None): function for function in functions}
    for function in functions:
        name = getattr(function, "name", None) or ""
        if name.startswith(PREFIX) and not is_helper(function):
            raise PowiUnsupportedError(
                f"the program defines {name}, a name reserved for the powi helpers",
                target=target,
            )
    user_names = {name for name in existing if name in FUNCTIONS}
    walker = _Calls(program, target, user_names)
    walker.run_program()
    if not walker.count:
        return 0
    program_functions = getattr(program, "functions", None)
    if not isinstance(program_functions, list):
        raise PowiUnsupportedError(
            f"the program has no function list to receive the powi helpers for {target}",
            target=target,
        )
    added = []
    for key in sorted(walker.needed, key=_order_key):
        name = helper_name(*key)
        if name in existing:
            continue
        function = clone(_parsed(key))
        function.annotations[HELPER_MARK] = _MARK
        added.append(function)
        existing[name] = function
    program_functions[0:0] = added
    return walker.count
