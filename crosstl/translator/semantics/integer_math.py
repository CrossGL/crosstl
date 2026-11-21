"""CrossGL's integer combinatorics functions: ``gcd``, ``lcm``, ``factorial``, ``comb`` and ``perm``.

Mojo's ``std.math`` defines them on ``Int`` (``math.mojo``: ``gcd``, ``lcm``, ``factorial``, ``comb``, ``perm``), as
Python's ``math`` module and C++17's ``std::gcd`` and ``std::lcm`` do. CrossGL defines them on its 32- and 64-bit
integer scalars and vectors, signed and unsigned, component by component, with the operands converted to their
common type by the usual arithmetic conversions (an 8- or 16-bit operand computes as the 32-bit type of its
signedness, and the result converts back to the operand's type):

- ``gcd(a, b)``: the greatest common divisor of ``|a|`` and ``|b|``; ``gcd(a, 0) = |a|`` and ``gcd(0, 0) = 0``.
- ``lcm(a, b)``: ``|a| / gcd(a, b) * |b|``, and 0 when either operand is 0.
- ``factorial(n)``: ``n!`` for ``0 <= n``.
- ``comb(n, k)``: the binomial coefficient ``n! / (k! (n - k)!)`` for ``0 <= k <= n``, and 0 when ``k > n`` or
  ``k < 0``.
- ``perm(n, k)``: ``n! / (n - k)!`` for ``0 <= k <= n``, and 0 when ``k > n`` or ``k < 0``, as Python's
  ``math.perm``. ``perm(n)`` is ``factorial(n)``, as Mojo's ``perm`` with its default ``k``.

Each result is defined where it fits the type, as Mojo's are where they fit ``Int``: a signed ``|a|`` of the most
negative value is its unsigned magnitude, and a result that does not fit is undefined. Negative operands of
``factorial`` and negative ``n`` of ``comb`` and ``perm`` are outside Mojo's domain too (it asserts); CrossGL gives
``factorial`` of a negative value 1 and the others 0.

No GPU language declares these functions, so :func:`lower_integer_math` rewrites every call into a call of a
CrossGL helper for every target, which it adds to the program once, ahead of the program's functions, with the
helpers it calls. The helpers compute in unsigned arithmetic, which wraps in every target, so that no target's
signed overflow, signed division or signed remainder rule applies. Their loops are bounded: ``gcd`` is Euclid's
algorithm, which takes at most 93 steps for 64-bit operands; ``factorial`` and ``perm`` multiply at most 21 factors
and ``comb`` at most 34, more than any result that fits 64 bits needs (``21!`` and ``C(68, 34)`` exceed
``2^64``), so an operand outside the domain still terminates. Mojo's own functions take ``Int`` where CrossGL's
take sized integers, so Mojo output calls the helpers as well. A call of a function the program defines itself
under one of these names is left alone, and a program that defines a function with a helper's reserved name is
refused with :class:`IntegerMathError`, as is a call whose operands are not integers or whose type cannot be
determined.
"""

from __future__ import annotations

import functools
import re

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .helpers import HELPER_MARK, is_helper
from .hub_types import HubType, arithmetic_result, call_name, iter_functions
from .rewrite import call, clone
from .scoped_walk import ScopedWalker

__all__ = [
    "FUNCTIONS",
    "IntegerMathError",
    "PREFIX",
    "helper_program_source",
    "lower_integer_math",
]

PREFIX = "cgl_rule_int_"
# CrossGL function -> the numbers of operands it takes.
FUNCTIONS = {
    "gcd": (2,),
    "lcm": (2,),
    "factorial": (1,),
    "comb": (2,),
    "perm": (1, 2),
}
# Targets that keep the call: CrossGL output writes the function as it stands.
_KEEPING_TARGETS = frozenset({"", "cgl", "crossgl"})
_MARK = "integer_math"
_LANE_NAMES = "xyzw"
_SCALAR_NAMES = {
    ("i", 32): "int",
    ("u", 32): "uint",
    ("i", 64): "int64_t",
    ("u", 64): "uint64_t",
}
_VECTOR_PREFIXES = {
    ("i", 32): "ivec",
    ("u", 32): "uvec",
    ("i", 64): "i64vec",
    ("u", 64): "u64vec",
}
_NARROW_NAMES = {
    ("i", 8): "int8_t",
    ("u", 8): "uint8_t",
    ("i", 16): "int16_t",
    ("u", 16): "uint16_t",
}


class IntegerMathError(TranslationDiagnostic):
    """A call of an integer combinatorics function that CrossGL cannot compute."""

    code = CODE_PREFIX + "integer-math-unsupported"


def _suffix(kind, bits):
    return f"{kind}{bits}"


# The unsigned helpers, in terms of ``U`` (the unsigned type) and ``$W`` (its suffix). ``U(...)`` converts, so the
# text needs no literal suffix of the width.
_UNSIGNED = {
    "gcd": (
        """U {p}gcd_u$W(U a, U b) {
    U x = a;
    U y = b;
    while (y != U(0)) {
        U r = x % y;
        x = y;
        y = r;
    }
    return x;
}"""
    ),
    "lcm": (
        """U {p}lcm_u$W(U a, U b) {
    U g = {p}gcd_u$W(a, b);
    if (g == U(0)) {
        return U(0);
    }
    return a / g * b;
}"""
    ),
    "factorial": (
        """U {p}factorial_u$W(U n) {
    U r = U(1);
    for (uint i = 2u; i <= 21u && U(i) <= n; i++) {
        r = r * U(i);
    }
    return r;
}"""
    ),
    "comb": (
        """U {p}comb_u$W(U n, U k) {
    if (k > n) {
        return U(0);
    }
    U m = n - k < k ? n - k : k;
    U r = U(1);
    for (uint i = 0u; i < 34u && U(i) < m; i++) {
        U top = n - U(i);
        U bottom = U(i) + U(1);
        U g = {p}gcd_u$W(top, bottom);
        r = r / (bottom / g) * (top / g);
    }
    return r;
}"""
    ),
    "perm": (
        """U {p}perm_u$W(U n, U k) {
    if (k > n) {
        return U(0);
    }
    U r = U(1);
    for (uint i = 0u; i < 21u && U(i) < k; i++) {
        r = r * (n - U(i));
    }
    return r;
}"""
    ),
}

# The signed helpers, in terms of ``S`` and ``U``: each takes the magnitudes, or guards the negative operands, and
# calls its unsigned helper. ``-(a + 1) + 1`` is the magnitude of a negative ``a`` without a signed overflow.
_SIGNED = {
    "magnitude": (
        """U {p}magnitude_i$W(S a) {
    if (a < S(0)) {
        return U(-(a + S(1))) + U(1);
    }
    return U(a);
}"""
    ),
    "gcd": (
        """S {p}gcd_i$W(S a, S b) {
    return S({p}gcd_u$W({p}magnitude_i$W(a), {p}magnitude_i$W(b)));
}"""
    ),
    "lcm": (
        """S {p}lcm_i$W(S a, S b) {
    return S({p}lcm_u$W({p}magnitude_i$W(a), {p}magnitude_i$W(b)));
}"""
    ),
    "factorial": (
        """S {p}factorial_i$W(S n) {
    if (n < S(0)) {
        return S(1);
    }
    return S({p}factorial_u$W(U(n)));
}"""
    ),
    "comb": (
        """S {p}comb_i$W(S n, S k) {
    if (k < S(0) || n < S(0)) {
        return S(0);
    }
    return S({p}comb_u$W(U(n), U(k)));
}"""
    ),
    "perm": (
        """S {p}perm_i$W(S n, S k) {
    if (k < S(0) || n < S(0)) {
        return S(0);
    }
    return S({p}perm_u$W(U(n), U(k)));
}"""
    ),
}


def _render(text, bits):
    """``text`` for ``bits``-bit operands, with the reserved prefix."""
    unsigned = _SCALAR_NAMES[("u", bits)]
    signed = _SCALAR_NAMES[("i", bits)]
    text = text.replace("{p}", PREFIX).replace("$W", str(bits))
    text = re.sub(r"\bU\b", unsigned, text)
    return re.sub(r"\bS\b", signed, text)


def helper_name(function, kind, bits):
    """The reserved name of ``function``'s scalar helper for ``kind`` (``i`` or ``u``) operands of ``bits`` bits."""
    return f"{PREFIX}{function}_{_suffix(kind, bits)}"


@functools.lru_cache(maxsize=None)
def _definitions():
    """Every scalar helper, in dependency order: name -> CrossGL definition."""
    definitions = {}
    for bits in (32, 64):
        for function in ("gcd", "lcm", "factorial", "comb", "perm"):
            definitions[helper_name(function, "u", bits)] = _render(
                _UNSIGNED[function], bits
            )
        for function in ("magnitude", "gcd", "lcm", "factorial", "comb", "perm"):
            definitions[helper_name(function, "i", bits)] = _render(
                _SIGNED[function], bits
            )
    return definitions


def _dependencies(name):
    """The helpers the definition of ``name`` calls, each before its callers, then ``name``."""
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


def _lift_source(helper, kind, bits, arity, lanes):
    vector = f"{_VECTOR_PREFIXES[(kind, bits)]}{lanes}"
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
    """A CrossGL program defining the scalar helpers ``names`` and every helper they call."""
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
    """The parsed ``FunctionNode`` of a definition (a name) or of a lift (``(helper, kind, bits, arity, lanes)``)."""
    from ..source_registry import SOURCE_REGISTRY, register_default_sources

    register_default_sources()
    text = _lift_source(*key) if isinstance(key, tuple) else _definitions()[key]
    program = SOURCE_REGISTRY.get("cgl").parse(
        "shader __cgl_rule_helpers {\n" + text + "\n}\n"
    )
    (function,) = program.functions
    return function


def _computed_type(value):
    """The 32- or 64-bit integer type that computes an integer operand type ``value``."""
    bits = 64 if value.bits > 32 else 32
    return HubType(value.kind, bits, value.lanes)


def _type_spelling(value):
    """The CrossGL spelling of an integer type."""
    key = (value.kind, value.bits)
    if value.lanes == 1:
        return _SCALAR_NAMES.get(key) or _NARROW_NAMES[key]
    return f"{_VECTOR_PREFIXES[key]}{value.lanes}"


class _Calls(ScopedWalker):
    """Rewrites the calls in place and records the helpers they need."""

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
        if len(arguments) not in FUNCTIONS[name]:
            raise IntegerMathError(
                f"{name} takes {' or '.join(map(str, FUNCTIONS[name]))} operands, not {len(arguments)}",
                target=self.target,
            )
        types = [self.scope.infer(argument) for argument in arguments]
        common = types[0]
        for value in types[1:]:
            common = arithmetic_result(common, value)
        if common is None or any(value is None for value in types):
            raise IntegerMathError(
                f"the operand types of {name} cannot be determined for {self.target}",
                target=self.target,
            )
        if common.kind not in ("i", "u") or any(
            value.kind not in ("i", "u") for value in types
        ):
            raise IntegerMathError(
                f"{name} takes integer operands; this call has a "
                f"{'floating-point' if common.kind == 'f' else 'boolean'} one",
                target=self.target,
            )
        if common.bits > 64 or common.lanes not in (1, 2, 3, 4):
            raise IntegerMathError(
                f"{name} on {common.lanes} {common.bits}-bit components has no helper for {self.target}",
                target=self.target,
            )
        function = name
        if name == "perm" and len(arguments) == 1:
            function = "factorial"
        computed = _computed_type(common)
        helper = helper_name(function, computed.kind, computed.bits)
        arity = len(arguments)
        for key in _dependencies(helper):
            if key not in self.needed:
                self.needed.append(key)
        if computed.lanes > 1:
            lift = (helper, computed.kind, computed.bits, arity, computed.lanes)
            if lift not in self.needed:
                self.needed.append(lift)
        operands = [
            _converted(argument, value, computed)
            for argument, value in zip(arguments, types)
        ]
        node.function = node.name = A.IdentifierNode(_lift_name(helper, computed.lanes))
        node.arguments = node.args = operands
        node.annotations["cgl_rule_type"] = computed
        self.count += 1
        if computed != common:
            # A narrow operand type computes in 32 bits; the result converts back to it.
            replacement = call(_type_spelling(common), [clone(node)])
            replacement.annotations["cgl_rule_type"] = common
            _replace(node, replacement)


def _converted(argument, value, computed):
    """``argument`` as an operand of the ``computed`` type: converted or splat when its type differs."""
    if value == computed:
        return argument
    return call(_type_spelling(computed), [argument])


def _replace(node, replacement):
    """Make ``node`` the call ``replacement`` describes, in place, so its parent needs no update."""
    node.function = node.name = replacement.function
    node.arguments = node.args = replacement.arguments
    node.annotations = dict(replacement.annotations)


def _order_key(key, order):
    """Definitions in dependency order, then the vector lifts."""
    if isinstance(key, tuple):
        return (1, order.index(key[0]), key[4])
    return (0, order.index(key), 0)


def lower_integer_math(program, target) -> int:
    """Rewrite ``gcd``, ``lcm``, ``factorial``, ``comb`` and ``perm`` calls for ``target``; returns their count."""
    target = str(target or "").lower()
    if target in _KEEPING_TARGETS:
        return 0
    functions = list(iter_functions(program))
    existing = {getattr(function, "name", None): function for function in functions}
    for function in functions:
        name = getattr(function, "name", None) or ""
        if name.startswith(PREFIX) and not is_helper(function):
            raise IntegerMathError(
                f"the program defines {name}, a name reserved for the integer math helpers",
                target=target,
            )
    user_names = {name for name in existing if name in FUNCTIONS}
    walker = _Calls(program, target, user_names)
    walker.run_program()
    if not walker.count:
        return 0
    program_functions = getattr(program, "functions", None)
    if not isinstance(program_functions, list):
        raise IntegerMathError(
            f"the program has no function list to receive the integer math helpers for {target}",
            target=target,
        )
    order = list(_definitions())
    added = []
    for key in sorted(walker.needed, key=lambda key: _order_key(key, order)):
        name = _lift_name(key[0], key[4]) if isinstance(key, tuple) else key
        if name in existing:
            continue
        function = clone(_parsed(key))
        function.annotations[HELPER_MARK] = _MARK
        added.append(function)
        existing[name] = function
    program_functions[0:0] = added
    return walker.count
