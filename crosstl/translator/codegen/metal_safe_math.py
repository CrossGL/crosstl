"""Metal output that keeps the source's IEEE floating-point rule under Metal's default fast math.

The Metal compiler compiles with fast math unless told otherwise: ``-fmetal-math-mode`` defaults to ``fast`` (MSL
4.1 section 1.6.3), and ``MTLCompileOptions.mathMode`` defaults to ``relaxed`` on Apple silicon and ``fast``
elsewhere. Fast math lets the compiler assume no NaNs, infinities or signed zeros and reassociate arithmetic, and
it selects the approximate implementations of the plain math functions: ``sin`` is accurate only to 2^-13
absolutely in [-pi, pi] (MSL 4.1 Table 8.2), ``fmod(-0.0, y)`` loses its sign and ``atan2`` mishandles zero
operands.

A source whose rule table entry carries ``fp_ieee`` (CUDA, HIP and OpenCL C, see ``rule_table.SOURCE_RULES``)
defines those operations without such relaxations. Rule carrying marks the program with
:data:`METAL_SAFE_MATH_ANNOTATION` for the Metal target, and the Metal generator then

* starts the output with ``#pragma METAL fp math_mode(safe)`` (:data:`METAL_SAFE_MATH_PRAGMA`), which makes every
  arithmetic operation after it safe whatever the compile options; the pragma precedes ``<metal_stdlib>`` so the
  library's inline arithmetic compiles safe as well;
* spells every call of a standard-library function that ``metal::precise`` declares, with ``float`` or
  ``floatN`` arguments of one type, as ``metal::precise::<name>``. The pragma does not change which overload a
  plain call selects: the library picks the fast ones from the compile options at preprocessing time.

``metal::precise`` declares only ``float`` overloads, so calls on ``half`` and on other types keep their plain
spelling. Under ``-fno-fast-math`` the plain ``float`` overloads are the precise ones, so the output computes the
same values in either configuration.
"""

from __future__ import annotations

__all__ = [
    "METAL_PRECISE_NAMESPACE_FUNCTIONS",
    "METAL_SAFE_MATH_ANNOTATION",
    "METAL_SAFE_MATH_PRAGMA",
    "metal_safe_math_requested",
    "precise_namespace_eligible",
]

METAL_SAFE_MATH_ANNOTATION = "metal.safe_math"
METAL_SAFE_MATH_PRAGMA = "#pragma METAL fp math_mode(safe)"

# The functions that ``namespace metal::precise`` of the <metal_math>, <metal_common> and <metal_geometric> headers
# declares for ``float`` and ``floatN`` arguments of one type. Functions with an integer or reference argument
# (``frexp``, ``ilogb``, ``ldexp``, ``modf``, ``sincos``) are left out, as is ``select``.
METAL_PRECISE_NAMESPACE_FUNCTIONS = frozenset(
    {
        "abs",
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atan2",
        "atanh",
        "ceil",
        "clamp",
        "copysign",
        "cos",
        "cosh",
        "cospi",
        "distance",
        "divide",
        "exp",
        "exp10",
        "exp2",
        "fabs",
        "fdim",
        "floor",
        "fma",
        "fmax",
        "fmax3",
        "fmedian3",
        "fmin",
        "fmin3",
        "fmod",
        "fract",
        "length",
        "log",
        "log10",
        "log2",
        "max",
        "max3",
        "median3",
        "min",
        "min3",
        "nextafter",
        "normalize",
        "pow",
        "powr",
        "rint",
        "round",
        "rsqrt",
        "saturate",
        "sin",
        "sinh",
        "sinpi",
        "sqrt",
        "tan",
        "tanh",
        "tanpi",
        "trunc",
    }
)
_BINARY32_TYPES = frozenset({"float", "float2", "float3", "float4"})


def metal_safe_math_requested(program) -> bool:
    """Whether rule carrying asked for Metal output that keeps IEEE semantics under fast math."""
    annotations = getattr(program, "annotations", None)
    return isinstance(annotations, dict) and bool(
        annotations.get(METAL_SAFE_MATH_ANNOTATION)
    )


def precise_namespace_eligible(name, argument_types) -> bool:
    """Whether ``name(arguments)`` has a ``metal::precise`` overload: one ``float``/``floatN`` type throughout."""
    if name not in METAL_PRECISE_NAMESPACE_FUNCTIONS or not argument_types:
        return False
    first = argument_types[0]
    return first in _BINARY32_TYPES and all(
        argument == first for argument in argument_types
    )
