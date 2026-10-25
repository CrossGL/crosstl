"""CrossGL helper functions that compute binary16 results exactly in binary32 registers (defect S7).

The half legalization pass (:mod:`.half`) keeps every ``half`` value in a ``float`` that holds a binary16 value. The
helpers here produce such values from binary32 arithmetic. Each one is exact under any faithful binary32 rounding and
under fast-math reassociation, because the rounding decisions use integer operations on the bit patterns, and every
floating-point operation they compare against is exact:

- ``cgl_half_round`` rounds a binary32 value to binary16, ties to even (IEEE 754-2019 section 4.3.1), including
  subnormals, overflow to infinity and quiet NaN. Rounding the binary32 sum, difference or product of two binary16
  values gives the correctly rounded binary16 result: those results are exact in binary32 or lie too far from a
  binary16 midpoint for the first rounding to matter.
- ``cgl_half_div`` and ``cgl_half_sqrt`` correct the rounded binary32 quotient or root, which Vulkan and WGSL only
  bound to 2.5 ULP and to the error of ``1 / inversesqrt`` (Vulkan "Precision and Operation of SPIR-V Instructions",
  WGSL section 15.7.4.1). They compare the operand with the product of a binary16 midpoint, which is exact because a
  midpoint has 12 significant bits and an operand 11.
- ``cgl_half_fma`` adds the exact binary32 product of two binary16 values to the third one in 32-bit integers with a
  sticky bit, and rounds the sum once (IEEE 754-2019 section 5.4.1, fusedMultiplyAdd). A zero product or a
  non-finite operand makes the binary32 sum exact. The GLSL emitter uses the same definition (:func:`glsl_source`).
- ``cgl_half_to_bits`` and ``cgl_half_from_bits`` convert between a binary16 value and its encoding.

Vector forms apply the scalar helper per lane, and the matrix form of ``cgl_half_round`` (``lanes`` is a
``(columns, rows)`` shape, named ``cgl_half_roundCxR``) applies the vector form per column.
"""

from __future__ import annotations

import functools

from .rewrite import clone

HALF_HELPER_MARK = "cgl_rule_helper"

_ROUND = """
float cgl_half_round(float x0) {
    uint bits = asuint(x0);
    uint sign = bits & 0x80000000u;
    uint magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) {
        return asfloat(sign | (magnitude == 0x7f800000u ? 0x7f800000u : (0x7fc00000u | (magnitude & 0x003fe000u))));
    }
    if (magnitude >= 0x477ff000u) {
        return asfloat(sign | 0x7f800000u);
    }
    if (magnitude >= 0x38800000u) {
        return asfloat(sign | ((magnitude + 0x00000fffu + ((magnitude >> 13u) & 1u)) & 0xffffe000u));
    }
    if (magnitude < 0x33000000u) {
        return asfloat(sign);
    }
    uint shift = 126u - (magnitude >> 23u);
    uint significand = (magnitude & 0x007fffffu) | 0x00800000u;
    uint rounded = significand >> shift;
    uint remainder = significand & ((1u << shift) - 1u);
    uint halfway = 1u << (shift - 1u);
    if (remainder > halfway || (remainder == halfway && (rounded & 1u) != 0u)) {
        rounded = rounded + 1u;
    }
    return asfloat(sign | asuint(float(rounded) * 5.9604644775390625e-8));
}
"""

_INDEX = """
uint cgl_half_index(float x0) {
    uint magnitude = asuint(x0) & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) {
        return 0x7c00u;
    }
    if (magnitude >= 0x38800000u) {
        return (magnitude >> 13u) - 0x0001c000u;
    }
    return uint(asfloat(magnitude) * 16777216.0);
}
"""

_VALUE = """
float cgl_half_value(uint x0) {
    if (x0 < 0x400u) {
        return float(x0) * 5.9604644775390625e-8;
    }
    return asfloat((((x0 >> 10u) + 112u) << 23u) | ((x0 & 0x3ffu) << 13u));
}
"""

_MIDPOINT = """
float cgl_half_midpoint(uint x0) {
    return (cgl_half_value(x0) + cgl_half_value(x0 + 1u)) * 0.5;
}
"""

_DIV = """
float cgl_half_div(float x0, float x1) {
    float quotient = x0 / x1;
    uint a = asuint(x0);
    uint b = asuint(x1);
    uint ma = a & 0x7fffffffu;
    uint mb = b & 0x7fffffffu;
    if (ma == 0u || mb == 0u || ma >= 0x7f800000u || mb >= 0x7f800000u) {
        return cgl_half_round(quotient);
    }
    float numerator = asfloat(ma);
    float denominator = asfloat(mb);
    uint k = cgl_half_index(cgl_half_round(asfloat(asuint(quotient) & 0x7fffffffu)));
    bool up = false;
    if (k < 0x7c00u) {
        float above = numerator - cgl_half_midpoint(k) * denominator;
        up = above > 0.0 || (above == 0.0 && (k & 1u) != 0u);
    }
    if (up) {
        k = k + 1u;
    } else if (k > 0u) {
        float below = numerator - cgl_half_midpoint(k - 1u) * denominator;
        if (below < 0.0 || (below == 0.0 && (k & 1u) != 0u)) {
            k = k - 1u;
        }
    }
    uint sign = (a ^ b) & 0x80000000u;
    if (k >= 0x7c00u) {
        return asfloat(sign | 0x7f800000u);
    }
    return asfloat(sign | asuint(cgl_half_value(k)));
}
"""

_SQRT = """
float cgl_half_sqrt(float x0) {
    float root = sqrt(x0);
    uint bits = asuint(x0);
    if (bits == 0u || bits >= 0x7f800000u) {
        return cgl_half_round(root);
    }
    uint k = cgl_half_index(cgl_half_round(root));
    float above_middle = cgl_half_midpoint(k);
    float above = x0 - above_middle * above_middle;
    if (above > 0.0 || (above == 0.0 && (k & 1u) != 0u)) {
        k = k + 1u;
    } else if (k > 0u) {
        float below_middle = cgl_half_midpoint(k - 1u);
        float below = x0 - below_middle * below_middle;
        if (below < 0.0 || (below == 0.0 && (k & 1u) != 0u)) {
            k = k - 1u;
        }
    }
    return cgl_half_value(k);
}
"""

_FMA = """
float cgl_half_fma(float x0, float x1, float x2) {
    float product = x0 * x1;
    uint p = asuint(product);
    uint c = asuint(x2);
    uint mp = p & 0x7fffffffu;
    uint mc = c & 0x7fffffffu;
    if (mp == 0u || mp >= 0x7f800000u || mc >= 0x7f800000u) {
        return product + x2;
    }
    uint big = ((mp & 0x007fffffu) | 0x00800000u) << 6u;
    uint exponent = mp >> 23u;
    uint sign = p & 0x80000000u;
    uint small = 0u;
    uint small_exponent = exponent;
    uint small_sign = sign;
    if (mc != 0u) {
        small = ((mc & 0x007fffffu) | 0x00800000u) << 6u;
        small_exponent = mc >> 23u;
        small_sign = c & 0x80000000u;
    }
    if (small_exponent > exponent || (small_exponent == exponent && small > big)) {
        uint swap = big;
        big = small;
        small = swap;
        swap = exponent;
        exponent = small_exponent;
        small_exponent = swap;
        swap = sign;
        sign = small_sign;
        small_sign = swap;
    }
    uint distance = exponent - small_exponent;
    if (distance > 30u) {
        small = 1u;
    } else if (distance > 0u) {
        uint lost = small & ((1u << distance) - 1u);
        small = (small >> distance) | (lost != 0u ? 1u : 0u);
    }
    uint sum = sign == small_sign ? big + small : big - small;
    if (sum == 0u) {
        return 0.0;
    }
    uint top = sum >> 8u;
    uint msb = top != 0u ? (asuint(float(top)) >> 23u) - 119u : (asuint(float(sum)) >> 23u) - 127u;
    uint shift = 30u - msb;
    uint normalized = sum << shift;
    uint scale = exponent - shift;
    uint drop = scale >= 112u ? 20u : 132u - scale;
    if (drop > 31u) {
        return asfloat(sign);
    }
    uint q = normalized >> drop;
    uint remainder = normalized & ((1u << drop) - 1u);
    uint halfway = 1u << (drop - 1u);
    if (remainder > halfway || (remainder == halfway && (q & 1u) != 0u)) {
        q = q + 1u;
    }
    if (q == 0u) {
        return asfloat(sign);
    }
    if (scale > 141u || (scale == 141u && q > 2047u)) {
        return asfloat(sign | 0x7f800000u);
    }
    return asfloat(sign | ((asuint(float(q)) + ((scale + drop) << 23u)) - (156u << 23u)));
}
"""

_TO_BITS = """
uint cgl_half_to_bits(float x0) {
    uint bits = asuint(x0);
    uint magnitude = bits & 0x7fffffffu;
    uint sign = (bits >> 16u) & 0x8000u;
    if (magnitude > 0x7f800000u) {
        uint payload = (magnitude >> 13u) & 0x3ffu;
        return sign | 0x7c00u | (payload == 0u ? 0x200u : payload);
    }
    return sign | cgl_half_index(x0);
}
"""

_FROM_BITS = """
float cgl_half_from_bits(uint x0) {
    uint sign = (x0 & 0x8000u) << 16u;
    uint k = x0 & 0x7fffu;
    if (k >= 0x7c00u) {
        return asfloat(sign | 0x7f800000u | ((k & 0x3ffu) << 13u));
    }
    return asfloat(sign | asuint(cgl_half_value(k)));
}
"""

# name: (source, arity, helpers it calls). The order is the definition order in the output.
SCALAR_HELPERS = {
    "cgl_half_round": (_ROUND, 1, ()),
    "cgl_half_index": (_INDEX, 1, ()),
    "cgl_half_value": (_VALUE, 1, ()),
    "cgl_half_midpoint": (_MIDPOINT, 1, ("cgl_half_value",)),
    "cgl_half_div": (
        _DIV,
        2,
        ("cgl_half_round", "cgl_half_index", "cgl_half_value", "cgl_half_midpoint"),
    ),
    "cgl_half_sqrt": (
        _SQRT,
        1,
        ("cgl_half_round", "cgl_half_index", "cgl_half_value", "cgl_half_midpoint"),
    ),
    "cgl_half_fma": (_FMA, 3, ()),
    "cgl_half_to_bits": (_TO_BITS, 1, ("cgl_half_index",)),
    "cgl_half_from_bits": (_FROM_BITS, 1, ("cgl_half_value",)),
}
# Helpers with a per-lane vector form; their operands and results are floats.
LANE_HELPERS = ("cgl_half_round", "cgl_half_div", "cgl_half_sqrt", "cgl_half_fma")
# Helpers with a per-column matrix form.
MATRIX_HELPERS = ("cgl_half_round",)
_LANES = "xyzw"


def helper_name(base: str, lanes) -> str:
    if isinstance(lanes, tuple):
        columns, rows = lanes
        return f"{base}{columns}x{rows}"
    return base if lanes == 1 else f"{base}{lanes}"


def matrix_name(shape) -> str:
    """The CrossGL spelling of the float matrix with ``shape`` ``(columns, rows)``."""
    columns, rows = shape
    return f"mat{columns}" if columns == rows else f"mat{columns}x{rows}"


def _matrix_source(base: str, shape) -> str:
    columns, rows = shape
    matrix = matrix_name(shape)
    per_column = ", ".join(
        f"{helper_name(base, rows)}(x0[{column}])" for column in range(columns)
    )
    return (
        f"{matrix} {helper_name(base, shape)}({matrix} x0) "
        f"{{ return {matrix}({per_column}); }}"
    )


def _vector_source(base: str, lanes: int) -> str:
    arity = SCALAR_HELPERS[base][1]
    vector = f"vec{lanes}"
    parameters = ", ".join(f"{vector} x{index}" for index in range(arity))
    per_lane = ", ".join(
        f"{base}(" + ", ".join(f"x{index}.{lane}" for index in range(arity)) + ")"
        for lane in _LANES[:lanes]
    )
    return (
        f"{vector} {helper_name(base, lanes)}({parameters}) "
        f"{{ return {vector}({per_lane}); }}"
    )


def helper_source(base: str, lanes) -> str:
    """The CrossGL definition of ``base`` for ``lanes`` lanes (a ``(columns, rows)`` shape for a matrix)."""
    if isinstance(lanes, tuple):
        if base not in MATRIX_HELPERS:
            raise ValueError(f"{base} has no matrix form")
        return _matrix_source(base, lanes)
    if lanes == 1:
        return SCALAR_HELPERS[base][0]
    return _vector_source(base, lanes)


@functools.lru_cache(maxsize=None)
def _parsed(base: str, lanes):
    from ..source_registry import SOURCE_REGISTRY, register_default_sources

    register_default_sources()
    text = "shader __cgl_half_helpers {\n" + helper_source(base, lanes) + "\n}\n"
    program = SOURCE_REGISTRY.get("cgl").parse(text)
    (function,) = program.functions
    return function


def helper_function(base: str, lanes):
    """A fresh copy of the helper's ``FunctionNode``, marked so rule carrying leaves its body alone."""
    function = clone(_parsed(base, lanes))
    function.annotations[HALF_HELPER_MARK] = "half"
    return function


def is_half_helper(function) -> bool:
    annotations = getattr(function, "annotations", None) or {}
    return annotations.get(HALF_HELPER_MARK) == "half"


def glsl_source(base: str, name: str) -> str:
    """The scalar helper ``base`` in GLSL 4.00 spelling, named ``name``; only self-contained helpers qualify."""
    source, _, dependencies = SCALAR_HELPERS[base]
    if dependencies:
        raise ValueError(f"{base} calls other helpers")
    return (
        source.strip()
        .replace(f"float {base}(", f"float {name}(", 1)
        .replace("asuint(", "floatBitsToUint(")
        .replace("asfloat(", "uintBitsToFloat(")
        + "\n"
    )


def closure(required):
    """Every ``(base, lanes)`` the required helpers need, in definition order (scalar helpers before vector helpers
    and vector helpers before matrix helpers)."""
    needed = set(required)
    for base, lanes in list(needed):
        if isinstance(lanes, tuple):
            needed.add((base, lanes[1]))
    for base, lanes in list(needed):
        if lanes != 1:
            needed.add((base, 1))
    pending = [base for base, lanes in needed if lanes == 1]
    while pending:
        for dependency in SCALAR_HELPERS[pending.pop()][2]:
            if (dependency, 1) not in needed:
                needed.add((dependency, 1))
                pending.append(dependency)
    order = list(SCALAR_HELPERS)

    def key(item):
        base, lanes = item
        shape = lanes if isinstance(lanes, tuple) else (lanes,)
        rank = 2 if isinstance(lanes, tuple) else int(lanes != 1)
        return rank, order.index(base), shape

    return sorted(needed, key=key)
