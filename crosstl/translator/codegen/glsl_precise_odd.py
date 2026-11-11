"""Binary32 GLSL helpers for Metal's ``precise::`` sinh, tanh, asinh and atanh.

GLSL defines these builtins through ``exp`` and ``log``, and Mesa evaluates
them that way. Besides losing the sign of zero and mapping ``tanh(NaN)`` to
-1, the formulas cancel near zero: ``(exp(x) - exp(-x)) / 2`` and
``log((1 + x) / (1 - x)) / 2`` keep only the absolute error of ``exp`` and
``log``, so ``sinh(0.0015)`` is off by about 1e-5 relatively. Metal's
precise implementations stay within a few units in the last place there.

Each helper therefore evaluates small magnitudes through an odd series whose
terms are all positive, so no rounding cancels, and leaves larger magnitudes
to the builtin, whose formula no longer cancels there:

- ``sinh(a)`` for ``a < 1`` sums the Taylor series to ``a^11 / 11!``; the
  first omitted term is below ``2^-31`` of the result.
- ``tanh(a)`` for ``a < 1`` is ``sinh(a) / sqrt(1 + sinh(a)^2)`` with that
  series.
- ``atanh(a)`` for ``a < 0.5`` sums ``a^(2k+1) / (2k+1)`` to ``a^23 / 23``;
  the first omitted term is below ``2^-28`` of the result.
- ``asinh(a)`` for ``a < 0.5`` is ``atanh(a / sqrt(1 + a^2))`` with that
  series, whose argument stays below 0.45.

Simulated in binary32 over every seventh binary32 magnitude in ``[2^-30, 1)``,
the series paths stay within 0.85 (sinh), 2.7 (tanh), 0.72 (atanh) and 2.5
(asinh) units in the last place of the exact result. The result takes the
operand's sign bit, and NaN passes through.
"""

SINH_COEFFICIENTS = (
    "0.16666666666666666667",
    "0.0083333333333333333333",
    "0.00019841269841269841270",
    "0.0000027557319223985890653",
    "0.000000025052108385441718775",
)
ATANH_COEFFICIENTS = tuple(f"{1.0 / (2 * k + 3):.20g}" for k in range(11))
SERIES_THRESHOLDS = {"sinh": "1.0", "tanh": "1.0", "asinh": "0.5", "atanh": "0.5"}
PRECISE_ODD_OPERATIONS = frozenset(SERIES_THRESHOLDS)


def _odd_series(value, squared, coefficients):
    """``value + value * squared * p(squared)`` with ``p`` in Horner form."""
    polynomial = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        polynomial = f"{coefficient} + {squared} * ({polynomial})"
    return f"{value} + {value} * {squared} * ({polynomial})"


def _small_magnitude_lines(operation, value_type):
    if operation == "sinh":
        return [
            f"{value_type} small = "
            + _odd_series("magnitude", "squared", SINH_COEFFICIENTS)
            + ";"
        ]
    if operation == "tanh":
        return [
            f"{value_type} hyperbolicSine = "
            + _odd_series("magnitude", "squared", SINH_COEFFICIENTS)
            + ";",
            f"{value_type} small = hyperbolicSine / sqrt(1.0 + hyperbolicSine * hyperbolicSine);",
        ]
    if operation == "atanh":
        return [
            f"{value_type} small = "
            + _odd_series("magnitude", "squared", ATANH_COEFFICIENTS)
            + ";"
        ]
    if operation == "asinh":
        return [
            f"{value_type} reduced = magnitude / sqrt(1.0 + squared);",
            f"{value_type} reducedSquared = reduced * reduced;",
            f"{value_type} small = "
            + _odd_series("reduced", "reducedSquared", ATANH_COEFFICIENTS)
            + ";",
        ]
    raise ValueError(f"Unsupported precise odd function: {operation}")


def glsl_precise_odd_helper(name, operation, value_type, width):
    """Return the GLSL helper ``name`` computing precise ``operation``.

    ``value_type`` is ``float`` or ``vecN`` for ``width`` N.
    """
    if operation not in PRECISE_ODD_OPERATIONS:
        raise ValueError(f"Unsupported precise odd function: {operation}")
    if width not in (1, 2, 3, 4):
        raise ValueError(f"Unsupported precise odd function width: {width}")
    threshold = SERIES_THRESHOLDS[operation]
    if width == 1:
        unsigned = "uint"
        selected = f"magnitude < {threshold} ? small : {operation}(magnitude)"
        nan = "isnan(value) ? value : result"
    else:
        unsigned = f"uvec{width}"
        selected = (
            f"mix({operation}(magnitude), small, "
            f"lessThan(magnitude, {value_type}({threshold})))"
        )
        nan = "mix(result, value, isnan(value))"
    lines = [
        f"{value_type} magnitude = abs(value);",
        f"{value_type} squared = magnitude * magnitude;",
        *_small_magnitude_lines(operation, value_type),
        f"{unsigned} bits = floatBitsToUint({selected}) & 0x7fffffffu;",
        f"{value_type} result = uintBitsToFloat("
        "bits | (floatBitsToUint(value) & 0x80000000u));",
        f"return {nan};",
    ]
    body = "".join(f"    {line}\n" for line in lines)
    return f"{value_type} {name}({value_type} value) {{\n{body}}}\n"
