"""Fixed-width interval sets for source integer conversions and arithmetic."""


def merge_integer_intervals(intervals):
    result = []
    for lower, upper in sorted(intervals):
        if lower > upper:
            raise ValueError("Integer interval minimum exceeds maximum.")
        if result and lower <= result[-1][1] + 1:
            result[-1] = result[-1][0], max(result[-1][1], upper)
        else:
            result.append((lower, upper))
    # Widen excessive fragmentation without losing any reachable value.
    if len(result) > 16:
        return [(result[0][0], result[-1][1])]
    return result


def convert_integer_intervals(intervals, bits, signed):
    """Retain low bits, then interpret the destination sign bit if present."""
    if not 1 <= bits <= 64:
        raise ValueError("Integer conversion requires a width from 1 to 64 bits.")
    modulus = 1 << bits
    unsigned = []
    for lower, upper in intervals:
        if lower > upper:
            raise ValueError("Integer interval minimum exceeds maximum.")
        if upper - lower >= modulus - 1:
            unsigned = [(0, modulus - 1)]
            break
        first, last = lower % modulus, upper % modulus
        if lower // modulus == upper // modulus:
            unsigned.append((first, last))
        else:
            unsigned.extend(((0, last), (first, modulus - 1)))
    unsigned = merge_integer_intervals(unsigned)
    if not signed:
        return unsigned
    midpoint = modulus >> 1
    result = []
    for lower, upper in unsigned:
        if lower < midpoint:
            result.append((lower, min(upper, midpoint - 1)))
        if upper >= midpoint:
            result.append((max(lower, midpoint) - modulus, upper - modulus))
    return merge_integer_intervals(result)


def bitwise_interval(operator, left, right):
    """Bound ``&``, ``|`` or ``^`` of two's-complement values in ``left`` and ``right``.

    ``x & y`` keeps only bits of a non-negative operand, so it lies in ``[0, x]``. For
    non-negative operands ``x | y`` and ``x ^ y`` set no bit above the wider operand's
    highest bit, and ``x | y`` is at least either operand. Other sign mixes are not bounded.
    """
    (a, b), (c, d) = left, right
    if operator == "&":
        if a < 0 and c < 0:
            return None
        upper = min(bound for low, bound in (left, right) if low >= 0)
        return 0, upper
    if operator not in {"|", "^"} or a < 0 or c < 0:
        return None
    upper = (1 << max(b, d).bit_length()) - 1
    return (max(a, c) if operator == "|" else 0), upper


def integer_arithmetic_intervals(operator, left, right, bits, signed):
    """Bound typed arithmetic; signed overflow cannot establish a range."""
    left = convert_integer_intervals(left, bits, signed)
    right = convert_integer_intervals(right, bits, signed)
    if not signed and operator in {"+", "-", "*"}:
        # Congruent signed representatives tighten bounds around unsigned wrap.
        left = convert_integer_intervals(left, bits, True)
        right = convert_integer_intervals(right, bits, True)
    result = []
    for a, b in left:
        for c, d in right:
            if operator == "+":
                bounds = a + c, b + d
            elif operator == "-":
                bounds = a - d, b - c
            elif operator == "*":
                products = a * c, a * d, b * c, b * d
                bounds = min(products), max(products)
            elif operator in {"/", "%"}:
                if c <= 0 <= d or (signed and a == -(1 << (bits - 1)) and c <= -1 <= d):
                    return None
                if operator == "/":

                    def quotient(x, y):
                        value = abs(x) // abs(y)
                        return -value if (x < 0) != (y < 0) else value

                    values = [quotient(x, y) for x in (a, b) for y in (c, d)]
                    bounds = min(values), max(values)
                elif a == b and c == d:
                    remainder = abs(a) % abs(c)
                    value = -remainder if a < 0 else remainder
                    bounds = value, value
                else:
                    maximum = max(abs(c), abs(d)) - 1
                    bounds = max(a, -maximum) if a < 0 else 0, (
                        min(b, maximum) if b > 0 else 0
                    )
            elif operator in {"&", "|", "^"}:
                bounds = bitwise_interval(operator, (a, b), (c, d))
                if bounds is None:
                    return None
            else:
                return None
            if signed and not -(1 << (bits - 1)) <= bounds[0] <= bounds[1] < (
                1 << (bits - 1)
            ):
                return None
            result.append(bounds)
    return convert_integer_intervals(result, bits, signed)
