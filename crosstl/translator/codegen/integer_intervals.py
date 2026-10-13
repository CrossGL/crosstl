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


def integer_arithmetic_intervals(operator, left, right, bits, signed):
    """Bound typed arithmetic; signed overflow cannot establish a range."""
    left = convert_integer_intervals(left, bits, signed)
    right = convert_integer_intervals(right, bits, signed)
    if not signed:
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
            else:
                return None
            if signed and not -(1 << (bits - 1)) <= bounds[0] <= bounds[1] < (
                1 << (bits - 1)
            ):
                return None
            result.append(bounds)
    return convert_integer_intervals(result, bits, signed)
