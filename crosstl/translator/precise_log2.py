"""Binary32 base-two logarithms with independent operand and accuracy policies."""


def binary32_log2_support(name, operand_profile, accuracy_profile):
    prefix = ""
    if operand_profile == "flush-subnormals":
        prefix = "    if (magnitude < 0x00800000u) { return asfloat(0xff800000u); }\n"
    elif operand_profile is None:
        prefix = "    if (magnitude != 0u && magnitude < 0x00800000u) { return log2(value); }\n"
    if accuracy_profile is None:
        if operand_profile != "preserve-subnormals":
            return (
                f"@metal_static\nfloat {name}(float value) {{\n"
                "    uint magnitude = asuint(value) & 0x7fffffffu;\n"
                + prefix
                + "    return log2(value);\n}\n"
            )
        prefix += "    if (magnitude == 0u || magnitude >= 0x00800000u) { return log2(value); }\n"
    normalization = """    int exponent = int(magnitude >> 23u) - 127;
    uint fraction = magnitude & 0x007fffffu;
    if (exponent == -127) {
        exponent = -126;
        while ((fraction & 0x00800000u) == 0u) {
            fraction <<= 1u;
            exponent -= 1;
        }
        fraction &= 0x007fffffu;
    }
    float reduced @precise = asfloat(0x3f800000u | fraction);
"""
    approximation = """    if (reduced > 1.4142135623730950488) {
        reduced *= 0.5;
        exponent += 1;
    }
    float ratio @precise = (reduced - 1.0) / (reduced + 1.0);
    float squared @precise = ratio * ratio;
    float polynomial @precise = 1.0 / 13.0;
    polynomial = 1.0 / 11.0 + squared * polynomial;
    polynomial = 1.0 / 9.0 + squared * polynomial;
    polynomial = 1.0 / 7.0 + squared * polynomial;
    polynomial = 1.0 / 5.0 + squared * polynomial;
    polynomial = 1.0 / 3.0 + squared * polynomial;
    float correction @precise = (2.0 * ratio * squared) * polynomial;
    float logarithm @precise = 2.0 * ratio + correction;
    return float(exponent) + logarithm * 1.4426950408889634074;
"""
    # Normalize in integer storage; the reduced odd series avoids cancellation
    # near one and never depends on arithmetic with a subnormal operand.
    return (
        f"@precise\n@metal_static\nfloat {name}(float value) {{\n"
        "    uint bits = asuint(value);\n"
        "    uint magnitude = bits & 0x7fffffffu;\n"
        + prefix
        + """    if (magnitude == 0u) { return asfloat(0xff800000u); }
    if (magnitude > 0x7f800000u || (bits & 0x80000000u) != 0u) {
        return asfloat(0x7fc00000u);
    }
    if (magnitude == 0x7f800000u) { return asfloat(magnitude); }
"""
        + normalization
        + (
            approximation
            if accuracy_profile == "portable-finite"
            else "    return float(exponent) + log2(reduced);\n"
        )
        + "}\n"
    )
