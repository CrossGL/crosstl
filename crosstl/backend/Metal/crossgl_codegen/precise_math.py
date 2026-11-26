"""Precise, round and log2 math helpers."""

from ....translator.precise_exp import binary32_exp_support
from ....translator.precise_log2 import (
    LOG2_PROFILED_OPERATIONS,
    LOG10_OF_TWO,
    binary32_log2_support,
)
from ....translator.precise_trig import (
    TRIG_HELPER_KEYS,
    binary32_trig_support,
)
from .support import (
    MetalPreciseMathLoweringError,
    MetalRoundLoweringError,
)


class MetalPreciseMathMixin:
    """Members of ``MetalToCrossGLConverter``: precise, round and log2 math helpers."""

    def map_metal_precise_atan2_function_name(self, args):
        arguments = list(args or [])
        types = [self.expression_metal_type(argument) for argument in arguments]
        location = getattr(arguments[0], "source_location", None) if arguments else None

        def unsupported(reason):
            return MetalPreciseMathLoweringError(
                "atan2", ", ".join(str(value) for value in types), reason, location
            )

        if len(arguments) != 2:
            raise unsupported("the operation requires exactly two operands")
        if self.current_function is None:
            raise unsupported(
                "global initializers cannot call the runtime arctangent helper"
            )
        widths = []
        for value_type in types:
            info = self.metal_math_builtin_type_info(value_type)
            if info is None or info["category"] not in {"floating", "bfloat"}:
                raise unsupported("both operands must have known floating-point types")
            width = info["width"]
            scalar = self.metal_scalar_arithmetic_type_info(info["element_type"])
            if width not in {1, 2, 3, 4} or not (
                scalar == ("floating", True, 32)
                or (
                    width == 1
                    and (
                        info["category"] == "bfloat" or scalar == ("floating", True, 16)
                    )
                )
            ):
                raise unsupported(
                    "precise arctangent requires binary32 scalar or vector operands"
                )
            widths.append(width)
        if widths[0] != widths[1] and 1 not in widths:
            raise unsupported("vector operands must have matching widths")
        width = max(widths)
        self.required_metal_precise_atan2_widths.add(width)
        self.required_metal_precise_atan_widths.add(1)
        return self.metal_precise_atan2_helper_name(width)

    def generate_metal_precise_atan2_call(self, expression, is_main=False):
        if (
            self.metal_math_builtin_namespace_mode(expression.name) != "precise"
            or self.resolve_metal_math_builtin_name(expression.name, expression.args)
            != "atan2"
        ):
            return None
        name = self.map_metal_precise_atan2_function_name(expression.args)
        types = [
            self.metal_math_builtin_type_info(self.expression_metal_type(argument))
            for argument in expression.args
        ]
        width = max(info["width"] for info in types)
        mapped = "float" if width == 1 else f"vec{width}"
        arguments = []
        for argument, info in zip(expression.args, types):
            value = self.generate_expression(argument, is_main)
            if info["width"] != width or self.metal_scalar_arithmetic_type_info(
                info["element_type"]
            ) != ("floating", True, 32):
                value = f"{mapped}({value})"
            arguments.append(value)
        return f"{name}({', '.join(arguments)})"

    def generate_metal_round_call(self, expression, is_main=False):
        if str(expression.name).rsplit("::", 1)[-1] != "round":
            return None
        selected = self.selected_metal_callable(expression)
        bfloat_wrapper = self.is_materialized_metal_stdlib_wrapper(selected) and (
            self.normalized_metal_type(
                self.resolve_type_alias(
                    self.selected_metal_callable_return_type(selected)
                )
            )
            in self.metal_source_bfloat_types
        )
        if bfloat_wrapper:
            result_type = "float"
        elif (
            self.resolve_metal_math_builtin_name(expression.name, expression.args)
            == "round"
        ):
            result_type = self.metal_math_builtin_result_type(expression)
        else:
            return None
        info = self.metal_math_builtin_type_info(result_type)
        element = (
            self.normalized_metal_type(self.resolve_type_alias(info["element_type"]))
            if info is not None
            else None
        )
        if (
            info is None
            or element not in {"float", "half"}
            or info["width"] not in {1, 2, 3, 4}
        ):
            raise MetalRoundLoweringError(
                result_type,
                "expected a float or half scalar or vector with at most four lanes",
                getattr(expression, "source_location", None),
            )
        if self.current_function is None:
            raise MetalRoundLoweringError(
                result_type,
                "global initializers cannot call the runtime rounding helper",
                getattr(expression, "source_location", None),
            )
        width = info["width"]
        self.required_metal_round_widths.add(width)
        argument = expression.args[0]
        value = self.generate_expression(argument, is_main)
        parameter_type = (
            self.metal_source_overload_parameter_type(selected.params[0])
            if bfloat_wrapper
            else result_type
        )
        if self.metal_source_overload_type_identity(
            self.expression_metal_type(argument)
        ) != self.metal_source_overload_type_identity(parameter_type):
            value = f"{self.map_type(parameter_type)}({value})"
        mapped = "float" if width == 1 else f"vec{width}"
        result = f"{self.metal_round_helper_name(width)}({mapped}({value}))"
        return (
            f"{self.map_type(result_type)}({result})" if element == "half" else result
        )

    def metal_round_helper_name(self, width):
        suffix = "" if width == 1 else str(width)
        return self.metal_precise_math_unique_helper_name(
            f"round-float{suffix}", f"__crossgl_metal_round_float{suffix}"
        )

    def generate_metal_round_support_code(self, indent=0):
        if not self.required_metal_round_widths:
            return ""
        scalar = self.metal_round_helper_name(1)
        code = f"@metal_static\nfloat {scalar}(float value) {{\n" + """
    uint bits = asuint(value);
    uint magnitude = bits & 0x7fffffffu;
    uint sign = bits & 0x80000000u;
    // Integral binary32 values, infinities and NaNs need no rounding.
    if (magnitude >= 0x4b000000u) { return value; }
    // Metal round produces positive zero for magnitudes below one half.
    if (magnitude < 0x3f000000u) { return 0.0; }
    if (magnitude < 0x3f800000u) { return asfloat(sign | 0x3f800000u); }
    // Integer-bit rounding avoids losing a half-unit near 2^23.
    uint shift = 150u - (magnitude >> 23u);
    uint unit = 1u << shift;
    uint rounded = (magnitude + (unit >> 1u)) & ~(unit - 1u);
    return asfloat(sign | rounded);
}
"""
        for width in sorted(self.required_metal_round_widths - {1}):
            lanes = ", ".join(f"{scalar}(value.{lane})" for lane in "xyzw"[:width])
            code += (
                f"@metal_static\nvec{width} {self.metal_round_helper_name(width)}(vec{width} value) {{\n"
                f"    return vec{width}({lanes});\n}}\n"
            )
        pad = "    " * indent
        return "".join(pad + line + "\n" for line in code.strip().splitlines())

    def generate_metal_log2_call(self, expression, is_main=False):
        # log10 shares the profiled base-two logarithm: log10(x) is
        # log2(x) * log10(2), so its result keeps the helper's finite
        # accuracy and operand policy instead of the target's log2 near one.
        operation = str(expression.name).rsplit("::", 1)[-1]
        if (
            operation not in LOG2_PROFILED_OPERATIONS
            or self.metal_math_builtin_namespace_mode(expression.name) == "fast"
            or (
                self.binary32_log2_operand_profile is None
                and self.binary32_log2_accuracy_profile is None
            )
        ):
            return None
        selected = self.selected_metal_callable(expression)
        bfloat_wrapper = self.is_materialized_metal_stdlib_wrapper(selected) and (
            self.normalized_metal_type(
                self.resolve_type_alias(
                    self.selected_metal_callable_return_type(selected)
                )
            )
            in self.metal_source_bfloat_types
        )
        if bfloat_wrapper:
            result_type = "float"
        elif (
            self.resolve_metal_math_builtin_name(expression.name, expression.args)
            == operation
        ):
            result_type = self.metal_math_builtin_result_type(expression)
        else:
            return None
        info = self.metal_math_builtin_type_info(result_type)
        if info is None:
            return None
        element = self.normalized_metal_type(
            self.resolve_type_alias(info["element_type"])
        )
        width = info["width"]
        if element not in {"float", "half"} or width not in {1, 2, 3, 4}:
            raise MetalPreciseMathLoweringError(
                operation,
                result_type,
                "the selected profile requires binary32 computation",
                getattr(expression, "source_location", None),
            )
        if self.current_function is None:
            raise MetalPreciseMathLoweringError(
                operation,
                result_type,
                "global initializers cannot call the runtime logarithm helper",
                getattr(expression, "source_location", None),
            )
        self.required_metal_log2_widths.add(width)
        argument = expression.args[0]
        value = self.generate_expression(argument, is_main)
        parameter_type = (
            self.metal_source_overload_parameter_type(selected.params[0])
            if bfloat_wrapper
            else result_type
        )
        if self.metal_source_overload_type_identity(
            self.expression_metal_type(argument)
        ) != self.metal_source_overload_type_identity(parameter_type):
            value = f"{self.map_type(parameter_type)}({value})"
        mapped = "float" if width == 1 else f"vec{width}"
        result = f"{self.metal_log2_helper_name(width)}({mapped}({value}))"
        if operation == "log10":
            result = f"({result} * {LOG10_OF_TWO})"
        return (
            f"{self.map_type(result_type)}({result})" if element == "half" else result
        )

    def metal_log2_helper_name(self, width):
        suffix = "" if width == 1 else str(width)
        return self.metal_precise_math_unique_helper_name(
            f"log2-float{suffix}", f"__crossgl_metal_log2_float{suffix}"
        )

    def generate_metal_log2_support_code(self, indent=0):
        if not self.required_metal_log2_widths:
            return ""
        scalar = self.metal_log2_helper_name(1)
        code = binary32_log2_support(
            scalar,
            self.binary32_log2_operand_profile,
            self.binary32_log2_accuracy_profile,
        )
        for width in sorted(self.required_metal_log2_widths - {1}):
            lanes = ", ".join(f"{scalar}(value.{lane})" for lane in "xyzw"[:width])
            code += (
                f"@precise\n@metal_static\nvec{width} {self.metal_log2_helper_name(width)}(vec{width} value) {{\n"
                f"    return vec{width}({lanes});\n}}\n"
            )
        pad = "    " * indent
        return "".join(pad + line + "\n" for line in code.strip().splitlines())

    def metal_precise_math_unique_helper_name(self, key, base_name):
        existing = self.metal_precise_math_helper_names.get(key)
        if existing is not None:
            return existing
        used_names = set(self.wide_vector_reserved_names)
        used_names.update(self.metal_precise_math_helper_names.values())
        used_names.update(
            self.sanitize_identifier(name) for name in self.user_function_names
        )
        helper_name = base_name
        while helper_name in used_names:
            helper_name += "_"
        self.metal_precise_math_helper_names[key] = helper_name
        self.wide_vector_reserved_names.add(helper_name)
        return helper_name

    def metal_precise_acos_ratio_helper_name(self):
        return self.metal_precise_math_unique_helper_name(
            "acos-ratio",
            "__crossgl_metal_precise_acos_ratio",
        )

    def metal_precise_acos_helper_name(self, width):
        self.metal_precise_acos_ratio_helper_name()
        scalar_name = self.metal_precise_math_unique_helper_name(
            "acos-float",
            "__crossgl_metal_precise_acos_float",
        )
        if width == 1:
            return scalar_name
        return self.metal_precise_math_unique_helper_name(
            f"acos-float{width}",
            f"__crossgl_metal_precise_acos_float{width}",
        )

    def metal_precise_asin_helper_name(self, width):
        self.metal_precise_acos_helper_name(1)
        scalar_name = self.metal_precise_math_unique_helper_name(
            "asin-float", "__crossgl_metal_precise_asin_float"
        )
        if width == 1:
            return scalar_name
        return self.metal_precise_math_unique_helper_name(
            f"asin-float{width}", f"__crossgl_metal_precise_asin_float{width}"
        )

    def generate_metal_precise_asin_support_code(self, indent=0):
        if not self.required_metal_precise_asin_widths:
            return ""
        pad = "    " * indent
        body_pad = "    " * (indent + 1)
        nested_pad = "    " * (indent + 2)
        scalar_name = self.metal_precise_asin_helper_name(1)
        ratio_name = self.metal_precise_acos_ratio_helper_name()
        acos_name = self.metal_precise_acos_helper_name(1)
        # Near zero use the shared fdlibm rational correction; subtracting
        # acos from pi/2 there would lose relative accuracy and signed zero.
        code = f"{pad}// Portable float32 lowering for Metal precise::asin.\n"
        code += f"{pad}@precise\n{pad}@metal_static\n"
        code += f"{pad}float {scalar_name}(float value) {{\n"
        code += f"{body_pad}uint bits = asuint(value);\n"
        code += f"{body_pad}uint magnitude_bits = bits & 0x7fffffffu;\n"
        code += f"{body_pad}if (magnitude_bits > 0x3f800000u) {{\n"
        code += f"{nested_pad}return asfloat(0x7fc00000u);\n{body_pad}}}\n"
        code += f"{body_pad}if (magnitude_bits < 0x39800000u) {{\n"
        code += f"{nested_pad}return value;\n{body_pad}}}\n"
        code += f"{body_pad}if (magnitude_bits < 0x3f000000u) {{\n"
        code += f"{nested_pad}float squared @precise = value * value;\n"
        code += f"{nested_pad}float correction @precise = {ratio_name}(squared);\n"
        code += f"{nested_pad}return value + value * correction;\n{body_pad}}}\n"
        code += f"{body_pad}float magnitude = asfloat(magnitude_bits);\n"
        code += f"{body_pad}float result @precise = 1.5707962513 - (\n"
        code += f"{nested_pad}{acos_name}(magnitude) - 0.000000075497894159\n"
        code += f"{body_pad});\n"
        code += f"{body_pad}return (bits >> 31u) != 0u ? -result : result;\n"
        code += f"{pad}}}\n\n"
        for width in sorted(self.required_metal_precise_asin_widths):
            if width == 1:
                continue
            vector_name = self.metal_precise_asin_helper_name(width)
            arguments = ",\n".join(
                f"{nested_pad}{scalar_name}(value.{component})"
                for component in "xyzw"[:width]
            )
            code += f"{pad}@precise\n{pad}@metal_static\n"
            code += f"{pad}vec{width} {vector_name}(vec{width} value) {{\n"
            code += f"{body_pad}return vec{width}(\n{arguments}\n{body_pad});\n"
            code += f"{pad}}}\n\n"
        return code

    def generate_metal_precise_math_support_code(self, indent=0):
        independent_code = self.generate_metal_precise_trig_support_code(indent)
        independent_code += self.generate_metal_sincos_support_code(indent)
        independent_code += self.generate_metal_precise_acosh_support_code(indent)
        independent_code += self.generate_metal_precise_atan_support_code(indent)
        independent_code += self.generate_metal_precise_atan2_support_code(indent)
        independent_code += self.generate_metal_power_support_code(indent)
        independent_code += self.generate_metal_round_support_code(indent)
        independent_code += self.generate_metal_precise_exp_support_code(indent)
        independent_code += self.generate_metal_precise_log_support_code(indent)
        independent_code += self.generate_metal_log2_support_code(indent)
        independent_code += self.generate_metal_precise_sqrt_support_code(indent)
        independent_code += self.generate_metal_precise_rsqrt_support_code(indent)
        widths = sorted(self.required_metal_precise_acos_widths)
        if not widths and not self.required_metal_precise_asin_widths:
            return independent_code

        ratio_name = self.metal_precise_acos_ratio_helper_name()
        scalar_name = self.metal_precise_acos_helper_name(1)
        pad = "    " * indent
        body_pad = "    " * (indent + 1)
        nested_pad = "    " * (indent + 2)

        # The float32 acos range reduction and coefficients are derived from
        # fdlibm. Copyright (C) 1993 by Sun Microsystems, Inc. All rights
        # reserved. Developed at SunPro. Permission to use, copy, modify, and
        # distribute this software is freely granted, provided this notice is
        # preserved.
        code = f"{pad}// Portable float32 lowering for Metal precise::acos.\n"
        code += f"{pad}// Derived from fdlibm. Copyright (C) 1993 by Sun\n"
        code += f"{pad}// Microsystems, Inc. All rights reserved. Developed at\n"
        code += f"{pad}// SunPro. Permission to use, copy, modify, and distribute\n"
        code += f"{pad}// this software is freely granted, provided this notice\n"
        code += f"{pad}// is preserved.\n"
        code += f"{pad}@source_license(fdlibm)\n"
        code += f"{pad}@precise\n"
        code += f"{pad}@metal_static\n"
        code += f"{pad}float {ratio_name}(float value) {{\n"
        code += (
            f"{body_pad}float numerator @precise = value * (\n"
            f"{nested_pad}0.16666586697 + value * (\n"
            f"{nested_pad}    -0.042743422091 + value * -0.008656363003\n"
            f"{nested_pad})\n"
            f"{body_pad});\n"
        )
        code += (
            f"{body_pad}float denominator @precise = " "1.0 + value * -0.70662963390;\n"
        )
        code += f"{body_pad}return numerator / denominator;\n"
        code += f"{pad}}}\n\n"

        code += f"{pad}@precise\n"
        code += f"{pad}@metal_static\n"
        code += f"{pad}float {scalar_name}(float value) {{\n"
        code += f"{body_pad}uint bits = asuint(value);\n"
        code += f"{body_pad}uint magnitude_bits = bits & 0x7fffffffu;\n"
        code += f"{body_pad}if (magnitude_bits >= 0x3f800000u) {{\n"
        code += f"{nested_pad}if (magnitude_bits == 0x3f800000u) {{\n"
        code += (
            f"{nested_pad}    return (bits >> 31u) != 0u\n"
            f"{nested_pad}        ? 3.14159265358979323846\n"
            f"{nested_pad}        : 0.0;\n"
        )
        code += f"{nested_pad}}}\n"
        code += f"{nested_pad}return asfloat(0x7fc00000u);\n"
        code += f"{body_pad}}}\n"
        code += f"{body_pad}if (magnitude_bits < 0x3f000000u) {{\n"
        code += f"{nested_pad}if (magnitude_bits <= 0x32800000u) {{\n"
        code += f"{nested_pad}    return 1.5707962513;\n"
        code += f"{nested_pad}}}\n"
        code += f"{nested_pad}float squared @precise = value * value;\n"
        code += f"{nested_pad}float correction @precise = {ratio_name}(squared);\n"
        code += (
            f"{nested_pad}return 1.5707962513 - (\n"
            f"{nested_pad}    value - (\n"
            f"{nested_pad}        0.000000075497894159 - value * correction\n"
            f"{nested_pad}    )\n"
            f"{nested_pad});\n"
        )
        code += f"{body_pad}}}\n"
        code += f"{body_pad}if ((bits >> 31u) != 0u) {{\n"
        code += (
            f"{nested_pad}float reduced @precise = (1.0 + value) * 0.5;\n"
            f"{nested_pad}float root @precise = sqrt(reduced);\n"
            f"{nested_pad}float correction @precise =\n"
            f"{nested_pad}    {ratio_name}(reduced) * root\n"
            f"{nested_pad}    - 0.000000075497894159;\n"
            f"{nested_pad}return 2.0 * (1.5707962513 - (root + correction));\n"
        )
        code += f"{body_pad}}}\n"
        code += (
            f"{body_pad}float reduced @precise = (1.0 - value) * 0.5;\n"
            f"{body_pad}float root @precise = sqrt(reduced);\n"
            f"{body_pad}float truncated_root @precise =\n"
            f"{body_pad}    asfloat(asuint(root) & 0xfffff000u);\n"
            f"{body_pad}float tail @precise =\n"
            f"{body_pad}    (reduced - truncated_root * truncated_root)\n"
            f"{body_pad}    / (root + truncated_root);\n"
            f"{body_pad}float correction @precise =\n"
            f"{body_pad}    {ratio_name}(reduced) * root + tail;\n"
            f"{body_pad}return 2.0 * (truncated_root + correction);\n"
        )
        code += f"{pad}}}\n\n"

        for width in widths:
            if width == 1:
                continue
            vector_name = self.metal_precise_acos_helper_name(width)
            components = "xyzw"[:width]
            arguments = ",\n".join(
                f"{nested_pad}{scalar_name}(value.{component})"
                for component in components
            )
            code += f"{pad}@precise\n"
            code += f"{pad}@metal_static\n"
            code += f"{pad}vec{width} {vector_name}(vec{width} value) {{\n"
            code += f"{body_pad}return vec{width}(\n{arguments}\n{body_pad});\n"
            code += f"{pad}}}\n\n"
        return (
            independent_code
            + code
            + self.generate_metal_precise_asin_support_code(indent)
        )

    def metal_precise_exp_helper_name(self, width):
        suffix = "" if width == 1 else str(width)
        return self.metal_precise_math_unique_helper_name(
            f"exp-float{suffix}", f"__crossgl_metal_precise_exp_float{suffix}"
        )

    def generate_metal_precise_exp_support_code(self, indent=0):
        if not self.required_metal_precise_exp_widths:
            return ""
        scalar = self.metal_precise_exp_helper_name(1)
        code = binary32_exp_support(scalar)
        for width in sorted(self.required_metal_precise_exp_widths - {1}):
            vector = self.metal_precise_exp_helper_name(width)
            arguments = ", ".join(f"{scalar}(value.{lane})" for lane in "xyzw"[:width])
            code += (
                f"@precise\n@metal_static\nvec{width} {vector}(vec{width} value) {{\n"
                f"    return vec{width}({arguments});\n}}\n"
            )
        pad = "    " * indent
        return "".join(pad + line + "\n" for line in code.strip().splitlines())

    def metal_precise_sqrt_helper_name(self, width):
        suffix = "" if width == 1 else str(width)
        return self.metal_precise_math_unique_helper_name(
            f"sqrt-float{suffix}", f"__crossgl_metal_precise_sqrt_float{suffix}"
        )

    def generate_metal_precise_sqrt_support_code(self, indent=0):
        if not self.required_metal_precise_sqrt_widths:
            return ""
        scalar = self.metal_precise_sqrt_helper_name(1)
        flush = "true" if self.binary32_sqrt_profile == "flush-subnormals" else "false"
        # Normalize to an even exponent, then take the integer square root of
        # the 48-bit radicand two bits at a time. For N = root^2 + remainder,
        # rounding sqrt(N) upward is exactly remainder > root; ties cannot occur.
        code = f"""@precise
@metal_static
float {scalar}(float value) {{
    uint bits = asuint(value);
    uint magnitude = bits & 0x7fffffffu;
    uint sign = bits & 0x80000000u;
    if ({flush} && magnitude < 0x00800000u) {{ magnitude = 0u; }}
    if (magnitude == 0u) {{ return asfloat(sign); }}
    if (magnitude > 0x7f800000u || sign != 0u) {{ return asfloat(0x7fc00000u); }}
    if (magnitude == 0x7f800000u) {{ return asfloat(magnitude); }}
    int exponent = int(magnitude >> 23u) - 127;
    uint significand = magnitude & 0x007fffffu;
    if (exponent == -127) {{
        exponent = -126;
        while ((significand & 0x00800000u) == 0u) {{
            significand <<= 1u;
            exponent -= 1;
        }}
    }} else {{
        significand |= 0x00800000u;
    }}
    if ((exponent & 1) != 0) {{
        significand <<= 1u;
        exponent -= 1;
    }}
    uint high = significand >> 9u;
    uint low = significand << 23u;
    uint root = 0u;
    uint remainder = 0u;
    for (int digit = 0; digit < 24; digit += 1) {{
        remainder = (remainder << 2u) | (high >> 14u);
        high = ((high << 2u) & 0xffffu) | (low >> 30u);
        low <<= 2u;
        root <<= 1u;
        uint trial = (root << 1u) | 1u;
        if (remainder >= trial) {{
            remainder -= trial;
            root |= 1u;
        }}
    }}
    if (remainder > root) {{ root += 1u; }}
    uint result = (uint(exponent / 2 + 127) << 23u) + root - 0x00800000u;
    return asfloat(result);
}}
"""
        for width in sorted(self.required_metal_precise_sqrt_widths - {1}):
            vector = self.metal_precise_sqrt_helper_name(width)
            arguments = ", ".join(f"{scalar}(value.{lane})" for lane in "xyzw"[:width])
            code += (
                f"@precise\n@metal_static\nvec{width} {vector}(vec{width} value) {{\n"
                f"    return vec{width}({arguments});\n}}\n"
            )
        pad = "    " * indent
        return "".join(pad + line + "\n" for line in code.strip().splitlines())

    def metal_precise_rsqrt_helper_name(self, width):
        suffix = "" if width == 1 else str(width)
        return self.metal_precise_math_unique_helper_name(
            f"rsqrt-float{suffix}", f"__crossgl_metal_precise_rsqrt_float{suffix}"
        )

    def generate_metal_precise_rsqrt_support_code(self, indent=0):
        if not self.required_metal_precise_rsqrt_widths:
            return ""
        scalar = self.metal_precise_rsqrt_helper_name(1)
        flush = "true" if self.binary32_rsqrt_profile == "flush-subnormals" else "false"
        # For even source exponent, the output significand is sqrt(2^71 / m).
        # Keep both division and integer-root remainders: if Q = n*n + S,
        # compare S + R/m with n + 1/4 to round once at the output midpoint.
        code = f"""@precise
@metal_static
float {scalar}(float value) {{
    uint bits = asuint(value);
    uint magnitude = bits & 0x7fffffffu;
    uint sign = bits & 0x80000000u;
    if ({flush} && magnitude < 0x00800000u) {{ magnitude = 0u; }}
    if (magnitude == 0u) {{ return asfloat(sign | 0x7f800000u); }}
    if (magnitude > 0x7f800000u || sign != 0u) {{ return asfloat(0x7fc00000u); }}
    if (magnitude == 0x7f800000u) {{ return 0.0f; }}
    int exponent = int(magnitude >> 23u) - 127;
    uint significand = magnitude & 0x007fffffu;
    if (exponent == -127) {{
        exponent = -126;
        while ((significand & 0x00800000u) == 0u) {{
            significand <<= 1u;
            exponent -= 1;
        }}
    }} else {{ significand |= 0x00800000u; }}
    if ((exponent & 1) != 0) {{
        significand <<= 1u;
        exponent -= 1;
    }}
    if (significand == 0x00800000u) {{
        return asfloat(uint(-exponent / 2 + 127) << 23u);
    }}
    uint remainder = 0x00800000u;
    uint high = 0u;
    uint low = 0u;
    for (int digit = 0; digit < 48; digit += 1) {{
        remainder <<= 1u;
        high = (high << 1u) | (low >> 31u);
        low <<= 1u;
        if (remainder >= significand) {{
            remainder -= significand;
            low |= 1u;
        }}
    }}
    uint root = 0u;
    uint residual = 0u;
    for (int digit = 0; digit < 24; digit += 1) {{
        residual = (residual << 2u) | (high >> 14u);
        high = ((high << 2u) & 0xffffu) | (low >> 30u);
        low <<= 2u;
        root <<= 1u;
        uint trial = (root << 1u) | 1u;
        if (residual >= trial) {{
            residual -= trial;
            root |= 1u;
        }}
    }}
    if (residual > root || (residual == root &&
        ((remainder << 2u) > significand ||
         ((remainder << 2u) == significand && (root & 1u) != 0u)))) {{
        root += 1u;
    }}
    return asfloat((uint(-exponent / 2 + 126) << 23u) + root - 0x00800000u);
}}
"""
        for width in sorted(self.required_metal_precise_rsqrt_widths - {1}):
            vector = self.metal_precise_rsqrt_helper_name(width)
            arguments = ", ".join(f"{scalar}(value.{lane})" for lane in "xyzw"[:width])
            code += (
                f"@precise\n@metal_static\nvec{width} {vector}(vec{width} value) {{\n"
                f"    return vec{width}({arguments});\n}}\n"
            )
        pad = "    " * indent
        return "".join(pad + line + "\n" for line in code.strip().splitlines())

    def metal_precise_log_helper_name(self, width):
        suffix = "" if width == 1 else str(width)
        return self.metal_precise_math_unique_helper_name(
            f"log-float{suffix}", f"__crossgl_metal_precise_log_float{suffix}"
        )

    def generate_metal_precise_log_support_code(self, indent=0):
        if not self.required_metal_precise_log_widths:
            return ""
        scalar = self.metal_precise_log_helper_name(1)
        flush = "true" if self.binary32_log_profile == "flush-subnormals" else "false"
        # Integer normalization avoids target subnormal arithmetic. Reducing
        # around one bounds |r| by (sqrt(2)-1)/(sqrt(2)+1); use the odd series
        # log(m) = 2*(r + r^3/3 + ...), and a split ln(2) for reconstruction.
        code = f"""@precise
@metal_static
float {scalar}(float value) {{
    uint bits = asuint(value);
    uint magnitude = bits & 0x7fffffffu;
    if ({flush} && magnitude < 0x00800000u) {{ magnitude = 0u; }}
    if (magnitude == 0u) {{ return asfloat(0xff800000u); }}
    if (magnitude > 0x7f800000u || (bits & 0x80000000u) != 0u) {{
        return asfloat(0x7fc00000u);
    }}
    if (magnitude == 0x7f800000u) {{ return asfloat(magnitude); }}
    int exponent = int(magnitude >> 23u) - 127;
    uint fraction = magnitude & 0x007fffffu;
    if (exponent == -127) {{
        exponent = -126;
        while ((fraction & 0x00800000u) == 0u) {{
            fraction <<= 1u;
            exponent -= 1;
        }}
        fraction &= 0x007fffffu;
    }}
    float reduced @precise = asfloat(0x3f800000u | fraction);
    if (reduced > 1.4142135623730950488) {{
        reduced *= 0.5;
        exponent += 1;
    }}
    float ratio @precise = (reduced - 1.0) / (reduced + 1.0);
    float squared @precise = ratio * ratio;
    float polynomial @precise = 1.0 / 13.0;
    polynomial = 1.0 / 11.0 + squared * polynomial;
    polynomial = 1.0 / 9.0 + squared * polynomial;
    polynomial = 1.0 / 7.0 + squared * polynomial;
    polynomial = 1.0 / 5.0 + squared * polynomial;
    polynomial = 1.0 / 3.0 + squared * polynomial;
    float correction @precise = (2.0 * ratio * squared) * polynomial;
    float tail @precise = float(exponent) * 0.000001428606765330187 + correction;
    return float(exponent) * 0.693145751953125 + (2.0 * ratio + tail);
}}
"""
        for width in sorted(self.required_metal_precise_log_widths - {1}):
            vector = self.metal_precise_log_helper_name(width)
            arguments = ", ".join(f"{scalar}(value.{lane})" for lane in "xyzw"[:width])
            code += (
                f"@precise\n@metal_static\nvec{width} {vector}(vec{width} value) {{\n"
                f"    return vec{width}({arguments});\n}}\n"
            )
        pad = "    " * indent
        return "".join(pad + line + "\n" for line in code.strip().splitlines())

    def metal_precise_atan_helper_name(self, width):
        suffix = "" if width == 1 else str(width)
        return self.metal_precise_math_unique_helper_name(
            f"atan-float{suffix}", f"__crossgl_metal_precise_atan_float{suffix}"
        )

    def generate_metal_precise_atan_support_code(self, indent=0):
        if not self.required_metal_precise_atan_widths:
            return ""
        scalar = self.metal_precise_atan_helper_name(1)
        # Reciprocal and pi/4 reductions bound the alternating series by
        # tan(pi/8). Bitwise endpoint handling preserves subnormals and -0.
        code = f"""@precise
@metal_static
float {scalar}(float value) {{
    uint bits = asuint(value);
    uint magnitude = bits & 0x7fffffffu;
    uint sign = bits & 0x80000000u;
    if (magnitude > 0x7f800000u || magnitude < 0x39800000u) {{
        return value;
    }}
    if (magnitude >= 0x4c800000u) {{
        return asfloat(sign | 0x3fc90fdbu);
    }}
    float positive = asfloat(magnitude);
    bool invert = positive > 1.0;
    float ratio @precise = invert ? 1.0 / positive : positive;
    bool reduce = ratio > 0.4142135623730950488;
    float reduced @precise = reduce ? (ratio - 1.0) / (ratio + 1.0) : ratio;
    float squared @precise = reduced * reduced;
    float series @precise = 1.0 / 17.0;
    series = -1.0 / 15.0 + squared * series;
    series = 1.0 / 13.0 + squared * series;
    series = -1.0 / 11.0 + squared * series;
    series = 1.0 / 9.0 + squared * series;
    series = -1.0 / 7.0 + squared * series;
    series = 1.0 / 5.0 + squared * series;
    series = -1.0 / 3.0 + squared * series;
    float angle @precise = reduced + (reduced * squared) * series;
    if (reduce) {{ angle = 0.7853981633974483096 + angle; }}
    if (invert) {{ angle = 1.5707963267948966192 - angle; }}
    return asfloat(asuint(angle) | sign);
}}
"""
        for width in sorted(self.required_metal_precise_atan_widths - {1}):
            vector = self.metal_precise_atan_helper_name(width)
            arguments = ", ".join(f"{scalar}(value.{lane})" for lane in "xyzw"[:width])
            code += (
                f"@precise\n@metal_static\nvec{width} {vector}(vec{width} value) {{\n"
                f"    return vec{width}({arguments});\n}}\n"
            )
        pad = "    " * indent
        return "".join(pad + line + "\n" for line in code.strip().splitlines())

    def metal_precise_atan2_helper_name(self, width):
        suffix = "" if width == 1 else str(width)
        return self.metal_precise_math_unique_helper_name(
            f"atan2-float{suffix}", f"__crossgl_metal_precise_atan2_float{suffix}"
        )

    def generate_metal_precise_atan2_support_code(self, indent=0):
        if not self.required_metal_precise_atan2_widths:
            return ""
        scalar = self.metal_precise_atan2_helper_name(1)
        atan = self.metal_precise_atan_helper_name(1)
        divide = self.metal_precise_math_unique_helper_name(
            "division-bits", "__crossgl_divide_bits"
        )
        flush = "true" if self.binary32_atan2_profile == "flush-subnormals" else "false"
        # Integer division prevents target denormal handling from changing the ratio.
        code = f"""@precise
@metal_static
float {scalar}(float y, float x) {{
    uint yBits = asuint(y);
    uint xBits = asuint(x);
    if ({flush}) {{
        if ((yBits & 0x7f800000u) == 0u) {{ yBits &= 0x80000000u; }}
        if ((xBits & 0x7f800000u) == 0u) {{ xBits &= 0x80000000u; }}
    }}
    uint yMagnitude = yBits & 0x7fffffffu;
    uint xMagnitude = xBits & 0x7fffffffu;
    uint sign = yBits & 0x80000000u;
    bool negativeX = (xBits & 0x80000000u) != 0u;
    if (yMagnitude > 0x7f800000u || xMagnitude > 0x7f800000u) {{
        return asfloat(0x7fc00000u);
    }}
    if (yMagnitude == 0u) {{
        return asfloat(sign | (negativeX ? 0x40490fdbu : 0u));
    }}
    if (xMagnitude == 0u) {{ return asfloat(sign | 0x3fc90fdbu); }}
    if (yMagnitude == 0x7f800000u) {{
        uint angleBits = xMagnitude == 0x7f800000u
            ? (negativeX ? 0x4016cbe4u : 0x3f490fdbu) : 0x3fc90fdbu;
        return asfloat(sign | angleBits);
    }}
    if (xMagnitude == 0x7f800000u) {{
        return asfloat(sign | (negativeX ? 0x40490fdbu : 0u));
    }}
    bool invert = yMagnitude > xMagnitude;
    uint smaller = invert ? xMagnitude : yMagnitude;
    uint larger = invert ? yMagnitude : xMagnitude;
    uint ratioBits = {divide}(smaller, larger, false);
    // At the normal/subnormal midpoint, atan(r) < r breaks the
    // division's upward tie. Only a maximal significand over a power
    // of two can form this exact midpoint from binary32 operands.
    if (ratioBits == 0x00800000u &&
        (smaller & 0x007fffffu) == 0x007fffffu &&
        (larger & 0x007fffffu) == 0u) {{
        ratioBits = 0x007fffffu;
    }}
    float ratio = asfloat(ratioBits);
    float angle @precise = {atan}(ratio);
    if (invert) {{ angle = 1.5707963267948966192 - angle; }}
    if (negativeX) {{ angle = 3.1415926535897932385 - angle; }}
    uint result = asuint(angle);
    if ({flush} && (result & 0x7f800000u) == 0u) {{ return asfloat(0u); }}
    return asfloat(result | sign);
}}
"""
        for width in sorted(self.required_metal_precise_atan2_widths - {1}):
            vector = self.metal_precise_atan2_helper_name(width)
            arguments = ", ".join(
                f"{scalar}(y.{lane}, x.{lane})" for lane in "xyzw"[:width]
            )
            code += (
                f"@precise\n@metal_static\nvec{width} {vector}(vec{width} y, vec{width} x) {{\n"
                f"    return vec{width}({arguments});\n}}\n"
            )
        pad = "    " * indent
        return "".join(pad + line + "\n" for line in code.strip().splitlines())

    def metal_precise_acosh_helper_name(self, width):
        suffix = "" if width == 1 else str(width)
        return self.metal_precise_math_unique_helper_name(
            f"acosh-float{suffix}", f"__crossgl_metal_precise_acosh_float{suffix}"
        )

    def generate_metal_precise_acosh_support_code(self, indent=0):
        if not self.required_metal_precise_acosh_widths:
            return ""
        scalar = self.metal_precise_acosh_helper_name(1)
        log1p = self.metal_precise_math_unique_helper_name(
            "acosh-log1p", "__crossgl_metal_precise_acosh_log1p"
        )
        # The near-one series avoids cancellation in x*x - 1. The large
        # branch avoids squaring overflow; the omitted tail is below one ULP.
        code = f"""@precise
@metal_static
float {log1p}(float value) {{
    if (value > 0.5) {{
        return log(1.0 + value);
    }}
    float reduced @precise = value / (2.0 + value);
    float squared @precise = reduced * reduced;
    float series @precise = 1.0 + squared * (
        0.33333333333333333333 + squared * (
            0.2 + squared * (
                0.14285714285714285714 + squared * (
                    0.11111111111111111111 + squared * 0.09090909090909090909
                )
            )
        )
    );
    return (2.0 * reduced) * series;
}}
@precise
@metal_static
float {scalar}(float value) {{
    if (isnan(value)) {{
        return value;
    }}
    if (value < 1.0) {{
        return asfloat(0x7fc00000u);
    }}
    if (value == 1.0) {{
        return 0.0;
    }}
    if (value > 4096.0) {{
        return log(value) + 0.69314718055994530942;
    }}
    float offset @precise = value - 1.0;
    if (offset <= 0.0625) {{
        float series @precise = 0.00189887152777777778;
        series = -0.00558035714285714286 + offset * series;
        series = 0.01875 + offset * series;
        series = -0.08333333333333333333 + offset * series;
        series = 1.0 + offset * series;
        return sqrt(2.0 * offset) * series;
    }}
    return {log1p}(offset + sqrt(offset * (value + 1.0)));
}}
"""
        for width in sorted(self.required_metal_precise_acosh_widths - {1}):
            vector = self.metal_precise_acosh_helper_name(width)
            arguments = ", ".join(f"{scalar}(value.{lane})" for lane in "xyzw"[:width])
            code += (
                f"@precise\n@metal_static\nvec{width} {vector}(vec{width} value) {{\n"
                f"    return vec{width}({arguments});\n}}\n"
            )
        pad = "    " * indent
        return "".join(pad + line + "\n" for line in code.strip().splitlines())

    def metal_precise_trig_helper_name(self, operation, width):
        suffix = "" if width == 1 else str(width)
        return self.metal_precise_math_unique_helper_name(
            f"{operation}-float{suffix}",
            f"__crossgl_metal_precise_{operation}_float{suffix}",
        )

    def generate_metal_precise_trig_support_code(self, indent=0):
        if not any(self.required_metal_precise_trig_widths.values()):
            return ""
        names = {
            key: self.metal_precise_math_unique_helper_name(
                f"trig-{key}", f"__crossgl_metal_trig_{key}"
            )
            for key in TRIG_HELPER_KEYS
        }
        code = binary32_trig_support(names)
        for operation, widths in self.required_metal_precise_trig_widths.items():
            if not widths:
                continue
            scalar = self.metal_precise_trig_helper_name(operation, 1)
            cosine = "true" if operation == "cos" else "false"
            code += (
                f"@precise\n@metal_static\nfloat {scalar}(float value) {{\n"
                f"    return {names['evaluate']}(value, {cosine});\n}}\n"
            )
            for width in sorted(widths - {1}):
                vector = self.metal_precise_trig_helper_name(operation, width)
                arguments = ", ".join(
                    f"{scalar}(value.{lane})" for lane in "xyzw"[:width]
                )
                code += (
                    f"@precise\n@metal_static\nvec{width} {vector}(vec{width} value) {{\n"
                    f"    return vec{width}({arguments});\n}}\n"
                )
        pad = "    " * indent
        return "".join(pad + line + "\n" for line in code.strip().splitlines())
