"""CrossGL emission of Direct3D trigonometric range reduction.

With ``trigonometric_reduction="revolutions"`` or ``"warp"`` the HLSL
frontend replaces ``sin``, ``cos`` and ``tan`` with CrossGL helpers that
reduce their argument the way a Direct3D implementation does: by whole
revolutions, or with the table-driven single-precision evaluation of WARP.
"""

import struct


class HLSLTrigonometricReductionMixin:
    """Trigonometric range-reduction members of ``HLSLToCrossGLConverter``."""

    REVOLUTION_REDUCED_INTRINSICS = frozenset({"sin", "cos", "tan"})

    REVOLUTION_REDUCTION_HELPER = "crosstl_hlsl_revolutions"

    WARP_TRIGONOMETRIC_HELPER = "crosstl_hlsl_warp"

    # WARP (the Direct3D 12 software rasterizer, "Microsoft Basic Render Driver")
    # evaluates sin and cos in single precision from a 32-entry table indexed by
    # the fractional revolution: S[k] = fl(sin(fl(k * fl(2 * pi / 32)))) and
    # C[k] = fl(cos(fl(k * fl(2 * pi / 32)))), stored as float32 bit patterns.
    # fmt: off
    WARP_SINE_TABLE_BITS = (
        0x00000000, 0x3E47C5C2, 0x3EC3EF16, 0x3F0E39DA,
        0x3F3504F3, 0x3F54DB32, 0x3F6C835E, 0x3F7B14BF,
        0x3F800000, 0x3F7B14BE, 0x3F6C835E, 0x3F54DB30,
        0x3F3504F3, 0x3F0E39D9, 0x3EC3EF10, 0x3E47C5C1,
        0xB3BBBD2E, 0xBE47C5CD, 0xBEC3EF15, 0xBF0E39DB,
        0xBF3504F5, 0xBF54DB34, 0xBF6C8361, 0xBF7B14BE,
        0xBF800000, 0xBF7B14BE, 0xBF6C835D, 0xBF54DB2F,
        0xBF3504EF, 0xBF0E39DB, 0xBEC3EF15, 0xBE47C5BC,
    )

    WARP_COSINE_TABLE_BITS = (
        0x3F800000, 0x3F7B14BE, 0x3F6C835E, 0x3F54DB31,
        0x3F3504F3, 0x3F0E39D9, 0x3EC3EF15, 0x3E47C5BC,
        0xB33BBD2E, 0xBE47C5C2, 0xBEC3EF18, 0xBF0E39DC,
        0xBF3504F3, 0xBF54DB32, 0xBF6C8360, 0xBF7B14BF,
        0xBF800000, 0xBF7B14BE, 0xBF6C835E, 0xBF54DB30,
        0xBF3504F1, 0xBF0E39D6, 0xBEC3EF0B, 0xBE47C5C6,
        0x324CDE2E, 0x3E47C5C8, 0x3EC3EF1B, 0x3F0E39DD,
        0x3F3504F7, 0x3F54DB31, 0x3F6C835F, 0x3F7B14BF,
    )

    # fmt: on
    # fl(1 / (2 * pi)), fl(2 * pi / 32), fl(-1 / 6) and fl(1 / 24).
    WARP_INVERSE_TWO_PI_BITS = 0x3E22F983

    WARP_SEGMENT_ANGLE_BITS = 0x3E490FDB

    WARP_SINE_CUBIC_BITS = 0xBE2AAAAB

    WARP_COSINE_QUARTIC_BITS = 0x3D2AAAAB

    def trigonometric_operand(self, operand_expr, rendered):
        """Reduce a runtime ``sin``/``cos``/``tan`` operand like Direct3D GPUs."""
        if self.trigonometric_reduction != "revolutions":
            return rendered
        if self.constant_initializer_depth or self.is_hlsl_constant_expression(
            operand_expr
        ):
            return rendered
        value_type, rendered = self.trigonometric_operand_type(operand_expr, rendered)
        self.revolution_reduction_types.add(value_type)
        return f"{self.REVOLUTION_REDUCTION_HELPER}_{value_type}({rendered})"

    def trigonometric_operand_type(self, operand_expr, rendered):
        """Return the CrossGL float type of a trigonometric operand and the operand."""
        width = self.hlsl_component_width(operand_expr)
        if width is None:
            raise ValueError(
                f"Cannot apply {self.trigonometric_reduction} trigonometric "
                f"evaluation to '{rendered}': the operand is not a known "
                "floating-point scalar or vector"
            )
        value_type = "float" if width == 1 else f"vec{width}"
        raw_type = self.expression_value_raw_type(operand_expr)
        family = self.numeric_type_family(raw_type)
        if family in {"int", "uint"} or self.raw_type_base(raw_type).startswith("bool"):
            rendered = f"{value_type}({rendered})"
        return value_type, rendered

    def trigonometric_call(self, func_name, operand_expr, rendered):
        """Render a runtime ``sin``/``cos``/``tan`` call in the selected mode."""
        if self.trigonometric_reduction != "warp":
            return f"{func_name}({self.trigonometric_operand(operand_expr, rendered)})"
        if self.constant_initializer_depth or self.is_hlsl_constant_expression(
            operand_expr
        ):
            return f"{func_name}({rendered})"
        value_type, rendered = self.trigonometric_operand_type(operand_expr, rendered)
        self.warp_trigonometric_uses.add((func_name, value_type))
        return f"{self.WARP_TRIGONOMETRIC_HELPER}_{func_name}_{value_type}({rendered})"

    @staticmethod
    def float32_literal(bits):
        """Render float32 ``bits`` as a CrossGL literal that rounds back to them."""
        value = struct.unpack("<f", struct.pack("<I", bits))[0]
        text = f"{value:.9g}"
        if not any(marker in text for marker in (".", "e", "n")):
            text += ".0"
        return text

    def generate_trigonometric_helpers(self):
        return (
            self.generate_revolution_reduction_helpers()
            + self.generate_warp_trigonometric_helpers()
        )

    def generate_revolution_reduction_helpers(self):
        """CrossGL helpers for :meth:`trigonometric_operand`.

        ``turns`` is rounded to single precision before the integral
        revolutions are removed, which is the reduction Direct3D hardware
        applies; the remaining angle lies in ``[-pi, pi]``.
        """
        code = ""
        for value_type in sorted(self.revolution_reduction_types):
            code += (
                f"    {value_type} {self.REVOLUTION_REDUCTION_HELPER}_{value_type}"
                f"({value_type} angle) {{\n"
                f"        {value_type} turns = angle * 0.15915494309189535;\n"
                "        return (turns - floor(turns + 0.5)) * 6.283185307179586;\n"
                "    }\n"
            )
        return code

    def generate_warp_trigonometric_helpers(self):
        """CrossGL helpers for :meth:`trigonometric_call` in ``"warp"`` mode.

        With ``t = fl(x / (2 * pi))`` the fractional revolution
        ``f = fl(t - floor(t))`` selects table entry ``k = floor(32 * f)``;
        ``b = fl((32 * f - k) * fl(2 * pi / 32))`` is the remaining angle.
        ``sin(x) = S[k] * cos(b) + C[k] * sin(b)`` and
        ``cos(x) = C[k] * cos(b) - S[k] * sin(b)`` with
        ``sin(b) = b + b^3 * fl(-1/6)`` and
        ``cos(b) = 1 + b^2 * (-1/2 + b^2 * fl(1/24))``, every operation rounded
        to single precision. ``tan(x)`` is ``sin(x) / cos(x)``. The
        intermediate values are ``precise`` so targets cannot contract them.
        """
        if not self.warp_trigonometric_uses:
            return ""
        prefix = self.WARP_TRIGONOMETRIC_HELPER
        functions = {name for name, _ in self.warp_trigonometric_uses}
        scalar = set(functions)
        if "tan" in scalar:
            scalar.update({"sin", "cos"})
        literal = self.float32_literal

        def table(name, bits):
            values = ", ".join(literal(value) for value in bits)
            return f"    static const float {prefix}_{name}_table[32] = {{{values}}};\n"

        code = table("sine", self.WARP_SINE_TABLE_BITS)
        code += table("cosine", self.WARP_COSINE_TABLE_BITS)
        for name in ("sin", "cos"):
            if name not in scalar:
                continue
            if name == "sin":
                result = (
                    f"{prefix}_sine_table[index] * cosine"
                    f" + {prefix}_cosine_table[index] * sine"
                )
            else:
                result = (
                    f"{prefix}_cosine_table[index] * cosine"
                    f" - {prefix}_sine_table[index] * sine"
                )
            code += (
                f"    float {prefix}_{name}_float(float angle) {{\n"
                "        precise float turns = angle * "
                f"{literal(self.WARP_INVERSE_TWO_PI_BITS)};\n"
                "        precise float fraction = turns - floor(turns);\n"
                "        precise float scaled = fraction * 32.0;\n"
                "        precise float segment = floor(scaled);\n"
                "        precise float offset = (scaled - segment) * "
                f"{literal(self.WARP_SEGMENT_ANGLE_BITS)};\n"
                "        precise float offset2 = offset * offset;\n"
                "        precise float sine = offset + (offset * offset2) * "
                f"{literal(self.WARP_SINE_CUBIC_BITS)};\n"
                "        precise float cosine = 1.0 + offset2 * (-0.5 + offset2 * "
                f"{literal(self.WARP_COSINE_QUARTIC_BITS)});\n"
                "        int index = int(segment) & 31;\n"
                f"        precise float result = {result};\n"
                "        return result;\n"
                "    }\n"
            )
        if "tan" in scalar:
            code += (
                f"    float {prefix}_tan_float(float angle) {{\n"
                f"        precise float result = {prefix}_sin_float(angle)"
                f" / {prefix}_cos_float(angle);\n"
                "        return result;\n"
                "    }\n"
            )
        components = "xyzw"
        for name, value_type in sorted(self.warp_trigonometric_uses):
            if value_type == "float":
                continue
            width = int(value_type[-1])
            items = ", ".join(
                f"{prefix}_{name}_float(angle.{components[i]})" for i in range(width)
            )
            code += (
                f"    {value_type} {prefix}_{name}_{value_type}({value_type} angle) {{\n"
                f"        return {value_type}({items});\n"
                "    }\n"
            )
        return code
