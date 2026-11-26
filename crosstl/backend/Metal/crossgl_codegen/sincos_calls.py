"""Metal ``sincos`` calls, imported as a helper that returns the sine and stores the cosine.

MSL declares ``T sincos(T x, thread T& cosval)``: it returns the sine of ``x`` and stores its cosine in ``cosval``
(Metal Shading Language Specification, section 6.5, "Math Functions"). CrossGL's ``sincos`` is HLSL's
``void sincos(x, out s, out c)``, a statement with three operands, so the importer used to carry the two-operand call
into CrossGL unchanged. Metal output wrote it back, but HLSL and Slang output copied a call DXC and slangc reject,
and GLSL, SPIR-V and CUDA output refused it.

The importer now writes one helper per value type and namespace, ``T helper(T value, out T cosine)``, which stores
``cos(value)`` and returns ``sin(value)``, and calls it where the source called ``sincos``. The call stays an
expression, so it may appear anywhere the source wrote it, and ``value`` is evaluated once. ``metal::fast::sincos``
calls ``fast::sin`` and ``fast::cos``; ``metal::precise::sincos`` calls the portable binary32 lowering of
``precise::sin`` and ``precise::cos``.
"""

from __future__ import annotations

import re

from .support import MetalSincosLoweringError

_FLOAT_ELEMENTS = frozenset({"float", "half"})


class MetalSincosMixin:
    """Members of ``MetalToCrossGLConverter``: the ``sincos`` helper."""

    def generate_metal_sincos_call(self, expression, is_main=False):
        """The CrossGL call that replaces a call to Metal's ``sincos``, or None for another call."""
        if str(expression.name).rsplit("::", 1)[-1] != "sincos":
            return None
        if (
            self.resolve_metal_math_builtin_name(expression.name, expression.args)
            != "sincos"
        ):
            return None
        source_location = getattr(expression, "source_location", None)
        arguments = list(expression.args or [])
        result_type = self.metal_math_builtin_result_type(expression)
        info = self.metal_math_builtin_type_info(result_type)
        element = (
            self.normalized_metal_type(self.resolve_type_alias(info["element_type"]))
            if info is not None
            else None
        )
        if (
            len(arguments) != 2
            or info is None
            or element not in _FLOAT_ELEMENTS
            or info["width"] not in {1, 2, 3, 4}
        ):
            raise MetalSincosLoweringError(
                result_type,
                "expected a value and a cosine destination of one float or half scalar or vector type with at "
                "most four lanes",
                source_location,
            )
        if self.current_function is None:
            raise MetalSincosLoweringError(
                result_type,
                "a program-scope initializer cannot call the sincos helper",
                source_location,
            )
        mode = self.metal_math_builtin_namespace_mode(expression.name)
        mapped_type = self.map_type(result_type)
        helper = self.metal_sincos_helper_name(mode, mapped_type, info["width"])
        value = self.generate_expression(arguments[0], is_main)
        if self.metal_source_overload_type_identity(
            self.expression_metal_type(arguments[0])
        ) != self.metal_source_overload_type_identity(result_type):
            value = f"{mapped_type}({value})"
        cosine = self.generate_expression(arguments[1], is_main)
        return f"{helper}({value}, {cosine})"

    def metal_sincos_helper_name(self, mode, mapped_type, width):
        """Register and name the helper for one namespace mode and CrossGL value type."""
        if mode == "precise":
            self.required_metal_precise_trig_widths["sin"].add(width)
            self.required_metal_precise_trig_widths["cos"].add(width)
        type_suffix = re.sub(r"\W+", "_", mapped_type)
        name = self.metal_precise_math_unique_helper_name(
            f"sincos-{mode}-{mapped_type}",
            f"__crossgl_metal_sincos_{mode}_{type_suffix}",
        )
        self.required_metal_sincos_helpers[name] = (mode, mapped_type, width)
        return name

    def generate_metal_sincos_support_code(self, indent=0):
        """The helpers the program's ``sincos`` calls need, after the precise trigonometric helpers they call."""
        if not self.required_metal_sincos_helpers:
            return ""
        code = ""
        for name, (mode, mapped_type, width) in sorted(
            self.required_metal_sincos_helpers.items()
        ):
            if mode == "precise":
                sine = self.metal_precise_trig_helper_name("sin", width)
                cosine = self.metal_precise_trig_helper_name("cos", width)
                attributes = "@precise\n@metal_static\n"
            elif mode == "fast":
                sine, cosine = "fast::sin", "fast::cos"
                attributes = "@metal_static\n"
            else:
                sine, cosine = "sin", "cos"
                attributes = "@metal_static\n"
            code += (
                f"{attributes}{mapped_type} {name}({mapped_type} value, out {mapped_type} cosine) {{\n"
                f"    cosine = {cosine}(value);\n"
                f"    return {sine}(value);\n"
                "}\n"
            )
        pad = "    " * indent
        return "".join(pad + line + "\n" for line in code.strip().splitlines())
