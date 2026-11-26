"""Binary64 math calls that Slang output refuses.

Slang declares its math functions for every floating-point type, ``double`` included, but slangc writes most of them
for SPIR-V as the GLSL.std.450 instruction of the same name, which takes 16- and 32-bit floating-point operands only
(GLSL.std.450 specification, section 2: "Result Type must be a scalar or vector of floating-point type ... 16 or 32
bits"). spirv-val rejects the module ("GLSL.std.450 Acos: expected Result Type to be a 16 or 32-bit scalar or vector
float type"), so Slang source that calls one of :data:`SLANG_BINARY32_ONLY_FUNCTIONS` on a ``double`` operand does not
compile for Slang's SPIR-V target. Those calls are refused. A call that
:mod:`crosstl.translator.semantics.libm_binary64` rewrites into its binary64 helpers never reaches the generator.

The list is what slangc 2026.20 rejects with ``SLANG_RUN_SPIRV_VALIDATION=1`` for a ``double`` operand. ``exp2`` is
not in it: slangc converts the operand to ``float`` and back, which compiles.
"""

from __future__ import annotations

import re

__all__ = ["SLANG_BINARY32_ONLY_FUNCTIONS", "slang_binary64_math_refusal"]

SLANG_BINARY32_ONLY_FUNCTIONS = frozenset(
    {
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atan2",
        "atanh",
        "cbrt",
        "cos",
        "cosh",
        "erf",
        "exp",
        "log",
        "log10",
        "log2",
        "pow",
        "sin",
        "sinh",
        "tan",
        "tanh",
    }
)

_DOUBLE = re.compile(
    r"(?:double|float64_t)(?:[234])?|vector<\s*(?:double|float64_t)\s*,\s*[234]\s*>"
)


def _is_binary64(codegen, argument):
    argument_type = codegen.expression_result_type(argument)
    if argument_type is None:
        return False
    mapped = codegen.convert_type(argument_type)
    return isinstance(mapped, str) and _DOUBLE.fullmatch(mapped.strip()) is not None


def slang_binary64_math_refusal(codegen, callee, arguments):
    """Refuse a call of a built-in in :data:`SLANG_BINARY32_ONLY_FUNCTIONS` on a binary64 operand, else ``None``."""
    if callee not in SLANG_BINARY32_ONLY_FUNCTIONS:
        return None
    if not any(_is_binary64(codegen, argument) for argument in arguments or ()):
        return None
    return codegen.refuse_lowering(
        f"Slang {callee}: slangc writes {callee} for SPIR-V as the GLSL.std.450 instruction, which takes 16- and "
        "32-bit floating-point operands only, so a binary64 operand does not compile",
        f"{callee}()",
    )
