"""The rule table: which operation rules each source defines, and how each target discharges them.

``SOURCE_RULES`` lists, per source language, the rule families whose result the language defines. A family that a
source leaves undefined carries no obligation. ``TARGET_ENTRIES`` maps every (family, target) pair to a
``TargetEntry``: ``native`` (the target's own operation refines the rule), ``polyfill`` (the templates in
``POLYFILLS`` or the helper functions in ``HELPERS``, over the target's trusted base) or ``refuse``. ``tests/conformance/rule_table_smt.py`` proves every
native and polyfill entry with Z3, and ``tests/test_translator/test_rule_carrying.py`` checks that the table is total
over the registered backends.

See docs/source/adr/0001-rule-carrying.md.
"""

from __future__ import annotations

from dataclasses import dataclass

WRAP = "wrap"
UDIV_ZERO = "udiv_zero"
CONVERT_SAT = "convert_sat"
FTOI_NAN = "ftoi_nan"
# Floating-point operations evaluated without a fast-math compilation mode: no mode that assumes NaNs, infinities and
# signed zeros are absent, reassociates, or selects approximate implementations of the math functions. The family has
# no intrinsics: a target discharges it for the whole program (program_families.py).
FP_IEEE = "fp_ieee"
FAMILIES = (WRAP, UDIV_ZERO, CONVERT_SAT, FTOI_NAN, FP_IEEE)
PROGRAM_FAMILIES = frozenset({FP_IEEE})


@dataclass(frozen=True)
class SourceRule:
    family: str
    citation: str


@dataclass(frozen=True)
class TargetEntry:
    """How a target discharges a rule family: ``kind`` is ``native``, ``polyfill`` or ``refuse``."""

    kind: str
    reason: str
    configuration: str | None = None


# Rule families each source language defines. CUDA, HIP, OpenCL C, Slang, Rust, Mojo and CrossGL itself leave signed
# overflow and division by zero undefined or unstated, so those operations carry no obligation; Metal leaves them
# undefined too. GLSL and SPIR-V leave unsigned division by zero unspecified (GLSL 4.60 section 5.9, SPIR-V OpUDiv).
# CUDA, HIP and OpenCL C compile without fast math unless asked to. Metal (fast math by default), HLSL (DXC marks
# float operations fast unless they are precise or -Gis is given), GLSL and SPIR-V (whose environments need not
# preserve NaNs, infinities or signed zeros) do not define fp_ieee.
SOURCE_RULES = {
    "cuda": (
        SourceRule(
            FP_IEEE,
            "CUDA Programming Guide, floating-point standard: nvcc compiles IEEE 754 arithmetic and the accurate "
            "math library (--use_fast_math, which implies --ftz=true, --prec-div=false and --prec-sqrt=false, is "
            "off by default); CUDA Math API: the error bounds of the single-precision functions",
        ),
    ),
    "hip": (
        SourceRule(
            FP_IEEE,
            "Clang Users Manual: -ffast-math is off by default, so HIP device code keeps NaNs, infinities and "
            "signed zeros and is not reassociated; HIP math API: the ULP bounds of the device functions",
        ),
    ),
    "opencl": (
        SourceRule(
            FP_IEEE,
            "OpenCL C 3.0 sections 7.2 and 7.4: the full profile's ULP bounds and IEEE 754 edge cases apply unless "
            "the program is built with -cl-fast-relaxed-math or -cl-finite-math-only",
        ),
    ),
    "metal": (
        SourceRule(
            FTOI_NAN,
            "MSL 4.1 section 8.6: converting a floating-point value to an integer type rounds toward zero, and a NaN "
            "converts to 0; a truncation the type cannot represent is undefined (section 2.23 converts with no "
            "saturation, and C++17 [conv.fpint]/1 applies)",
        ),
    ),
    "opengl": (
        SourceRule(
            WRAP,
            "GLSL 4.60 section 4.1.3 and ESSL 3.00 section 4.1.3: addition, subtraction and multiplication "
            "that overflow give the low-order 32 bits of the result",
        ),
    ),
    "directx": (
        SourceRule(
            WRAP,
            "Direct3D 11 functional specification 22.12.1 iadd and 22.12.6 imul: no carry or borrow beyond "
            "the 32-bit values is performed",
        ),
        SourceRule(
            UDIV_ZERO,
            "Direct3D 11.3 functional specification 22.12.9 udiv: unsigned division by zero returns "
            "0xFFFFFFFF for both the quotient and the remainder",
        ),
    ),
    "vulkan": (
        SourceRule(
            WRAP,
            "SPIR-V 1.6 section 3.3.13 OpIAdd, OpISub, OpIMul, OpSNegate: the result is the low-order bits "
            "of the correct result unless the instruction carries NoSignedWrap",
        ),
    ),
}

TARGET_ENTRIES = {
    (WRAP, "opengl"): TargetEntry(
        "native", "GLSL 4.60 section 4.1.3: signed overflow wraps"
    ),
    (WRAP, "webgl"): TargetEntry(
        "native", "ESSL 3.00 section 4.1.3: signed overflow wraps"
    ),
    (WRAP, "wgsl"): TargetEntry(
        "native",
        "WGSL section 6.2.3: overflow of concrete integer types is modulo 2^bitwidth",
    ),
    (WRAP, "vulkan"): TargetEntry(
        "polyfill",
        "SPIR-V OpIAdd wraps, but MoltenVK and KosmicKrisp translate signed OpIAdd and OpIMul to MSL signed "
        "arithmetic, whose overflow is undefined, and Apple's compiler folds x + 1 > x (conformance cells "
        "s_add_wrap, h_loop_wrap). Unsigned OpIAdd between bit casts fixes MoltenVK; KosmicKrisp drops the "
        "bit casts and stays a known driver defect (docs/source/adr/0001-rule-carrying.md, section 4)",
    ),
    (WRAP, "metal"): TargetEntry(
        "polyfill",
        "MSL 4.1 section 3.1: the result of signed integer overflow is undefined",
    ),
    (WRAP, "directx"): TargetEntry(
        "polyfill",
        "DXC 1.9 marks HLSL signed add, sub and mul nsw in DXIL, so overflow is poison and x + 1 > x folds",
    ),
    (WRAP, "cuda"): TargetEntry(
        "polyfill", "CUDA C++ follows C++17 [expr]/4: signed overflow is undefined"
    ),
    (WRAP, "hip"): TargetEntry(
        "polyfill", "HIP C++ follows C++17 [expr]/4: signed overflow is undefined"
    ),
    (WRAP, "slang"): TargetEntry(
        "polyfill",
        "Slang does not specify signed overflow and lowers int arithmetic to targets that leave it undefined",
    ),
    (WRAP, "mojo"): TargetEntry(
        "polyfill",
        "the Mojo manual does not state the result of signed Int32 overflow; UInt32 arithmetic is modular",
    ),
    (WRAP, "rust"): TargetEntry(
        "native",
        "Rust integer overflow wraps when overflow checks are off (the release profile) and panics otherwise; "
        "it never yields a different value",
        configuration="overflow-checks = false",
    ),
}

_UDIV_ZERO_UNDEFINED = "unsigned division by zero is undefined"
TARGET_ENTRIES.update(
    {
        (UDIV_ZERO, "opengl"): TargetEntry(
            "polyfill",
            "GLSL 4.60 section 5.9: integer division by zero results in an unspecified value",
        ),
        (UDIV_ZERO, "webgl"): TargetEntry(
            "polyfill",
            "ESSL 3.00 section 5.9: division by zero results in an unspecified value",
        ),
        (UDIV_ZERO, "wgsl"): TargetEntry(
            "polyfill",
            "WGSL section 8.7: an unsigned e1 / 0 is e1 and e1 % 0 is 0, not 0xFFFFFFFF",
        ),
        (UDIV_ZERO, "vulkan"): TargetEntry(
            "polyfill",
            "SPIR-V 1.6 section 3.3.13 OpUDiv and OpUMod: behavior is undefined if Operand 2 is 0",
        ),
        (UDIV_ZERO, "metal"): TargetEntry(
            "polyfill",
            "MSL 4.1 is based on C++17, whose [expr.mul]/4 leaves division by zero undefined; Apple GPUs do not "
            "return 0xFFFFFFFF (conformance cell h_udiv_zero)",
        ),
        (UDIV_ZERO, "directx"): TargetEntry(
            "polyfill",
            "DXC 1.9 lowers HLSL unsigned / and % to LLVM udiv and urem, whose zero divisor is undefined "
            "behaviour: a divisor known to be zero folds to undef and the validator rejects the store",
        ),
        (UDIV_ZERO, "cuda"): TargetEntry(
            "polyfill", "CUDA C++ follows C++17 [expr.mul]/4: " + _UDIV_ZERO_UNDEFINED
        ),
        (UDIV_ZERO, "hip"): TargetEntry(
            "polyfill", "HIP C++ follows C++17 [expr.mul]/4: " + _UDIV_ZERO_UNDEFINED
        ),
        (UDIV_ZERO, "slang"): TargetEntry(
            "polyfill",
            "Slang does not specify integer division by zero and lowers it to targets that leave it undefined",
        ),
        (UDIV_ZERO, "mojo"): TargetEntry(
            "polyfill",
            "the Mojo manual does not state the result of UInt32 division by zero",
        ),
        (UDIV_ZERO, "rust"): TargetEntry(
            "polyfill",
            "The Rust Reference, arithmetic operators: integer division and remainder by zero panic",
        ),
    }
)

_FTOI_CXX = (
    "C++17 [conv.fpint]/1: converting a floating-point value whose truncation the integer type cannot represent "
    "is undefined"
)
TARGET_ENTRIES.update(
    {
        (CONVERT_SAT, "opengl"): TargetEntry(
            "polyfill",
            "GLSL 4.60 section 5.4.1 leaves converting a negative floating-point value to uint undefined and does "
            "not define the conversion of NaN or of values outside the integer range",
        ),
        (CONVERT_SAT, "webgl"): TargetEntry(
            "polyfill",
            "ESSL 3.00 section 5.4.1 leaves converting a negative floating-point value to uint undefined and does "
            "not define the conversion of NaN or of values outside the integer range",
        ),
        (CONVERT_SAT, "wgsl"): TargetEntry(
            "polyfill",
            "the WGSL f32 to i32 and u32 conversions are not relied on for NaN and out-of-range values; the helper "
            "converts only values whose truncation is representable",
        ),
        (CONVERT_SAT, "vulkan"): TargetEntry(
            "polyfill",
            "SPIR-V 1.6 OpConvertFToS and OpConvertFToU saturate only under the SaturatedConversion decoration, "
            "which needs the Kernel capability that Vulkan does not offer",
        ),
        (CONVERT_SAT, "metal"): TargetEntry(
            "polyfill", "MSL 4.1 is based on C++17; " + _FTOI_CXX
        ),
        (CONVERT_SAT, "directx"): TargetEntry(
            "polyfill",
            "DXC 1.9 lowers HLSL float-to-integer casts to LLVM fptosi and fptoui, whose NaN and out-of-range "
            "results are poison",
        ),
        (CONVERT_SAT, "cuda"): TargetEntry("polyfill", "CUDA C++ follows " + _FTOI_CXX),
        (CONVERT_SAT, "hip"): TargetEntry("polyfill", "HIP C++ follows " + _FTOI_CXX),
        (CONVERT_SAT, "slang"): TargetEntry(
            "polyfill",
            "Slang does not specify out-of-range float-to-integer conversion and lowers it to targets that leave "
            "it undefined",
        ),
        (CONVERT_SAT, "mojo"): TargetEntry(
            "polyfill",
            "the Mojo manual does not state the result of an out-of-range float-to-integer cast",
        ),
        (CONVERT_SAT, "rust"): TargetEntry(
            "polyfill",
            "The Rust Reference's numeric cast saturates float-to-integer `as` casts, but the hub conversion is "
            "not proven to reach the Rust output as an `as` cast",
        ),
    }
)

# A NaN operand is what the ftoi_nan rule constrains; the convert_sat helpers also map it to 0, and they clamp the
# values the rule leaves undefined, so they refine it on every target whose own conversion leaves NaN undefined.
_FTOI_NAN_CXX = (
    "C++17 [conv.fpint]/1 leaves converting a NaN to an integer type undefined"
)
TARGET_ENTRIES.update(
    {
        (FTOI_NAN, "opengl"): TargetEntry(
            "polyfill",
            "GLSL 4.60 section 5.4.1 does not define converting a NaN to an integer type",
        ),
        (FTOI_NAN, "webgl"): TargetEntry(
            "polyfill",
            "ESSL 3.00 section 5.4.1 does not define converting a NaN to an integer type",
        ),
        (FTOI_NAN, "wgsl"): TargetEntry(
            "polyfill",
            "the WGSL f32 to i32 and u32 conversions are not relied on for NaN; the helper converts only values "
            "whose truncation is representable",
        ),
        (FTOI_NAN, "vulkan"): TargetEntry(
            "polyfill",
            "SPIR-V 1.6 OpConvertFToS and OpConvertFToU are undefined for a NaN operand without the "
            "SaturatedConversion decoration, which needs the Kernel capability that Vulkan does not offer",
        ),
        (FTOI_NAN, "metal"): TargetEntry(
            "native",
            "MSL 4.1 section 8.6: converting a NaN to an integer type gives 0, and conversions round toward zero",
        ),
        (FTOI_NAN, "directx"): TargetEntry(
            "polyfill",
            "DXC 1.9 lowers HLSL float-to-integer casts to LLVM fptosi and fptoui, whose result for a NaN is "
            "poison",
        ),
        (FTOI_NAN, "cuda"): TargetEntry(
            "polyfill", "CUDA C++ follows " + _FTOI_NAN_CXX
        ),
        (FTOI_NAN, "hip"): TargetEntry("polyfill", "HIP C++ follows " + _FTOI_NAN_CXX),
        (FTOI_NAN, "slang"): TargetEntry(
            "polyfill",
            "Slang does not specify converting a NaN to an integer type and lowers it to targets that leave it "
            "undefined",
        ),
        (FTOI_NAN, "mojo"): TargetEntry(
            "polyfill",
            "the Mojo manual does not state the result of casting a NaN to an integer type",
        ),
        (FTOI_NAN, "rust"): TargetEntry(
            "polyfill",
            "The Rust Reference's numeric cast maps NaN to 0 in float-to-integer `as` casts, but the hub conversion "
            "is not proven to reach the Rust output as an `as` cast",
        ),
    }
)

# fp_ieee asks only that no fast-math compilation mode applies; the accuracy each language guarantees for its math
# functions is a separate rule.
TARGET_ENTRIES.update(
    {
        (FP_IEEE, "metal"): TargetEntry(
            "polyfill",
            "MSL 4.1 section 1.6.3: the compiler defaults to fast math, and MTLCompileOptions.mathMode is relaxed on "
            "Apple silicon and fast elsewhere; the output starts with #pragma METAL fp math_mode(safe) and calls "
            "the metal::precise functions (codegen/metal_safe_math.py)",
        ),
        (FP_IEEE, "directx"): TargetEntry(
            "native",
            "HLSL cannot select DXC's floating-point mode in source; -Gis forces IEEE strictness",
            configuration="dxc -Gis",
        ),
        (FP_IEEE, "opengl"): TargetEntry(
            "native", "GLSL 4.60 defines no fast-math compilation mode"
        ),
        (FP_IEEE, "webgl"): TargetEntry(
            "native", "ESSL 3.00 defines no fast-math compilation mode"
        ),
        (FP_IEEE, "wgsl"): TargetEntry(
            "native", "WGSL defines no fast-math compilation mode"
        ),
        (FP_IEEE, "vulkan"): TargetEntry(
            "native",
            "SPIR-V relaxes floating-point operations only through FPFastMathMode decorations, which CrossTL does "
            "not emit",
        ),
        (FP_IEEE, "cuda"): TargetEntry(
            "native",
            "nvcc compiles IEEE 754 arithmetic and the accurate math library unless --use_fast_math is given",
            configuration="nvcc without --use_fast_math",
        ),
        (FP_IEEE, "hip"): TargetEntry(
            "native",
            "Clang compiles HIP without fast math unless -ffast-math is given",
            configuration="hipcc without -ffast-math",
        ),
        (FP_IEEE, "slang"): TargetEntry(
            "native",
            "slangc's default floating-point mode defers to the downstream compiler, which may be fast; -fp-mode "
            "precise selects precise",
            configuration="slangc -fp-mode precise",
        ),
        (FP_IEEE, "mojo"): TargetEntry(
            "native", "the Mojo manual documents no fast-math compilation mode"
        ),
        (FP_IEEE, "rust"): TargetEntry(
            "native",
            "Rust has no fast-math compilation mode: its floating-point operators are IEEE 754 operations",
        ),
    }
)

# Hub intrinsics of the wrap family: name -> (operator, arity). Statement forms (compound assignment, increment
# and decrement) reuse the binary templates; see annotate.py and lower.py.
WRAP_INTRINSICS = {
    "__cgl_wrap_add": ("+", 2),
    "__cgl_wrap_sub": ("-", 2),
    "__cgl_wrap_mul": ("*", 2),
    "__cgl_wrap_neg": ("-", 1),
}
# Hub intrinsics of the udiv_zero family: unsigned 32-bit division and remainder whose zero divisor gives all ones.
UDIV_ZERO_INTRINSICS = {
    "__cgl_udiv_ones": ("/", 2),
    "__cgl_urem_ones": ("%", 2),
}
BINARY_UDIV_ZERO_INTRINSIC = {"/": "__cgl_udiv_ones", "%": "__cgl_urem_ones"}
# Hub intrinsics of the convert_sat family (OpenCL C 6.4.3.3 saturated conversions): name -> (conversion, arity).
# Each converts one operand type to one 32-bit integer type, clamping to its range and mapping NaN to 0. Sources
# write the generic forms in CONVERT_SAT_GENERIC; annotation resolves them by operand type (see conversions.py).
CONVERT_SAT_INTRINSICS = {
    "__cgl_sat_ftoi": ("ftoi", 1),
    "__cgl_sat_ftou": ("ftou", 1),
    "__cgl_sat_itou": ("itou", 1),
    "__cgl_sat_utoi": ("utoi", 1),
}
# Hub intrinsics of the ftoi_nan family (MSL 4.1 section 8.6): a float-to-integer conversion whose NaN operand gives 0.
# The destination may be an 8-, 16- or 32-bit integer type; see nan_conversions.py.
FTOI_NAN_INTRINSICS = {
    "__cgl_nan0_ftoi": ("ftoi", 1),
    "__cgl_nan0_ftou": ("ftou", 1),
}
# Operand and result scalar types of each conversion intrinsic.
CONVERSION_SIGNATURES = {
    "__cgl_sat_ftoi": ("float", "int"),
    "__cgl_sat_ftou": ("float", "uint"),
    "__cgl_sat_itou": ("int", "uint"),
    "__cgl_sat_utoi": ("uint", "int"),
    "__cgl_nan0_ftoi": ("float", "int"),
    "__cgl_nan0_ftou": ("float", "uint"),
}
# Generic conversions to ``int`` or ``uint`` written by sources, resolved by operand type (see conversions.py):
# ``__cgl_convert_<result>[_sat][_<mode>]``, where the optional mode rounds a float operand to an integral value first
# (``rte`` to nearest even, ``rtz`` toward zero, ``rtp`` up, ``rtn`` down): name -> (result, saturate, mode).
CONVERT_SAT_GENERIC = {
    f"__cgl_convert_{result}{'_sat' if saturate else ''}{'_' + mode if mode else ''}": (
        result,
        saturate,
        mode,
    )
    for result in ("int", "uint")
    for saturate in (True, False)
    for mode in ("", "rte", "rtz", "rtp", "rtn")
}
INTRINSICS = {
    **WRAP_INTRINSICS,
    **UDIV_ZERO_INTRINSICS,
    **CONVERT_SAT_INTRINSICS,
    **FTOI_NAN_INTRINSICS,
}
INTRINSIC_FAMILY = {
    **{name: WRAP for name in WRAP_INTRINSICS},
    **{name: UDIV_ZERO for name in UDIV_ZERO_INTRINSICS},
    **{name: CONVERT_SAT for name in CONVERT_SAT_INTRINSICS},
    **{name: CONVERT_SAT for name in CONVERT_SAT_GENERIC},
    **{name: FTOI_NAN for name in FTOI_NAN_INTRINSICS},
}
BINARY_WRAP_INTRINSIC = {
    "+": "__cgl_wrap_add",
    "-": "__cgl_wrap_sub",
    "*": "__cgl_wrap_mul",
}

# CrossGL expression templates over the parameters x0 and x1. Each parameter occurs exactly once, so substitution
# never evaluates an operand twice. The trusted base is unsigned 32-bit arithmetic and bit reinterpretation.
POLYFILLS = {
    "__cgl_wrap_add": "asint(asuint(x0) + asuint(x1))",
    "__cgl_wrap_sub": "asint(asuint(x0) - asuint(x1))",
    "__cgl_wrap_mul": "asint(asuint(x0) * asuint(x1))",
    "__cgl_wrap_neg": "asint(0u - asuint(x0))",
}


@dataclass(frozen=True)
class Helper:
    """A polyfill that needs an operand more than once: a reserved helper function the lowering pass injects.

    ``body`` is a CrossGL expression over the parameters ``x0`` and ``x1``, which have the types in ``parameters``
    (default: two of ``scalar``); the helper returns ``result`` (default: ``scalar``). Vector operations call a
    per-lane lift of the scalar helper, named ``name`` followed by the lane count.
    """

    name: str
    scalar: str
    body: str
    parameters: tuple[str, ...] | None = None
    result: str | None = None

    @property
    def parameter_types(self) -> tuple[str, ...]:
        return self.parameters if self.parameters is not None else (self.scalar,) * 2

    @property
    def result_type(self) -> str:
        return self.result if self.result is not None else self.scalar


# Magnitude bits of a binary32 value, and the bit patterns of 2^31 and 2^32. A conversion helper decides NaN and range
# on the bits, which fast-math compilation cannot fold away, and converts only a value whose truncation is
# representable: the converted operand is sanitized to 0.0 otherwise, so the conversion stays defined where a target
# evaluates both arms of the conditional.
_MAGNITUDE = "(asuint(x0) & 0x7FFFFFFFu)"
_FTOI_BODY = (
    f"{_MAGNITUDE} < 0x4F000000u ? int(asfloat({_MAGNITUDE} < 0x4F000000u ? asuint(x0) : 0u)) : "
    f"({_MAGNITUDE} > 0x7F800000u ? 0 : (asuint(x0) >= 0x80000000u ? asint(0x80000000u) : 2147483647))"
)
# Every pattern above +infinity is a positive NaN or a negative value (including -0.0), and all of them give 0.
_FTOU_BODY = (
    "asuint(x0) > 0x7F800000u ? 0u : (asuint(x0) < 0x4F800000u ? "
    "uint(asfloat(asuint(x0) < 0x4F800000u ? asuint(x0) : 0u)) : 0xFFFFFFFFu)"
)


# The divisor is made nonzero before the division, so the helper stays defined where a target evaluates both arms of
# the conditional (SPIR-V OpSelect, WGSL select).
HELPERS = {
    "__cgl_udiv_ones": Helper(
        "cgl_rule_udiv_ones",
        "uint",
        "x1 == 0u ? 0xFFFFFFFFu : x0 / (x1 == 0u ? 1u : x1)",
    ),
    "__cgl_urem_ones": Helper(
        "cgl_rule_urem_ones",
        "uint",
        "x1 == 0u ? 0xFFFFFFFFu : x0 % (x1 == 0u ? 1u : x1)",
    ),
    "__cgl_sat_ftoi": Helper(
        "cgl_rule_sat_ftoi", "float", _FTOI_BODY, ("float",), "int"
    ),
    "__cgl_sat_ftou": Helper(
        "cgl_rule_sat_ftou", "float", _FTOU_BODY, ("float",), "uint"
    ),
    "__cgl_sat_itou": Helper(
        "cgl_rule_sat_itou", "int", "x0 < 0 ? 0u : uint(x0)", ("int",), "uint"
    ),
    "__cgl_sat_utoi": Helper(
        "cgl_rule_sat_utoi",
        "uint",
        "x0 > 0x7FFFFFFFu ? 2147483647 : int(x0)",
        ("uint",),
        "int",
    ),
}
# The ftoi_nan polyfills are the saturating helpers themselves (one definition per program; see helpers.py).
HELPERS["__cgl_nan0_ftoi"] = HELPERS["__cgl_sat_ftoi"]
HELPERS["__cgl_nan0_ftou"] = HELPERS["__cgl_sat_ftou"]


def source_families(source: str) -> frozenset:
    return frozenset(rule.family for rule in SOURCE_RULES.get(source, ()))


def target_entry(family: str, target: str) -> TargetEntry:
    entry = TARGET_ENTRIES.get((family, target))
    if entry is None:
        return TargetEntry(
            "refuse", f"the rule table has no entry for {family!r} on {target!r}"
        )
    return entry
