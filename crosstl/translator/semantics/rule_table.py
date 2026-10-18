"""The rule table: which operation rules each source defines, and how each target discharges them.

``SOURCE_RULES`` lists, per source language, the rule families whose result the language defines. A family that a
source leaves undefined carries no obligation. ``TARGET_ENTRIES`` maps every (family, target) pair to a
``TargetEntry``: ``native`` (the target's own operation refines the rule), ``polyfill`` (the templates in
``POLYFILLS``, over the target's trusted base) or ``refuse``. ``tests/conformance/rule_table_smt.py`` proves every
native and polyfill entry with Z3, and ``tests/test_translator/test_rule_carrying.py`` checks that the table is total
over the registered backends.

See docs/source/adr/0001-rule-carrying.md.
"""

from __future__ import annotations

from dataclasses import dataclass

WRAP = "wrap"
FAMILIES = (WRAP,)


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


# Rule families each source language defines. Sources not listed (Metal, CUDA, HIP, OpenCL C, Slang, Rust, Mojo
# and CrossGL itself) leave signed overflow undefined or unstated, so their operations carry no wrap obligation.
SOURCE_RULES = {
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

# Hub intrinsics of the wrap family: name -> (operator, arity). Statement forms (compound assignment, increment
# and decrement) reuse the binary templates; see annotate.py and lower.py.
WRAP_INTRINSICS = {
    "__cgl_wrap_add": ("+", 2),
    "__cgl_wrap_sub": ("-", 2),
    "__cgl_wrap_mul": ("*", 2),
    "__cgl_wrap_neg": ("-", 1),
}
INTRINSIC_FAMILY = {name: WRAP for name in WRAP_INTRINSICS}
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


def source_families(source: str) -> frozenset:
    return frozenset(rule.family for rule in SOURCE_RULES.get(source, ()))


def target_entry(family: str, target: str) -> TargetEntry:
    entry = TARGET_ENTRIES.get((family, target))
    if entry is None:
        return TargetEntry(
            "refuse", f"the rule table has no entry for {family!r} on {target!r}"
        )
    return entry
