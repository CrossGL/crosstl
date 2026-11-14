"""Built-in functions the HLSL, Metal, CUDA, HIP and Mojo generators may call by name, and the refusal of the rest.

The importers carry the built-in functions of the languages they read into the IR as plain calls
(:data:`~crosstl.translator.semantics.builtin_function_names.BUILTIN_FUNCTION_NAMES`), and each generator lowers the
ones it knows. A call a generator had no lowering for was written as it stands: Metal output called HLSL's ``lit``,
``msad4``, ``InterlockedAdd`` and ``GetDimensions``, GLSL's ``EmitVertex`` and CUDA's ``acosf``, and HLSL output
called MSL's ``absdiff`` and ``simd_sum``, GLSL's ``EmitVertex`` and C++'s ``alignof``. DXC and the Metal compiler
reject each of those as an undeclared identifier; a target that declared a function of the same name with another
meaning would have accepted it silently.

A generator also asks for the spelling of calls it never writes (a call a later lowering replaces, or an
intermediate it discards), so the refusal is decided on the output, as
:func:`~.written_type_names.refuse_written_foreign_types` decides it for types. :func:`written_foreign_call_names`
lists each unqualified call in the rendered program whose callee is a built-in function of a language the importers
read, is not a function the target defines and is not a function, macro or type the program declares;
:func:`refuse_written_foreign_calls` refuses each of them.

The target sets are the free functions of the specification inventories (``tests/conformance/inventory/data``), as
:mod:`~crosstl.translator.semantics.builtin_function_names` holds them, plus the operators that read as calls. An
object method (HLSL's ``Sample``, MSL's ``sample``) is called through its object, so a free call to one is refused.

- HLSL: the intrinsics DXC declares as free functions, and ``sizeof``, an operator of HLSL 2021. A ray payload
  field's access qualifiers (`` : read(caller, closesthit) : write(miss)``) are not calls
  (:data:`HLSL_PAYLOAD_ACCESS_QUALIFIER`).
- CUDA and HIP: the device library and runtime functions of the CUDA Math API and runtime headers, which HIP
  mirrors, the warp intrinsics, and the C++ operators ``sizeof`` and ``alignof``.
- MSL: the Metal standard library functions, ``as_type``, the ``max_anisotropy`` and ``lod_clamp`` sampler
  constructor arguments, and the C++14 operators ``sizeof`` and ``alignof`` ([expr.sizeof], [expr.alignof]).
"""

from __future__ import annotations

import re

from ..semantics.builtin_function_names import (
    BUILTIN_FUNCTION_NAMES,
    CUDA_FUNCTION_NAMES,
    HLSL_INTRINSIC_FUNCTION_NAMES,
    METAL_LIBRARY_FUNCTION_NAMES,
)
from ..semantics.hlsl_ray_payloads import PAYLOAD_STAGES
from ..semantics.sampler_states import SAMPLER_STATE_CALL_ARGUMENTS
from .lowering_refusals import refuse_unsupported

__all__ = [
    "CUDA_CALLABLE_FUNCTION_NAMES",
    "HLSL_CALLABLE_FUNCTION_NAMES",
    "HLSL_PAYLOAD_ACCESS_QUALIFIER",
    "METAL_CALLABLE_FUNCTION_NAMES",
    "MOJO_CALLABLE_FUNCTION_NAMES",
    "MOJO_NON_CALLS",
    "refuse_written_foreign_calls",
    "written_foreign_call_names",
]

HLSL_CALLABLE_FUNCTION_NAMES = HLSL_INTRINSIC_FUNCTION_NAMES | frozenset({"sizeof"})
# CUDA and HIP: the device library and runtime functions ``cuda_runtime.h`` and ``hip/hip_runtime.h`` declare, with
# the warp intrinsics (:data:`~crosstl.translator.semantics.builtin_function_names.CUDA_FUNCTION_NAMES`), and the C++
# operators ``sizeof`` and ``alignof`` ([expr.sizeof], [expr.alignof]).
CUDA_CALLABLE_FUNCTION_NAMES = CUDA_FUNCTION_NAMES | frozenset({"sizeof", "alignof"})
METAL_CALLABLE_FUNCTION_NAMES = (
    METAL_LIBRARY_FUNCTION_NAMES
    | frozenset(SAMPLER_STATE_CALL_ARGUMENTS)
    | frozenset({"as_type", "sizeof", "alignof"})
)

# Mojo: the output imports ``math`` and ``gpu`` whole (``from math import *``, ``from gpu import *``) and, when it
# reinterprets bits, ``memory.bitcast``, and the ``std.bit`` functions it calls. The callable built-in functions are
# the free functions of the Mojo standard library's ``math`` module, the prelude's built-in functions (``abs``,
# ``min``, ``max``, ``round``, ``pow``, ``divmod``, ``len``, ``range``, ``print``, ``any``, ``all``), ``gpu.barrier``,
# ``memory.bitcast`` and the ``std.bit`` counts and reversal. Every other
# helper the generator writes (``dot_product``, ``lerp``, ``_crossgl_saturate_f32``) is a function the output
# declares with ``fn``.
MOJO_CALLABLE_FUNCTION_NAMES = frozenset("""
    acos acosh align_down align_up asin asinh atan atan2 atanh cbrt ceil ceildiv clamp copysign cos cosh erf erfc
    exp exp2 expm1 factorial floor fma frexp gamma gcd hypot iota isclose isqrt j0 j1 lcm ldexp lgamma log log10
    log1p log2 logb modf recip remainder rsqrt scalb sin sinh sqrt tan tanh trunc ulp y0 y1
    abs all any divmod len max min pow print range round
    barrier bitcast
    pop_count bit_reverse count_leading_zeros count_trailing_zeros
    """.split())
# A Mojo comment runs from ``#`` to the end of the line (string literals are blanked first), and a keyword operator
# or statement keyword followed by a parenthesis (``a and (b)``, ``not (c)``, ``return (d)``) is not a call.
MOJO_COMMENT = re.compile(r"#[^\n]*")
MOJO_KEYWORD = re.compile(
    r"\b(?:and|or|not|in|is|if|elif|else|while|for|return|raise|assert|del|yield|await|with)\b"
)
MOJO_NON_CALLS = (MOJO_COMMENT, MOJO_KEYWORD)

_PAYLOAD_STAGE = "|".join(PAYLOAD_STAGES)
# A ray payload field's access qualifier, which follows a colon like a semantic and lists payload stages
# (DirectX Raytracing specification, "Payload access qualifiers"): `float3 color : read(caller) : write(miss);`.
HLSL_PAYLOAD_ACCESS_QUALIFIER = re.compile(
    rf":\s*(?:read|write)\s*\(\s*(?:(?:{_PAYLOAD_STAGE})(?:\s*,\s*(?:{_PAYLOAD_STAGE}))*)?\s*\)"
)

# Comments, string and character literals and C++11 attributes (``[[buffer(0)]]``, ``[[texture(1)]]``), blanked
# before the output is searched: an attribute argument list is not a call.
_NON_CODE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'|\[\[.*?\]\]", re.S
)
_PREPROCESSOR_LINE = re.compile(r"^[ \t]*#[^\n]*", re.M)
_DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)", re.M)
_TYPE_DECLARATIONS = (
    re.compile(
        r"\b(?:struct|class|union|enum|typename)\s+(?:class\s+|struct\s+)?([A-Za-z_]\w*)"
    ),
    re.compile(r"\busing\s+([A-Za-z_]\w*)\s*="),
    re.compile(r"\btypedef\b[^;{}]*?\b([A-Za-z_]\w*)\s*(?:\[[^\]]*\]\s*)*;"),
)
# An unqualified call: an identifier, not after ``.``, ``->`` or ``::``, followed by an argument list.
_CALL = re.compile(r"(?<![\w.:])(?<!->)([A-Za-z_]\w*)[ \t]*\(")
# Words that may stand before a call at the start of a statement without making it a declaration.
_STATEMENT_WORDS = frozenset(
    {"return", "else", "do", "case", "default", "goto", "throw", "co_return", "new"}
)
# A word of a declaration's prefix: a qualifier, a possibly qualified or templated type, a pointer or reference
# mark, or an attribute in brackets (HLSL's ``[numthreads(8, 8, 1)]``).
_DECLARATION_WORD = re.compile(
    r"\[[^\[\]]*\]|template[ \t]*<[^;{}]*>|[A-Za-z_][\w:]*(?:<[^;{}]*>)?[&*]*|[&*]+"
)


def _code_text(code):
    return _NON_CODE.sub(lambda match: re.sub(r"[^\n]", " ", match.group(0)), code)


def _is_declaration_prefix(prefix):
    """Whether ``prefix``, the text of a line before ``name(``, makes the line a declaration of ``name``."""
    words = []
    position = 0
    prefix = prefix.strip()
    while position < len(prefix):
        if prefix[position] in " \t":
            position += 1
            continue
        match = _DECLARATION_WORD.match(prefix, position)
        if match is None:
            return False
        words.append(match.group(0))
        position = match.end()
    if not words or words[0] in _STATEMENT_WORDS:
        return False
    return not all(word.startswith("[") for word in words)


def _declared_names(text, names):
    """The members of ``names`` that ``text`` declares as a function, a macro or a type."""
    declared = set(_DEFINE.findall(text)) & names
    body = _PREPROCESSOR_LINE.sub("", text)
    for pattern in _TYPE_DECLARATIONS:
        declared.update(name for name in pattern.findall(body) if name in names)
    for match in _CALL.finditer(body):
        name = match.group(1)
        if name not in names or name in declared:
            continue
        line_start = body.rfind("\n", 0, match.start()) + 1
        if _is_declaration_prefix(body[line_start : match.start()]):
            declared.add(name)
    return declared


def written_foreign_call_names(code, callable_names, non_calls=()):
    """The built-in functions ``code`` calls unqualified that neither the target nor the program defines, sorted.

    ``callable_names`` is the set of built-in functions the target defines, and ``non_calls`` the patterns of the
    target's syntax that read as a call but are not one. A call in a comment, a string literal, a preprocessor
    line or an attribute, a method or qualified call (``tex.Sample(...)``, ``metal::fast::sin(...)``) and a call to
    a function, macro or type the output declares is not a call to a foreign built-in function.
    """
    if not isinstance(code, str):
        return []
    text = _code_text(code)
    for pattern in non_calls:
        text = pattern.sub(lambda match: re.sub(r"[^\n]", " ", match.group(0)), text)
    body = _PREPROCESSOR_LINE.sub(
        lambda match: re.sub(r"[^\n]", " ", match.group(0)), text
    )
    called = {
        match.group(1)
        for match in _CALL.finditer(body)
        if match.group(1) in BUILTIN_FUNCTION_NAMES
        and match.group(1) not in callable_names
    }
    if not called:
        return []
    return sorted(called - _declared_names(text, frozenset(called)))


def refuse_written_foreign_calls(code, callable_names, target_label, non_calls=()):
    """Refuse, in the rendering in progress, each foreign built-in function the rendered ``code`` still calls."""
    for name in written_foreign_call_names(code, callable_names, non_calls):
        refuse_unsupported(
            target_label,
            "function",
            f"call to '{name}', a built-in function of another shading language that {target_label} does not "
            "define and the program does not declare",
            name,
        )
