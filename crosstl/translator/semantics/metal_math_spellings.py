"""MSL math functions that the other shading languages spell differently or lack.

CrossGL imported from Metal keeps MSL's math function names, and each generator maps them to its own spelling. Three
groups had no mapping, so the HLSL and GLSL generators emitted them verbatim (an undeclared function) and the
SPIR-V generator refused them. :func:`lower_metal_math_spellings` handles them for every target except Metal:

* ``powr(x, y)`` is ``pow(x, y)``. MSL defines ``powr`` for ``x >= 0`` (MSL 3.2 section 6.5), where ``pow`` of
  GLSL (8.2), HLSL and SPIR-V (GLSL.std.450 ``Pow``) computes the same value.
* ``exp10(x)`` becomes ``exp2(x * log2(10))``, the form GLSL and HLSL themselves give ``pow`` (GLSL 4.60 section 4.7.1
  bounds ``pow`` by ``exp2(y * log2(x))``), within MSL's fast-math bound for ``exp10``.
* ``sinpi``, ``cospi``, ``tanpi``, ``asinpi``, ``acospi``, ``atanpi`` and ``atan2pi`` are exact at integers and
  half-integers in MSL (``sinpi(1.0)`` is ``0``), which ``sin(x * pi)`` is not. They are refused with
  :class:`MetalMathFunctionUnsupportedError` until an exact range reduction lowers them. CUDA and HIP keep ``sinpi``
  and ``cospi``: both math libraries declare ``sinpi``/``cospi`` for ``double`` and ``sinpif``/``cospif`` for
  ``float``, exact at integers and half-integers, and their generators pick the spelling for the operand's precision
  and apply it per component to vectors. Neither declares the other five, so those stay refused there too.

A program that defines a function with one of these names keeps its calls.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import call_name, iter_functions
from .rewrite import child_slots, get_child, set_child

__all__ = [
    "MetalMathFunctionUnsupportedError",
    "PI_FUNCTIONS",
    "lower_metal_math_spellings",
]

# log2(10), rounded to binary64; the product rounds once more to the operand's type.
LOG2_10 = 3.321928094887362
PI_FUNCTIONS = frozenset(
    {"sinpi", "cospi", "tanpi", "asinpi", "acospi", "atanpi", "atan2pi"}
)
_ARITY = {"powr": 2, "exp10": 1, **{name: 1 for name in PI_FUNCTIONS}, "atan2pi": 2}
_CARRYING = frozenset({"metal", "cgl", "crossgl", ""})
# Pi functions a target's own math library provides, so the call is kept for its generator.
_NATIVE_PI_FUNCTIONS = {
    "cuda": frozenset({"sinpi", "cospi"}),
    "hip": frozenset({"sinpi", "cospi"}),
}


class MetalMathFunctionUnsupportedError(TranslationDiagnostic):
    """An MSL math function whose exact values the target has no function or lowering for."""

    def __init__(self, target, name):
        self.missing_capabilities = (f"{target}.metal-math-{name}",)
        super().__init__(
            f"{name} is exact at integers and half-integers, which {target} has no function for, and CrossTL "
            "does not lower it with an exact range reduction yet",
            target=target,
            code=f"{CODE_PREFIX}{target}-metal-math-function-unsupported",
        )


def _identifier(name):
    return A.IdentifierNode(name)


def _replacement(node, name, target):
    arguments = list(getattr(node, "arguments", None) or [])
    if len(arguments) != _ARITY[name]:
        return node
    if name in PI_FUNCTIONS:
        if name in _NATIVE_PI_FUNCTIONS.get(target, ()):
            return node
        raise MetalMathFunctionUnsupportedError(target, name)
    if name == "powr":
        return A.FunctionCallNode(_identifier("pow"), arguments)
    scaled = A.BinaryOpNode(
        arguments[0], "*", A.LiteralNode(LOG2_10, A.PrimitiveType("float"))
    )
    return A.FunctionCallNode(_identifier("exp2"), [scaled])


def _rewrite(node, replace):
    for field, key in list(child_slots(node)):
        child = get_child(node, field, key)
        _rewrite(child, replace)
        replacement = replace(child)
        if replacement is not child:
            set_child(node, field, key, replacement)


def lower_metal_math_spellings(program, target) -> int:
    """Rewrite or refuse ``program``'s MSL-only math calls for ``target``; returns how many were rewritten."""
    target = str(target or "").lower()
    if target in _CARRYING or not isinstance(program, A.ShaderNode):
        return 0
    functions = list(iter_functions(program))
    names = set(_ARITY) - {getattr(function, "name", None) for function in functions}
    changed = 0

    def replace(node):
        nonlocal changed
        if not isinstance(node, A.FunctionCallNode):
            return node
        name = call_name(node)
        if name not in names:
            return node
        replacement = _replacement(node, name, target)
        changed += replacement is not node
        return replacement

    for function in functions:
        _rewrite(function, replace)
    return changed
