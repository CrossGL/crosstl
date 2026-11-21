"""Metal ballot masks converted to their ``vote_t`` integers, imported through CrossGL's ``WaveActiveBallot``.

``simd_ballot`` and ``simd_active_threads_mask`` return a ``simd_vote``, and ``quad_ballot`` and
``quad_active_threads_mask`` return a ``quad_vote`` (Metal Shading Language Specification, sections 6.9.2 and 6.9.3).
Both wrap a bit mask that converts to its ``vote_t`` only explicitly: ``simd_vote::vote_t`` is 64 bits with bit ``n``
for SIMD lane ``n``, and ``quad_vote::vote_t`` is a ``ushort`` whose constructor keeps the low four bits, bit ``n``
for lane ``n`` of the quad (the ``metal_simdgroup`` and ``metal_quadgroup`` headers). An explicit conversion accepts
only the ``vote_t`` type itself, so ``uint(simd_ballot(p))`` does not compile.

CrossGL has no type for either mask. Its ``WaveActiveBallot`` returns a ``uvec4`` that holds SIMD lane ``n`` as bit
``n % 32`` of component ``n / 32``. An explicit conversion of one of the four ballot calls to its ``vote_t`` imports as
a helper over that ``uvec4``: the ``simd_vote`` helper joins components ``x`` and ``y`` into 64 bits, and the
``quad_vote`` helper takes the four bits of the caller's quad, which is SIMD lanes ``lane & ~3`` to ``(lane & ~3) + 3``.
A conversion of any other ``simd_vote`` or ``quad_vote`` value has no CrossGL form and is refused.

A mask's ``any()`` is true when an active thread of the SIMD-group or quad set its bit, so ``any()`` called on a
ballot imports as ``WaveActiveAnyTrue`` or ``QuadAny`` of the predicate, and on an active-threads mask as ``true``.
A mask's ``all()`` is not ``simd_all`` or ``quad_all``: on an Apple M2 Max, ``simd_ballot(p).all()``,
``quad_ballot(p).all()`` and even ``simd_active_threads_mask().all()`` are false in a group with an inactive thread,
where ``simd_all(p)`` and ``quad_all(p)`` read only the active threads. ``all()`` and either method called on any
other mask are refused.
"""

from __future__ import annotations

from typing import Any

from .MetalAst import FunctionCallNode, MethodCallNode

__all__ = [
    "generate_vote_mask_support_code",
    "render_vote_mask_conversion",
    "render_vote_mask_method",
]

# Ballot call -> (its mask, whether it takes a predicate).
_BALLOT_CALLS = {
    "simd_ballot": ("simd", True),
    "simd_active_threads_mask": ("simd", False),
    "quad_ballot": ("quad", True),
    "quad_active_threads_mask": ("quad", False),
}
# Mask -> (its vote_t spelling, the CrossGL type of that vote_t).
_VOTE_TYPES = {
    "simd": ("simd_vote::vote_t", "uint64"),
    "quad": ("quad_vote::vote_t", "uint16"),
}
# Mask -> the CrossGL reduction that ``any()`` of a ballot's mask computes from its predicate.
_ANY_REDUCTIONS = {"simd": "WaveActiveAnyTrue", "quad": "QuadAny"}
_HELPERS = {
    "simd": ("simd-vote-bits", "__crossgl_metal_simd_vote_bits"),
    "quad": ("quad-vote-bits", "__crossgl_metal_quad_vote_bits"),
}


def _unqualified(name: Any) -> str | None:
    if not isinstance(name, str):
        return None
    name = "".join(name.split())
    while name.startswith("::"):
        name = name[2:]
    if name.startswith("metal::"):
        name = name[len("metal::") :]
    return name


def _vote_type_mask(target_type: Any) -> str | None:
    """The mask whose ``vote_t`` ``target_type`` names, or None."""
    spelled = _unqualified(target_type)
    for mask, (vote_type, _crossgl_type) in _VOTE_TYPES.items():
        if spelled == vote_type:
            return mask
    return None


def _ballot_call(operand: Any) -> tuple[str, bool] | None:
    if not isinstance(operand, FunctionCallNode):
        return None
    return _BALLOT_CALLS.get(_unqualified(getattr(operand, "name", None)))


def render_vote_mask_conversion(
    converter: Any, target_type: Any, operand: Any, is_main: bool = False
) -> str | None:
    """CrossGL for an explicit conversion of a Metal ballot mask to an integer, or None when it is not one.

    ``target_type`` is the conversion's type, from ``static_cast<T>(x)``, ``(T)x`` or ``T(x)``. The conversion is one
    when ``T`` is a ``vote_t`` or when ``x`` is a ballot call whose ``vote_t`` CrossGL spells as ``T``.
    """
    ballot = _ballot_call(operand)
    vote_mask = _vote_type_mask(target_type)
    if ballot is None:
        if vote_mask is None:
            return None
        return converter.refuse_import(
            f"Metal converts a {vote_mask}_vote value to {_VOTE_TYPES[vote_mask][0]}, but only a direct "
            f"conversion of a {vote_mask}_ballot or {vote_mask}_active_threads_mask call has a CrossGL form; "
            "CrossGL has no type for the mask itself",
            "0",
        )
    mask, takes_predicate = ballot
    crossgl_type = _VOTE_TYPES[mask][1]
    if vote_mask is None:
        if converter.map_type(target_type) != crossgl_type:
            # Metal accepts no other conversion of the mask, so the call is not a ballot conversion to render here.
            return None
    elif vote_mask != mask:
        return None
    arguments = list(getattr(operand, "args", None) or [])
    if takes_predicate:
        if len(arguments) != 1:
            return None
        predicate = converter.generate_expression(arguments[0], is_main)
    else:
        if arguments:
            return None
        predicate = "true"
    required = converter.__dict__.setdefault("required_metal_vote_masks", set())
    required.add(mask)
    helper = converter.metal_precise_math_unique_helper_name(*_HELPERS[mask])
    ballot_value = f"WaveActiveBallot({predicate})"
    if mask == "simd":
        return f"{helper}({ballot_value})"
    return f"{helper}({ballot_value}, WaveGetLaneIndex())"


def render_vote_mask_method(
    converter: Any, expr: Any, is_main: bool = False
) -> str | None:
    """CrossGL for ``any()`` called on a Metal ballot call, or None when ``expr`` calls no mask method on one."""
    if not isinstance(expr, MethodCallNode) or expr.method not in ("all", "any"):
        return None
    ballot = _ballot_call(expr.object)
    if ballot is None or expr.args:
        return None
    mask, takes_predicate = ballot
    arguments = list(getattr(expr.object, "args", None) or [])
    if len(arguments) != int(takes_predicate):
        return None
    if expr.method == "all":
        return converter.refuse_import(
            f"Metal {mask}_vote::all() is false in a {'SIMD-group' if mask == 'simd' else 'quad'} with an "
            "inactive thread, which no CrossGL wave or quad reduction computes",
            "false",
        )
    if not takes_predicate:
        # The calling thread is active, so it set its bit of the active-threads mask.
        return "true"
    predicate = converter.generate_expression(arguments[0], is_main)
    return f"{_ANY_REDUCTIONS[mask]}({predicate})"


def generate_vote_mask_support_code(converter: Any, indent: int = 0) -> str:
    """The CrossGL helpers :func:`render_vote_mask_conversion` called, in a fixed order."""
    required = converter.__dict__.get("required_metal_vote_masks") or set()
    code = ""
    if "simd" in required:
        helper = converter.metal_precise_math_unique_helper_name(*_HELPERS["simd"])
        code += (
            f"@metal_static\nuint64 {helper}(uvec4 ballot) {{\n"
            "    return uint64(ballot.x) | (uint64(ballot.y) << 32u);\n"
            "}\n"
        )
    if "quad" in required:
        helper = converter.metal_precise_math_unique_helper_name(*_HELPERS["quad"])
        code += (
            f"@metal_static\nuint16 {helper}(uvec4 ballot, uint lane) {{\n"
            "    uint first = lane & 0xfffffffcu;\n"
            "    uint word = first < 32u ? ballot.x : ballot.y;\n"
            "    return uint16((word >> (first & 31u)) & 15u);\n"
            "}\n"
        )
    pad = "    " * indent
    return "".join(pad + line + "\n" for line in code.strip().splitlines())
