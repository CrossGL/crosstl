"""Metal ballot masks imported as CrossGL ``WaveActiveBallot`` values.

``simd_ballot`` and ``simd_active_threads_mask`` return a ``simd_vote``, and ``quad_ballot`` and
``quad_active_threads_mask`` return a ``quad_vote`` (Metal Shading Language Specification, sections 6.9.2 and 6.9.3).
Both wrap a bit mask that converts to its ``vote_t`` only explicitly: ``simd_vote::vote_t`` is 64 bits with bit ``n``
for SIMD lane ``n``, and ``quad_vote::vote_t`` is a ``ushort`` whose constructor keeps the low four bits, bit ``n``
for lane ``n`` of the quad (the ``metal_simdgroup`` and ``metal_quadgroup`` headers). An explicit conversion accepts
only the ``vote_t`` type itself, so ``uint(simd_ballot(p))`` does not compile.

CrossGL has no type for either mask. Its ``WaveActiveBallot`` returns a ``uvec4`` that holds SIMD lane ``n`` as bit
``n % 32`` of component ``n / 32``, so a mask imports as that ``uvec4``: each of the four ballot calls becomes
``WaveActiveBallot`` of its predicate, or of ``true`` for an active-threads mask, and a ``simd_vote`` or ``quad_vote``
variable, parameter or member becomes a ``uvec4``. A ``quad_vote`` holds the ballot of the whole SIMD-group; the
caller's quad is selected where the mask is read, by the same thread. An explicit conversion of a mask to its
``vote_t`` imports as a helper over the ``uvec4``: the ``simd_vote`` helper joins components ``x`` and ``y`` into 64
bits, and the ``quad_vote`` helper takes the four bits of the caller's quad, which is SIMD lanes ``lane & ~3`` to
``(lane & ~3) + 3``. A mask constructed from a ``vote_t`` has no CrossGL form and is refused.

A mask's ``any()`` is true when its ``vote_t`` is not zero. Called on a ballot, it imports as ``WaveActiveAnyTrue`` or
``QuadAny`` of the predicate, and on an active-threads mask as ``true``, because the calling thread set its own bit;
called on a stored mask, it compares the helper's ``vote_t`` with zero. A mask's ``all()`` is not ``simd_all`` or
``quad_all``: on an Apple M2 Max, ``simd_ballot(p).all()``, ``quad_ballot(p).all()`` and even
``simd_active_threads_mask().all()`` are false in a group with an inactive thread, where ``simd_all(p)`` and
``quad_all(p)`` read only the active threads. ``all()`` is refused.
"""

from __future__ import annotations

from typing import Any

from .MetalAst import BinaryOpNode, FunctionCallNode, MethodCallNode, UnaryOpNode

__all__ = [
    "MASK_CROSSGL_TYPES",
    "generate_vote_mask_support_code",
    "render_ballot_call",
    "render_vote_mask_conversion",
    "render_vote_mask_method",
    "vote_mask_call_metal_type",
    "vote_mask_default_initializer",
]

# Ballot call -> (its mask, whether it takes a predicate).
_BALLOT_CALLS = {
    "simd_ballot": ("simd", True),
    "simd_active_threads_mask": ("simd", False),
    "quad_ballot": ("quad", True),
    "quad_active_threads_mask": ("quad", False),
}
# Mask -> (its Metal type, its vote_t spelling, the CrossGL type of that vote_t).
_VOTE_TYPES = {
    "simd": ("simd_vote", "simd_vote::vote_t", "uint64"),
    "quad": ("quad_vote", "quad_vote::vote_t", "uint16"),
}
# The CrossGL spelling of each Metal mask type and vote_t, for the importer's type table.
MASK_CROSSGL_TYPES = {
    **{metal_type: "uvec4" for metal_type, _vote, _bits in _VOTE_TYPES.values()},
    **{vote: bits for _metal_type, vote, bits in _VOTE_TYPES.values()},
}
_VOTE_BITS = {bits for _metal_type, _vote, bits in _VOTE_TYPES.values()}
_MASK_TYPES = {metal_type: mask for mask, (metal_type, _v, _b) in _VOTE_TYPES.items()}
_TYPE_QUALIFIERS = {"const", "volatile", "thread", "device", "threadgroup", "constant"}
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
    for mask, (_metal_type, vote_type, _crossgl_type) in _VOTE_TYPES.items():
        if spelled == vote_type:
            return mask
    return None


def _mask_of_type(metal_type: Any) -> str | None:
    """The mask a Metal value of ``metal_type`` holds, or None when it is not a ``simd_vote`` or ``quad_vote``."""
    if not isinstance(metal_type, str):
        return None
    words = [
        word
        for word in metal_type.replace("&", " ").split()
        if word not in _TYPE_QUALIFIERS
    ]
    if len(words) != 1:
        return None
    return _MASK_TYPES.get(_unqualified(words[0]))


def _ballot_call(converter: Any, operand: Any) -> tuple[str, bool] | None:
    """The mask and predicate flag of a call to one of Metal's four ballot functions, or None."""
    if not isinstance(operand, FunctionCallNode):
        return None
    ballot = _BALLOT_CALLS.get(_unqualified(getattr(operand, "name", None)))
    if ballot is None:
        return None
    if len(getattr(operand, "args", None) or ()) != int(ballot[1]):
        return None
    if not str(operand.name).lstrip(":").startswith("metal::"):
        overloads = converter.metal_user_function_overloads(operand.name)
        if not all(
            converter.is_materialized_metal_stdlib_wrapper(function)
            for function in overloads
        ):
            return None
    return ballot


def vote_mask_call_metal_type(converter: Any, expr: Any) -> str | None:
    """The Metal mask type of a ballot call, or None when ``expr`` calls no ballot function."""
    ballot = _ballot_call(converter, expr)
    return None if ballot is None else _VOTE_TYPES[ballot[0]][0]


def _value_mask(converter: Any, operand: Any) -> str | None:
    """The mask ``operand`` evaluates to, or None when it is no ``simd_vote`` or ``quad_vote`` value."""
    if isinstance(operand, (bool, int, float, BinaryOpNode, UnaryOpNode)):
        # Metal declares no operator on a mask, so an operation never yields one.
        return None
    ballot = _ballot_call(converter, operand)
    if ballot is not None:
        return ballot[0]
    return _mask_of_type(converter.expression_metal_type(operand))


def _read_mask(converter: Any, mask: str, value: str) -> str:
    """CrossGL for the ``vote_t`` of the ``uvec4`` ballot ``value`` holding a ``mask`` mask."""
    required = converter.__dict__.setdefault("required_metal_vote_masks", set())
    required.add(mask)
    helper = converter.metal_precise_math_unique_helper_name(*_HELPERS[mask])
    if mask == "simd":
        return f"{helper}({value})"
    return f"{helper}({value}, WaveGetLaneIndex())"


def render_ballot_call(converter: Any, expr: Any, is_main: bool = False) -> str | None:
    """CrossGL for a Metal ballot call or mask constructor, or None when ``expr`` is neither."""
    if not isinstance(expr, FunctionCallNode):
        return None
    constructed = _MASK_TYPES.get(_unqualified(getattr(expr, "name", None)))
    if constructed is not None:
        return converter.refuse_import(
            f"Metal constructs a {_VOTE_TYPES[constructed][0]} from its vote_t, but CrossGL has no type for the "
            "mask; it imports only the masks Metal's ballot functions return",
            "uvec4(0u)",
        )
    ballot = _ballot_call(converter, expr)
    if ballot is None:
        return None
    _mask, takes_predicate = ballot
    predicate = (
        converter.generate_expression(expr.args[0], is_main)
        if takes_predicate
        else "true"
    )
    return f"WaveActiveBallot({predicate})"


def vote_mask_default_initializer(converter: Any, declaration: Any) -> str | None:
    """The CrossGL initializer of a default-constructed Metal mask local, or None when ``declaration`` is none.

    ``simd_vote`` and ``quad_vote`` default-construct from a zero ``vote_t``, so a mask declared without an initializer
    starts as a zero ballot. An array of masks has no single CrossGL initializer and is refused.
    """
    qualifiers = {
        str(q).lower() for q in getattr(declaration, "qualifiers", None) or ()
    }
    if "threadgroup" in qualifiers or "extern" in qualifiers:
        return None
    metal_type = converter.metal_declaration_expression_type(declaration)
    if _mask_of_type(metal_type) is not None:
        return "uvec4(0u)"
    element = str(metal_type or "").split("[", 1)
    if len(element) == 2 and _mask_of_type(element[0]) is not None:
        converter.refuse_import(
            f"Metal default-constructs each element of the mask array {declaration.name} to a zero vote_t, "
            "which no CrossGL array declaration states"
        )
    return None


def _conversion_bits(converter: Any, target_type: Any) -> str | None:
    """The CrossGL type of ``vote_t`` that converting to ``target_type`` yields, or None when it yields neither.

    ``target_type`` may be a function name, so it is mapped only once the importer knows it as a type.
    """
    vote_mask = _vote_type_mask(target_type)
    if vote_mask is not None:
        return _VOTE_TYPES[vote_mask][2]
    if converter.metal_constructor_result_type(target_type) is None:
        return None
    mapped = converter.map_type(target_type)
    return mapped if mapped in _VOTE_BITS else None


def render_vote_mask_conversion(
    converter: Any, target_type: Any, operand: Any, is_main: bool = False
) -> str | None:
    """CrossGL for an explicit conversion of a Metal mask to its ``vote_t``, or None when it is not one.

    ``target_type`` is the conversion's type, from ``static_cast<T>(x)``, ``(T)x``, ``x.operator T()`` or ``T(x)``,
    where ``T`` may also name a function the program calls with ``x``. Only a conversion to a ``vote_t`` type can
    convert a mask, and a conversion to ``vote_t`` of a value that is no mask is an ordinary integer conversion, so
    None leaves both other cases to the caller.
    """
    bits = _conversion_bits(converter, target_type)
    if bits is None:
        return None
    mask = _value_mask(converter, operand)
    if mask is None:
        return None
    _metal_type, vote_type, crossgl_type = _VOTE_TYPES[mask]
    if bits != crossgl_type:
        return converter.refuse_import(
            f"Metal converts a {_VOTE_TYPES[mask][0]} only to {vote_type}, not to {target_type}",
            "0",
        )
    return _read_mask(converter, mask, converter.generate_expression(operand, is_main))


def render_vote_mask_method(
    converter: Any, expr: Any, is_main: bool = False
) -> str | None:
    """CrossGL for ``any()`` or ``all()`` called on a Metal mask, or None when ``expr`` calls neither on one."""
    if not isinstance(expr, MethodCallNode) or expr.method not in ("all", "any"):
        return None
    if expr.args:
        return None
    ballot = _ballot_call(converter, expr.object)
    mask = ballot[0] if ballot is not None else _value_mask(converter, expr.object)
    if mask is None:
        return None
    if expr.method == "all":
        return converter.refuse_import(
            f"Metal {mask}_vote::all() is false in a {'SIMD-group' if mask == 'simd' else 'quad'} with an "
            "inactive thread, which no CrossGL wave or quad reduction computes",
            "false",
        )
    if ballot is None:
        bits = _read_mask(
            converter, mask, converter.generate_expression(expr.object, is_main)
        )
        return f"({bits} != {_VOTE_TYPES[mask][2]}(0u))"
    if not ballot[1]:
        # The calling thread is active, so it set its bit of the active-threads mask.
        return "true"
    predicate = converter.generate_expression(expr.object.args[0], is_main)
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
