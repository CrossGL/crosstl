"""CrossGL forms of the CUDA and HIP bit intrinsics ``__clz``, ``__clzll``, ``__byte_perm`` and ``__funnelshift_*``.

CUDA Math API, "Integer Intrinsics", defines them, and HIP declares the same functions with the same results
(``hip/amd_detail/amd_device_functions.h``). CrossGL has none of them, so both importers write them with CrossGL's
own bit operations, which every target declares:

- ``__clz(x)`` counts the leading zero bits of the 32-bit ``x``, 0 to 32. It becomes ``(31 - findMSB(u32(x)))``:
  ``findMSB`` of an unsigned value is the index of its highest set bit and -1 for zero (GLSL 4.60 section 8.8), so
  zero gives 32. The ``u32`` conversion keeps a negative ``x`` from taking the signed ``findMSB``, which finds the
  highest clear bit.
- ``__clzll(x)`` counts the leading zero bits of the 64-bit ``x``, 0 to 64. ``findMSB`` takes 32-bit operands only
  (SPIR-V GLSL.std.450 ``FindUMsb``), so the form tests the high word and counts in the low word when it is zero.
  It reads ``x`` twice.
- ``__byte_perm(x, y, s)``: byte ``n`` of the result is byte ``(s >> 4n) & 7`` of the eight bytes of ``y:x``, where
  ``x`` holds bytes 0 to 3. Only the low three bits of each selector nibble count: nvcc masks ``s`` with ``0x7777``
  before PTX ``prmt``, and HIP's ``__byte_perm`` masks each nibble with 7. A literal selector picks its bytes when
  the importer runs; any other selector picks them in the program. Both read ``x`` and ``y`` more than once.
- ``__funnelshift_l(lo, hi, shift)`` is the high word of ``hi:lo`` shifted left by ``shift & 31``, and
  ``__funnelshift_r`` the low word of ``hi:lo`` shifted right by ``shift & 31``. The ``lc`` and ``rc`` forms shift by
  ``min(shift, 32)`` instead. The forms stay in 32-bit words where they can, because WGSL and GLSL ES have no
  64-bit integers. A literal shift is applied when the importer runs. Any other shift reads ``shift`` more than
  once, so it stays in 32-bit words when no argument has side effects: ``(lo >> 1) >> (31 - s)`` is
  ``lo >> (32 - s)`` without the shift by 32 that ``s == 0`` would need, which C++, GLSL and SPIR-V leave undefined.
  Otherwise each argument is read once in the 64-bit form ``u32(((u64(hi) << 32) | u64(lo)) << s >> 32)``.

Every operand passes through ``u32`` as it does through the ``unsigned int`` parameter of the intrinsic (C++
[conv.integral]). An intrinsic that would read an argument with side effects twice is refused with the importer's
typed diagnostic, because CrossGL expressions have no temporaries to hold it.
"""

from __future__ import annotations

from .pointer_math_import import is_pure_expression

__all__ = ["BIT_INTRINSICS", "format_bit_intrinsic"]

# Intrinsic name -> number of arguments.
BIT_INTRINSICS = {
    "__clz": 1,
    "__clzll": 1,
    "__byte_perm": 3,
    "__funnelshift_l": 3,
    "__funnelshift_lc": 3,
    "__funnelshift_r": 3,
    "__funnelshift_rc": 3,
}


def format_bit_intrinsic(converter, name, nodes, args, language):
    """The CrossGL text of a bit intrinsic call, or ``None`` when ``name`` is not one with its argument count.

    ``nodes`` are the parsed arguments and ``args`` their CrossGL text. ``language`` (``"CUDA"`` or ``"HIP"``) names
    the source in a refusal, which ``converter.refuse_import`` records.
    """
    if not isinstance(name, str):
        return None
    bare = name[2:] if name.startswith("::") else name
    if BIT_INTRINSICS.get(bare) != len(args) or len(nodes) != len(args):
        return None
    if bare == "__clz":
        return f"(31 - findMSB({_unsigned(args[0])}))"
    if bare == "__clzll":
        if not is_pure_expression(nodes[0]):
            return _refuse_impure(converter, language, bare, args, "x")
        return _count_leading_zeros_64(args[0])
    if bare == "__byte_perm":
        return _byte_perm(converter, language, nodes, args)
    return _funnel_shift(bare, nodes, args)


def _refuse_impure(converter, language, name, args, operand):
    return converter.refuse_import(
        f"{language} integer intrinsic {name}({', '.join(args)}) reads its operand {operand} more than once, and "
        "the operand has side effects that CrossGL would repeat",
        "0u",
    )


def _unsigned(text):
    """``text`` as a ``u32`` operand: a literal with a ``u`` suffix as it is, anything else converted."""
    stripped = str(text).strip()
    literal = _integer_literal(stripped)
    if literal is not None and 0 <= literal <= 0xFFFFFFFF:
        return f"{literal:#x}u" if literal > 9 else f"{literal}u"
    return f"u32({stripped})"


def _integer_literal(text):
    """The value of an integer literal such as ``0x7777u``, ``(3)`` or ``1'000``, or ``None``."""
    text = str(text).strip().replace("'", "")
    while text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    while text and text[-1] in "uUlL":
        text = text[:-1]
    if not text or not text[0].isdigit():
        return None
    try:
        return (
            int(text, 0)
            if not (len(text) > 1 and text[0] == "0" and text[1].isdigit())
            else int(text, 8)
        )
    except ValueError:
        return None


def _count_leading_zeros_64(value):
    high = f"u32(u64({value}) >> 32u)"
    low = f"u32(u64({value}))"
    return f"(({high} != 0u) ? (31 - findMSB({high})) : (63 - findMSB({low})))"


def _byte_perm(converter, language, nodes, args):
    for index, operand in ((0, "x"), (1, "y")):
        if not is_pure_expression(nodes[index]):
            return _refuse_impure(converter, language, "__byte_perm", args, operand)
    sources = (_unsigned(args[0]), _unsigned(args[1]))
    selector = _integer_literal(args[2])
    if selector is not None:
        return _static_byte_perm(sources, selector)
    if not is_pure_expression(nodes[2]):
        return _refuse_impure(converter, language, "__byte_perm", args, "s")
    selector_text = _unsigned(args[2])
    terms = []
    for position in range(4):
        nibble = (
            f"(({selector_text} >> {4 * position}u) & 7u)"
            if position
            else f"({selector_text} & 7u)"
        )
        byte = (
            f"(((({nibble} < 4u) ? {sources[0]} : {sources[1]}) >> "
            f"(({nibble} & 3u) * 8u)) & 0xffu)"
        )
        terms.append(f"({byte} << {8 * position}u)" if position else byte)
    return f"({' | '.join(terms)})"


def _static_byte_perm(sources, selector):
    terms = []
    for position in range(4):
        index = (selector >> (4 * position)) & 7
        source = sources[index // 4]
        offset = 8 * (index % 4)
        byte = f"(({source} >> {offset}u) & 0xffu)" if offset else f"({source} & 0xffu)"
        terms.append(f"({byte} << {8 * position}u)" if position else byte)
    return f"({' | '.join(terms)})"


def _funnel_shift(name, nodes, args):
    low, high, shift = (_unsigned(arg) for arg in args)
    left = name.startswith("__funnelshift_l")
    clamped = name.endswith("c")
    pure_low, pure_high, pure_shift = (is_pure_expression(node) for node in nodes)
    count = _integer_literal(args[2])
    if count is not None:
        count = min(count, 32) if clamped else count & 31
        if 0 < count < 32:
            if left:
                return f"(({high} << {count}u) | ({low} >> {32 - count}u))"
            return f"(({low} >> {count}u) | ({high} << {32 - count}u))"
        # A shift by 0 or 32 is one of the words; the other is dropped, so it must have no side effects.
        kept_low = (count == 32) == left
        if pure_high if kept_low else pure_low:
            return low if kept_low else high
    elif pure_low and pure_high and pure_shift:
        masked = f"({shift} & 31u)"
        if left:
            shifted = f"(({high} << {masked}) | (({low} >> 1u) >> (31u - {masked})))"
        else:
            shifted = f"(({low} >> {masked}) | (({high} << 1u) << (31u - {masked})))"
        if not clamped:
            return shifted
        return f"(({shift} >= 32u) ? {low if left else high} : {shifted})"
    amount = f"min({shift}, 32u)" if clamped else f"({shift} & 31u)"
    source = f"((u64({high}) << 32u) | u64({low}))"
    if left:
        return f"u32(({source} << {amount}) >> 32u)"
    return f"u32({source} >> {amount})"
