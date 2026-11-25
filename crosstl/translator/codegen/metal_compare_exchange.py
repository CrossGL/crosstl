"""Strong compare-exchange loops for the Metal compare-exchange helpers.

Metal has only the weak ``atomic_compare_exchange_weak_explicit``, which may fail spuriously, so each helper retries
while the value it observed still equals the comparand and returns the value it observed.

CrossGL's ``atomicCompSwap`` on a ``float`` compares the bits of the operands, as HLSL's
``InterlockedCompareExchangeFloatBitwise`` and ``InterlockedCompareStoreFloatBitwise`` do (shader model 6.6;
``semantics/atomic_forms.py``). A ``float`` ``==`` is not that comparison: it equates ``-0.0`` with ``0.0`` and never
matches a NaN. The ``float`` loop therefore exchanges the 32-bit word that holds the value through ``atomic_uint``,
converting the operands and the result with ``as_type`` (MSL 4.1 section 2.19), so both the exchange and the retry
test compare bits.
"""

from __future__ import annotations

__all__ = ["compare_exchange_loop", "supports_compare_exchange"]

# The component types whose compare-exchange a helper writes. ``float`` goes through its bits.
_WORD = {"int": "int", "uint": "uint", "float": "uint"}


def supports_compare_exchange(component_type):
    """Whether a compare-exchange helper can be written for ``component_type``."""
    return component_type in _WORD


def compare_exchange_loop(
    atomic_pointer_type, target_expr, component_type, name="atomicTarget"
):
    """The body of a helper ``T f(..., T compareValue, T value)`` that compare-exchanges ``target_expr``.

    ``atomic_pointer_type`` is the address-space-qualified atomic pointer type without its component, such as
    ``"device atomic_"``; ``target_expr`` addresses the stored value, and ``name`` is the local that holds the
    atomic pointer.
    """
    word = _WORD[component_type]
    if word == component_type:
        return (
            f"    {atomic_pointer_type}{word}* {name} = "
            f"reinterpret_cast<{atomic_pointer_type}{word}*>({target_expr});\n"
            f"    {component_type} original;\n"
            "    do {\n"
            "        original = compareValue;\n"
            f"    }} while (!atomic_compare_exchange_weak_explicit({name}, "
            "&original, value, memory_order_relaxed, memory_order_relaxed) "
            "&& original == compareValue);\n"
            "    return original;\n"
        )
    return (
        f"    {atomic_pointer_type}{word}* {name} = "
        f"reinterpret_cast<{atomic_pointer_type}{word}*>({target_expr});\n"
        f"    {word} compareWord = as_type<{word}>(compareValue);\n"
        f"    {word} valueWord = as_type<{word}>(value);\n"
        f"    {word} original;\n"
        "    do {\n"
        "        original = compareWord;\n"
        f"    }} while (!atomic_compare_exchange_weak_explicit({name}, "
        "&original, valueWord, memory_order_relaxed, memory_order_relaxed) "
        "&& original == compareWord);\n"
        f"    return as_type<{component_type}>(original);\n"
    )
