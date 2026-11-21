"""CUDA device functions that write through a pointer, called from CrossGL with an ``out`` operand.

CrossGL has no address-of for a function argument, so a CrossGL program passes the variable a CUDA device function
writes as an ``out`` operand: ``__stcg(slot, value)`` for CUDA's ``void __stcg(T* ptr, T value)``, and
``__match_all_sync(mask, value, pred)`` for ``unsigned int __match_all_sync(unsigned int mask, T value, int* pred)``
(CUDA C++ Programming Guide, "Store Functions Using Cache Hints" and "Warp Match Functions"). Written as is, the call
passed an ``int`` where the function takes an ``int*`` and matched no overload. The CUDA generator passes the
operand's address instead, as the C math functions that write through pointers do (:mod:`.c_pointer_math`). An
operand that already is an address or a pointer is passed unchanged.

HIP declares neither family in the ROCm releases CrossTL validates against. A cache hint changes only where the
stored value is cached, never the value a later load observes, so a HIP store with a cache hint is the plain store
``slot = value``. HIP has no warp match function, and ``__match_all_sync`` is refused there.

An operand that is neither an address, a pointer nor an assignable value, a ``bool`` store (no CUDA overload takes
``bool*``) and a call with the wrong number of operands are refused.
"""

from __future__ import annotations

from ..ast import ArrayAccessNode, IdentifierNode, MemberAccessNode, UnaryOpNode

__all__ = [
    "POINTER_OPERAND_FUNCTIONS",
    "cuda_pointer_operand_call",
    "hip_pointer_operand_call",
]

# Function -> (operand count, index of the operand CUDA takes by pointer).
POINTER_OPERAND_FUNCTIONS = {
    "__stcg": (2, 0),
    "__stcs": (2, 0),
    "__stwb": (2, 0),
    "__stwt": (2, 0),
    "__match_all_sync": (3, 2),
}
_STORES = frozenset({"__stcg", "__stcs", "__stwb", "__stwt"})
_ASSIGNABLE = (IdentifierNode, MemberAccessNode, ArrayAccessNode)
_BOOL_TYPES = frozenset({"bool"})


def _type_text(codegen, node):
    try:
        found = codegen.expression_result_type(node)
    except (AttributeError, TypeError, ValueError):
        return ""
    if found is None:
        return ""
    if not isinstance(found, str):
        found = codegen.type_name_string(found)
    return " ".join(str(found).split())


def _operand(codegen, func_name, raw, text, refusal):
    """``(kind, text)``: ``"address"`` for an address or pointer, ``"value"`` for an assignable value."""
    if isinstance(raw, UnaryOpNode) and raw.operator == "&" and not raw.is_postfix:
        return "address", text
    if _type_text(codegen, raw).endswith("*"):
        return "address", text
    if isinstance(raw, _ASSIGNABLE):
        return "value", text
    raise refusal(
        f"{func_name} writes through its pointer operand, and '{text}' is neither an address, a pointer "
        "nor an assignable value"
    )


def _check(codegen, func_name, raw_args, refusal, target):
    count, index = POINTER_OPERAND_FUNCTIONS[func_name]
    if len(raw_args) != count:
        raise refusal(f"{func_name} takes {count} operands, got {len(raw_args)}")
    if func_name in _STORES and _type_text(codegen, raw_args[1]) in _BOOL_TYPES:
        raise refusal(f"{target} has no {func_name} overload that stores a bool")
    return index


def cuda_pointer_operand_call(codegen, func_name, raw_args, args, refusal):
    """The CUDA call for ``func_name``, or None when it is not one of :data:`POINTER_OPERAND_FUNCTIONS`."""
    if func_name not in POINTER_OPERAND_FUNCTIONS:
        return None
    index = _check(codegen, func_name, raw_args, refusal, "CUDA")
    kind, text = _operand(codegen, func_name, raw_args[index], args[index], refusal)
    operands = list(args)
    if kind == "value":
        operands[index] = f"&({text})"
    return f"{func_name}({', '.join(operands)})"


def hip_pointer_operand_call(codegen, func_name, raw_args, args, refusal):
    """The HIP spelling of ``func_name``, or None when it is not one of :data:`POINTER_OPERAND_FUNCTIONS`."""
    if func_name not in POINTER_OPERAND_FUNCTIONS:
        return None
    if func_name not in _STORES:
        raise refusal(
            f"HIP has no warp match function, so {func_name} has no HIP spelling"
        )
    index = _check(codegen, func_name, raw_args, refusal, "HIP")
    kind, text = _operand(codegen, func_name, raw_args[index], args[index], refusal)
    raw = raw_args[index]
    if kind == "address" and isinstance(raw, UnaryOpNode) and raw.operator == "&":
        target = codegen.visit(raw.operand)
    elif kind == "address":
        target = f"*({text})"
    else:
        target = text
    return f"({target} = {args[1]})"
