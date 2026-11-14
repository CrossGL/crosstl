"""No-contraction scopes of CUDA, HIP and OpenCL C sources.

C-family GPU languages let the compiler contract ``a * b + c`` into one fused multiply-add unless the source
forbids it locally:

- CUDA and HIP ``__fadd_rn``, ``__fsub_rn``, ``__fmul_rn`` and their binary64 forms ``__dadd_rn``, ``__dsub_rn``
  and ``__dmul_rn`` are "never merged into a single multiply-add instruction" (CUDA Math API, single and double
  precision intrinsics; HIP documents the same intrinsics).
- ``#pragma OPENCL FP_CONTRACT OFF`` (OpenCL C 6.15.2.1) and ``#pragma STDC FP_CONTRACT OFF`` (C11 7.12.2) forbid
  contraction from the pragma to the next ``FP_CONTRACT`` pragma, or to the end of the compound statement that
  contains it.

CrossGL has no per-operation contraction control. An importer therefore marks each function that contains such
a scope ``@precise``, which evaluates all of the function's floating-point arithmetic as written. Contraction is
only ever permitted, never required, so evaluating the rest of the function without it keeps its meaning.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "NO_CONTRACTION_INTRINSICS",
    "PRECISE_FUNCTION_ATTRIBUTE",
    "FpContractState",
    "function_requires_no_contraction",
    "fp_contract_pragma_state",
]

NO_CONTRACTION_INTRINSICS = frozenset(
    {
        "__fadd_rn",
        "__fsub_rn",
        "__fmul_rn",
        "__dadd_rn",
        "__dsub_rn",
        "__dmul_rn",
    }
)

PRECISE_FUNCTION_ATTRIBUTE = "@precise"

_FP_CONTRACT_NAMESPACES = ("OPENCL", "STDC")


class FpContractState:
    """The ``FP_CONTRACT`` setting in effect at file scope while an importer walks a translation unit."""

    def __init__(self):
        self.off = False

    def update(self, node: Any) -> bool:
        """Apply ``node`` if it is a file-scope ``FP_CONTRACT`` pragma; return whether it was one."""
        state = fp_contract_pragma_state(node)
        if state is None:
            return False
        self.off = state == "OFF"
        return True


def fp_contract_pragma_state(node: Any) -> str | None:
    """Return ``ON``, ``OFF`` or ``DEFAULT`` for an ``FP_CONTRACT`` pragma node, otherwise ``None``."""
    if str(getattr(node, "directive", "")).strip() != "pragma":
        return None
    words = str(getattr(node, "content", "")).replace("\t", " ").split()
    if len(words) != 3 or words[0] not in _FP_CONTRACT_NAMESPACES:
        return None
    if words[1] != "FP_CONTRACT" or words[2] not in {"ON", "OFF", "DEFAULT"}:
        return None
    return words[2]


def function_requires_no_contraction(
    function: Any, *, file_scope_off: bool = False
) -> bool:
    """Return whether ``function`` contains a no-contraction scope (see the module docstring)."""
    if file_scope_off:
        return True
    body = getattr(function, "body", None)
    if body is None:
        return False
    for node in _walk(body):
        name = getattr(node, "name", None)
        if (
            isinstance(name, str)
            and name in NO_CONTRACTION_INTRINSICS
            and hasattr(node, "args")
        ):
            return True
        if fp_contract_pragma_state(node) == "OFF":
            return True
    return False


def _walk(root: Any):
    """Yield every AST node under ``root`` once, ignoring parent links."""
    stack = [root]
    seen = set()
    while stack:
        node = stack.pop()
        if node is None or isinstance(node, (str, bytes, int, float, bool)):
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, dict):
            stack.extend(node.values())
            continue
        if isinstance(node, (list, tuple, set, frozenset)):
            stack.extend(node)
            continue
        fields = getattr(node, "__dict__", None)
        if fields is None:
            continue
        yield node
        stack.extend(value for key, value in fields.items() if key != "parent")
