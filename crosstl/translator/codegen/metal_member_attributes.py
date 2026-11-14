"""Placement of Metal struct member attributes on array declarators.

In C++ an attribute after an array declarator's extent appertains to the array type, and an attribute right after
the declarator-id appertains to the member ([dcl.array]/1, [dcl.meaning]/1). MSL's ``[[id(n)]]`` names a member's
argument buffer index, so ``device float* params [[id(0)]][4];`` is valid, while ``device float* params[4]
[[id(0)]];`` is rejected with "'id' attribute cannot be applied to types".
"""

from __future__ import annotations

import re

__all__ = ["attach_member_attributes"]


def attach_member_attributes(declaration: str, name: str, attributes: str) -> str:
    """``declaration`` with ``attributes`` after the declarator ``name`` when array extents follow it."""
    if not attributes or not name:
        return f"{declaration}{attributes}"
    matches = list(re.finditer(rf"(?<![\w.]){re.escape(name)}\b(?=\s*\[)", declaration))
    if not matches:
        return f"{declaration}{attributes}"
    end = matches[-1].end()
    return f"{declaration[:end]}{attributes}{declaration[end:]}"
