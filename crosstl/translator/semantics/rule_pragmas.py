"""The rule families of a CrossGL program, declared with ``#pragma cgl_rules(...)``.

A translation from Metal, GLSL, HLSL or SPIR-V carries the rules that its source language defines
(``rule_table.SOURCE_RULES``). Written out as CrossGL, the program used to lose them: ``.cgl`` sources carry no rule,
so translating the saved CrossGL gave a different program than translating the source directly. A Metal
``int(x)`` converted NaN to 0 one way and was undefined the other.

The CrossGL output of an imported program therefore starts with ``#pragma cgl_rules(<families>)`` naming the source's
families (:func:`rule_pragma`). :func:`take_declared_families` reads the pragma from a parsed program and removes
it, so that no target prints it, and rule carrying applies the declared families as the source's own. A family the
rule table does not define is refused.
"""

from __future__ import annotations

import re

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .rule_table import FAMILIES, source_families

__all__ = ["UnknownRuleFamilyError", "rule_pragma", "take_declared_families"]

PRAGMA = "cgl_rules"
_PRAGMA_RE = re.compile(rf"^{PRAGMA}\s*\((?P<families>[^()]*)\)\s*$")


class UnknownRuleFamilyError(TranslationDiagnostic):
    """``#pragma cgl_rules`` names a family the rule table does not define, or is malformed."""

    def __init__(self, content, target=None):
        super().__init__(
            f"'#pragma {content}' does not name rule families of {', '.join(FAMILIES)}",
            target=target,
            code=f"{CODE_PREFIX}unknown-rule-family",
        )


def rule_pragma(source: str) -> str | None:
    """``#pragma cgl_rules(...)`` for the rule families ``source`` defines, or ``None`` when it defines none."""
    families = sorted(source_families(source))
    if not families:
        return None
    return f"#pragma {PRAGMA}({', '.join(families)})"


def take_declared_families(program, target=None) -> frozenset:
    """The families the program's ``#pragma cgl_rules`` lines declare; the lines are removed from the program."""
    preprocessors = getattr(program, "preprocessors", None) or []
    declared, kept = set(), []
    for node in preprocessors:
        content = str(getattr(node, "content", "") or "").strip()
        if getattr(node, "directive", None) != "pragma" or not content.startswith(
            PRAGMA
        ):
            kept.append(node)
            continue
        match = _PRAGMA_RE.match(content)
        if match is None:
            raise UnknownRuleFamilyError(content, target)
        names = {name.strip() for name in match.group("families").split(",")}
        names.discard("")
        if not names <= set(FAMILIES):
            raise UnknownRuleFamilyError(content, target)
        declared |= names
    if len(kept) != len(preprocessors):
        program.preprocessors = kept
    return frozenset(declared)
