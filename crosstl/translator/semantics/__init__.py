"""Rule carrying: carry the source language's operation rules through CrossGL and discharge them per target.

``apply_rule_carrying(program, source, target)`` annotates the CrossGL AST with the rules ``source`` defines
(:mod:`.annotate`) and lowers them for ``target`` (:mod:`.lower`) as the rule table says (:mod:`.rule_table`). It
runs on every translation between parsing the CrossGL AST and target code generation. See
docs/source/adr/0001-rule-carrying.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .annotate import Annotator
from .lower import Lowerer, RuleInexpressibleError
from .narrow import NarrowIntegerUnsupportedError, legalize_narrow_integers
from .rule_table import FAMILIES, SOURCE_RULES, TARGET_ENTRIES

__all__ = [
    "FAMILIES",
    "NarrowIntegerUnsupportedError",
    "RuleInexpressibleError",
    "RuleReport",
    "SOURCE_RULES",
    "TARGET_ENTRIES",
    "apply_rule_carrying",
]


@dataclass
class RuleReport:
    """What the passes did: operations annotated per family, how they were discharged, and what stayed unknown."""

    source: str
    target: str
    annotated: dict = field(default_factory=dict)
    discharged: dict = field(default_factory=dict)
    unresolved: int = 0
    configurations: set = field(default_factory=set)

    def count(self, family):
        self.annotated[family] = self.annotated.get(family, 0) + 1

    def to_json(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "annotated": dict(sorted(self.annotated.items())),
            "discharged": dict(sorted(self.discharged.items())),
            "unresolved": self.unresolved,
            "configurations": sorted(self.configurations),
        }


def apply_rule_carrying(program, source, target) -> RuleReport:
    """Annotate ``program`` (a CrossGL ``ShaderNode``) with ``source``'s rules and lower them for ``target``.

    Narrow integers are legalized first for targets without narrow registers (:mod:`.narrow`). The AST is
    rewritten in place. Raises :class:`RuleInexpressibleError` when the target cannot express a rule, and
    :class:`NarrowIntegerUnsupportedError` for narrow-integer storage the target cannot lay out.
    The report's JSON form is also stored in ``program.annotations["rule_carrying"]``.
    """
    report = RuleReport(source=str(source or ""), target=str(target or ""))
    legalize_narrow_integers(program, report.source, report.target)
    Annotator(program, report.source, report).run()
    Lowerer(program, report.target, report).run()
    annotations = getattr(program, "annotations", None)
    if isinstance(annotations, dict):
        annotations["rule_carrying"] = report.to_json()
    return report
