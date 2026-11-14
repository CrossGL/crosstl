"""Discharge of the ``fp_ieee`` rule family, which a target meets for the whole program at once.

A source that defines ``fp_ieee`` (``rule_table.SOURCE_RULES``) evaluates floating-point operations without a
fast-math compilation mode. The family has no per-operation intrinsics: :func:`discharge_program_families` looks the
target's entry up once per program. A ``polyfill`` entry marks the program for its generator (Metal: the
``metal.safe_math`` annotation read by ``codegen/metal_safe_math.py``), a ``native`` entry records the compile
configuration it relies on in the rule report, and a ``refuse`` entry raises :class:`RuleInexpressibleError`.
"""

from __future__ import annotations

from .lower import RuleInexpressibleError
from .rule_table import FP_IEEE, target_entry

__all__ = ["PROGRAM_ANNOTATIONS", "discharge_program_families"]

# Program annotation each target's polyfill of fp_ieee sets for its generator.
PROGRAM_ANNOTATIONS = {"metal": "metal.safe_math"}


def discharge_program_families(program, families, report) -> None:
    """Discharge the program-wide families among ``families`` for ``report.target``."""
    if FP_IEEE not in families:
        return
    target = report.target
    entry = target_entry(FP_IEEE, target)
    if entry.kind == "refuse":
        raise RuleInexpressibleError(
            f"{report.source or 'the source'} defines the {FP_IEEE} rule, and {target} cannot express it: "
            f"{entry.reason}",
            target=target,
        )
    if entry.configuration:
        report.configurations.add(entry.configuration)
    if entry.kind == "polyfill":
        annotation = PROGRAM_ANNOTATIONS.get(target)
        annotations = getattr(program, "annotations", None)
        if annotation is None or not isinstance(annotations, dict):
            raise RuleInexpressibleError(
                f"{report.source or 'the source'} defines the {FP_IEEE} rule, and the {target} generator has "
                "no way to keep it",
                target=target,
            )
        annotations[annotation] = True
    report.count(FP_IEEE)
    report.discharged[entry.kind] = report.discharged.get(entry.kind, 0) + 1
