"""One artifact per entry for targets whose compilation unit holds one entry point per stage.

A GLSL shader's entry point is ``main`` (GLSL 4.60 and GLSL ES 3.20 section 6.1), so the GLSL generators refuse a
program with two entries of one stage (:mod:`crosstl.translator.codegen.glsl_stage_entries`). A project unit with such
entries and no configured entry selection is translated for those targets one entry at a time, exactly as if each
discovered entry had been selected: each entry gets its own artifact (``<target>/<unit>/<entry>``) with its own
bindings, instead of one refused artifact for the whole file.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from crosstl.translator.codegen.registry import BACKEND_REGISTRY
from crosstl.translator.entry_discovery import ENTRY_DISCOVERY_AVAILABLE

__all__ = ["holds_one_entry_per_stage", "host_sizes_unit", "per_entry_split"]

_HOST_SIZE_RULES = (
    "workgroup_size_rules",
    "entry_workgroup_size_rules",
    "subgroup_width_rules",
)


def holds_one_entry_per_stage(target: str) -> bool:
    """Whether ``target``'s generator refuses a program with two entries of one stage."""
    spec = BACKEND_REGISTRY.get(target)
    return spec is not None and bool(
        getattr(spec.codegen_class, "one_entry_per_stage", False)
    )


def host_sizes_unit(config: Any, unit: Any) -> bool:
    """Whether the project gives ``unit`` a host workgroup size or subgroup width.

    Those sizes are attributed to entries by the dispatch planner, which refuses a size it cannot attribute
    (``workgroup-size-entry-ambiguous``); splitting the unit here would attribute it to every entry instead.
    """
    if getattr(config, "workgroup_size", None) is not None:
        return True
    if getattr(config, "variant_workgroup_sizes", None):
        return True
    path = str(getattr(unit, "relative_path", ""))
    for field in _HOST_SIZE_RULES:
        for pattern in getattr(config, field, None) or {}:
            if str(pattern) == path or any(mark in str(pattern) for mark in "*?["):
                return True
    return False


def per_entry_split(unit: Any, target: str) -> tuple[str, ...]:
    """The entries ``unit`` is translated by, one per artifact, for ``target``; empty for a whole-file artifact.

    Only a unit whose discovered entries repeat a stage is split, and then every entry is, since each artifact can
    hold just the one entry its ``main`` stands for. A unit whose entries could not be discovered is translated
    whole, and the generator's refusal names its entries.
    """
    if not holds_one_entry_per_stage(target):
        return ()
    discovery = getattr(unit, "entry_discovery", None)
    if discovery is None or discovery.status != ENTRY_DISCOVERY_AVAILABLE:
        return ()
    entries = tuple(discovery.entries)
    if max(Counter(entry.stage for entry in entries).values(), default=0) < 2:
        return ()
    return tuple(dict.fromkeys(entry.name for entry in entries))
