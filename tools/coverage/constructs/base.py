"""The record each language-level feature is listed as."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Construct:
    category: str
    name: str
    anchors: tuple[str, ...]
    stage: str = "compute"


def c(category: str, name: str, *anchors: str, stage: str = "compute") -> Construct:
    return Construct(category, name, anchors, stage)
