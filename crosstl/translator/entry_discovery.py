"""Backend-neutral source entry-point discovery contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

ENTRY_DISCOVERY_AVAILABLE = "available"
ENTRY_DISCOVERY_FAILED = "failed"
ENTRY_DISCOVERY_UNAVAILABLE = "unavailable"
ENTRY_DISCOVERY_STATUSES = frozenset(
    (
        ENTRY_DISCOVERY_AVAILABLE,
        ENTRY_DISCOVERY_FAILED,
        ENTRY_DISCOVERY_UNAVAILABLE,
    )
)


def source_position(text: str, offset: int) -> tuple[int, int]:
    """Return a one-based line and column for an offset in source text."""
    bounded_offset = max(0, min(offset, len(text)))
    line = text.count("\n", 0, bounded_offset) + 1
    previous_newline = text.rfind("\n", 0, bounded_offset)
    column = (
        bounded_offset + 1
        if previous_newline < 0
        else bounded_offset - previous_newline
    )
    return line, column


@dataclass(frozen=True)
class SourceEntryLocation:
    """Location in the source representation inspected by a frontend."""

    line: int
    column: int
    offset: int
    length: int
    coordinate_space: str = "preprocessed-source"

    def to_json(self) -> dict[str, Any]:
        return {
            "line": self.line,
            "column": self.column,
            "offset": self.offset,
            "length": self.length,
            "coordinateSpace": self.coordinate_space,
        }


@dataclass(frozen=True)
class SourceEntryProvenance:
    """How a source declaration exposes a host-visible entry point."""

    kind: str
    declared_name: str
    template_arguments: Sequence[str] = ()

    def to_json(self) -> dict[str, Any]:
        payload = {
            "kind": self.kind,
            "declaredName": self.declared_name,
        }
        if self.template_arguments:
            payload["templateArguments"] = list(self.template_arguments)
        return payload


@dataclass(frozen=True)
class SourceEntryPoint:
    """One host-visible entry point discovered in a source translation unit."""

    name: str
    stage: str
    location: SourceEntryLocation
    provenance: SourceEntryProvenance

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "location": self.location.to_json(),
            "provenance": self.provenance.to_json(),
        }


@dataclass(frozen=True)
class SourceEntryDiscoveryDiagnostic:
    """Frontend diagnostic produced while enumerating source entries."""

    severity: str
    code: str
    message: str
    location: SourceEntryLocation
    missing_capabilities: Sequence[str] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "location": self.location.to_json(),
        }
        if self.missing_capabilities:
            payload["missingCapabilities"] = list(self.missing_capabilities)
        if self.details:
            payload["details"] = dict(self.details)
        return payload


@dataclass(frozen=True)
class SourceEntryDiscovery:
    """Entry discovery result returned by a registered source frontend."""

    source_backend: str
    source_path: str
    status: str
    entries: Sequence[SourceEntryPoint] = ()
    diagnostics: Sequence[SourceEntryDiscoveryDiagnostic] = ()

    def __post_init__(self) -> None:
        if self.status not in ENTRY_DISCOVERY_STATUSES:
            raise ValueError(
                f"Unsupported source entry discovery status: {self.status}"
            )
        if self.status == ENTRY_DISCOVERY_UNAVAILABLE and (
            self.entries or self.diagnostics
        ):
            raise ValueError(
                "Unavailable source entry discovery cannot contain entries or diagnostics"
            )

    @classmethod
    def unavailable(
        cls, *, source_backend: str, source_path: str
    ) -> SourceEntryDiscovery:
        return cls(
            source_backend=source_backend,
            source_path=source_path,
            status=ENTRY_DISCOVERY_UNAVAILABLE,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "sourceBackend": self.source_backend,
            "sourcePath": self.source_path,
            "entryCount": len(self.entries),
            "entries": [entry.to_json() for entry in self.entries],
            "diagnosticCount": len(self.diagnostics),
            "diagnostics": [diagnostic.to_json() for diagnostic in self.diagnostics],
        }
