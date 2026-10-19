"""Typed translation diagnostics: a stable code, a source span, the target and the reason.

Every construct CrossTL cannot carry to a target is refused with a :class:`TranslationDiagnostic`. The code is a
stable, documented string (``project.translate.<kind>``) that tests and reports key on; the message is for people
and may change. The class also exposes ``project_diagnostic_code``, the attribute the project pipeline and the
corpus runner already read from older typed errors, so a refusal raised here is classified as typed everywhere.
Non-fatal findings are collected as :class:`Diagnostic` records with a severity instead of being raised.
"""

from __future__ import annotations

from dataclasses import dataclass

CODE_PREFIX = "project.translate."


@dataclass(frozen=True)
class SourceSpan:
    """A 1-based position in a source file; ``file`` is ``None`` for an anonymous in-memory source."""

    file: str | None
    line: int
    column: int

    def __str__(self) -> str:
        return f"{self.file or '<source>'}:{self.line}:{self.column}"

    def to_json(self) -> dict:
        return {"file": self.file, "line": self.line, "column": self.column}


@dataclass(frozen=True)
class Diagnostic:
    """A non-fatal finding, for example a macro redefinition the target compiler only warns about."""

    severity: str
    code: str
    message: str
    span: SourceSpan | None = None

    def __str__(self) -> str:
        where = f"{self.span}: " if self.span else ""
        return f"{where}{self.severity}: {self.message} [{self.code}]"

    def to_json(self) -> dict:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "span": self.span.to_json() if self.span else None,
        }


class TranslationDiagnostic(ValueError):
    """A typed refusal. Subclasses set ``code``; ``reason`` says why, ``span`` where, ``target`` for which output."""

    code = CODE_PREFIX + "unsupported"

    def __init__(
        self,
        reason: str,
        *,
        span: SourceSpan | None = None,
        target: str | None = None,
        code: str | None = None,
    ) -> None:
        if code is not None:
            if not code.startswith(CODE_PREFIX):
                raise ValueError(
                    f"Diagnostic codes start with {CODE_PREFIX!r}: {code!r}"
                )
            self.code = code
        self.reason = reason
        self.span = span
        self.target = target
        where = f"{span}: " if span else ""
        for_target = f" (target {target})" if target else ""
        super().__init__(f"{where}{reason}{for_target} [{self.code}]")

    @property
    def project_diagnostic_code(self) -> str:
        return self.code

    def to_json(self) -> dict:
        return {
            "code": self.code,
            "reason": self.reason,
            "span": self.span.to_json() if self.span else None,
            "target": self.target,
        }
