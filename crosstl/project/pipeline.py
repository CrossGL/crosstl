"""Project-scale shader and GPU source porting pipeline for CrossTL."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Optional, Sequence, Tuple

from crosstl._crosstl import translate
from crosstl.translator.codegen import (
    backend_names,
    get_backend_extension,
    normalize_backend_name,
)
from crosstl.translator.plugin_loader import discover_backend_plugins
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources

REPORT_KIND = "crosstl-project-portability-report"
REPORT_INSPECTION_KIND = "crosstl-project-report-inspection"
REPORT_SCHEMA_VERSION = 1
REPORT_GENERATOR_PIPELINE = "project-porting"
REPORT_ARTIFACT_PROVENANCE_PIPELINES = ("single-file-translate",)
REPORT_ARTIFACT_PROVENANCE_INTERMEDIATES = ("crossgl",)
REPORT_MIGRATION_SCOPE = "shader-kernel-translation"
REPORT_MIGRATION_NON_GOALS = (
    "automatic runtime API migration",
    "application build-system rewrites",
    "backend framework integration",
)
REPORT_MIGRATION_ACTION_KINDS = ("manual-runtime-integration",)
REPORT_PACKAGE_NAME = "crosstl"
UNKNOWN_PACKAGE_VERSION = "unknown"
EXTERNAL_CORPUS_SCHEMA_VERSION = 1
DEFAULT_CONFIG_NAME = "crosstl.toml"
DEFAULT_OUTPUT_DIR = "crosstl-out"
OUTPUT_DIR_OUTSIDE_PROJECT_CODE = "project.config.output-dir-outside-project"
INTERNAL_EXCLUDE_PATTERNS = (DEFAULT_CONFIG_NAME,)
DEFAULT_EXCLUDE_PATTERNS = (
    ".git/**",
    ".hg/**",
    ".svn/**",
    ".venv/**",
    "build/**",
    "dist/**",
    "node_modules/**",
    "worktrees/**",
    f"{DEFAULT_OUTPUT_DIR}/**",
)

TOOLCHAIN_BY_BACKEND = {
    "cuda": ("nvcc",),
    "directx": ("dxc",),
    "hip": ("hipcc",),
    "metal": ("xcrun",),
    "mojo": ("mojo",),
    "opengl": ("glslangValidator",),
    "rust": ("rustc",),
    "slang": ("slangc",),
    "vulkan": ("spirv-val", "spirv-as"),
}

CROSSL_TARGETS = {"cgl", "crossgl"}
SHA256_HEX_LENGTH = 64
LOWERCASE_HEX_DIGITS = frozenset("0123456789abcdef")
VALIDATION_ARTIFACT_STATUSES = frozenset(("ok", "failed"))
SOURCE_HASH_VALIDATION_STATUSES = frozenset(
    ("ok", "missing", "mismatch", "not-recorded", "outside-project")
)
GENERATED_HASH_VALIDATION_STATUSES = frozenset(
    ("ok", "missing", "mismatch", "not-applicable", "not-recorded", "outside-project")
)
VARIANT_OUTPUT_SAFE_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ImportError:  # pragma: no cover - exercised only on Python < 3.11
        try:
            import tomli as tomllib
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Reading crosstl.toml on Python < 3.11 requires tomli"
            ) from exc
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _as_str_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        result = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError(f"{field_name} entries must be strings")
            result.append(item)
        return result
    raise ValueError(f"{field_name} must be a string or list of strings")


def _as_optional_str(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ValueError(f"{field_name} must be a string")


def _as_optional_non_empty_str(value: Any, *, field_name: str) -> str | None:
    result = _as_optional_str(value, field_name=field_name)
    if result is not None and not result.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return result


def _as_str_mapping(value: Any, *, field_name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a table")

    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ValueError(f"{field_name} entries must map strings to strings")
        result[key] = item
    return result


def _variant_defines(variants: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, value in variants.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "crosstl.toml [project.variants] keys must be non-empty strings"
            )
        if not isinstance(value, Mapping):
            raise ValueError(f"crosstl.toml [project.variants.{name}] must be a table")
        result[name] = _as_str_mapping(
            value, field_name=f"crosstl.toml [project.variants.{name}]"
        )
    return result


def _normalized_targets(targets: Sequence[str]) -> list[str]:
    discover_backend_plugins()
    normalized_targets = []
    for target in targets:
        normalized = normalize_backend_name(target) or target.strip().lower()
        if normalized not in normalized_targets:
            normalized_targets.append(normalized)
    return normalized_targets


def _supported_target_names() -> list[str]:
    discover_backend_plugins()
    return sorted(set(backend_names()) | CROSSL_TARGETS)


def _relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _path_matches(path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _is_repository_relative_glob(pattern: str) -> bool:
    normalized = pattern.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part and part != "."]
    return (
        not Path(pattern).is_absolute()
        and not PureWindowsPath(pattern).is_absolute()
        and not PureWindowsPath(pattern).drive
        and ".." not in parts
    )


def _is_repository_relative_report_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part and part != "."]
    return (
        not Path(path).is_absolute()
        and not PureWindowsPath(path).is_absolute()
        and not PureWindowsPath(path).drive
        and ".." not in parts
    )


def _repository_relative_globs(patterns: Sequence[str]) -> list[str]:
    return [pattern for pattern in patterns if _is_repository_relative_glob(pattern)]


def _internal_exclude_patterns(config: ProjectConfig) -> tuple[str, ...]:
    patterns = list(INTERNAL_EXCLUDE_PATTERNS)
    output_path = config.output_path
    if _is_relative_to(output_path, config.root):
        output_relative = _relpath(output_path, config.root)
        if output_relative and output_relative != ".":
            patterns.extend((output_relative, f"{output_relative}/**"))
    return tuple(patterns)


def _source_hash(path: Path) -> dict[str, str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"algorithm": "sha256", "value": digest}


def _hash_matches_report(
    actual: Mapping[str, str], expected: Mapping[str, Any]
) -> bool:
    return (actual["algorithm"], actual["value"]) == (
        expected.get("algorithm"),
        expected.get("value"),
    )


def _resolve_report_path(config: ProjectConfig, path_value: Any) -> Path:
    path = Path(str(path_value))
    if not path.is_absolute():
        path = config.root / path
    return path.resolve()


def _package_version() -> str:
    try:
        return importlib_metadata.version(REPORT_PACKAGE_NAME)
    except importlib_metadata.PackageNotFoundError:
        return UNKNOWN_PACKAGE_VERSION


def _variant_output_segment(variant: str) -> str:
    safe = "".join(
        character if character in VARIANT_OUTPUT_SAFE_CHARS else "_"
        for character in variant.strip()
    ).strip("._-")
    if not safe:
        safe = "variant"
    if safe == variant and safe not in {".", ".."}:
        return safe
    digest = hashlib.sha256(variant.encode("utf-8")).hexdigest()[:8]
    return f"{safe[:48]}-{digest}"


def _file_span(path: Path, report_path: str) -> SourceLocation:
    text = path.read_text(encoding="utf-8", errors="replace")
    line = 1
    column = 1
    for character in text:
        if character == "\n":
            line += 1
            column = 1
        else:
            column += 1
    return SourceLocation(
        file=report_path,
        length=len(text),
        end_line=line,
        end_column=column,
        end_offset=len(text),
    )


def _artifact_report_path(path: Path, config: ProjectConfig) -> str:
    return (
        _relpath(path, config.root) if _is_relative_to(path, config.root) else str(path)
    )


def _diagnostic_counts(diagnostics: Sequence[ProjectDiagnostic]) -> dict[str, int]:
    counts = {"note": 0, "warning": 0, "error": 0}
    for diagnostic in diagnostics:
        counts[diagnostic.severity] = counts.get(diagnostic.severity, 0) + 1
    return counts


def _diagnostic_counts_by_code(
    diagnostics: Sequence[ProjectDiagnostic],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for diagnostic in diagnostics:
        counts[diagnostic.code] = counts.get(diagnostic.code, 0) + 1
    return dict(sorted(counts.items()))


def _missing_capability_counts(
    diagnostics: Sequence[ProjectDiagnostic],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for diagnostic in diagnostics:
        for capability in diagnostic.missing_capabilities:
            counts[capability] = counts.get(capability, 0) + 1
    return dict(sorted(counts.items()))


def _diagnostic_payload_counts(
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts = {"note": 0, "warning": 0, "error": 0}
    for diagnostic in diagnostics:
        severity = str(diagnostic.get("severity", "note"))
        counts[severity] = counts.get(severity, 0) + 1
    return counts


def _unit_counts_by_source_backend(
    units: Sequence[ProjectTranslationUnit],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for unit in units:
        counts[unit.source_backend] = counts.get(unit.source_backend, 0) + 1
    return dict(sorted(counts.items()))


def _artifact_counts_by_target(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for artifact in artifacts:
        target = str(artifact.get("target", "unknown"))
        row = counts.setdefault(
            target,
            {
                "artifactCount": 0,
                "translatedCount": 0,
                "failedCount": 0,
            },
        )
        row["artifactCount"] += 1
        if artifact.get("status") == "translated":
            row["translatedCount"] += 1
        elif artifact.get("status") == "failed":
            row["failedCount"] += 1
    return {target: counts[target] for target in sorted(counts)}


def _source_map_counts(artifacts: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    source_map_count = sum(1 for artifact in artifacts if artifact.get("sourceMap"))
    return {
        "sourceMapCount": source_map_count,
        "fineGrainedSourceMapCount": 0,
    }


def _external_corpus_empty_summary() -> dict[str, Any]:
    return {
        "entryCount": 0,
        "presentCount": 0,
        "missingCount": 0,
        "discoveredUnitCount": 0,
        "undiscoveredPresentCount": 0,
        "entriesBySourceBackend": {},
        "entriesByTarget": {},
        "artifactsByTarget": {},
    }


def _external_corpus_manifest_path(config: ProjectConfig) -> Path | None:
    manifest = config.external_corpus_manifest
    if not manifest:
        return None
    path = Path(manifest)
    if not path.is_absolute():
        path = config.root / path
    return path.resolve()


def _load_external_corpus_manifest(
    config: ProjectConfig,
) -> tuple[Mapping[str, Any] | None, str | None]:
    manifest_path = _external_corpus_manifest_path(config)
    if manifest_path is None:
        return None, None
    if not _is_relative_to(manifest_path, config.root):
        return None, "outside-project"
    if not manifest_path.exists():
        return None, "missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "invalid"
    if not isinstance(manifest, Mapping):
        return None, "invalid"
    if manifest.get("schemaVersion") != EXTERNAL_CORPUS_SCHEMA_VERSION:
        return None, "invalid"
    if not isinstance(manifest.get("entries"), list):
        return None, "invalid"
    return manifest, None


def _external_corpus_manifest_entry_reasons(entry: Any) -> list[str]:
    if not isinstance(entry, Mapping):
        return ["entry must be an object"]

    reasons = []
    path = entry.get("path")
    if not _is_non_empty_string(path):
        reasons.append("path must be a non-empty repository-relative string")
    else:
        normalized_path = path.replace("\\", "/")
        if not _is_repository_relative_report_path(normalized_path):
            reasons.append("path must be repository-relative")

    source_backend = entry.get("sourceBackend")
    if source_backend is not None and not isinstance(source_backend, str):
        reasons.append("sourceBackend must be a string")

    targets = entry.get("targets")
    if targets is not None:
        valid_targets = isinstance(targets, str) or (
            isinstance(targets, Sequence)
            and not isinstance(targets, (bytes, bytearray))
            and all(isinstance(target, str) for target in targets)
        )
        if not valid_targets:
            reasons.append("targets must be a string or list of strings")

    return reasons


def _valid_external_corpus_manifest_entries(
    manifest: Mapping[str, Any],
) -> list[tuple[int, Mapping[str, Any]]]:
    entries = []
    for index, entry in enumerate(manifest.get("entries", [])):
        if _external_corpus_manifest_entry_reasons(entry):
            continue
        entries.append((index, entry))
    return entries


def _external_corpus_manifest_reference(config: ProjectConfig) -> str:
    return str(config.external_corpus_manifest or "")


def _manifest_entry_targets(
    entry: Mapping[str, Any], fallback_targets: Sequence[str]
) -> list[str]:
    targets = entry.get("targets")
    if isinstance(targets, str):
        return _normalized_targets([targets])
    if isinstance(targets, Sequence) and not isinstance(targets, (bytes, bytearray)):
        valid_targets = [target for target in targets if isinstance(target, str)]
        if valid_targets:
            return _normalized_targets(valid_targets)
    return list(fallback_targets)


def _external_corpus_entries_by_source_backend(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        source_backend = str(entry.get("sourceBackend", "unknown"))
        counts[source_backend] = counts.get(source_backend, 0) + 1
    return dict(sorted(counts.items()))


def _external_corpus_entries_by_target(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        for target in entry.get("targets", []):
            target_name = str(target)
            counts[target_name] = counts.get(target_name, 0) + 1
    return dict(sorted(counts.items()))


def _external_corpus_artifacts(
    entries: Sequence[Mapping[str, Any]], artifacts: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    entry_targets: dict[str, set[str]] = {}
    for entry in entries:
        path = entry.get("path")
        if not _is_non_empty_string(path):
            continue
        targets = {
            target for target in entry.get("targets", []) if isinstance(target, str)
        }
        entry_targets.setdefault(path, set()).update(targets)

    corpus_artifacts = []
    for artifact in artifacts:
        source = artifact.get("source")
        target = artifact.get("target")
        if not _is_non_empty_string(source) or source not in entry_targets:
            continue
        targets = entry_targets[source]
        if targets and target not in targets:
            continue
        corpus_artifacts.append(artifact)
    return corpus_artifacts


def _external_corpus_entry_artifacts(
    entry: Mapping[str, Any], artifacts: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    path = entry.get("path")
    if not _is_non_empty_string(path):
        return []
    targets = {target for target in entry.get("targets", []) if isinstance(target, str)}
    return [
        artifact
        for artifact in artifacts
        if artifact.get("source") == path
        and (not targets or artifact.get("target") in targets)
    ]


def _external_corpus_report(
    config: ProjectConfig,
    units: Sequence[ProjectTranslationUnit],
    artifacts: Sequence[Mapping[str, Any]],
    targets: Sequence[str],
) -> dict[str, Any] | None:
    if not config.external_corpus_manifest:
        return None

    manifest, status = _load_external_corpus_manifest(config)
    if manifest is None:
        return {
            "schemaVersion": EXTERNAL_CORPUS_SCHEMA_VERSION,
            "manifest": _external_corpus_manifest_reference(config),
            "status": status or "invalid",
            "entries": [],
            "summary": _external_corpus_empty_summary(),
        }

    units_by_path = {unit.relative_path: unit for unit in units}
    artifacts_by_source: dict[str, list[Mapping[str, Any]]] = {}
    for artifact in artifacts:
        source = str(artifact.get("source", ""))
        if source:
            artifacts_by_source.setdefault(source, []).append(artifact)

    entries = []
    for index, raw_entry in _valid_external_corpus_manifest_entries(manifest):
        path = str(raw_entry.get("path", "")).replace("\\", "/")
        entry_targets = _manifest_entry_targets(raw_entry, targets)
        source_backend = str(raw_entry.get("sourceBackend", "unknown"))
        entry_artifacts = [
            artifact
            for artifact in artifacts_by_source.get(path, [])
            if not entry_targets or artifact.get("target") in entry_targets
        ]
        entry_path = config.root / path
        present = (
            bool(path)
            and _is_repository_relative_report_path(path)
            and entry_path.exists()
        )
        discovered = path in units_by_path
        entry_payload = {
            "id": str(raw_entry.get("id") or path or f"entry-{index + 1}"),
            "path": path,
            "sourceBackend": source_backend,
            "targets": entry_targets,
            "present": present,
            "discovered": discovered,
            "artifactCount": len(entry_artifacts),
            "translatedCount": sum(
                1
                for artifact in entry_artifacts
                if artifact.get("status") == "translated"
            ),
            "failedCount": sum(
                1 for artifact in entry_artifacts if artifact.get("status") == "failed"
            ),
        }
        for field_name in ("repository", "commit", "sourceUrl"):
            value = raw_entry.get(field_name)
            if isinstance(value, str) and value:
                entry_payload[field_name] = value
        entries.append(entry_payload)

    summary = {
        "entryCount": len(entries),
        "presentCount": sum(1 for entry in entries if entry["present"]),
        "missingCount": sum(1 for entry in entries if not entry["present"]),
        "discoveredUnitCount": sum(1 for entry in entries if entry["discovered"]),
        "undiscoveredPresentCount": sum(
            1 for entry in entries if entry["present"] and not entry["discovered"]
        ),
        "entriesBySourceBackend": _external_corpus_entries_by_source_backend(entries),
        "entriesByTarget": _external_corpus_entries_by_target(entries),
        "artifactsByTarget": _artifact_counts_by_target(
            _external_corpus_artifacts(entries, artifacts)
        ),
    }
    payload = {
        "schemaVersion": EXTERNAL_CORPUS_SCHEMA_VERSION,
        "manifest": _external_corpus_manifest_reference(config),
        "status": "ok",
        "entries": entries,
        "summary": summary,
    }
    for field_name in ("name", "description"):
        value = manifest.get(field_name)
        if isinstance(value, str) and value:
            payload[field_name] = value
    return payload


def _resolved_include_dir(config: ProjectConfig, include_dir: str) -> Path:
    path = Path(include_dir)
    if not path.is_absolute():
        path = config.root / path
    return path.resolve()


def _resolved_include_dirs(config: ProjectConfig) -> list[str]:
    include_dirs = []
    for include_dir in config.include_dirs:
        include_dirs.append(str(_resolved_include_dir(config, include_dir)))
    return include_dirs


def _frontend_include_dirs(config: ProjectConfig) -> list[str]:
    include_dirs = []
    for include_dir in config.include_dirs:
        absolute_dir = _resolved_include_dir(config, include_dir)
        if (
            _is_relative_to(absolute_dir, config.root)
            and absolute_dir.exists()
            and absolute_dir.is_dir()
        ):
            include_dirs.append(str(absolute_dir))
    return include_dirs


def _resolved_source_root(config: ProjectConfig, source_root: str) -> Path:
    path = Path(source_root)
    if not path.is_absolute():
        path = config.root / path
    return path.resolve()


def _config_location(config: ProjectConfig) -> SourceLocation:
    if config.config_path:
        file = (
            _relpath(config.config_path, config.root)
            if _is_relative_to(config.config_path, config.root)
            else str(config.config_path)
        )
        return SourceLocation(file=file)
    return SourceLocation(file=".")


@dataclass(frozen=True)
class ProjectConfig:
    """Configuration for scanning and translating a shader repository."""

    root: Path
    config_path: Path | None = None
    source_roots: Sequence[str] = (".",)
    include_patterns: Sequence[str] = ()
    exclude_patterns: Sequence[str] = DEFAULT_EXCLUDE_PATTERNS
    targets: Sequence[str] = ()
    output_dir: str = DEFAULT_OUTPUT_DIR
    source_overrides: Mapping[str, str] = field(default_factory=dict)
    include_dirs: Sequence[str] = ()
    defines: Mapping[str, str] = field(default_factory=dict)
    variants: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    external_corpus_manifest: str | None = None

    def normalized_targets(self) -> list[str]:
        return _normalized_targets(self.targets)

    @property
    def output_path(self) -> Path:
        output = Path(self.output_dir)
        if output.is_absolute():
            return output
        return self.root / output


@dataclass(frozen=True)
class SourceLocation:
    """Compiler-compatible source span used by project diagnostics."""

    file: str
    line: int = 1
    column: int = 1
    offset: int = 0
    length: int = 0
    end_line: int = 1
    end_column: int = 1
    end_offset: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "column": self.column,
            "offset": self.offset,
            "length": self.length,
            "endLine": self.end_line,
            "endColumn": self.end_column,
            "endOffset": self.end_offset,
        }


@dataclass(frozen=True)
class ProjectDiagnostic:
    """Structured project diagnostic compatible with compiler diagnostics."""

    severity: str
    code: str
    message: str
    location: SourceLocation
    target: str | None = None
    missing_capabilities: Sequence[str] = ()

    def to_json(self) -> dict[str, Any]:
        payload = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "location": self.location.to_json(),
        }
        if self.target:
            payload["target"] = self.target
        if self.missing_capabilities:
            payload["missingCapabilities"] = list(self.missing_capabilities)
        return payload


def _configuration_diagnostics(config: ProjectConfig) -> list[ProjectDiagnostic]:
    diagnostics: list[ProjectDiagnostic] = []
    location = _config_location(config)
    if not _is_relative_to(config.output_path, config.root):
        diagnostics.append(
            ProjectDiagnostic(
                severity="error",
                code=OUTPUT_DIR_OUTSIDE_PROJECT_CODE,
                message=(
                    f"Configured output directory '{config.output_dir}' resolves "
                    f"outside the repository: {config.output_path.resolve()}"
                ),
                location=location,
                missing_capabilities=["artifact.manifest"],
            )
        )
    if config.external_corpus_manifest:
        manifest_path = _external_corpus_manifest_path(config)
        if manifest_path is not None and not _is_relative_to(
            manifest_path, config.root
        ):
            diagnostics.append(
                ProjectDiagnostic(
                    severity="error",
                    code="project.config.external-corpus-outside-project",
                    message=(
                        "Configured external corpus manifest resolves outside "
                        f"the repository: {manifest_path}"
                    ),
                    location=location,
                    missing_capabilities=["external.corpus"],
                )
            )
        else:
            manifest, status = _load_external_corpus_manifest(config)
            if status in {"missing", "invalid"}:
                diagnostics.append(
                    ProjectDiagnostic(
                        severity="warning",
                        code=f"project.config.external-corpus-{status}",
                        message=(
                            "Configured external corpus manifest is "
                            f"{status}: {config.external_corpus_manifest}"
                        ),
                        location=location,
                        missing_capabilities=["external.corpus"],
                    )
                )
            elif manifest is not None:
                diagnostics.extend(_external_corpus_entry_diagnostics(config, manifest))
    return diagnostics


def _external_corpus_entry_diagnostics(
    config: ProjectConfig, manifest: Mapping[str, Any]
) -> list[ProjectDiagnostic]:
    diagnostics: list[ProjectDiagnostic] = []
    location = _config_location(config)
    for index, entry in enumerate(manifest.get("entries", [])):
        reasons = _external_corpus_manifest_entry_reasons(entry)
        if not reasons:
            continue
        diagnostics.append(
            ProjectDiagnostic(
                severity="warning",
                code="project.config.external-corpus-entry-invalid",
                message=(
                    f"External corpus manifest entry {index + 1} is invalid "
                    "and will be skipped: " + "; ".join(reasons)
                ),
                location=location,
                missing_capabilities=["external.corpus"],
            )
        )
    return diagnostics


def _source_root_diagnostics(config: ProjectConfig) -> list[ProjectDiagnostic]:
    diagnostics: list[ProjectDiagnostic] = []
    location = _config_location(config)
    for source_root in config.source_roots:
        absolute_root = _resolved_source_root(config, source_root)
        if not _is_relative_to(absolute_root, config.root):
            diagnostics.append(
                ProjectDiagnostic(
                    severity="error",
                    code="project.config.source-root-outside-project",
                    message=(
                        f"Configured source root '{source_root}' resolves outside "
                        f"the repository: {absolute_root}"
                    ),
                    location=location,
                    missing_capabilities=["repo.scan"],
                )
            )
            continue
        if not absolute_root.exists():
            diagnostics.append(
                ProjectDiagnostic(
                    severity="warning",
                    code="project.scan.missing-source-root",
                    message=f"Configured source root does not exist: {source_root}",
                    location=location,
                    missing_capabilities=["repo.scan"],
                )
            )
    return diagnostics


def _include_dir_diagnostics(config: ProjectConfig) -> list[ProjectDiagnostic]:
    diagnostics: list[ProjectDiagnostic] = []
    location = _config_location(config)
    for include_dir in config.include_dirs:
        absolute_dir = _resolved_include_dir(config, include_dir)
        if not _is_relative_to(absolute_dir, config.root):
            diagnostics.append(
                ProjectDiagnostic(
                    severity="warning",
                    code="project.config.include-dir-outside-project",
                    message=(
                        f"Configured include directory '{include_dir}' resolves "
                        f"outside the repository: {absolute_dir}"
                    ),
                    location=location,
                    missing_capabilities=["include.resolution"],
                )
            )
            continue
        if not absolute_dir.exists():
            diagnostics.append(
                ProjectDiagnostic(
                    severity="warning",
                    code="project.config.missing-include-dir",
                    message=(
                        f"Configured include directory does not exist: {include_dir}"
                    ),
                    location=location,
                    missing_capabilities=["include.resolution"],
                )
            )
    return diagnostics


def _scan_pattern_diagnostics(config: ProjectConfig) -> list[ProjectDiagnostic]:
    diagnostics: list[ProjectDiagnostic] = []
    location = _config_location(config)
    for pattern in config.include_patterns:
        if _is_repository_relative_glob(pattern):
            continue
        diagnostics.append(
            ProjectDiagnostic(
                severity="error",
                code="project.config.include-pattern-outside-project",
                message=(
                    f"Configured include pattern '{pattern}' is not "
                    "repository-relative."
                ),
                location=location,
                missing_capabilities=["repo.scan"],
            )
        )
    for pattern in config.source_overrides:
        if _is_repository_relative_glob(pattern):
            continue
        diagnostics.append(
            ProjectDiagnostic(
                severity="error",
                code="project.config.source-override-pattern-outside-project",
                message=(
                    f"Configured source override pattern '{pattern}' is not "
                    "repository-relative."
                ),
                location=location,
                missing_capabilities=["source.override"],
            )
        )
    return diagnostics


def _target_diagnostics(
    config: ProjectConfig, targets: Sequence[str]
) -> list[ProjectDiagnostic]:
    diagnostics: list[ProjectDiagnostic] = []
    supported_targets = _supported_target_names()
    supported = set(supported_targets)
    for target in targets:
        if target in supported:
            continue
        diagnostics.append(
            ProjectDiagnostic(
                severity="error",
                code="project.config.unsupported-target",
                message=(
                    f"Target backend '{target}' is not supported. Supported "
                    f"targets: {', '.join(supported_targets)}"
                ),
                location=_config_location(config),
                target=target,
                missing_capabilities=["target.backend"],
            )
        )
    return diagnostics


@dataclass(frozen=True)
class ProjectTranslationUnit:
    """A discovered source file that can be translated as one unit."""

    path: Path
    relative_path: str
    source_backend: str
    extension: str
    source_override: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload = {
            "id": self.relative_path,
            "path": self.relative_path,
            "sourceBackend": self.source_backend,
            "extension": self.extension,
        }
        if self.source_override:
            payload["sourceOverride"] = self.source_override
        return payload


@dataclass(frozen=True)
class ProjectScan:
    """Result of scanning a repository for shader translation units."""

    config: ProjectConfig
    units: Sequence[ProjectTranslationUnit]
    skipped: Sequence[dict[str, Any]] = ()
    diagnostics: Sequence[ProjectDiagnostic] = ()

    def to_report(
        self, targets: Sequence[str] | None = None
    ) -> ProjectPortabilityReport:
        report_targets = (
            _normalized_targets(targets)
            if targets is not None
            else self.config.normalized_targets()
        )
        diagnostics = list(self.diagnostics)
        diagnostics.extend(_target_diagnostics(self.config, report_targets))
        return ProjectPortabilityReport(
            config=self.config,
            targets=report_targets,
            units=self.units,
            skipped=self.skipped,
            artifacts=[],
            diagnostics=diagnostics,
            validation={"toolchains": [], "artifacts": []},
            migration_actions=_runtime_migration_actions(self.units, report_targets),
        )


@dataclass(frozen=True)
class ProjectPortabilityReport:
    """Machine-readable report emitted by the project porting pipeline."""

    config: ProjectConfig
    targets: Sequence[str]
    units: Sequence[ProjectTranslationUnit]
    skipped: Sequence[dict[str, Any]]
    artifacts: Sequence[dict[str, Any]]
    diagnostics: Sequence[ProjectDiagnostic]
    validation: Mapping[str, Any]
    migration_actions: Sequence[dict[str, Any]]
    generated_at: int = field(default_factory=lambda: int(time.time()))

    def to_json(self) -> dict[str, Any]:
        artifact_count = len(self.artifacts)
        translated_count = sum(
            1 for artifact in self.artifacts if artifact.get("status") == "translated"
        )
        failed_count = sum(
            1 for artifact in self.artifacts if artifact.get("status") == "failed"
        )
        diagnostics = [diagnostic.to_json() for diagnostic in self.diagnostics]
        source_map_counts = _source_map_counts(self.artifacts)
        external_corpus = _external_corpus_report(
            self.config, self.units, self.artifacts, self.targets
        )
        payload = {
            "schemaVersion": REPORT_SCHEMA_VERSION,
            "kind": REPORT_KIND,
            "generatedAt": self.generated_at,
            "generator": {
                "name": "CrossTL",
                "pipeline": REPORT_GENERATOR_PIPELINE,
                "packageVersion": _package_version(),
            },
            "project": {
                "root": str(self.config.root),
                "config": (
                    str(self.config.config_path) if self.config.config_path else None
                ),
                "sourceRoots": list(self.config.source_roots),
                "includePatterns": list(self.config.include_patterns),
                "excludePatterns": list(self.config.exclude_patterns),
                "targets": list(self.targets),
                "outputDir": str(self.config.output_path),
                "sourceOverrides": dict(sorted(self.config.source_overrides.items())),
                "sourceOverrideCount": len(self.config.source_overrides),
                "includeDirs": list(self.config.include_dirs),
                "defines": dict(sorted(self.config.defines.items())),
                "defineCount": len(self.config.defines),
                "variants": {
                    name: dict(sorted(defines.items()))
                    for name, defines in sorted(self.config.variants.items())
                },
                "variantCount": len(self.config.variants),
                "externalCorpusManifest": self.config.external_corpus_manifest,
            },
            "summary": {
                "unitCount": len(self.units),
                "skippedCount": len(self.skipped),
                "targetCount": len(self.targets),
                "artifactCount": artifact_count,
                "translatedCount": translated_count,
                "failedCount": failed_count,
                "diagnosticCounts": _diagnostic_counts(self.diagnostics),
                "diagnosticsByCode": _diagnostic_counts_by_code(self.diagnostics),
                "missingCapabilityCounts": _missing_capability_counts(self.diagnostics),
                "unitsBySourceBackend": _unit_counts_by_source_backend(self.units),
                "artifactsByTarget": _artifact_counts_by_target(self.artifacts),
                **source_map_counts,
            },
            "units": [unit.to_json() for unit in self.units],
            "skipped": list(self.skipped),
            "artifacts": list(self.artifacts),
            "diagnosticCounts": _diagnostic_counts(self.diagnostics),
            "diagnostics": diagnostics,
            "validation": dict(self.validation),
            "migration": {
                "scope": REPORT_MIGRATION_SCOPE,
                "nonGoals": list(REPORT_MIGRATION_NON_GOALS),
                "actions": list(self.migration_actions),
            },
        }
        if external_corpus is not None:
            payload["externalCorpus"] = external_corpus
        return payload

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n")


def load_project_config(
    root: str | os.PathLike[str], config: str | os.PathLike[str] | None = None
) -> ProjectConfig:
    """Load ``crosstl.toml`` if present, otherwise return default scan settings."""
    root_path = Path(root).resolve()
    config_path = Path(config).resolve() if config else root_path / DEFAULT_CONFIG_NAME
    if not config_path.exists():
        return ProjectConfig(root=root_path, config_path=None)

    raw = _load_toml(config_path)
    project = raw.get("project", raw)
    if not isinstance(project, Mapping):
        raise ValueError("crosstl.toml [project] must be a table")

    sources = _as_str_mapping(
        project.get("sources"), field_name="crosstl.toml [project.sources]"
    )
    defines = _as_str_mapping(
        project.get("defines"), field_name="crosstl.toml [project.defines]"
    )
    variants = project.get("variants", {})
    if not isinstance(variants, Mapping):
        raise ValueError("crosstl.toml [project.variants] must be a table")
    output_dir = _as_optional_non_empty_str(
        project.get("output_dir"),
        field_name="crosstl.toml project.output_dir",
    )
    external_corpus_manifest = _as_optional_str(
        project.get("external_corpus_manifest"),
        field_name="crosstl.toml project.external_corpus_manifest",
    )

    excludes = _as_str_list(project.get("exclude"), field_name="project.exclude")
    if not excludes:
        excludes = list(DEFAULT_EXCLUDE_PATTERNS)
    return ProjectConfig(
        root=root_path,
        config_path=config_path,
        source_roots=_as_str_list(
            project.get("source_roots", "."), field_name="project.source_roots"
        )
        or (".",),
        include_patterns=_as_str_list(
            project.get("include"), field_name="project.include"
        ),
        exclude_patterns=excludes,
        targets=_as_str_list(project.get("targets"), field_name="project.targets"),
        output_dir=output_dir or DEFAULT_OUTPUT_DIR,
        source_overrides=sources,
        include_dirs=_as_str_list(
            project.get("include_dirs"), field_name="project.include_dirs"
        ),
        defines=defines,
        variants=_variant_defines(variants),
        external_corpus_manifest=external_corpus_manifest,
    )


def _override_for_path(relative_path: str, config: ProjectConfig) -> str | None:
    for pattern, backend in config.source_overrides.items():
        if fnmatch.fnmatch(relative_path, pattern):
            return backend
    return None


def _iter_scan_candidates(config: ProjectConfig) -> list[Path]:
    register_default_sources()
    discover_backend_plugins()
    known_extensions = tuple(SOURCE_REGISTRY.extensions())
    explicit_include_patterns = bool(config.include_patterns)
    include_patterns = _repository_relative_globs(config.include_patterns)
    source_override_patterns = set(
        _repository_relative_globs(tuple(config.source_overrides))
    )
    if not explicit_include_patterns:
        include_patterns = [f"**/*{extension}" for extension in known_extensions]
        include_patterns.extend(source_override_patterns)

    candidates: set[Path] = set()
    for source_root in config.source_roots:
        absolute_root = _resolved_source_root(config, source_root)
        if not _is_relative_to(absolute_root, config.root):
            continue
        if not absolute_root.exists():
            continue
        for pattern in include_patterns:
            if explicit_include_patterns or pattern in source_override_patterns:
                candidates.update(
                    path
                    for path in config.root.glob(pattern)
                    if path.is_file() and _is_relative_to(path, absolute_root)
                )
            candidates.update(
                path for path in absolute_root.glob(pattern) if path.is_file()
            )
    return sorted(candidates)


def scan_project(config_or_root: ProjectConfig | str | os.PathLike[str]) -> ProjectScan:
    """Discover supported shader/GPU source files in a repository."""
    config = (
        config_or_root
        if isinstance(config_or_root, ProjectConfig)
        else load_project_config(config_or_root)
    )
    units: list[ProjectTranslationUnit] = []
    skipped: list[dict[str, Any]] = []
    diagnostics: list[ProjectDiagnostic] = _configuration_diagnostics(config)
    diagnostics.extend(_source_root_diagnostics(config))
    diagnostics.extend(_include_dir_diagnostics(config))
    diagnostics.extend(_scan_pattern_diagnostics(config))
    internal_exclude_patterns = _internal_exclude_patterns(config)

    for path in _iter_scan_candidates(config):
        try:
            relative_path = _relpath(path, config.root)
        except ValueError:
            continue
        if _path_matches(relative_path, config.exclude_patterns):
            continue
        if _path_matches(relative_path, internal_exclude_patterns):
            continue

        override = _override_for_path(relative_path, config)
        source_spec = (
            SOURCE_REGISTRY.get(override)
            if override
            else SOURCE_REGISTRY.get_by_extension(str(path))
        )
        if override and not source_spec:
            skipped.append(
                {
                    "path": relative_path,
                    "reason": "unsupported-source-override",
                    "sourceOverride": override,
                }
            )
            diagnostics.append(
                ProjectDiagnostic(
                    severity="error",
                    code="project.config.unsupported-source-override",
                    message=(
                        f"Source override for {relative_path} references "
                        f"unsupported backend '{override}'."
                    ),
                    location=_config_location(config),
                    missing_capabilities=["source.override"],
                )
            )
            continue
        if not source_spec:
            skipped.append({"path": relative_path, "reason": "unsupported-extension"})
            diagnostics.append(
                ProjectDiagnostic(
                    severity="warning",
                    code="project.scan.unsupported-source",
                    message=f"No registered source backend for {relative_path}",
                    location=SourceLocation(file=relative_path),
                    missing_capabilities=["source.discovery"],
                )
            )
            continue

        units.append(
            ProjectTranslationUnit(
                path=path,
                relative_path=relative_path,
                source_backend=source_spec.name,
                extension=path.suffix.lower(),
                source_override=override,
            )
        )

    if not units:
        diagnostics.append(
            ProjectDiagnostic(
                severity="warning",
                code="project.scan.empty",
                message="No supported shader or GPU source files were discovered",
                location=_config_location(config),
                missing_capabilities=["repo.scan"],
            )
        )
    return ProjectScan(
        config=config, units=units, skipped=skipped, diagnostics=diagnostics
    )


def _artifact_path(
    config: ProjectConfig,
    unit: ProjectTranslationUnit,
    target: str,
    variant: str | None = None,
) -> Path:
    extension = _artifact_target_extension(target)
    relative = Path(unit.relative_path)
    base = config.output_path / target
    if variant is not None:
        base = base / _variant_output_segment(variant)
    return base / relative.with_suffix(extension)


def _artifact_target_extension(target: str) -> str:
    normalized_target = _normalized_targets([target])[0]
    if normalized_target in CROSSL_TARGETS:
        return ".cgl"
    return get_backend_extension(normalized_target) or ".out"


def _variant_jobs(
    config: ProjectConfig,
) -> list[tuple[str | None, dict[str, str]]]:
    if not config.variants:
        return [(None, dict(config.defines))]
    return [
        (name, {**config.defines, **dict(defines)})
        for name, defines in sorted(config.variants.items())
    ]


def _artifact_source_map(
    config: ProjectConfig,
    unit: ProjectTranslationUnit,
    target: str,
    output_path: Path,
) -> dict[str, Any]:
    artifact_path = _artifact_report_path(output_path, config)
    source_span = _file_span(unit.path, unit.relative_path)
    generated_span = _file_span(output_path, artifact_path)
    return {
        "schemaVersion": 1,
        "kind": "crosstl-artifact-source-map",
        "mappingGranularity": "file",
        "target": target,
        "source": source_span.to_json(),
        "generated": generated_span.to_json(),
        "mappings": [
            {
                "source": source_span.to_json(),
                "generated": generated_span.to_json(),
            }
        ],
    }


def _runtime_migration_targets(
    units: Sequence[ProjectTranslationUnit],
    targets: Sequence[str],
    artifacts: Sequence[Mapping[str, Any]] | None = None,
) -> list[str]:
    if not units or not targets:
        return []
    supported_targets = set(_supported_target_names())
    selected_targets = [
        target for target in _normalized_targets(targets) if target in supported_targets
    ]
    if artifacts is None:
        return selected_targets

    translated_targets = {
        artifact.get("target")
        for artifact in artifacts
        if isinstance(artifact, Mapping) and artifact.get("status") == "translated"
    }
    return [target for target in selected_targets if target in translated_targets]


def _runtime_migration_actions(
    units: Sequence[ProjectTranslationUnit],
    targets: Sequence[str],
    artifacts: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    action_targets = _runtime_migration_targets(units, targets, artifacts)
    if not action_targets:
        return []
    return [
        {
            "kind": "manual-runtime-integration",
            "severity": "note",
            "message": (
                "CrossTL translated shader/kernel source artifacts only; review "
                "host runtime API calls, resource binding setup, build scripts, "
                "and backend framework integration separately."
            ),
            "targets": action_targets,
        }
    ]


def _has_error_diagnostic(diagnostics: Sequence[ProjectDiagnostic], code: str) -> bool:
    return any(
        diagnostic.severity == "error" and diagnostic.code == code
        for diagnostic in diagnostics
    )


def translate_project(
    config_or_root: ProjectConfig | str | os.PathLike[str],
    *,
    targets: Sequence[str] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    format_output: bool = False,
    validate: bool = False,
) -> ProjectPortabilityReport:
    """Translate all discovered project units to one or more target backends."""
    config = (
        config_or_root
        if isinstance(config_or_root, ProjectConfig)
        else load_project_config(config_or_root)
    )
    if output_dir is not None:
        config = ProjectConfig(
            root=config.root,
            config_path=config.config_path,
            source_roots=config.source_roots,
            include_patterns=config.include_patterns,
            exclude_patterns=config.exclude_patterns,
            targets=config.targets,
            output_dir=str(output_dir),
            source_overrides=config.source_overrides,
            include_dirs=config.include_dirs,
            defines=config.defines,
            variants=config.variants,
            external_corpus_manifest=config.external_corpus_manifest,
        )

    selected_targets = _normalized_targets(
        targets if targets is not None else config.targets
    )
    if not selected_targets:
        selected_targets = ["cgl"]

    scan = scan_project(config)
    diagnostics: list[ProjectDiagnostic] = list(scan.diagnostics)
    diagnostics.extend(_target_diagnostics(config, selected_targets))
    artifacts: list[dict[str, Any]] = []
    include_paths = _frontend_include_dirs(config)
    output_dir_blocked = _has_error_diagnostic(
        diagnostics, OUTPUT_DIR_OUTSIDE_PROJECT_CODE
    )

    variant_jobs = _variant_jobs(config)

    for unit in scan.units:
        for target in selected_targets:
            for variant, defines in variant_jobs:
                output_path = _artifact_path(config, unit, target, variant)
                artifact = {
                    "source": unit.relative_path,
                    "sourceBackend": unit.source_backend,
                    "target": target,
                    "path": _artifact_report_path(output_path, config),
                    "status": "translated",
                    "sourceHash": _source_hash(unit.path),
                    "provenance": {
                        "pipeline": "single-file-translate",
                        "intermediate": (
                            "crossgl"
                            if unit.source_backend not in {"cgl", "crossgl"}
                            and target not in {"cgl", "crossgl"}
                            else None
                        ),
                    },
                }
                if variant is not None:
                    artifact["variant"] = variant
                if output_dir_blocked:
                    artifact["status"] = "failed"
                    artifact["error"] = (
                        "Configured output directory resolves outside the repository; "
                        "artifact was not written."
                    )
                    artifacts.append(artifact)
                    continue
                try:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    translate(
                        str(unit.path),
                        backend=target,
                        save_shader=str(output_path),
                        format_output=format_output,
                        source_backend=unit.source_backend,
                        include_paths=include_paths,
                        defines=defines,
                    )
                    artifact["generatedHash"] = _source_hash(output_path)
                    artifact["sourceMap"] = _artifact_source_map(
                        config, unit, target, output_path
                    )
                except Exception as exc:  # noqa: BLE001
                    # Project translation reports per-artifact failures so one bad
                    # unit does not hide the rest of the repository's migration state.
                    artifact["status"] = "failed"
                    artifact["error"] = str(exc)
                    diagnostics.append(
                        ProjectDiagnostic(
                            severity="error",
                            code="project.translate.failed",
                            message=str(exc),
                            location=SourceLocation(file=unit.relative_path),
                            target=target,
                            missing_capabilities=["batch.translation"],
                        )
                    )
                artifacts.append(artifact)

    validation = (
        _validate_artifacts(artifacts, selected_targets, config)
        if validate
        else {"toolchains": [], "artifacts": []}
    )
    diagnostics.extend(validation.pop("_diagnostics", []))
    return ProjectPortabilityReport(
        config=config,
        targets=selected_targets,
        units=scan.units,
        skipped=scan.skipped,
        artifacts=artifacts,
        diagnostics=diagnostics,
        validation=validation,
        migration_actions=_runtime_migration_actions(
            scan.units, selected_targets, artifacts
        ),
    )


def _tool_status(target: str) -> dict[str, Any]:
    tools = TOOLCHAIN_BY_BACKEND.get(target, ())
    if not tools:
        return {
            "target": target,
            "status": "not-configured",
            "tools": [],
            "message": "No validation toolchain hook is configured for this target.",
        }
    resolved = []
    for tool in tools:
        path = shutil.which(tool)
        resolved.append({"name": tool, "path": path, "available": path is not None})
    available = all(tool["available"] for tool in resolved)
    return {
        "target": target,
        "status": "available" if available else "unavailable",
        "tools": resolved,
    }


def _status_counts(
    records: Sequence[Any], field_name: str, statuses: frozenset[str]
) -> dict[str, int]:
    counts = {status: 0 for status in sorted(statuses)}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        status = record.get(field_name)
        if isinstance(status, str) and status in counts:
            counts[status] += 1
    return counts


def _validation_summary(artifact_checks: Sequence[Any]) -> dict[str, Any]:
    status_counts = _status_counts(
        artifact_checks, "status", VALIDATION_ARTIFACT_STATUSES
    )
    return {
        "artifactCount": len(artifact_checks),
        "okCount": status_counts["ok"],
        "failedCount": status_counts["failed"],
        "sourceHashStatusCounts": _status_counts(
            artifact_checks, "sourceHashStatus", SOURCE_HASH_VALIDATION_STATUSES
        ),
        "generatedHashStatusCounts": _status_counts(
            artifact_checks,
            "generatedHashStatus",
            GENERATED_HASH_VALIDATION_STATUSES,
        ),
    }


def _validate_artifacts(
    artifacts: Sequence[Mapping[str, Any]],
    targets: Sequence[str],
    config: ProjectConfig,
) -> dict[str, Any]:
    diagnostics: list[ProjectDiagnostic] = []
    toolchains = [_tool_status(target) for target in targets]
    artifact_checks = []

    for artifact in artifacts:
        artifact_path = _resolve_report_path(config, artifact["path"])
        artifact_inside_project = _is_relative_to(artifact_path, config.root)
        if not artifact_inside_project:
            diagnostics.append(
                ProjectDiagnostic(
                    severity="error",
                    code="project.validate.artifact-outside-project",
                    message=(
                        "Artifact path resolves outside the repository: "
                        f"{artifact['path']}"
                    ),
                    location=SourceLocation(file=str(artifact["source"])),
                    target=str(artifact["target"]),
                    missing_capabilities=["artifact.manifest"],
                )
            )
        exists = artifact_path.exists() if artifact_inside_project else False
        source_hash_status = "not-recorded"
        source_hash = artifact.get("sourceHash")
        if isinstance(source_hash, Mapping):
            source_path = _resolve_report_path(config, artifact["source"])
            source_inside_project = _is_relative_to(source_path, config.root)
            if not source_inside_project:
                source_hash_status = "outside-project"
                diagnostics.append(
                    ProjectDiagnostic(
                        severity="error",
                        code="project.validate.source-outside-project",
                        message=(
                            "Source path resolves outside the repository: "
                            f"{artifact['source']}"
                        ),
                        location=SourceLocation(file=str(artifact["source"])),
                        target=str(artifact["target"]),
                        missing_capabilities=["source.provenance"],
                    )
                )
            elif not source_path.exists():
                source_hash_status = "missing"
                diagnostics.append(
                    ProjectDiagnostic(
                        severity="error",
                        code="project.validate.missing-source",
                        message=(
                            "Expected source artifact is missing: "
                            f"{artifact['source']}"
                        ),
                        location=SourceLocation(file=str(artifact["source"])),
                        target=str(artifact["target"]),
                        missing_capabilities=["source.provenance"],
                    )
                )
            else:
                actual_source_hash = _source_hash(source_path)
                if not _hash_matches_report(actual_source_hash, source_hash):
                    source_hash_status = "mismatch"
                    diagnostics.append(
                        ProjectDiagnostic(
                            severity="error",
                            code="project.validate.source-hash-mismatch",
                            message=(
                                "Source artifact hash does not match report: "
                                f"{artifact['source']}"
                            ),
                            location=SourceLocation(file=str(artifact["source"])),
                            target=str(artifact["target"]),
                            missing_capabilities=["source.provenance"],
                        )
                    )
                else:
                    source_hash_status = "ok"
        if (
            artifact_inside_project
            and not exists
            and artifact.get("status") == "translated"
        ):
            diagnostics.append(
                ProjectDiagnostic(
                    severity="error",
                    code="project.validate.missing-artifact",
                    message=f"Expected translated artifact is missing: {artifact['path']}",
                    location=SourceLocation(file=str(artifact["source"])),
                    target=str(artifact["target"]),
                    missing_capabilities=["artifact.manifest"],
                )
            )
        generated_hash_status = (
            "not-recorded"
            if artifact.get("status") == "translated"
            else "not-applicable"
        )
        generated_hash = artifact.get("generatedHash")
        if artifact.get("status") == "translated":
            if not artifact_inside_project:
                generated_hash_status = "outside-project"
            elif not exists:
                generated_hash_status = "missing"
            elif isinstance(generated_hash, Mapping):
                actual_hash = _source_hash(artifact_path)
                if not _hash_matches_report(actual_hash, generated_hash):
                    generated_hash_status = "mismatch"
                    diagnostics.append(
                        ProjectDiagnostic(
                            severity="error",
                            code="project.validate.generated-hash-mismatch",
                            message=(
                                "Generated artifact hash does not match report: "
                                f"{artifact['path']}"
                            ),
                            location=SourceLocation(file=str(artifact["source"])),
                            target=str(artifact["target"]),
                            missing_capabilities=["artifact.manifest"],
                        )
                    )
                else:
                    generated_hash_status = "ok"
        if artifact.get("status") == "failed":
            error = str(artifact.get("error", "")).strip()
            message = (
                f"Artifact translation failed before validation: {artifact['path']}"
            )
            if error:
                message = f"{message}: {error}"
            diagnostics.append(
                ProjectDiagnostic(
                    severity="error",
                    code="project.validate.failed-artifact",
                    message=message,
                    location=SourceLocation(file=str(artifact["source"])),
                    target=str(artifact["target"]),
                    missing_capabilities=["batch.translation"],
                )
            )
        artifact_check = {
            "source": artifact["source"],
            "target": artifact["target"],
            "path": artifact["path"],
            "exists": exists,
            "status": (
                "ok"
                if artifact_inside_project
                and exists
                and artifact.get("status") == "translated"
                and source_hash_status in {"ok", "not-recorded"}
                and generated_hash_status in {"ok", "not-recorded"}
                else "failed"
            ),
            "sourceHashStatus": source_hash_status,
            "generatedHashStatus": generated_hash_status,
        }
        if artifact.get("variant") is not None:
            artifact_check["variant"] = artifact["variant"]
        artifact_checks.append(artifact_check)

    for toolchain in toolchains:
        if toolchain["status"] == "unavailable":
            diagnostics.append(
                ProjectDiagnostic(
                    severity="warning",
                    code="project.validate.toolchain-unavailable",
                    message=f"No validation toolchain is available for target {toolchain['target']}",
                    location=_config_location(config),
                    target=toolchain["target"],
                    missing_capabilities=["toolchain.validation"],
                )
            )

    return {
        "toolchains": toolchains,
        "artifacts": artifact_checks,
        "summary": _validation_summary(artifact_checks),
        "_diagnostics": diagnostics,
    }


def validate_project_report(
    report_path: str | os.PathLike[str], *, run_toolchains: bool = False
) -> dict[str, Any]:
    """Validate artifact existence and optional toolchain availability for a report."""
    path = Path(report_path)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        diagnostic = _invalid_report_diagnostic(
            path, [f"could not read JSON report: {exc}"]
        )
        return _validation_report_payload(
            path,
            [diagnostic.to_json()],
            {"toolchains": [], "artifacts": []},
        )

    contract_diagnostics = _report_contract_diagnostics(path, report)
    if contract_diagnostics:
        diagnostics = [diagnostic.to_json() for diagnostic in contract_diagnostics]
        return _validation_report_payload(
            path,
            diagnostics,
            {"toolchains": [], "artifacts": []},
        )

    root = Path(report["project"]["root"])
    targets = report["project"].get("targets", [])
    config = ProjectConfig(
        root=root, output_dir=report["project"].get("outputDir", DEFAULT_OUTPUT_DIR)
    )
    validation = _validate_artifacts(report.get("artifacts", []), targets, config)
    if run_toolchains:
        toolchain_runs = _run_toolchain_smoke(report.get("artifacts", []), root)
        validation["toolchainRuns"] = toolchain_runs
        validation["_diagnostics"].extend(_toolchain_run_diagnostics(toolchain_runs))
    diagnostic_objects = validation.pop("_diagnostics", [])
    source_diagnostics = [
        dict(diagnostic)
        for diagnostic in report.get("diagnostics", [])
        if isinstance(diagnostic, Mapping)
    ]
    diagnostics = source_diagnostics + [
        diagnostic.to_json() for diagnostic in diagnostic_objects
    ]
    return _validation_report_payload(path, diagnostics, validation)


def inspect_project_report(
    report_path: str | os.PathLike[str],
    *,
    run_toolchains: bool = False,
    max_diagnostics: int = 20,
    max_failed_artifacts: int = 20,
) -> dict[str, Any]:
    """Build a concise inspection summary for a project portability report."""
    path = Path(report_path)
    validation_report = validate_project_report(path, run_toolchains=run_toolchains)
    diagnostic_limit = max(0, max_diagnostics)
    failed_artifact_limit = max(0, max_failed_artifacts)
    diagnostics = list(validation_report.get("diagnostics", []))
    payload: dict[str, Any] = {
        "schemaVersion": REPORT_SCHEMA_VERSION,
        "kind": REPORT_INSPECTION_KIND,
        "sourceReport": str(path),
        "generatedAt": int(time.time()),
        "success": bool(validation_report.get("success")),
        "report": {"available": False},
        "diagnosticCount": len(diagnostics),
        "truncatedDiagnosticCount": max(0, len(diagnostics) - diagnostic_limit),
        "failedArtifactCount": 0,
        "truncatedFailedArtifactCount": 0,
        "failedArtifacts": [],
        "diagnostics": diagnostics[:diagnostic_limit],
        "validation": {
            "success": bool(validation_report.get("success")),
            "diagnosticCounts": dict(validation_report.get("diagnosticCounts", {})),
            "result": dict(validation_report.get("validation", {})),
        },
    }

    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return payload
    if not isinstance(report, Mapping):
        return payload

    summary = report.get("summary")
    project = report.get("project")
    generator = report.get("generator")
    payload["report"] = {
        "available": True,
        "kind": report.get("kind"),
        "generatedAt": report.get("generatedAt"),
        "generator": dict(generator) if isinstance(generator, Mapping) else {},
        "project": _inspection_project_summary(project),
        "summary": dict(summary) if isinstance(summary, Mapping) else {},
    }

    artifacts = report.get("artifacts", [])
    failed_artifacts_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    if isinstance(artifacts, Sequence) and not isinstance(
        artifacts, (str, bytes, bytearray)
    ):
        for artifact in artifacts:
            if not isinstance(artifact, Mapping) or artifact.get("status") != "failed":
                continue
            failed = _inspection_failed_artifact(artifact)
            failed_artifacts_by_key[_inspection_failed_artifact_key(failed)] = failed

    validation_result = validation_report.get("validation", {})
    validation_artifacts = (
        validation_result.get("artifacts")
        if isinstance(validation_result, Mapping)
        else None
    )
    if isinstance(validation_artifacts, Sequence) and not isinstance(
        validation_artifacts, (str, bytes, bytearray)
    ):
        for artifact in validation_artifacts:
            if not isinstance(artifact, Mapping) or artifact.get("status") != "failed":
                continue
            failed = _inspection_failed_artifact(artifact)
            failed["validationStatus"] = "failed"
            key = _inspection_failed_artifact_key(failed)
            existing = failed_artifacts_by_key.get(key)
            if existing:
                existing.update(
                    {
                        field_name: failed[field_name]
                        for field_name in (
                            "exists",
                            "sourceHashStatus",
                            "generatedHashStatus",
                            "validationStatus",
                        )
                        if field_name in failed
                    }
                )
            else:
                failed_artifacts_by_key[key] = failed
    failed_artifacts = list(failed_artifacts_by_key.values())
    payload["failedArtifactCount"] = len(failed_artifacts)
    payload["truncatedFailedArtifactCount"] = max(
        0, len(failed_artifacts) - failed_artifact_limit
    )
    payload["failedArtifacts"] = failed_artifacts[:failed_artifact_limit]

    migration = report.get("migration")
    if isinstance(migration, Mapping):
        payload["migration"] = {
            "scope": migration.get("scope"),
            "actions": (
                list(migration.get("actions", []))
                if isinstance(migration.get("actions"), list)
                else []
            ),
        }

    external_corpus = report.get("externalCorpus")
    if isinstance(external_corpus, Mapping):
        external_summary = external_corpus.get("summary")
        payload["externalCorpus"] = {
            "status": external_corpus.get("status"),
            "summary": (
                dict(external_summary) if isinstance(external_summary, Mapping) else {}
            ),
        }
    return payload


def _inspection_project_summary(project: Any) -> dict[str, Any]:
    if not isinstance(project, Mapping):
        return {}
    summary = {
        "root": project.get("root"),
        "targets": (
            list(project.get("targets", []))
            if isinstance(project.get("targets"), list)
            else []
        ),
        "outputDir": project.get("outputDir"),
    }
    for field_name in (
        "sourceOverrideCount",
        "defineCount",
        "variantCount",
        "externalCorpusManifest",
    ):
        if field_name in project:
            summary[field_name] = project[field_name]
    return summary


def _inspection_failed_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    failed = {
        "source": artifact.get("source"),
        "target": artifact.get("target"),
        "path": artifact.get("path"),
    }
    if "variant" in artifact:
        failed["variant"] = artifact.get("variant")
    if "error" in artifact:
        failed["error"] = artifact.get("error")
    for field_name in ("exists", "sourceHashStatus", "generatedHashStatus"):
        if field_name in artifact:
            failed[field_name] = artifact.get(field_name)
    return failed


def _inspection_failed_artifact_key(artifact: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        artifact.get("source"),
        artifact.get("target"),
        artifact.get("path"),
        artifact.get("variant"),
    )


def _toolchain_run_diagnostics(
    runs: Sequence[Mapping[str, Any]],
) -> list[ProjectDiagnostic]:
    diagnostics: list[ProjectDiagnostic] = []
    for run in runs:
        if run.get("returncode") == 0:
            continue
        target = str(run.get("target", "unknown"))
        artifact_path = str(run.get("path", ""))
        diagnostics.append(
            ProjectDiagnostic(
                severity="error",
                code="project.validate.toolchain-failed",
                message=(
                    f"Validation toolchain for target {target} rejected "
                    f"{artifact_path}."
                ),
                location=SourceLocation(file=artifact_path),
                target=target,
                missing_capabilities=["toolchain.validation"],
            )
        )
    return diagnostics


def _validation_report_payload(
    path: Path,
    diagnostics: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schemaVersion": REPORT_SCHEMA_VERSION,
        "kind": "crosstl-project-validation-report",
        "sourceReport": str(path),
        "generatedAt": int(time.time()),
        "success": not any(
            diagnostic.get("severity") == "error" for diagnostic in diagnostics
        ),
        "diagnosticCounts": _diagnostic_payload_counts(diagnostics),
        "diagnostics": list(diagnostics),
        "validation": dict(validation),
    }


def _invalid_report_diagnostic(path: Path, reasons: Sequence[str]) -> ProjectDiagnostic:
    return ProjectDiagnostic(
        severity="error",
        code="project.validate.invalid-report",
        message="Invalid project portability report: " + "; ".join(reasons),
        location=SourceLocation(file=str(path)),
        missing_capabilities=["artifact.manifest"],
    )


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_sha256_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in LOWERCASE_HEX_DIGITS for character in value)
    )


def _hash_contract_reasons(prefix: str, value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{prefix} must be an object"]

    reasons = []
    if value.get("algorithm") != "sha256":
        reasons.append(f"{prefix}.algorithm must be sha256")
    if not _is_sha256_digest(value.get("value")):
        reasons.append(f"{prefix}.value must be a lowercase 64-character hex digest")
    return reasons


def _source_hash_contract_reasons(
    index: int, artifact: Mapping[str, Any], *, required: bool = False
) -> list[str]:
    if "sourceHash" not in artifact and not required:
        return []
    return _hash_contract_reasons(
        f"artifacts[{index}].sourceHash", artifact.get("sourceHash")
    )


def _generated_hash_contract_reasons(
    index: int, artifact: Mapping[str, Any], *, required: bool = False
) -> list[str]:
    if "generatedHash" not in artifact and not required:
        return []
    return _hash_contract_reasons(
        f"artifacts[{index}].generatedHash", artifact.get("generatedHash")
    )


def _expected_provenance_intermediate(artifact: Mapping[str, Any]) -> str | None:
    source_backend = artifact.get("sourceBackend")
    target = artifact.get("target")
    if not _is_non_empty_string(source_backend) or not _is_non_empty_string(target):
        return None

    normalized_source_backend = _normalized_targets([source_backend])[0]
    normalized_target = _normalized_targets([target])[0]
    if (
        normalized_source_backend not in CROSSL_TARGETS
        and normalized_target not in CROSSL_TARGETS
    ):
        return "crossgl"
    return None


def _provenance_contract_reasons(
    index: int, artifact: Mapping[str, Any], *, required: bool = False
) -> list[str]:
    if "provenance" not in artifact:
        if required:
            return [f"artifacts[{index}].provenance must be an object"]
        return []

    prefix = f"artifacts[{index}].provenance"
    provenance = artifact.get("provenance")
    if not isinstance(provenance, Mapping):
        return [f"{prefix} must be an object"]

    reasons = []
    pipeline = provenance.get("pipeline")
    if not _is_non_empty_string(pipeline):
        reasons.append(f"{prefix}.pipeline must be a string")
    elif pipeline not in REPORT_ARTIFACT_PROVENANCE_PIPELINES:
        reasons.append(
            "{}.pipeline must be one of {}".format(
                prefix, ", ".join(REPORT_ARTIFACT_PROVENANCE_PIPELINES)
            )
        )
    intermediate = provenance.get("intermediate")
    if intermediate is not None and not _is_non_empty_string(intermediate):
        reasons.append(f"{prefix}.intermediate must be a string or null")
    elif intermediate is not None and intermediate not in (
        REPORT_ARTIFACT_PROVENANCE_INTERMEDIATES
    ):
        reasons.append(
            "{}.intermediate must be one of {} or null".format(
                prefix, ", ".join(REPORT_ARTIFACT_PROVENANCE_INTERMEDIATES)
            )
        )
    elif (
        _is_non_empty_string(artifact.get("sourceBackend"))
        and _is_non_empty_string(artifact.get("target"))
        and intermediate != _expected_provenance_intermediate(artifact)
    ):
        reasons.append(
            f"{prefix}.intermediate must match artifacts[{index}].sourceBackend "
            f"and artifacts[{index}].target"
        )
    return reasons


def _count_field_contract_reasons(
    prefix: str, value: Any, expected: int, source_name: str
) -> list[str]:
    if not _is_non_negative_int(value):
        return [f"{prefix} must be a non-negative integer"]
    if value != expected:
        return [f"{prefix} must match {source_name}"]
    return []


def _mapping_field_contract_reasons(
    prefix: str, value: Any, expected: Mapping[str, Any], source_name: str
) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{prefix} must be an object"]
    if dict(value) != dict(expected):
        return [f"{prefix} must match {source_name}"]
    return []


def _payload_diagnostic_counts(
    diagnostics: Sequence[Any],
) -> dict[str, int] | None:
    if not all(isinstance(diagnostic, Mapping) for diagnostic in diagnostics):
        return None
    return _diagnostic_payload_counts(diagnostics)


def _payload_diagnostic_counts_by_code(
    diagnostics: Sequence[Any],
) -> dict[str, int] | None:
    if not all(isinstance(diagnostic, Mapping) for diagnostic in diagnostics):
        return None

    counts: dict[str, int] = {}
    for diagnostic in diagnostics:
        code = diagnostic.get("code")
        if _is_non_empty_string(code):
            counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items()))


def _payload_missing_capability_counts(
    diagnostics: Sequence[Any],
) -> dict[str, int] | None:
    if not all(isinstance(diagnostic, Mapping) for diagnostic in diagnostics):
        return None

    counts: dict[str, int] = {}
    for diagnostic in diagnostics:
        missing_capabilities = diagnostic.get("missingCapabilities", [])
        if not isinstance(missing_capabilities, list):
            continue
        for capability in missing_capabilities:
            if _is_non_empty_string(capability):
                counts[capability] = counts.get(capability, 0) + 1
    return dict(sorted(counts.items()))


def _payload_unit_counts_by_source_backend(units: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for unit in units:
        if not isinstance(unit, Mapping):
            continue
        source_backend = unit.get("sourceBackend")
        if not _is_non_empty_string(source_backend):
            continue
        counts[source_backend] = counts.get(source_backend, 0) + 1
    return dict(sorted(counts.items()))


def _payload_artifact_records(artifacts: Sequence[Any]) -> list[Mapping[str, Any]]:
    return [artifact for artifact in artifacts if isinstance(artifact, Mapping)]


UnitDeclaration = Tuple[int, Mapping[str, Any]]


def _record_path(record: Any) -> str | None:
    if not isinstance(record, Mapping):
        return None
    path = record.get("path")
    if not _is_non_empty_string(path):
        return None
    if not _is_repository_relative_report_path(path):
        return None
    return path


def _duplicate_path_contract_reasons(prefix: str, records: Sequence[Any]) -> list[str]:
    reasons = []
    paths: dict[str, int] = {}
    for index, record in enumerate(records):
        path = _record_path(record)
        if path is None:
            continue
        previous_index = paths.get(path)
        if previous_index is None:
            paths[path] = index
            continue
        reasons.append(
            f"{prefix}[{index}].path duplicates {prefix}[{previous_index}].path"
        )
    return reasons


def _duplicate_field_contract_reasons(
    prefix: str, records: Sequence[Any], field_name: str
) -> list[str]:
    reasons = []
    values: dict[str, int] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        value = record.get(field_name)
        if not _is_non_empty_string(value):
            continue
        previous_index = values.get(value)
        if previous_index is None:
            values[value] = index
            continue
        reasons.append(
            f"{prefix}[{index}].{field_name} duplicates "
            f"{prefix}[{previous_index}].{field_name}"
        )
    return reasons


def _unit_skipped_path_contract_reasons(
    units: Sequence[Any], skipped: Sequence[Any]
) -> list[str]:
    unit_paths: dict[str, int] = {}
    for index, unit in enumerate(units):
        path = _record_path(unit)
        if path is not None and path not in unit_paths:
            unit_paths[path] = index
    reasons = []
    for index, skipped_record in enumerate(skipped):
        path = _record_path(skipped_record)
        if path is None:
            continue
        unit_index = unit_paths.get(path)
        if unit_index is not None:
            reasons.append(f"skipped[{index}].path duplicates units[{unit_index}].path")
    return reasons


def _declared_units_by_path(
    units: Any, *, require_full_metadata: bool
) -> dict[str, UnitDeclaration] | None:
    if not require_full_metadata or not isinstance(units, list):
        return None
    declarations = {}
    for index, unit in enumerate(units):
        if not isinstance(unit, Mapping):
            return None
        path = unit.get("path")
        if not _is_non_empty_string(path) or not _is_repository_relative_report_path(
            path
        ):
            return None
        declarations.setdefault(path, (index, unit))
    return declarations


ArtifactIdentity = Tuple[str, str, str, Optional[str]]
DeclaredArtifact = Tuple[int, Mapping[str, Any]]


def _artifact_identity(record: Mapping[str, Any]) -> ArtifactIdentity | None:
    source = record.get("source")
    target = record.get("target")
    path = record.get("path")
    if not (
        _is_non_empty_string(source)
        and _is_non_empty_string(target)
        and _is_non_empty_string(path)
    ):
        return None

    variant = record.get("variant")
    if "variant" in record:
        if not _is_non_empty_string(variant):
            return None
        variant_identity = variant
    else:
        variant_identity = None
    return source, _normalized_targets([target])[0], path, variant_identity


def _duplicate_identity_contract_reasons(
    prefix: str, records: Sequence[Any]
) -> list[str]:
    reasons = []
    identities: dict[ArtifactIdentity, int] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        identity = _artifact_identity(record)
        if identity is None:
            continue
        previous_index = identities.get(identity)
        if previous_index is None:
            identities[identity] = index
            continue
        reasons.append(
            f"{prefix}[{index}] duplicates {prefix}[{previous_index}] identity"
        )
    return reasons


def _duplicate_toolchain_target_contract_reasons(records: Sequence[Any]) -> list[str]:
    reasons = []
    targets: dict[str, int] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        target = record.get("target")
        if not _is_non_empty_string(target):
            continue
        normalized_target = _normalized_targets([target])[0]
        previous_index = targets.get(normalized_target)
        if previous_index is None:
            targets[normalized_target] = index
            continue
        reasons.append(
            "validation.toolchains[{}] duplicates "
            "validation.toolchains[{}].target".format(index, previous_index)
        )
    return reasons


def _validation_toolchain_coverage_contract_reasons(
    toolchains: Sequence[Any], project_targets: Sequence[Any]
) -> list[str]:
    validation_targets = set()
    for toolchain in toolchains:
        if not isinstance(toolchain, Mapping):
            continue
        target = toolchain.get("target")
        if _is_non_empty_string(target):
            validation_targets.add(_normalized_targets([target])[0])

    reasons = []
    for target_index, target in enumerate(project_targets):
        if not _is_non_empty_string(target):
            continue
        if _normalized_targets([target])[0] not in validation_targets:
            reasons.append(
                f"validation.toolchains must include project.targets[{target_index}]"
            )
    return reasons


def _validation_artifact_coverage_contract_reasons(
    artifact_checks: Sequence[Any],
    declared_artifacts_by_identity: Mapping[ArtifactIdentity, DeclaredArtifact] | None,
) -> list[str]:
    if declared_artifacts_by_identity is None:
        return []

    validation_identities: set[ArtifactIdentity] = set()
    for artifact_check in artifact_checks:
        if not isinstance(artifact_check, Mapping):
            continue
        identity = _artifact_identity(artifact_check)
        if identity is not None:
            validation_identities.add(identity)

    reasons = []
    for identity, (artifact_index, _) in declared_artifacts_by_identity.items():
        if identity not in validation_identities:
            reasons.append(
                f"validation.artifacts must include report.artifacts[{artifact_index}]"
            )
    return reasons


def _diagnostic_counts_contract_reasons(
    prefix: str, value: Any, diagnostics: Sequence[Any]
) -> list[str]:
    expected = _payload_diagnostic_counts(diagnostics)
    if expected is None:
        return []
    return _mapping_field_contract_reasons(prefix, value, expected, "diagnostics")


def _diagnostic_code_counts_contract_reasons(
    prefix: str, value: Any, diagnostics: Sequence[Any]
) -> list[str]:
    expected = _payload_diagnostic_counts_by_code(diagnostics)
    if expected is None:
        return []
    return _mapping_field_contract_reasons(prefix, value, expected, "diagnostics")


def _missing_capability_counts_contract_reasons(
    prefix: str, value: Any, diagnostics: Sequence[Any]
) -> list[str]:
    expected = _payload_missing_capability_counts(diagnostics)
    if expected is None:
        return []
    return _mapping_field_contract_reasons(prefix, value, expected, "diagnostics")


def _string_list_contract_reasons(prefix: str, value: Any) -> list[str]:
    if not isinstance(value, list) or any(
        not _is_non_empty_string(item) for item in value
    ):
        return [f"{prefix} must be a list of strings"]
    return []


def _target_list_contract_reasons(
    prefix: str, value: Any, *, require_canonical: bool
) -> list[str]:
    if not isinstance(value, list) or any(
        not _is_non_empty_string(item) for item in value
    ):
        return [f"{prefix} must be a list of backend names"]
    if not require_canonical:
        return []

    if _normalized_targets(value) != value:
        return [f"{prefix} must use normalized backend names without duplicates"]
    return []


def _diagnostic_location_contract_reasons(prefix: str, value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{prefix} must be an object"]

    reasons = []
    file = value.get("file")
    if not _is_non_empty_string(file):
        reasons.append(f"{prefix}.file must be a string")
    elif not _is_repository_relative_report_path(file):
        reasons.append(f"{prefix}.file must be repository-relative")
    for field_name in (
        "line",
        "column",
        "offset",
        "length",
        "endLine",
        "endColumn",
        "endOffset",
    ):
        if not _is_non_negative_int(value.get(field_name)):
            reasons.append(f"{prefix}.{field_name} must be a non-negative integer")
    if reasons:
        return reasons

    reasons.extend(_span_consistency_contract_reasons(prefix, value))
    return reasons


def _span_consistency_contract_reasons(
    prefix: str, value: Mapping[str, Any]
) -> list[str]:
    reasons = []
    line = value["line"]
    column = value["column"]
    offset = value["offset"]
    length = value["length"]
    end_line = value["endLine"]
    end_column = value["endColumn"]
    end_offset = value["endOffset"]
    if end_offset != offset + length:
        reasons.append(f"{prefix}.endOffset must equal {prefix}.offset plus length")
    if end_line < line:
        reasons.append(f"{prefix}.endLine must be after or equal to {prefix}.line")
    elif end_line == line and end_column < column:
        reasons.append(
            f"{prefix}.endColumn must be greater than or equal to "
            f"{prefix}.column when endLine equals line"
        )
    return reasons


def _repository_path_contract_reasons(prefix: str, value: Any) -> list[str]:
    if not _is_non_empty_string(value):
        return [f"{prefix} must be a string"]
    if not _is_repository_relative_report_path(value):
        return [f"{prefix} must be repository-relative"]
    return []


def _unit_contract_reasons(index: int, unit: Any) -> list[str]:
    prefix = f"units[{index}]"
    if not isinstance(unit, Mapping):
        return [f"{prefix} must be an object"]

    reasons = []
    path = unit.get("path")
    reasons.extend(_repository_path_contract_reasons(f"{prefix}.path", path))

    unit_id = unit.get("id")
    if not _is_non_empty_string(unit_id):
        reasons.append(f"{prefix}.id must be a string")
    elif _is_non_empty_string(path) and unit_id != path:
        reasons.append(f"{prefix}.id must match {prefix}.path")

    if not _is_non_empty_string(unit.get("sourceBackend")):
        reasons.append(f"{prefix}.sourceBackend must be a string")
    extension = unit.get("extension")
    if not isinstance(extension, str):
        reasons.append(f"{prefix}.extension must be a string")
    elif _is_non_empty_string(path) and _is_repository_relative_report_path(path):
        expected_extension = Path(path).suffix.lower()
        if extension != expected_extension:
            reasons.append(f"{prefix}.extension must match {prefix}.path suffix")
    if "sourceOverride" in unit and not _is_non_empty_string(
        unit.get("sourceOverride")
    ):
        reasons.append(f"{prefix}.sourceOverride must be a string")
    return reasons


def _skipped_contract_reasons(index: int, skipped: Any) -> list[str]:
    prefix = f"skipped[{index}]"
    if not isinstance(skipped, Mapping):
        return [f"{prefix} must be an object"]

    reasons = []
    reasons.extend(
        _repository_path_contract_reasons(f"{prefix}.path", skipped.get("path"))
    )
    if not _is_non_empty_string(skipped.get("reason")):
        reasons.append(f"{prefix}.reason must be a string")
    if "sourceOverride" in skipped and not _is_non_empty_string(
        skipped.get("sourceOverride")
    ):
        reasons.append(f"{prefix}.sourceOverride must be a string")
    return reasons


def _config_string_list_contract_reasons(prefix: str, value: Any) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return [f"{prefix} must be a list of strings"]
    return []


def _string_mapping_contract_reasons(prefix: str, value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{prefix} must be an object"]
    if any(not isinstance(item, str) for item in value.values()):
        return [f"{prefix} values must be strings"]
    return []


def _variant_mapping_contract_reasons(prefix: str, value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{prefix} must be an object"]

    reasons = []
    for variant_name, defines in value.items():
        variant_prefix = f"{prefix}.{variant_name}"
        if not _is_non_empty_string(variant_name):
            reasons.append(f"{prefix} keys must be non-empty strings")
        if not isinstance(defines, Mapping):
            reasons.append(f"{variant_prefix} must be an object")
            continue
        if any(not isinstance(item, str) for item in defines.values()):
            reasons.append(f"{variant_prefix} values must be strings")
    return reasons


def _optional_project_field(
    project: Mapping[str, Any], key: str, *, required: bool
) -> bool:
    return required or key in project


def _project_metadata_contract_reasons(
    project: Mapping[str, Any], *, require_full_metadata: bool
) -> list[str]:
    reasons = []
    if _optional_project_field(project, "config", required=require_full_metadata):
        config_path = project.get("config")
        if config_path is not None and not isinstance(config_path, str):
            reasons.append("project.config must be a string or null")

    for field_name in (
        "sourceRoots",
        "includePatterns",
        "excludePatterns",
        "includeDirs",
    ):
        if _optional_project_field(project, field_name, required=require_full_metadata):
            reasons.extend(
                _config_string_list_contract_reasons(
                    f"project.{field_name}", project.get(field_name)
                )
            )

    defines = project.get("defines")
    defines_is_mapping = isinstance(defines, Mapping)
    if _optional_project_field(project, "defines", required=require_full_metadata):
        reasons.extend(_string_mapping_contract_reasons("project.defines", defines))

    if _optional_project_field(project, "defineCount", required=require_full_metadata):
        define_count = project.get("defineCount")
        if not _is_non_negative_int(define_count):
            reasons.append("project.defineCount must be a non-negative integer")
        elif defines_is_mapping and define_count != len(defines):
            reasons.append("project.defineCount must match project.defines")

    source_overrides = project.get("sourceOverrides")
    source_overrides_is_mapping = isinstance(source_overrides, Mapping)
    if _optional_project_field(
        project, "sourceOverrides", required=require_full_metadata
    ):
        reasons.extend(
            _string_mapping_contract_reasons(
                "project.sourceOverrides", source_overrides
            )
        )

    if _optional_project_field(
        project, "sourceOverrideCount", required=require_full_metadata
    ):
        source_override_count = project.get("sourceOverrideCount")
        if not _is_non_negative_int(source_override_count):
            reasons.append("project.sourceOverrideCount must be a non-negative integer")
        elif source_overrides_is_mapping and source_override_count != len(
            source_overrides
        ):
            reasons.append(
                "project.sourceOverrideCount must match project.sourceOverrides"
            )

    variants = project.get("variants")
    variants_is_mapping = isinstance(variants, Mapping)
    if _optional_project_field(project, "variants", required=require_full_metadata):
        reasons.extend(_variant_mapping_contract_reasons("project.variants", variants))

    if _optional_project_field(project, "variantCount", required=require_full_metadata):
        variant_count = project.get("variantCount")
        if not _is_non_negative_int(variant_count):
            reasons.append("project.variantCount must be a non-negative integer")
        elif variants_is_mapping and variant_count != len(variants):
            reasons.append("project.variantCount must match project.variants")

    if _optional_project_field(
        project, "externalCorpusManifest", required=require_full_metadata
    ):
        external_corpus_manifest = project.get("externalCorpusManifest")
        if external_corpus_manifest is not None and not isinstance(
            external_corpus_manifest, str
        ):
            reasons.append("project.externalCorpusManifest must be a string or null")

    return reasons


def _generator_contract_reasons(
    report: Mapping[str, Any], *, require_generator: bool
) -> list[str]:
    if "generator" not in report:
        return ["generator must be an object"] if require_generator else []

    generator = report.get("generator")
    if not isinstance(generator, Mapping):
        return ["generator must be an object"]

    reasons = []
    if not _is_non_empty_string(generator.get("name")):
        reasons.append("generator.name must be a string")
    if generator.get("pipeline") != REPORT_GENERATOR_PIPELINE:
        reasons.append(f"generator.pipeline must be {REPORT_GENERATOR_PIPELINE}")
    if require_generator or "packageVersion" in generator:
        if not _is_non_empty_string(generator.get("packageVersion")):
            reasons.append("generator.packageVersion must be a string")
    return reasons


def _tool_status_contract_reasons(
    index: int,
    toolchain: Any,
    *,
    declared_targets: set[str] | None = None,
) -> list[str]:
    prefix = f"validation.toolchains[{index}]"
    if not isinstance(toolchain, Mapping):
        return [f"{prefix} must be an object"]

    reasons = []
    status = toolchain.get("status")
    status_is_valid = status in {"available", "unavailable", "not-configured"}
    target = toolchain.get("target")
    if not _is_non_empty_string(target):
        reasons.append(f"{prefix}.target must be a string")
    elif declared_targets is not None:
        normalized_target = _normalized_targets([target])[0]
        if normalized_target not in declared_targets:
            reasons.append(f"{prefix}.target must be listed in project.targets")
    if not status_is_valid:
        reasons.append(
            f"{prefix}.status must be available, unavailable, or not-configured"
        )
    tools = toolchain.get("tools")
    tools_are_valid = True
    if not isinstance(tools, list):
        reasons.append(f"{prefix}.tools must be a list")
        tools_are_valid = False
    else:
        tool_name_indexes: dict[str, int] = {}
        for tool_index, tool in enumerate(tools):
            tool_prefix = f"{prefix}.tools[{tool_index}]"
            if not isinstance(tool, Mapping):
                reasons.append(f"{tool_prefix} must be an object")
                tools_are_valid = False
                continue
            name = tool.get("name")
            if not _is_non_empty_string(name):
                reasons.append(f"{tool_prefix}.name must be a string")
                tools_are_valid = False
            elif name in tool_name_indexes:
                reasons.append(
                    f"{tool_prefix}.name duplicates "
                    f"{prefix}.tools[{tool_name_indexes[name]}].name"
                )
                tools_are_valid = False
            else:
                tool_name_indexes[name] = tool_index
            path = tool.get("path")
            if path is not None and not _is_non_empty_string(path):
                reasons.append(f"{tool_prefix}.path must be a string or null")
                tools_are_valid = False
            if not isinstance(tool.get("available"), bool):
                reasons.append(f"{tool_prefix}.available must be a boolean")
                tools_are_valid = False
    if status_is_valid and tools_are_valid:
        expected_status = (
            "not-configured"
            if not tools
            else (
                "available"
                if all(tool["available"] for tool in tools)
                else "unavailable"
            )
        )
        if status != expected_status:
            reasons.append(f"{prefix}.status must match tools availability")
    if "message" in toolchain and not _is_non_empty_string(toolchain.get("message")):
        reasons.append(f"{prefix}.message must be a string")
    return reasons


def _validation_artifact_status_matches_record(artifact: Mapping[str, Any]) -> bool:
    status = artifact.get("status")
    if artifact.get("exists") is not True:
        return status == "failed"
    source_hash_status = artifact.get("sourceHashStatus", "not-recorded")
    generated_hash_status = artifact.get("generatedHashStatus", "not-recorded")
    expected_status = (
        "ok"
        if source_hash_status in {"ok", "not-recorded"}
        and generated_hash_status in {"ok", "not-recorded"}
        else "failed"
    )
    return status == expected_status


def _validation_artifact_contract_reasons(
    index: int,
    artifact: Any,
    *,
    declared_targets: set[str] | None = None,
    declared_variants: set[str] | None = None,
    declared_artifact_identities: set[ArtifactIdentity] | None = None,
    declared_artifacts_by_identity: (
        Mapping[ArtifactIdentity, DeclaredArtifact] | None
    ) = None,
    require_hash_statuses: bool = False,
) -> list[str]:
    prefix = f"validation.artifacts[{index}]"
    if not isinstance(artifact, Mapping):
        return [f"{prefix} must be an object"]

    reasons = []
    for field_name in ("source", "target", "path"):
        if not _is_non_empty_string(artifact.get(field_name)):
            reasons.append(f"{prefix}.{field_name} must be a string")
    source = artifact.get("source")
    if _is_non_empty_string(source) and not _is_repository_relative_report_path(source):
        reasons.append(f"{prefix}.source must be repository-relative")
    path = artifact.get("path")
    if _is_non_empty_string(path) and not _is_repository_relative_report_path(path):
        reasons.append(f"{prefix}.path must be repository-relative")
    target = artifact.get("target")
    if _is_non_empty_string(target) and declared_targets is not None:
        normalized_target = _normalized_targets([target])[0]
        if normalized_target not in declared_targets:
            reasons.append(f"{prefix}.target must be listed in project.targets")
    if not isinstance(artifact.get("exists"), bool):
        reasons.append(f"{prefix}.exists must be a boolean")
    status = artifact.get("status")
    status_is_valid = status in {"ok", "failed"}
    if not status_is_valid:
        reasons.append(f"{prefix}.status must be ok or failed")
    variant = artifact.get("variant")
    if "variant" in artifact:
        if not _is_non_empty_string(variant):
            reasons.append(f"{prefix}.variant must be a string")
        elif declared_variants is not None and variant not in declared_variants:
            reasons.append(f"{prefix}.variant must be listed in project.variants")
    identity = _artifact_identity(artifact)
    if (
        identity is not None
        and declared_artifact_identities is not None
        and identity not in declared_artifact_identities
    ):
        reasons.append(f"{prefix} must reference an artifact in report.artifacts")
    referenced_artifact: DeclaredArtifact | None = None
    if identity is not None and declared_artifacts_by_identity is not None:
        referenced_artifact = declared_artifacts_by_identity.get(identity)
    source_hash_status = artifact.get("sourceHashStatus")
    if require_hash_statuses and "sourceHashStatus" not in artifact:
        reasons.append(
            f"{prefix}.sourceHashStatus must be recorded "
            "when validation.summary is present"
        )
    if (
        "sourceHashStatus" in artifact
        and source_hash_status not in SOURCE_HASH_VALIDATION_STATUSES
    ):
        reasons.append(
            f"{prefix}.sourceHashStatus must be one of "
            f"{', '.join(sorted(SOURCE_HASH_VALIDATION_STATUSES))}"
        )
    generated_hash_status = artifact.get("generatedHashStatus")
    if require_hash_statuses and "generatedHashStatus" not in artifact:
        reasons.append(
            f"{prefix}.generatedHashStatus must be recorded "
            "when validation.summary is present"
        )
    if (
        "generatedHashStatus" in artifact
        and generated_hash_status not in GENERATED_HASH_VALIDATION_STATUSES
    ):
        reasons.append(
            f"{prefix}.generatedHashStatus must be one of "
            f"{', '.join(sorted(GENERATED_HASH_VALIDATION_STATUSES))}"
        )
    if (
        status_is_valid
        and isinstance(artifact.get("exists"), bool)
        and (
            "sourceHashStatus" not in artifact
            or source_hash_status in SOURCE_HASH_VALIDATION_STATUSES
        )
        and (
            "generatedHashStatus" not in artifact
            or generated_hash_status in GENERATED_HASH_VALIDATION_STATUSES
        )
        and not _validation_artifact_status_matches_record(artifact)
    ):
        reasons.append(f"{prefix}.status must match exists and hash statuses")
    if (
        status == "ok"
        and referenced_artifact is not None
        and referenced_artifact[1].get("status") == "failed"
    ):
        reasons.append(
            f"{prefix}.status must match report.artifacts[{referenced_artifact[0]}].status"
        )
    return reasons


def _validation_summary_contract_reasons(
    summary: Any, artifact_checks: Sequence[Any]
) -> list[str]:
    if not isinstance(summary, Mapping):
        return ["validation.summary must be an object"]

    expected = _validation_summary(artifact_checks)
    reasons = []
    for field_name in ("artifactCount", "okCount", "failedCount"):
        reasons.extend(
            _count_field_contract_reasons(
                f"validation.summary.{field_name}",
                summary.get(field_name),
                expected[field_name],
                "validation.artifacts",
            )
        )
    for field_name in ("sourceHashStatusCounts", "generatedHashStatusCounts"):
        reasons.extend(
            _mapping_field_contract_reasons(
                f"validation.summary.{field_name}",
                summary.get(field_name),
                expected[field_name],
                "validation.artifacts",
            )
        )
    return reasons


def _toolchain_run_contract_reasons(
    index: int,
    run: Any,
    *,
    declared_targets: set[str] | None = None,
    declared_variants: set[str] | None = None,
    declared_artifact_identities: set[ArtifactIdentity] | None = None,
) -> list[str]:
    prefix = f"validation.toolchainRuns[{index}]"
    if not isinstance(run, Mapping):
        return [f"{prefix} must be an object"]

    reasons = []
    for field_name in ("source", "target", "path"):
        if not _is_non_empty_string(run.get(field_name)):
            reasons.append(f"{prefix}.{field_name} must be a string")
    source = run.get("source")
    if _is_non_empty_string(source) and not _is_repository_relative_report_path(source):
        reasons.append(f"{prefix}.source must be repository-relative")
    path = run.get("path")
    if _is_non_empty_string(path) and not _is_repository_relative_report_path(path):
        reasons.append(f"{prefix}.path must be repository-relative")
    target = run.get("target")
    if _is_non_empty_string(target) and declared_targets is not None:
        normalized_target = _normalized_targets([target])[0]
        if normalized_target not in declared_targets:
            reasons.append(f"{prefix}.target must be listed in project.targets")
    variant = run.get("variant")
    if "variant" in run:
        if not _is_non_empty_string(variant):
            reasons.append(f"{prefix}.variant must be a string")
        elif declared_variants is not None and variant not in declared_variants:
            reasons.append(f"{prefix}.variant must be listed in project.variants")
    identity = _artifact_identity(run)
    if (
        identity is not None
        and declared_artifact_identities is not None
        and identity not in declared_artifact_identities
    ):
        reasons.append(f"{prefix} must reference an artifact in report.artifacts")

    command = run.get("command")
    if not isinstance(command, list) or any(
        not _is_non_empty_string(part) for part in command
    ):
        reasons.append(f"{prefix}.command must be a list of strings")

    returncode = run.get("returncode")
    if not isinstance(returncode, int) or isinstance(returncode, bool):
        reasons.append(f"{prefix}.returncode must be an integer")

    status = run.get("status")
    if status not in {"ok", "failed"}:
        reasons.append(f"{prefix}.status must be ok or failed")
    elif isinstance(returncode, int) and not isinstance(returncode, bool):
        expected_status = "ok" if returncode == 0 else "failed"
        if status != expected_status:
            reasons.append(f"{prefix}.status must match returncode")

    for field_name in ("stdout", "stderr"):
        if not isinstance(run.get(field_name), str):
            reasons.append(f"{prefix}.{field_name} must be a string")
    return reasons


def _validation_contract_reasons(
    report: Mapping[str, Any], *, require_validation: bool
) -> list[str]:
    if "validation" not in report:
        return ["validation must be an object"] if require_validation else []

    validation = report.get("validation")
    if not isinstance(validation, Mapping):
        return ["validation must be an object"]

    reasons = []
    project = report.get("project")
    project_targets = project.get("targets", []) if isinstance(project, Mapping) else []
    project_targets_valid = isinstance(project_targets, list) and all(
        _is_non_empty_string(target) for target in project_targets
    )
    declared_targets = (
        set(_normalized_targets(project_targets)) if project_targets_valid else None
    )
    project_has_variants = isinstance(project, Mapping) and "variants" in project
    project_variants = (
        project.get("variants", {}) if isinstance(project, Mapping) else {}
    )
    project_variants_valid = isinstance(project_variants, Mapping) and all(
        _is_non_empty_string(name) for name in project_variants
    )
    declared_variants = (
        set(project_variants)
        if project_has_variants and project_variants_valid
        else None
    )
    artifacts = report.get("artifacts", [])
    declared_artifacts_by_identity: dict[ArtifactIdentity, DeclaredArtifact] | None = (
        None
    )
    declared_artifact_identities: set[ArtifactIdentity] | None = None
    if isinstance(artifacts, list):
        declared_artifacts_by_identity = {}
        for artifact_index, artifact in enumerate(artifacts):
            if not isinstance(artifact, Mapping):
                continue
            identity = _artifact_identity(artifact)
            if identity is None:
                continue
            declared_artifacts_by_identity.setdefault(
                identity, (artifact_index, artifact)
            )
        declared_artifact_identities = set(declared_artifacts_by_identity)
    toolchains = validation.get("toolchains")
    if not isinstance(toolchains, list):
        reasons.append("validation.toolchains must be a list")
    else:
        for index, toolchain in enumerate(toolchains):
            reasons.extend(
                _tool_status_contract_reasons(
                    index,
                    toolchain,
                    declared_targets=declared_targets,
                )
            )
        reasons.extend(_duplicate_toolchain_target_contract_reasons(toolchains))

    artifact_checks = validation.get("artifacts")
    summarized_validation = "summary" in validation
    if not isinstance(artifact_checks, list):
        reasons.append("validation.artifacts must be a list")
    else:
        for index, artifact in enumerate(artifact_checks):
            reasons.extend(
                _validation_artifact_contract_reasons(
                    index,
                    artifact,
                    declared_targets=declared_targets,
                    declared_variants=declared_variants,
                    declared_artifact_identities=declared_artifact_identities,
                    declared_artifacts_by_identity=declared_artifacts_by_identity,
                    require_hash_statuses=summarized_validation,
                )
            )
        reasons.extend(
            _duplicate_identity_contract_reasons(
                "validation.artifacts", artifact_checks
            )
        )

    if "summary" in validation:
        if isinstance(toolchains, list) and project_targets_valid:
            reasons.extend(
                _validation_toolchain_coverage_contract_reasons(
                    toolchains,
                    project_targets,
                )
            )
        if isinstance(artifact_checks, list):
            reasons.extend(
                _validation_artifact_coverage_contract_reasons(
                    artifact_checks,
                    declared_artifacts_by_identity,
                )
            )
        reasons.extend(
            _validation_summary_contract_reasons(
                validation.get("summary"),
                artifact_checks if isinstance(artifact_checks, list) else [],
            )
        )

    if "toolchainRuns" in validation:
        toolchain_runs = validation.get("toolchainRuns")
        if not isinstance(toolchain_runs, list):
            reasons.append("validation.toolchainRuns must be a list")
        else:
            for index, run in enumerate(toolchain_runs):
                reasons.extend(
                    _toolchain_run_contract_reasons(
                        index,
                        run,
                        declared_targets=declared_targets,
                        declared_variants=declared_variants,
                        declared_artifact_identities=declared_artifact_identities,
                    )
                )
            reasons.extend(
                _duplicate_identity_contract_reasons(
                    "validation.toolchainRuns", toolchain_runs
                )
            )
    return reasons


def _external_corpus_entry_contract_reasons(
    index: int,
    entry: Any,
    *,
    artifacts: Sequence[Mapping[str, Any]] | None = None,
    root_path: Path | None = None,
    declared_units_by_path: Mapping[str, UnitDeclaration] | None = None,
) -> list[str]:
    prefix = f"externalCorpus.entries[{index}]"
    if not isinstance(entry, Mapping):
        return [f"{prefix} must be an object"]

    reasons = []
    for field_name in ("id", "path", "sourceBackend"):
        if not _is_non_empty_string(entry.get(field_name)):
            reasons.append(f"{prefix}.{field_name} must be a string")
    path = entry.get("path")
    path_is_valid = False
    if _is_non_empty_string(path):
        if not _is_repository_relative_report_path(path):
            reasons.append(f"{prefix}.path must be repository-relative")
        else:
            path_is_valid = True
    reasons.extend(
        _string_list_contract_reasons(f"{prefix}.targets", entry.get("targets"))
    )
    for field_name in ("present", "discovered"):
        if not isinstance(entry.get(field_name), bool):
            reasons.append(f"{prefix}.{field_name} must be a boolean")
    for field_name in ("artifactCount", "translatedCount", "failedCount"):
        if not _is_non_negative_int(entry.get(field_name)):
            reasons.append(f"{prefix}.{field_name} must be a non-negative integer")
    if (
        root_path is not None
        and path_is_valid
        and isinstance(entry.get("present"), bool)
    ):
        expected_present = (root_path / path).exists()
        if entry.get("present") != expected_present:
            reasons.append(f"{prefix}.present must match project.root")
    if (
        declared_units_by_path is not None
        and path_is_valid
        and isinstance(entry.get("discovered"), bool)
    ):
        expected_discovered = path in declared_units_by_path
        if entry.get("discovered") != expected_discovered:
            reasons.append(f"{prefix}.discovered must match units")
    if declared_units_by_path is not None and path_is_valid:
        declared_unit = declared_units_by_path.get(path)
        source_backend = entry.get("sourceBackend")
        if declared_unit is not None and _is_non_empty_string(source_backend):
            unit_index, unit = declared_unit
            if source_backend != unit.get("sourceBackend"):
                reasons.append(
                    f"{prefix}.sourceBackend must match "
                    f"units[{unit_index}].sourceBackend"
                )
    if artifacts is not None and not reasons:
        entry_artifacts = _external_corpus_entry_artifacts(entry, artifacts)
        expected_counts = {
            "artifactCount": len(entry_artifacts),
            "translatedCount": sum(
                1
                for artifact in entry_artifacts
                if artifact.get("status") == "translated"
            ),
            "failedCount": sum(
                1 for artifact in entry_artifacts if artifact.get("status") == "failed"
            ),
        }
        for field_name, expected in expected_counts.items():
            if entry.get(field_name) != expected:
                reasons.append(f"{prefix}.{field_name} must match report.artifacts")
    for field_name in ("repository", "commit", "sourceUrl"):
        if field_name in entry and not _is_non_empty_string(entry.get(field_name)):
            reasons.append(f"{prefix}.{field_name} must be a string")
    return reasons


def _external_corpus_summary_contract_reasons(
    summary: Any, entries: Sequence[Any], artifacts: Any
) -> list[str]:
    if not isinstance(summary, Mapping):
        return ["externalCorpus.summary must be an object"]

    entry_records = [entry for entry in entries if isinstance(entry, Mapping)]
    reasons = []
    expected_counts = {
        "entryCount": len(entries),
        "presentCount": sum(1 for entry in entry_records if entry.get("present")),
        "missingCount": sum(1 for entry in entry_records if not entry.get("present")),
        "discoveredUnitCount": sum(
            1 for entry in entry_records if entry.get("discovered")
        ),
        "undiscoveredPresentCount": sum(
            1
            for entry in entry_records
            if entry.get("present") and not entry.get("discovered")
        ),
    }
    for field_name, expected in expected_counts.items():
        reasons.extend(
            _count_field_contract_reasons(
                f"externalCorpus.summary.{field_name}",
                summary.get(field_name),
                expected,
                "externalCorpus.entries",
            )
        )
    reasons.extend(
        _mapping_field_contract_reasons(
            "externalCorpus.summary.entriesBySourceBackend",
            summary.get("entriesBySourceBackend"),
            _external_corpus_entries_by_source_backend(entry_records),
            "externalCorpus.entries",
        )
    )
    reasons.extend(
        _mapping_field_contract_reasons(
            "externalCorpus.summary.entriesByTarget",
            summary.get("entriesByTarget"),
            _external_corpus_entries_by_target(entry_records),
            "externalCorpus.entries",
        )
    )
    if isinstance(artifacts, list):
        artifact_records = _payload_artifact_records(artifacts)
        reasons.extend(
            _mapping_field_contract_reasons(
                "externalCorpus.summary.artifactsByTarget",
                summary.get("artifactsByTarget"),
                _artifact_counts_by_target(
                    _external_corpus_artifacts(entry_records, artifact_records)
                ),
                "externalCorpus.entries",
            )
        )
    elif not isinstance(summary.get("artifactsByTarget"), Mapping):
        reasons.append("externalCorpus.summary.artifactsByTarget must be an object")
    return reasons


def _external_corpus_contract_reasons(
    report: Mapping[str, Any],
    artifacts: Any,
    *,
    require_external_corpus: bool,
    root_path: Path | None = None,
    declared_units_by_path: Mapping[str, UnitDeclaration] | None = None,
) -> list[str]:
    if "externalCorpus" not in report:
        return ["externalCorpus must be an object"] if require_external_corpus else []

    external_corpus = report.get("externalCorpus")
    if not isinstance(external_corpus, Mapping):
        return ["externalCorpus must be an object"]

    reasons = []
    if external_corpus.get("schemaVersion") != EXTERNAL_CORPUS_SCHEMA_VERSION:
        reasons.append(
            f"externalCorpus.schemaVersion must be {EXTERNAL_CORPUS_SCHEMA_VERSION}"
        )
    if not _is_non_empty_string(external_corpus.get("manifest")):
        reasons.append("externalCorpus.manifest must be a string")
    project = report.get("project")
    project_manifest = (
        project.get("externalCorpusManifest") if isinstance(project, Mapping) else None
    )
    external_manifest = external_corpus.get("manifest")
    if _is_non_empty_string(project_manifest) and _is_non_empty_string(
        external_manifest
    ):
        if external_manifest != project_manifest:
            reasons.append(
                "externalCorpus.manifest must match project.externalCorpusManifest"
            )
    status = external_corpus.get("status")
    if status not in {
        "ok",
        "missing",
        "invalid",
        "outside-project",
    }:
        reasons.append(
            "externalCorpus.status must be ok, missing, invalid, or outside-project"
        )
    for field_name in ("name", "description"):
        if field_name in external_corpus and not _is_non_empty_string(
            external_corpus.get(field_name)
        ):
            reasons.append(f"externalCorpus.{field_name} must be a string")

    entries = external_corpus.get("entries")
    if not isinstance(entries, list):
        reasons.append("externalCorpus.entries must be a list")
        entries = []
    else:
        if status in {"missing", "invalid", "outside-project"} and entries:
            reasons.append("externalCorpus.entries must be empty when status is not ok")
        artifact_records = (
            _payload_artifact_records(artifacts)
            if isinstance(artifacts, list)
            else None
        )
        for index, entry in enumerate(entries):
            reasons.extend(
                _external_corpus_entry_contract_reasons(
                    index,
                    entry,
                    artifacts=artifact_records,
                    root_path=root_path,
                    declared_units_by_path=declared_units_by_path,
                )
            )
        reasons.extend(
            _duplicate_field_contract_reasons("externalCorpus.entries", entries, "id")
        )
        reasons.extend(
            _duplicate_path_contract_reasons("externalCorpus.entries", entries)
        )
    reasons.extend(
        _external_corpus_summary_contract_reasons(
            external_corpus.get("summary"), entries, artifacts
        )
    )
    return reasons


def _migration_contract_reasons(
    report: Mapping[str, Any], *, require_migration: bool
) -> list[str]:
    if "migration" not in report:
        return ["migration must be an object"] if require_migration else []

    migration = report.get("migration")
    if not isinstance(migration, Mapping):
        return ["migration must be an object"]

    reasons = []
    project = report.get("project")
    project_targets = project.get("targets", []) if isinstance(project, Mapping) else []
    project_targets_valid = isinstance(project_targets, list) and all(
        _is_non_empty_string(target) for target in project_targets
    )
    declared_targets = (
        set(_normalized_targets(project_targets)) if project_targets_valid else set()
    )
    if migration.get("scope") != REPORT_MIGRATION_SCOPE:
        reasons.append(f"migration.scope must be {REPORT_MIGRATION_SCOPE}")
    non_goal_reasons = _string_list_contract_reasons(
        "migration.nonGoals", migration.get("nonGoals")
    )
    reasons.extend(non_goal_reasons)
    if not non_goal_reasons and list(migration["nonGoals"]) != list(
        REPORT_MIGRATION_NON_GOALS
    ):
        reasons.append("migration.nonGoals must match documented report non-goals")

    actions = migration.get("actions")
    if not isinstance(actions, list):
        reasons.append("migration.actions must be a list")
    else:
        for index, action in enumerate(actions):
            prefix = f"migration.actions[{index}]"
            if not isinstance(action, Mapping):
                reasons.append(f"{prefix} must be an object")
                continue
            kind = action.get("kind")
            if not _is_non_empty_string(kind):
                reasons.append(f"{prefix}.kind must be a string")
            elif kind not in REPORT_MIGRATION_ACTION_KINDS:
                reasons.append(
                    "{}.kind must be one of {}".format(
                        prefix, ", ".join(REPORT_MIGRATION_ACTION_KINDS)
                    )
                )
            if not _is_non_empty_string(action.get("message")):
                reasons.append(f"{prefix}.message must be a string")
            severity = action.get("severity")
            if severity not in {"note", "warning", "error"}:
                reasons.append(f"{prefix}.severity must be note, warning, or error")
            target_reasons = _string_list_contract_reasons(
                f"{prefix}.targets", action.get("targets")
            )
            reasons.extend(target_reasons)
            action_targets = action.get("targets")
            if not target_reasons:
                normalized_action_targets = _normalized_targets(action_targets)
                if normalized_action_targets != action_targets:
                    reasons.append(
                        f"{prefix}.targets must use normalized backend names "
                        "without duplicates"
                    )
                elif project_targets_valid:
                    for target in normalized_action_targets:
                        if target not in declared_targets:
                            reasons.append(
                                f"{prefix}.targets must be listed in project.targets"
                            )
                            break
    return reasons


def _summary_contract_reasons(
    report: Mapping[str, Any],
    project: Any,
    units: Any,
    skipped: Any,
    artifacts: Any,
    diagnostics: Any,
) -> list[str]:
    if "summary" not in report:
        return []

    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        return ["summary must be an object"]

    reasons = []
    if isinstance(units, list):
        reasons.extend(
            _count_field_contract_reasons(
                "summary.unitCount",
                summary.get("unitCount"),
                len(units),
                "units length",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.unitsBySourceBackend",
                summary.get("unitsBySourceBackend"),
                _payload_unit_counts_by_source_backend(units),
                "units",
            )
        )
    if isinstance(skipped, list):
        reasons.extend(
            _count_field_contract_reasons(
                "summary.skippedCount",
                summary.get("skippedCount"),
                len(skipped),
                "skipped length",
            )
        )
    if isinstance(project, Mapping):
        targets = project.get("targets", [])
        if isinstance(targets, list):
            reasons.extend(
                _count_field_contract_reasons(
                    "summary.targetCount",
                    summary.get("targetCount"),
                    len(targets),
                    "project.targets length",
                )
            )
    if isinstance(artifacts, list):
        artifact_records = _payload_artifact_records(artifacts)
        translated_count = sum(
            1 for artifact in artifact_records if artifact.get("status") == "translated"
        )
        failed_count = sum(
            1 for artifact in artifact_records if artifact.get("status") == "failed"
        )
        reasons.extend(
            _count_field_contract_reasons(
                "summary.artifactCount",
                summary.get("artifactCount"),
                len(artifacts),
                "artifacts length",
            )
        )
        reasons.extend(
            _count_field_contract_reasons(
                "summary.translatedCount",
                summary.get("translatedCount"),
                translated_count,
                "translated artifacts",
            )
        )
        reasons.extend(
            _count_field_contract_reasons(
                "summary.failedCount",
                summary.get("failedCount"),
                failed_count,
                "failed artifacts",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.artifactsByTarget",
                summary.get("artifactsByTarget"),
                _artifact_counts_by_target(artifact_records),
                "artifacts",
            )
        )
        source_map_counts = _source_map_counts(artifact_records)
        reasons.extend(
            _count_field_contract_reasons(
                "summary.sourceMapCount",
                summary.get("sourceMapCount"),
                source_map_counts["sourceMapCount"],
                "artifact source maps",
            )
        )
        reasons.extend(
            _count_field_contract_reasons(
                "summary.fineGrainedSourceMapCount",
                summary.get("fineGrainedSourceMapCount"),
                source_map_counts["fineGrainedSourceMapCount"],
                "artifact source maps",
            )
        )
    if isinstance(diagnostics, list):
        reasons.extend(
            _diagnostic_counts_contract_reasons(
                "summary.diagnosticCounts",
                summary.get("diagnosticCounts"),
                diagnostics,
            )
        )
        if "diagnosticsByCode" in summary:
            reasons.extend(
                _diagnostic_code_counts_contract_reasons(
                    "summary.diagnosticsByCode",
                    summary.get("diagnosticsByCode"),
                    diagnostics,
                )
            )
        if "missingCapabilityCounts" in summary:
            reasons.extend(
                _missing_capability_counts_contract_reasons(
                    "summary.missingCapabilityCounts",
                    summary.get("missingCapabilityCounts"),
                    diagnostics,
                )
            )
    return reasons


def _source_map_span_reasons(prefix: str, value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{prefix} must be an object"]

    reasons = []
    if not _is_non_empty_string(value.get("file")):
        reasons.append(f"{prefix}.file must be a string")
    for field_name in (
        "line",
        "column",
        "offset",
        "length",
        "endLine",
        "endColumn",
        "endOffset",
    ):
        if not _is_non_negative_int(value.get(field_name)):
            reasons.append(f"{prefix}.{field_name} must be a non-negative integer")
    if reasons:
        return reasons

    reasons.extend(_span_consistency_contract_reasons(prefix, value))
    return reasons


def _source_map_anchor_reasons(
    index: int, source_map: Mapping[str, Any], artifact: Mapping[str, Any]
) -> list[str]:
    prefix = f"artifacts[{index}].sourceMap"
    reasons = []
    source = source_map.get("source")
    generated = source_map.get("generated")

    if isinstance(source, Mapping) and _is_non_empty_string(source.get("file")):
        if _is_non_empty_string(artifact.get("source")):
            if source.get("file") != artifact.get("source"):
                reasons.append(
                    f"{prefix}.source.file must match artifacts[{index}].source"
                )
    if isinstance(generated, Mapping) and _is_non_empty_string(generated.get("file")):
        if _is_non_empty_string(artifact.get("path")):
            if generated.get("file") != artifact.get("path"):
                reasons.append(
                    f"{prefix}.generated.file must match artifacts[{index}].path"
                )

    mappings = source_map.get("mappings")
    if not isinstance(mappings, list):
        return reasons

    for mapping_index, mapping in enumerate(mappings):
        mapping_prefix = f"{prefix}.mappings[{mapping_index}]"
        if not isinstance(mapping, Mapping):
            continue
        mapping_source = mapping.get("source")
        if isinstance(source, Mapping) and isinstance(mapping_source, Mapping):
            if dict(mapping_source) != dict(source):
                reasons.append(f"{mapping_prefix}.source must match {prefix}.source")
        mapping_generated = mapping.get("generated")
        if isinstance(generated, Mapping) and isinstance(mapping_generated, Mapping):
            if dict(mapping_generated) != dict(generated):
                reasons.append(
                    f"{mapping_prefix}.generated must match {prefix}.generated"
                )
    return reasons


def _source_map_contract_reasons(
    index: int, artifact: Mapping[str, Any], *, required: bool = False
) -> list[str]:
    if "sourceMap" not in artifact:
        if required:
            return [f"artifacts[{index}].sourceMap must be an object"]
        return []

    prefix = f"artifacts[{index}].sourceMap"
    source_map = artifact.get("sourceMap")
    if not isinstance(source_map, Mapping):
        return [f"{prefix} must be an object"]

    reasons = []
    if source_map.get("schemaVersion") != 1:
        reasons.append(f"{prefix}.schemaVersion must be 1")
    if source_map.get("kind") != "crosstl-artifact-source-map":
        reasons.append(f"{prefix}.kind must be crosstl-artifact-source-map")
    if source_map.get("mappingGranularity") != "file":
        reasons.append(f"{prefix}.mappingGranularity must be file")

    target = source_map.get("target")
    if not _is_non_empty_string(target):
        reasons.append(f"{prefix}.target must be a string")
    elif _is_non_empty_string(artifact.get("target")) and target != artifact["target"]:
        reasons.append(f"{prefix}.target must match artifacts[{index}].target")

    reasons.extend(
        _source_map_span_reasons(f"{prefix}.source", source_map.get("source"))
    )
    reasons.extend(
        _source_map_span_reasons(f"{prefix}.generated", source_map.get("generated"))
    )

    mappings = source_map.get("mappings")
    if not isinstance(mappings, list):
        reasons.append(f"{prefix}.mappings must be a list")
    elif not mappings:
        reasons.append(f"{prefix}.mappings must not be empty")
    elif len(mappings) != 1:
        reasons.append(f"{prefix}.mappings must contain one file-level mapping")
    else:
        for mapping_index, mapping in enumerate(mappings):
            mapping_prefix = f"{prefix}.mappings[{mapping_index}]"
            if not isinstance(mapping, Mapping):
                reasons.append(f"{mapping_prefix} must be an object")
                continue
            reasons.extend(
                _source_map_span_reasons(
                    f"{mapping_prefix}.source", mapping.get("source")
                )
            )
            reasons.extend(
                _source_map_span_reasons(
                    f"{mapping_prefix}.generated", mapping.get("generated")
                )
            )

    reasons.extend(_source_map_anchor_reasons(index, source_map, artifact))
    return reasons


def _report_contract_diagnostics(path: Path, report: Any) -> list[ProjectDiagnostic]:
    if not isinstance(report, Mapping):
        return [_invalid_report_diagnostic(path, ["expected a JSON object"])]

    reasons = []
    has_summary = "summary" in report
    if report.get("schemaVersion") != REPORT_SCHEMA_VERSION:
        reasons.append(f"expected schemaVersion {REPORT_SCHEMA_VERSION}")
    if report.get("kind") != REPORT_KIND:
        reasons.append(f"expected kind {REPORT_KIND}")
    if has_summary or "generatedAt" in report:
        if not _is_non_negative_int(report.get("generatedAt")):
            reasons.append("generatedAt must be a non-negative integer")
    reasons.extend(_generator_contract_reasons(report, require_generator=has_summary))

    project = report.get("project")
    root_path: Path | None = None
    project_output_path: Path | None = None
    if not isinstance(project, Mapping):
        reasons.append("missing project object")
    else:
        root = project.get("root")
        if not _is_non_empty_string(root):
            reasons.append("missing project.root")
        else:
            root_path = Path(root)
            if not root_path.is_absolute():
                reasons.append("project.root must be an absolute path")
            elif not root_path.exists():
                reasons.append("project.root does not exist")
            elif not root_path.is_dir():
                reasons.append("project.root must be a directory")
        targets = project.get("targets", [])
        target_reasons = _target_list_contract_reasons(
            "project.targets", targets, require_canonical=has_summary
        )
        reasons.extend(target_reasons)
        output_dir = project.get("outputDir", DEFAULT_OUTPUT_DIR)
        if not _is_non_empty_string(output_dir):
            reasons.append("project.outputDir must be a string")
        elif root_path is not None and root_path.is_absolute() and root_path.is_dir():
            output_path = Path(output_dir)
            if not output_path.is_absolute():
                output_path = root_path / output_path
            if not _is_relative_to(output_path, root_path):
                reasons.append("project.outputDir must resolve inside project.root")
            else:
                project_output_path = output_path.resolve()
        reasons.extend(
            _project_metadata_contract_reasons(
                project, require_full_metadata=has_summary
            )
        )

    units = report.get("units", [])
    if has_summary and "units" not in report:
        reasons.append("units must be a list")
    if has_summary or "units" in report:
        if not isinstance(units, list):
            reasons.append("units must be a list")
        else:
            for index, unit in enumerate(units):
                reasons.extend(_unit_contract_reasons(index, unit))
            reasons.extend(_duplicate_path_contract_reasons("units", units))

    skipped = report.get("skipped", [])
    if has_summary and "skipped" not in report:
        reasons.append("skipped must be a list")
    if has_summary or "skipped" in report:
        if not isinstance(skipped, list):
            reasons.append("skipped must be a list")
        else:
            for index, skipped_record in enumerate(skipped):
                reasons.extend(_skipped_contract_reasons(index, skipped_record))
            reasons.extend(_duplicate_path_contract_reasons("skipped", skipped))
            if isinstance(units, list):
                reasons.extend(_unit_skipped_path_contract_reasons(units, skipped))

    project_targets = project.get("targets", []) if isinstance(project, Mapping) else []
    project_targets_valid = isinstance(project_targets, list) and all(
        _is_non_empty_string(target) for target in project_targets
    )
    declared_targets = (
        set(_normalized_targets(project_targets)) if project_targets_valid else set()
    )

    artifacts = report.get("artifacts", [])
    if has_summary and "artifacts" not in report:
        reasons.append("artifacts must be a list")
    if not isinstance(artifacts, list):
        reasons.append("artifacts must be a list")
    else:
        declared_units_by_path = _declared_units_by_path(
            units, require_full_metadata=has_summary
        )
        project_variants = (
            project.get("variants", {}) if isinstance(project, Mapping) else {}
        )
        project_variants_valid = isinstance(project_variants, Mapping) and all(
            _is_non_empty_string(name) for name in project_variants
        )
        declared_variants = set(project_variants) if project_variants_valid else set()
        artifact_identities: dict[ArtifactIdentity, int] = {}
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, Mapping):
                reasons.append(f"artifacts[{index}] must be an object")
                continue
            for field_name in ("source", "target", "path", "status"):
                if not _is_non_empty_string(artifact.get(field_name)):
                    reasons.append(f"artifacts[{index}].{field_name} must be a string")
            source = artifact.get("source")
            if _is_non_empty_string(source) and not _is_repository_relative_report_path(
                source
            ):
                reasons.append(f"artifacts[{index}].source must be repository-relative")
            elif (
                declared_units_by_path is not None
                and _is_non_empty_string(source)
                and source not in declared_units_by_path
            ):
                reasons.append(f"artifacts[{index}].source must be listed in units")
            path = artifact.get("path")
            target = artifact.get("target")
            variant = artifact.get("variant")
            if _is_non_empty_string(path) and not _is_repository_relative_report_path(
                path
            ):
                reasons.append(f"artifacts[{index}].path must be repository-relative")
            elif (
                root_path is not None
                and project_output_path is not None
                and _is_non_empty_string(path)
                and not _is_relative_to(root_path / path, project_output_path)
            ):
                reasons.append(
                    f"artifacts[{index}].path must be under project.outputDir"
                )
            elif (
                root_path is not None
                and project_output_path is not None
                and _is_non_empty_string(path)
                and _is_non_empty_string(target)
            ):
                expected_output_base = (
                    project_output_path / _normalized_targets([target])[0]
                )
                if _is_non_empty_string(variant):
                    expected_output_base = (
                        expected_output_base / _variant_output_segment(variant)
                    )
                if not _is_relative_to(root_path / path, expected_output_base):
                    reasons.append(
                        f"artifacts[{index}].path must be under "
                        "project.outputDir target/variant directory"
                    )
                elif Path(path).suffix.lower() != _artifact_target_extension(target):
                    reasons.append(
                        f"artifacts[{index}].path suffix must match "
                        f"artifacts[{index}].target"
                    )
                elif _is_non_empty_string(
                    source
                ) and _is_repository_relative_report_path(source):
                    expected_relative = Path(source.replace("\\", "/")).with_suffix(
                        _artifact_target_extension(target)
                    )
                    expected_path = (expected_output_base / expected_relative).resolve()
                    if (root_path / path).resolve() != expected_path:
                        reasons.append(
                            f"artifacts[{index}].path must match "
                            "project.outputDir target/variant directory plus "
                            f"artifacts[{index}].source"
                        )
            source_backend = artifact.get("sourceBackend")
            if has_summary or "sourceBackend" in artifact:
                if not _is_non_empty_string(source_backend):
                    reasons.append(f"artifacts[{index}].sourceBackend must be a string")
                elif (
                    declared_units_by_path is not None
                    and _is_non_empty_string(source)
                    and source in declared_units_by_path
                ):
                    unit_index, declared_unit = declared_units_by_path[source]
                    if source_backend != declared_unit.get("sourceBackend"):
                        reasons.append(
                            f"artifacts[{index}].sourceBackend must match "
                            f"units[{unit_index}].sourceBackend"
                        )
            if _is_non_empty_string(target) and project_targets_valid:
                normalized_target = _normalized_targets([target])[0]
                if normalized_target not in declared_targets:
                    reasons.append(
                        f"artifacts[{index}].target must be listed in project.targets"
                    )
            status = artifact.get("status")
            if isinstance(status, str) and status not in {"translated", "failed"}:
                reasons.append(
                    f"artifacts[{index}].status must be translated or failed"
                )
            if "error" in artifact or (has_summary and status == "failed"):
                if not _is_non_empty_string(artifact.get("error")):
                    reasons.append(f"artifacts[{index}].error must be a string")
            if has_summary and status == "translated" and "error" in artifact:
                reasons.append(
                    f"artifacts[{index}].error must be omitted "
                    "for translated artifacts"
                )
            if has_summary and status == "failed":
                if "generatedHash" in artifact:
                    reasons.append(
                        f"artifacts[{index}].generatedHash must be omitted "
                        "for failed artifacts"
                    )
                if "sourceMap" in artifact:
                    reasons.append(
                        f"artifacts[{index}].sourceMap must be omitted "
                        "for failed artifacts"
                    )
            if "variant" in artifact:
                if not _is_non_empty_string(variant):
                    reasons.append(f"artifacts[{index}].variant must be a string")
                elif (
                    has_summary
                    and project_variants_valid
                    and (variant not in declared_variants)
                ):
                    reasons.append(
                        f"artifacts[{index}].variant must be listed in project.variants"
                    )
            identity = _artifact_identity(artifact)
            if identity is not None:
                previous_index = artifact_identities.get(identity)
                if previous_index is None:
                    artifact_identities[identity] = index
                else:
                    reasons.append(
                        f"artifacts[{index}] duplicates artifacts[{previous_index}] "
                        "identity"
                    )
            reasons.extend(
                _source_hash_contract_reasons(
                    index,
                    artifact,
                    required=has_summary,
                )
            )
            reasons.extend(
                _generated_hash_contract_reasons(
                    index,
                    artifact,
                    required=has_summary and status == "translated",
                )
            )
            reasons.extend(
                _provenance_contract_reasons(index, artifact, required=has_summary)
            )
            reasons.extend(
                _source_map_contract_reasons(
                    index,
                    artifact,
                    required=has_summary and status == "translated",
                )
            )

    diagnostics = report.get("diagnostics", [])
    if has_summary and "diagnostics" not in report:
        reasons.append("diagnostics must be a list")
    if not isinstance(diagnostics, list):
        reasons.append("diagnostics must be a list")
    else:
        for index, diagnostic in enumerate(diagnostics):
            if not isinstance(diagnostic, Mapping):
                reasons.append(f"diagnostics[{index}] must be an object")
                continue
            severity = diagnostic.get("severity")
            if severity not in {"note", "warning", "error"}:
                reasons.append(
                    f"diagnostics[{index}].severity must be note, warning, or error"
                )
            for field_name in ("code", "message"):
                if not _is_non_empty_string(diagnostic.get(field_name)):
                    reasons.append(
                        f"diagnostics[{index}].{field_name} must be a string"
                    )
            reasons.extend(
                _diagnostic_location_contract_reasons(
                    f"diagnostics[{index}].location", diagnostic.get("location")
                )
            )
            if "target" in diagnostic and not _is_non_empty_string(
                diagnostic.get("target")
            ):
                reasons.append(f"diagnostics[{index}].target must be a string")
            elif (
                has_summary
                and "target" in diagnostic
                and project_targets_valid
                and _normalized_targets([diagnostic["target"]])[0]
                not in declared_targets
            ):
                reasons.append(
                    f"diagnostics[{index}].target must be listed in project.targets"
                )
            missing_capabilities = diagnostic.get("missingCapabilities", [])
            if not isinstance(missing_capabilities, list) or any(
                not _is_non_empty_string(capability)
                for capability in missing_capabilities
            ):
                reasons.append(
                    f"diagnostics[{index}].missingCapabilities must be a list of strings"
                )

    reasons.extend(_validation_contract_reasons(report, require_validation=has_summary))
    require_external_corpus = (
        has_summary
        and isinstance(project, Mapping)
        and project.get("externalCorpusManifest") is not None
    )
    external_corpus_root_path = (
        root_path
        if root_path is not None and root_path.is_absolute() and root_path.is_dir()
        else None
    )
    external_corpus_units_by_path = _declared_units_by_path(
        units, require_full_metadata=has_summary
    )
    reasons.extend(
        _external_corpus_contract_reasons(
            report,
            artifacts,
            require_external_corpus=require_external_corpus,
            root_path=external_corpus_root_path,
            declared_units_by_path=external_corpus_units_by_path,
        )
    )
    if "diagnosticCounts" in report and isinstance(diagnostics, list):
        reasons.extend(
            _diagnostic_counts_contract_reasons(
                "diagnosticCounts", report.get("diagnosticCounts"), diagnostics
            )
        )
    reasons.extend(
        _summary_contract_reasons(
            report, project, units, skipped, artifacts, diagnostics
        )
    )
    reasons.extend(_migration_contract_reasons(report, require_migration=has_summary))

    return [_invalid_report_diagnostic(path, reasons)] if reasons else []


def _run_toolchain_smoke(
    artifacts: Sequence[Mapping[str, Any]], root: Path
) -> list[dict[str, Any]]:
    runs = []
    for artifact in artifacts:
        if artifact.get("status") != "translated":
            continue
        target = str(artifact.get("target"))
        tools = TOOLCHAIN_BY_BACKEND.get(target, ())
        if not tools or not shutil.which(tools[0]):
            continue
        artifact_path = Path(str(artifact["path"]))
        if not artifact_path.is_absolute():
            artifact_path = root / artifact_path
        artifact_path = artifact_path.resolve()
        if not _is_relative_to(artifact_path, root):
            continue
        if not artifact_path.is_file():
            continue
        if target == "opengl":
            command = [tools[0], "--stdin"]
        elif target == "vulkan":
            command = [tools[0], str(artifact_path)]
        elif target == "directx":
            command = [tools[0], "-help"]
        elif target == "metal":
            command = [tools[0], "metal", "-v"]
        else:
            continue
        completed = subprocess.run(
            command,
            cwd=str(root),
            input=(
                artifact_path.read_text(encoding="utf-8", errors="replace")
                if command[-1] == "--stdin"
                else None
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        run = {
            "source": str(artifact.get("source", "")),
            "target": target,
            "path": str(artifact["path"]),
            "command": command,
            "returncode": completed.returncode,
            "status": "ok" if completed.returncode == 0 else "failed",
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }
        if artifact.get("variant") is not None:
            run["variant"] = artifact["variant"]
        runs.append(run)
    return runs
