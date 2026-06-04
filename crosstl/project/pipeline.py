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
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

from crosstl._crosstl import translate
from crosstl.translator.codegen import (
    backend_names,
    get_backend_extension,
    normalize_backend_name,
)
from crosstl.translator.plugin_loader import discover_backend_plugins
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources

REPORT_KIND = "crosstl-project-portability-report"
REPORT_SCHEMA_VERSION = 1
REPORT_GENERATOR_PIPELINE = "project-porting"
REPORT_MIGRATION_SCOPE = "shader-kernel-translation"
REPORT_MIGRATION_NON_GOALS = (
    "automatic runtime API migration",
    "application build-system rewrites",
    "backend framework integration",
)
REPORT_MIGRATION_ACTION_KINDS = ("manual-runtime-integration",)
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


def _variant_defines(variants: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, value in variants.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"crosstl.toml [project.variants.{name}] must be a table")
        result[str(name)] = {str(key): str(value) for key, value in value.items()}
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
        and ".." not in parts
    )


def _is_repository_relative_report_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part and part != "."]
    return (
        not Path(path).is_absolute()
        and not PureWindowsPath(path).is_absolute()
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
    return SourceLocation(file=str(config.root))


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
    if config.variants:
        diagnostics.append(
            ProjectDiagnostic(
                severity="warning",
                code="project.config.variants-not-applied",
                message=(
                    "Project named variants are recorded in the report but variant "
                    "expansion through backend preprocessors is not implemented yet."
                ),
                location=location,
                missing_capabilities=["macro.variants"],
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
        return {
            "schemaVersion": REPORT_SCHEMA_VERSION,
            "kind": REPORT_KIND,
            "generator": {
                "name": "CrossTL",
                "pipeline": REPORT_GENERATOR_PIPELINE,
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
                "includeDirs": list(self.config.include_dirs),
                "defines": dict(sorted(self.config.defines.items())),
                "defineCount": len(self.config.defines),
                "variants": {
                    name: dict(sorted(defines.items()))
                    for name, defines in sorted(self.config.variants.items())
                },
                "variantCount": len(self.config.variants),
            },
            "summary": {
                "unitCount": len(self.units),
                "skippedCount": len(self.skipped),
                "targetCount": len(self.targets),
                "artifactCount": artifact_count,
                "translatedCount": translated_count,
                "failedCount": failed_count,
                "diagnosticCounts": _diagnostic_counts(self.diagnostics),
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

    sources = project.get("sources", {})
    if not isinstance(sources, Mapping):
        raise ValueError("crosstl.toml [project.sources] must be a table")
    defines = project.get("defines", {})
    if not isinstance(defines, Mapping):
        raise ValueError("crosstl.toml [project.defines] must be a table")
    variants = project.get("variants", {})
    if not isinstance(variants, Mapping):
        raise ValueError("crosstl.toml [project.variants] must be a table")

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
        output_dir=str(project.get("output_dir", DEFAULT_OUTPUT_DIR)),
        source_overrides={str(key): str(value) for key, value in sources.items()},
        include_dirs=_as_str_list(
            project.get("include_dirs"), field_name="project.include_dirs"
        ),
        defines={str(key): str(value) for key, value in defines.items()},
        variants=_variant_defines(variants),
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
                location=SourceLocation(file=str(config.root)),
                missing_capabilities=["repo.scan"],
            )
        )
    return ProjectScan(
        config=config, units=units, skipped=skipped, diagnostics=diagnostics
    )


def _artifact_path(
    config: ProjectConfig, unit: ProjectTranslationUnit, target: str
) -> Path:
    extension = (
        ".cgl"
        if target in {"cgl", "crossgl"}
        else get_backend_extension(target) or ".out"
    )
    relative = Path(unit.relative_path)
    return config.output_path / target / relative.with_suffix(extension)


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


def _runtime_migration_actions(
    units: Sequence[ProjectTranslationUnit], targets: Sequence[str]
) -> list[dict[str, Any]]:
    if not units or not targets:
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
            "targets": list(targets),
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
    include_paths = _resolved_include_dirs(config)
    output_dir_blocked = _has_error_diagnostic(
        diagnostics, OUTPUT_DIR_OUTSIDE_PROJECT_CODE
    )

    for unit in scan.units:
        for target in selected_targets:
            output_path = _artifact_path(config, unit, target)
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
                    defines=config.defines,
                )
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
        migration_actions=_runtime_migration_actions(scan.units, selected_targets),
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


def _validate_artifacts(
    artifacts: Sequence[Mapping[str, Any]],
    targets: Sequence[str],
    config: ProjectConfig,
) -> dict[str, Any]:
    diagnostics: list[ProjectDiagnostic] = []
    toolchains = [_tool_status(target) for target in targets]
    artifact_checks = []

    for artifact in artifacts:
        artifact_path = Path(str(artifact["path"]))
        if not artifact_path.is_absolute():
            artifact_path = config.root / artifact_path
        artifact_path = artifact_path.resolve()
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
        artifact_checks.append(
            {
                "source": artifact["source"],
                "target": artifact["target"],
                "path": artifact["path"],
                "exists": exists,
                "status": (
                    "ok"
                    if artifact_inside_project
                    and exists
                    and artifact.get("status") == "translated"
                    else "failed"
                ),
            }
        )
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

    for toolchain in toolchains:
        if toolchain["status"] == "unavailable":
            diagnostics.append(
                ProjectDiagnostic(
                    severity="warning",
                    code="project.validate.toolchain-unavailable",
                    message=f"No validation toolchain is available for target {toolchain['target']}",
                    location=SourceLocation(file=str(config.root)),
                    target=toolchain["target"],
                    missing_capabilities=["toolchain.validation"],
                )
            )

    return {
        "toolchains": toolchains,
        "artifacts": artifact_checks,
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


def _source_hash_contract_reasons(index: int, artifact: Mapping[str, Any]) -> list[str]:
    if "sourceHash" not in artifact:
        return []

    prefix = f"artifacts[{index}].sourceHash"
    source_hash = artifact.get("sourceHash")
    if not isinstance(source_hash, Mapping):
        return [f"{prefix} must be an object"]

    reasons = []
    if source_hash.get("algorithm") != "sha256":
        reasons.append(f"{prefix}.algorithm must be sha256")
    if not _is_sha256_digest(source_hash.get("value")):
        reasons.append(f"{prefix}.value must be a lowercase 64-character hex digest")
    return reasons


def _provenance_contract_reasons(index: int, artifact: Mapping[str, Any]) -> list[str]:
    if "provenance" not in artifact:
        return []

    prefix = f"artifacts[{index}].provenance"
    provenance = artifact.get("provenance")
    if not isinstance(provenance, Mapping):
        return [f"{prefix} must be an object"]

    reasons = []
    if not _is_non_empty_string(provenance.get("pipeline")):
        reasons.append(f"{prefix}.pipeline must be a string")
    intermediate = provenance.get("intermediate")
    if intermediate is not None and not _is_non_empty_string(intermediate):
        reasons.append(f"{prefix}.intermediate must be a string or null")
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


def _diagnostic_counts_contract_reasons(
    prefix: str, value: Any, diagnostics: Sequence[Any]
) -> list[str]:
    expected = _payload_diagnostic_counts(diagnostics)
    if expected is None:
        return []
    return _mapping_field_contract_reasons(prefix, value, expected, "diagnostics")


def _string_list_contract_reasons(prefix: str, value: Any) -> list[str]:
    if not isinstance(value, list) or any(
        not _is_non_empty_string(item) for item in value
    ):
        return [f"{prefix} must be a list of strings"]
    return []


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
    if not isinstance(unit.get("extension"), str):
        reasons.append(f"{prefix}.extension must be a string")
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
    return reasons


def _tool_status_contract_reasons(index: int, toolchain: Any) -> list[str]:
    prefix = f"validation.toolchains[{index}]"
    if not isinstance(toolchain, Mapping):
        return [f"{prefix} must be an object"]

    reasons = []
    if not _is_non_empty_string(toolchain.get("target")):
        reasons.append(f"{prefix}.target must be a string")
    if toolchain.get("status") not in {"available", "unavailable", "not-configured"}:
        reasons.append(
            f"{prefix}.status must be available, unavailable, or not-configured"
        )
    tools = toolchain.get("tools")
    if not isinstance(tools, list):
        reasons.append(f"{prefix}.tools must be a list")
    else:
        for tool_index, tool in enumerate(tools):
            tool_prefix = f"{prefix}.tools[{tool_index}]"
            if not isinstance(tool, Mapping):
                reasons.append(f"{tool_prefix} must be an object")
                continue
            if not _is_non_empty_string(tool.get("name")):
                reasons.append(f"{tool_prefix}.name must be a string")
            path = tool.get("path")
            if path is not None and not _is_non_empty_string(path):
                reasons.append(f"{tool_prefix}.path must be a string or null")
            if not isinstance(tool.get("available"), bool):
                reasons.append(f"{tool_prefix}.available must be a boolean")
    if "message" in toolchain and not _is_non_empty_string(toolchain.get("message")):
        reasons.append(f"{prefix}.message must be a string")
    return reasons


def _validation_artifact_contract_reasons(index: int, artifact: Any) -> list[str]:
    prefix = f"validation.artifacts[{index}]"
    if not isinstance(artifact, Mapping):
        return [f"{prefix} must be an object"]

    reasons = []
    for field_name in ("source", "target", "path"):
        if not _is_non_empty_string(artifact.get(field_name)):
            reasons.append(f"{prefix}.{field_name} must be a string")
    if not isinstance(artifact.get("exists"), bool):
        reasons.append(f"{prefix}.exists must be a boolean")
    if artifact.get("status") not in {"ok", "failed"}:
        reasons.append(f"{prefix}.status must be ok or failed")
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
    toolchains = validation.get("toolchains")
    if not isinstance(toolchains, list):
        reasons.append("validation.toolchains must be a list")
    else:
        for index, toolchain in enumerate(toolchains):
            reasons.extend(_tool_status_contract_reasons(index, toolchain))

    artifact_checks = validation.get("artifacts")
    if not isinstance(artifact_checks, list):
        reasons.append("validation.artifacts must be a list")
    else:
        for index, artifact in enumerate(artifact_checks):
            reasons.extend(_validation_artifact_contract_reasons(index, artifact))
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
    if migration.get("scope") != REPORT_MIGRATION_SCOPE:
        reasons.append(f"migration.scope must be {REPORT_MIGRATION_SCOPE}")
    reasons.extend(
        _string_list_contract_reasons("migration.nonGoals", migration.get("nonGoals"))
    )

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
            reasons.extend(
                _string_list_contract_reasons(
                    f"{prefix}.targets", action.get("targets")
                )
            )
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


def _source_map_contract_reasons(index: int, artifact: Mapping[str, Any]) -> list[str]:
    if "sourceMap" not in artifact:
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
    reasons.extend(_generator_contract_reasons(report, require_generator=has_summary))

    project = report.get("project")
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
        if not isinstance(targets, list) or any(
            not _is_non_empty_string(target) for target in targets
        ):
            reasons.append("project.targets must be a list of backend names")
        output_dir = project.get("outputDir", DEFAULT_OUTPUT_DIR)
        if not _is_non_empty_string(output_dir):
            reasons.append("project.outputDir must be a string")
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

    skipped = report.get("skipped", [])
    if has_summary and "skipped" not in report:
        reasons.append("skipped must be a list")
    if has_summary or "skipped" in report:
        if not isinstance(skipped, list):
            reasons.append("skipped must be a list")
        else:
            for index, skipped_record in enumerate(skipped):
                reasons.extend(_skipped_contract_reasons(index, skipped_record))

    artifacts = report.get("artifacts", [])
    if has_summary and "artifacts" not in report:
        reasons.append("artifacts must be a list")
    if not isinstance(artifacts, list):
        reasons.append("artifacts must be a list")
    else:
        project_targets = (
            project.get("targets", []) if isinstance(project, Mapping) else []
        )
        project_targets_valid = isinstance(project_targets, list) and all(
            _is_non_empty_string(target) for target in project_targets
        )
        declared_targets = (
            set(_normalized_targets(project_targets))
            if project_targets_valid
            else set()
        )
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, Mapping):
                reasons.append(f"artifacts[{index}] must be an object")
                continue
            for field_name in ("source", "target", "path", "status"):
                if not _is_non_empty_string(artifact.get(field_name)):
                    reasons.append(f"artifacts[{index}].{field_name} must be a string")
            if "sourceBackend" in artifact and not _is_non_empty_string(
                artifact.get("sourceBackend")
            ):
                reasons.append(f"artifacts[{index}].sourceBackend must be a string")
            target = artifact.get("target")
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
            reasons.extend(_source_hash_contract_reasons(index, artifact))
            reasons.extend(_provenance_contract_reasons(index, artifact))
            reasons.extend(_source_map_contract_reasons(index, artifact))

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
            location = diagnostic.get("location")
            if not isinstance(location, Mapping):
                reasons.append(f"diagnostics[{index}].location must be an object")
            elif not _is_non_empty_string(location.get("file")):
                reasons.append(f"diagnostics[{index}].location.file must be a string")
            if "target" in diagnostic and not _is_non_empty_string(
                diagnostic.get("target")
            ):
                reasons.append(f"diagnostics[{index}].target must be a string")
            missing_capabilities = diagnostic.get("missingCapabilities", [])
            if not isinstance(missing_capabilities, list) or any(
                not _is_non_empty_string(capability)
                for capability in missing_capabilities
            ):
                reasons.append(
                    f"diagnostics[{index}].missingCapabilities must be a list of strings"
                )

    reasons.extend(_validation_contract_reasons(report, require_validation=has_summary))
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
        runs.append(
            {
                "source": str(artifact.get("source", "")),
                "target": target,
                "path": str(artifact["path"]),
                "command": command,
                "returncode": completed.returncode,
                "status": "ok" if completed.returncode == 0 else "failed",
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }
        )
    return runs
