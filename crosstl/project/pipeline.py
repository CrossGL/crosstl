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
from pathlib import Path
from typing import Any, Mapping, Sequence

from crosstl._crosstl import translate
from crosstl.translator.codegen import (
    get_backend_extension,
    normalize_backend_name,
)
from crosstl.translator.plugin_loader import discover_backend_plugins
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources

REPORT_KIND = "crosstl-project-portability-report"
REPORT_SCHEMA_VERSION = 1
DEFAULT_CONFIG_NAME = "crosstl.toml"
DEFAULT_OUTPUT_DIR = "crosstl-out"
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


def _source_hash(path: Path) -> dict[str, str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"algorithm": "sha256", "value": digest}


def _diagnostic_counts(diagnostics: Sequence[ProjectDiagnostic]) -> dict[str, int]:
    counts = {"note": 0, "warning": 0, "error": 0}
    for diagnostic in diagnostics:
        counts[diagnostic.severity] = counts.get(diagnostic.severity, 0) + 1
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


def _resolved_include_dirs(config: ProjectConfig) -> list[str]:
    include_dirs = []
    for include_dir in config.include_dirs:
        path = Path(include_dir)
        if not path.is_absolute():
            path = config.root / path
        include_dirs.append(str(path.resolve()))
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
        discover_backend_plugins()
        targets = []
        for target in self.targets:
            normalized = normalize_backend_name(target) or target.strip().lower()
            if normalized not in targets:
                targets.append(normalized)
        return targets

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
            list(targets) if targets is not None else self.config.normalized_targets()
        )
        return ProjectPortabilityReport(
            config=self.config,
            targets=report_targets,
            units=self.units,
            skipped=self.skipped,
            artifacts=[],
            diagnostics=self.diagnostics,
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
        return {
            "schemaVersion": REPORT_SCHEMA_VERSION,
            "kind": REPORT_KIND,
            "generator": {
                "name": "CrossTL",
                "pipeline": "project-porting",
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
                "defineCount": len(self.config.defines),
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
            },
            "units": [unit.to_json() for unit in self.units],
            "skipped": list(self.skipped),
            "artifacts": list(self.artifacts),
            "diagnosticCounts": _diagnostic_counts(self.diagnostics),
            "diagnostics": diagnostics,
            "validation": dict(self.validation),
            "migration": {
                "scope": "shader-kernel-translation",
                "nonGoals": [
                    "automatic runtime API migration",
                    "application build-system rewrites",
                    "backend framework integration",
                ],
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
    include_patterns = list(config.include_patterns)
    explicit_include_patterns = bool(include_patterns)
    if not explicit_include_patterns:
        include_patterns = [f"**/*{extension}" for extension in known_extensions]

    candidates: set[Path] = set()
    for source_root in config.source_roots:
        absolute_root = _resolved_source_root(config, source_root)
        if not _is_relative_to(absolute_root, config.root):
            continue
        if not absolute_root.exists():
            continue
        for pattern in include_patterns:
            if explicit_include_patterns:
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

    for path in _iter_scan_candidates(config):
        try:
            relative_path = _relpath(path, config.root)
        except ValueError:
            continue
        if _path_matches(relative_path, config.exclude_patterns):
            continue
        if _path_matches(relative_path, INTERNAL_EXCLUDE_PATTERNS):
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

    selected_targets = (
        list(targets) if targets is not None else config.normalized_targets()
    )
    if not selected_targets:
        selected_targets = ["cgl"]
    selected_targets = [
        normalize_backend_name(target) or target.strip().lower()
        for target in selected_targets
    ]

    scan = scan_project(config)
    diagnostics: list[ProjectDiagnostic] = list(scan.diagnostics)
    artifacts: list[dict[str, Any]] = []
    include_paths = _resolved_include_dirs(config)

    for unit in scan.units:
        for target in selected_targets:
            output_path = _artifact_path(config, unit, target)
            artifact = {
                "source": unit.relative_path,
                "sourceBackend": unit.source_backend,
                "target": target,
                "path": (
                    _relpath(output_path, config.root)
                    if _is_relative_to(output_path, config.root)
                    else str(output_path)
                ),
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
        exists = artifact_path.exists()
        artifact_checks.append(
            {
                "source": artifact["source"],
                "target": artifact["target"],
                "path": artifact["path"],
                "exists": exists,
                "status": (
                    "ok"
                    if exists and artifact.get("status") == "translated"
                    else "failed"
                ),
            }
        )
        if not exists and artifact.get("status") == "translated":
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
    report = json.loads(path.read_text(encoding="utf-8"))
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
    diagnostics = [diagnostic.to_json() for diagnostic in diagnostic_objects]
    return {
        "schemaVersion": REPORT_SCHEMA_VERSION,
        "kind": "crosstl-project-validation-report",
        "sourceReport": str(path),
        "generatedAt": int(time.time()),
        "success": not any(
            diagnostic["severity"] == "error" for diagnostic in diagnostics
        ),
        "diagnosticCounts": _diagnostic_counts(
            [
                ProjectDiagnostic(
                    severity=diagnostic["severity"],
                    code=diagnostic["code"],
                    message=diagnostic["message"],
                    location=SourceLocation(file=diagnostic["location"]["file"]),
                )
                for diagnostic in diagnostics
            ]
        ),
        "diagnostics": diagnostics,
        "validation": validation,
    }


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
