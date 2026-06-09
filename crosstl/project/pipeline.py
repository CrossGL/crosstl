"""Project-scale shader and GPU source porting pipeline for CrossTL."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from importlib import metadata as importlib_metadata
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

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
SOURCE_REMAP_SCHEMA_VERSION = 1
REPORT_FIELDS = frozenset(
    (
        "schemaVersion",
        "kind",
        "generatedAt",
        "generator",
        "project",
        "summary",
        "units",
        "skipped",
        "artifacts",
        "diagnosticCounts",
        "diagnostics",
        "validation",
        "migration",
        "artifactMatrix",
        "externalCorpus",
    )
)
REPORT_INSPECTION_FIELDS = frozenset(
    (
        "schemaVersion",
        "kind",
        "sourceReport",
        "sourceReportHash",
        "generatedAt",
        "success",
        "report",
        "sourceMaps",
        "artifactProvenance",
        "defineProcessing",
        "skippedSources",
        "includeDependencies",
        "includePathProcessing",
        "artifactMatrix",
        "diagnosticCount",
        "truncatedDiagnosticCount",
        "failedArtifactCount",
        "truncatedFailedArtifactCount",
        "failedArtifacts",
        "diagnostics",
        "validation",
        "migration",
        "externalCorpus",
    )
)
REPORT_GENERATOR_FIELDS = frozenset(("name", "pipeline", "packageVersion"))
REPORT_PROJECT_FIELDS = frozenset(
    (
        "root",
        "config",
        "configHash",
        "sourceRoots",
        "sourceRootCount",
        "sourceRootStatus",
        "sourceRootStatusCounts",
        "includePatterns",
        "includePatternCount",
        "excludePatterns",
        "excludePatternCount",
        "targets",
        "outputDir",
        "sourceOverrides",
        "sourceOverrideCount",
        "includeDirs",
        "includeDirCount",
        "includeDirStatus",
        "includeDirStatusCounts",
        "defines",
        "defineCount",
        "variants",
        "variantCount",
        "variantDefineCounts",
        "selectedVariants",
        "externalCorpusManifest",
    )
)
REPORT_SOURCE_ROOT_STATUS_FIELDS = frozenset(
    ("path", "resolvedPath", "status", "scanVisible")
)
REPORT_INCLUDE_DIR_STATUS_FIELDS = frozenset(
    ("path", "resolvedPath", "status", "frontendVisible")
)
REPORT_SUMMARY_FIELDS = frozenset(
    (
        "unitCount",
        "skippedCount",
        "targetCount",
        "artifactCount",
        "translatedCount",
        "failedCount",
        "diagnosticCounts",
        "diagnosticsByCode",
        "diagnosticsByTarget",
        "diagnosticsBySourceBackend",
        "diagnosticsByVariant",
        "missingCapabilityCounts",
        "unitsBySourceBackend",
        "unitsByExtension",
        "unitsBySourceOverride",
        "includeDependencyCount",
        "includeDependenciesByKind",
        "includeDependenciesByStatus",
        "includeDependenciesByResolvedFrom",
        "includeDependenciesBySourceBackend",
        "includeDependenciesBySourceBackendStatus",
        "includeDependenciesByVariant",
        "skippedByReason",
        "skippedByExtension",
        "skippedBySourceOverride",
        "artifactsBySourceBackend",
        "artifactsByVariant",
        "artifactsByTarget",
        "artifactProvenanceByPipeline",
        "artifactProvenanceByIntermediate",
        "artifactProvenanceIntermediateBySourceBackend",
        "artifactProvenanceIntermediateByTarget",
        "artifactProvenanceIntermediateByVariant",
        "sourceMapCount",
        "fineGrainedSourceMapCount",
        "sourceMapsByGranularity",
        "sourceMapsByTarget",
        "sourceMapsBySourceBackend",
        "sourceMapsByVariant",
        "sourceRemapCount",
        "sourceRemapMappingCount",
        "sourceRemapsByGranularity",
        "sourceRemapsByTarget",
        "sourceRemapsBySourceBackend",
        "sourceRemapsByVariant",
        "defineProcessingByStatus",
        "defineProcessingBySourceBackend",
        "defineProcessingByTarget",
        "defineProcessingByVariant",
        "includePathProcessingByStatus",
        "includePathProcessingBySourceBackend",
        "includePathProcessingByTarget",
        "includePathProcessingByVariant",
    )
)
REPORT_MIGRATION_FIELDS = frozenset(
    (
        "scope",
        "nonGoals",
        "actionCount",
        "actionsByKind",
        "actionsBySeverity",
        "actionsByTarget",
        "actions",
    )
)
REPORT_MIGRATION_ACTION_FIELDS = frozenset(("kind", "severity", "message", "targets"))
REPORT_ARTIFACT_MATRIX_FIELDS = frozenset(
    (
        "unitCount",
        "targetCount",
        "variantCount",
        "variantMode",
        "expectedArtifactCount",
        "emittedArtifactCount",
        "translatedCount",
        "failedCount",
        "identityCoverageAvailable",
        "missingArtifactCount",
        "extraArtifactCount",
        "complete",
        "statusByTarget",
        "statusBySourceBackend",
        "statusByVariant",
    )
)
REPORT_EXTERNAL_CORPUS_FIELDS = frozenset(
    (
        "schemaVersion",
        "manifest",
        "status",
        "entries",
        "summary",
        "name",
        "description",
    )
)
REPORT_EXTERNAL_CORPUS_ENTRY_FIELDS = frozenset(
    (
        "id",
        "path",
        "sourceBackend",
        "targets",
        "present",
        "discovered",
        "artifactCount",
        "translatedCount",
        "failedCount",
        "repository",
        "commit",
        "sourceUrl",
    )
)
REPORT_EXTERNAL_CORPUS_SUMMARY_FIELDS = frozenset(
    (
        "manifestEntryCount",
        "validEntryCount",
        "invalidEntryCount",
        "entryCount",
        "presentCount",
        "missingCount",
        "discoveredUnitCount",
        "undiscoveredPresentCount",
        "entriesBySourceBackend",
        "entriesByTarget",
        "artifactsByTarget",
    )
)
REPORT_DIAGNOSTIC_FIELDS = frozenset(
    (
        "severity",
        "code",
        "message",
        "location",
        "originalLocation",
        "target",
        "sourceBackend",
        "variant",
        "missingCapabilities",
    )
)
REPORT_ARTIFACT_FIELDS = frozenset(
    (
        "source",
        "sourceBackend",
        "target",
        "path",
        "status",
        "defines",
        "defineProcessing",
        "includePathProcessing",
        "sourceHash",
        "sourceSizeBytes",
        "provenance",
        "variant",
        "error",
        "generatedHash",
        "generatedSizeBytes",
        "sourceMap",
        "sourceRemap",
    )
)
REPORT_ARTIFACT_DEFINE_PROCESSING_FIELDS = frozenset(
    ("status", "frontend", "supportsDefines", "defineCount")
)
REPORT_ARTIFACT_INCLUDE_PATH_PROCESSING_FIELDS = frozenset(
    ("status", "frontend", "supportsIncludePaths", "includePathCount")
)
REPORT_HASH_FIELDS = frozenset(("algorithm", "value"))
REPORT_ARTIFACT_PROVENANCE_FIELDS = frozenset(("pipeline", "intermediate"))
REPORT_ARTIFACT_SOURCE_REMAP_FIELDS = frozenset(
    (
        "schemaVersion",
        "path",
        "target",
        "generatedFile",
        "mappingGranularity",
        "mappingCount",
        "sizeBytes",
        "hash",
    )
)
REPORT_UNIT_FIELDS = frozenset(
    (
        "id",
        "path",
        "sourceBackend",
        "extension",
        "sourceOverride",
        "sourceHash",
        "sourceSizeBytes",
        "includeDependencies",
    )
)
REPORT_INCLUDE_DEPENDENCY_FIELDS = frozenset(
    (
        "source",
        "include",
        "kind",
        "status",
        "line",
        "column",
        "resolvedPath",
        "resolvedHash",
        "resolvedSizeBytes",
        "resolvedFrom",
        "resolvedFromDefine",
        "variant",
    )
)
REPORT_INCLUDE_DEPENDENCY_SCAN_IDENTITY_FIELDS = (
    "source",
    "include",
    "kind",
    "status",
    "line",
    "column",
    "resolvedPath",
    "resolvedFrom",
    "resolvedFromDefine",
    "variant",
)
REPORT_SKIPPED_FIELDS = frozenset(("path", "reason", "sourceOverride"))
VALIDATION_REPORT_KIND = "crosstl-project-validation-report"
VALIDATION_REPORT_FIELDS = frozenset(
    (
        "schemaVersion",
        "kind",
        "sourceReport",
        "sourceReportHash",
        "generatedAt",
        "success",
        "project",
        "diagnosticCounts",
        "diagnosticsByCode",
        "diagnosticsByTarget",
        "diagnosticsBySourceBackend",
        "diagnosticsByVariant",
        "missingCapabilityCounts",
        "artifactStatusByTarget",
        "artifactStatusBySourceBackend",
        "artifactStatusByVariant",
        "toolchainStatusCounts",
        "toolchainRunStatusCounts",
        "toolchainRunStatusByTarget",
        "toolchainRunStatusBySourceBackend",
        "toolchainRunStatusByCheckKind",
        "toolchainRunStatusByTool",
        "toolchainRunStatusByVariant",
        "sourceHashStatusCounts",
        "sourceSizeStatusCounts",
        "generatedHashStatusCounts",
        "generatedSizeStatusCounts",
        "sourceMapStatusCounts",
        "sourceRemapStatusCounts",
        "diagnostics",
        "validation",
    )
)
VALIDATION_FIELDS = frozenset(("toolchains", "artifacts", "summary", "toolchainRuns"))
VALIDATION_TOOLCHAIN_FIELDS = frozenset(("target", "status", "tools", "message"))
VALIDATION_TOOL_FIELDS = frozenset(("name", "path", "available"))
VALIDATION_ARTIFACT_FIELDS = frozenset(
    (
        "source",
        "target",
        "path",
        "exists",
        "status",
        "variant",
        "sourceBackend",
        "sourceHashStatus",
        "sourceSizeStatus",
        "generatedHashStatus",
        "generatedSizeStatus",
        "sourceMapStatus",
        "sourceRemapStatus",
    )
)
VALIDATION_SUMMARY_FIELDS = frozenset(
    (
        "artifactCount",
        "okCount",
        "failedCount",
        "sourceHashStatusCounts",
        "sourceSizeStatusCounts",
        "generatedHashStatusCounts",
        "generatedSizeStatusCounts",
        "sourceMapStatusCounts",
        "sourceRemapStatusCounts",
    )
)
VALIDATION_TOOLCHAIN_RUN_FIELDS = frozenset(
    (
        "source",
        "target",
        "path",
        "variant",
        "sourceBackend",
        "command",
        "checkKind",
        "returncode",
        "status",
        "stdout",
        "stderr",
    )
)
VALIDATION_TOOLCHAIN_RUN_CHECK_KINDS = frozenset(("artifact", "tool-availability"))
SOURCE_MAP_SPAN_FIELDS = (
    "file",
    "line",
    "column",
    "offset",
    "length",
    "endLine",
    "endColumn",
    "endOffset",
)
SOURCE_MAP_PAYLOAD_FIELDS = frozenset(
    (
        "schemaVersion",
        "kind",
        "mappingGranularity",
        "target",
        "source",
        "generated",
        "mappings",
    )
)
SOURCE_MAP_MAPPING_FIELDS = frozenset(("source", "generated"))
SOURCE_MAP_SPAN_FIELD_SET = frozenset(SOURCE_MAP_SPAN_FIELDS)
SOURCE_MAP_GRANULARITIES = ("file", "line", "statement", "token")
COMPILER_SOURCE_REMAP_PAYLOAD_FIELDS = frozenset(
    ("schemaVersion", "generatedFile", "mappings")
)
COMPILER_SOURCE_REMAP_MAPPING_FIELDS = frozenset(("generated", "original"))
COMPILER_SOURCE_REMAP_SPAN_FIELDS = frozenset(SOURCE_MAP_SPAN_FIELDS)
REPORT_GENERATOR_PIPELINE = "project-porting"
REPORT_ARTIFACT_PROVENANCE_PIPELINES = ("single-file-translate",)
REPORT_ARTIFACT_PROVENANCE_INTERMEDIATES = ("crossgl",)
REPORT_MIGRATION_SCOPE = "shader-kernel-translation"
REPORT_MIGRATION_NON_GOALS = (
    "automatic runtime API migration",
    "application build-system rewrites",
    "backend framework integration",
)
REPORT_MIGRATION_ACTION_KINDS = (
    "manual-runtime-integration",
    "manual-include-resolution",
)
REPORT_PACKAGE_NAME = "crosstl"
UNKNOWN_PACKAGE_VERSION = "unknown"
EXTERNAL_CORPUS_SCHEMA_VERSION = 1
ARTIFACT_MATRIX_INSPECTION_SAMPLE_LIMIT = 20
ARTIFACT_PROVENANCE_INSPECTION_SAMPLE_LIMIT = 20
DEFINE_PROCESSING_INSPECTION_SAMPLE_LIMIT = 20
EXTERNAL_CORPUS_INSPECTION_SAMPLE_LIMIT = 20
INCLUDE_DEPENDENCY_INSPECTION_SAMPLE_LIMIT = 20
INCLUDE_PATH_PROCESSING_INSPECTION_SAMPLE_LIMIT = 20
MIGRATION_ACTION_INSPECTION_SAMPLE_LIMIT = 20
SKIPPED_SOURCE_INSPECTION_SAMPLE_LIMIT = 20
SOURCE_MAP_INSPECTION_SAMPLE_LIMIT = 20
VALIDATION_INSPECTION_SAMPLE_LIMIT = 20
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
TOOLCHAIN_AVAILABILITY_COMMANDS = {
    "cuda": ("nvcc", "--version"),
    "directx": ("dxc", "-help"),
    "hip": ("hipcc", "--version"),
    "metal": ("xcrun", "metal", "-v"),
    "mojo": ("mojo", "--version"),
    "rust": ("rustc", "--version"),
    "slang": ("slangc", "--version"),
}
TOOLCHAIN_SMOKE_TIMEOUT_SECONDS = 30
TOOLCHAIN_TIMEOUT_RETURNCODE = 124

CROSSL_TARGETS = {"cgl", "crossgl"}
SHA256_HEX_LENGTH = 64
GIT_COMMIT_HEX_LENGTH = 40
LOWERCASE_HEX_DIGITS = frozenset("0123456789abcdef")
VALIDATION_ARTIFACT_STATUSES = frozenset(("ok", "failed"))
SOURCE_HASH_VALIDATION_STATUSES = frozenset(
    ("ok", "missing", "mismatch", "not-recorded", "outside-project")
)
SOURCE_SIZE_VALIDATION_STATUSES = frozenset(
    ("ok", "missing", "mismatch", "not-recorded", "outside-project")
)
GENERATED_HASH_VALIDATION_STATUSES = frozenset(
    ("ok", "missing", "mismatch", "not-applicable", "not-recorded", "outside-project")
)
GENERATED_SIZE_VALIDATION_STATUSES = frozenset(
    ("ok", "missing", "mismatch", "not-applicable", "not-recorded", "outside-project")
)
SOURCE_MAP_VALIDATION_STATUSES = frozenset(
    ("ok", "mismatch", "not-applicable", "not-checked", "not-recorded")
)
SOURCE_REMAP_VALIDATION_STATUSES = frozenset(
    (
        "ok",
        "hash-mismatch",
        "invalid",
        "mismatch",
        "missing",
        "not-applicable",
        "not-recorded",
        "outside-project",
    )
)
SOURCE_ROOT_STATUSES = frozenset(
    ("active", "missing", "not-directory", "outside-project")
)
INCLUDE_DIR_STATUSES = frozenset(
    ("active", "missing", "not-directory", "outside-project")
)
INCLUDE_DEPENDENCY_KINDS = frozenset(("dynamic", "local", "system"))
INCLUDE_DEPENDENCY_STATUSES = frozenset(
    ("dynamic", "missing", "outside-project", "resolved", "system")
)
INCLUDE_DEPENDENCY_RESOLUTION_SOURCES = frozenset(("include-dir", "source"))
DEFINE_PROCESSING_STATUSES = frozenset(("forwarded", "not-requested", "not-supported"))
INCLUDE_PATH_PROCESSING_STATUSES = frozenset(
    ("forwarded", "not-requested", "not-supported")
)
VALIDATION_TOOLCHAIN_RUN_STATUSES = frozenset(("ok", "failed"))
VARIANT_OUTPUT_SAFE_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)
INCLUDE_DIRECTIVE_RE = re.compile(r"^\s*#\s*include\s+(?P<body>.+?)\s*$")
INCLUDE_CONDITIONAL_DIRECTIVE_RE = re.compile(
    r"^\s*#\s*(?P<directive>if|ifdef|ifndef|elif|else|endif)\b(?P<body>.*?)\s*$"
)
INCLUDE_LITERAL_RE = re.compile(r'^(?P<open>["<])(?P<path>[^">]+)(?P<close>[">])')
INCLUDE_CONDITION_TOKEN_RE = re.compile(
    r"defined|&&|\|\||==|!=|<=|>=|<|>|!|\(|\)|[+-]?[0-9]+|" r"[A-Za-z_][A-Za-z0-9_]*|\S"
)
DEFINE_DIRECTIVE_RE = re.compile(
    r"^\s*#\s*(?P<directive>define|undef)\s+" r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b"
)
PREPROCESSOR_DIAGNOSTIC_DIRECTIVE_RE = re.compile(
    r"^\s*#\s*(?P<directive>error|warning)\b(?P<body>.*?)\s*$"
)
REPORT_PATH_BARE_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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


def _mapping_key_path(prefix: str, key: str) -> str:
    if REPORT_PATH_BARE_KEY_RE.match(key):
        return f"{prefix}.{key}"
    return f"{prefix}[{json.dumps(key)}]"


def _as_str_list(value: Any, *, field_name: str) -> list[str]:
    def append_item(result: list[str], item: Any) -> None:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} entries must be strings")
        if not item.strip():
            raise ValueError(f"{field_name} entries must be non-empty strings")
        result.append(item)

    if value is None:
        return []
    if isinstance(value, str):
        result: list[str] = []
        append_item(result, value)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        result: list[str] = []
        for item in value:
            append_item(result, item)
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
        if not isinstance(key, str) or not key.strip() or not isinstance(item, str):
            raise ValueError(
                f"{field_name} entries must map non-empty strings to strings"
            )
        result[key] = item
    return result


def _variant_defines(variants: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, value in variants.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "crosstl.toml [project.variants] keys must be non-empty strings"
            )
        variant_path = _mapping_key_path("project.variants", name)
        if not isinstance(value, Mapping):
            raise ValueError(f"crosstl.toml [{variant_path}] must be a table")
        result[name] = _as_str_mapping(
            value, field_name=f"crosstl.toml [{variant_path}]"
        )
    return result


def _variant_define_counts(variants: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, defines in variants.items():
        if not _is_non_empty_string(name) or not isinstance(defines, Mapping):
            continue
        counts[name] = len(defines)
    return dict(sorted(counts.items()))


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


def _is_absolute_or_windows_drive_path(path: str) -> bool:
    return (
        Path(path).is_absolute()
        or PureWindowsPath(path).is_absolute()
        or bool(PureWindowsPath(path).drive)
    )


def _normalize_project_relative_path(path: str) -> str:
    if _is_absolute_or_windows_drive_path(path):
        return path
    return path.replace("\\", "/")


def _normalize_project_relative_paths(paths: Sequence[str]) -> list[str]:
    return [_normalize_project_relative_path(path) for path in paths]


def _normalize_project_relative_path_mapping(
    paths: Mapping[str, str],
) -> dict[str, str]:
    return {
        _normalize_project_relative_path(path): backend
        for path, backend in paths.items()
    }


def _project_config_path(root: Path, path: str) -> Path:
    if _is_absolute_or_windows_drive_path(path):
        return Path(path)
    return root / _normalize_project_relative_path(path)


def _project_output_path(root: Path, output_dir: str) -> Path:
    return _project_config_path(root, output_dir)


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


def _is_stable_relative_posix_path(path: str) -> bool:
    parts = path.split("/")
    return (
        bool(path)
        and "\\" not in path
        and not path.startswith("/")
        and not PureWindowsPath(path).drive
        and all(part not in {"", ".", ".."} for part in parts)
    )


def _is_report_identity_path(path: str) -> bool:
    return _is_stable_relative_posix_path(path)


def _is_diagnostic_location_path(path: str) -> bool:
    return path == "." or _is_report_identity_path(path)


def _repository_relative_globs(patterns: Sequence[str]) -> list[str]:
    return [
        _normalize_project_relative_path(pattern)
        for pattern in patterns
        if _is_repository_relative_glob(pattern)
    ]


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


def _optional_source_hash(path: Path) -> dict[str, str] | None:
    try:
        return _source_hash(path)
    except OSError:
        return None


def _hash_matches_report(
    actual: Mapping[str, str], expected: Mapping[str, Any]
) -> bool:
    return (actual["algorithm"], actual["value"]) == (
        expected.get("algorithm"),
        expected.get("value"),
    )


def _hash_mismatch_context(
    actual: Mapping[str, str], expected: Mapping[str, Any]
) -> str:
    return (
        f"expected {expected.get('algorithm')}:{expected.get('value')}, "
        f"actual {actual['algorithm']}:{actual['value']}"
    )


def _size_mismatch_context(expected: int, actual: int) -> str:
    return f"expected {expected} bytes, actual {actual} bytes"


def _value_mismatch_context(expected: Any, actual: Any) -> str:
    return f"expected {expected}, actual {actual}"


def _allowed_value_context(allowed: Iterable[str]) -> str:
    return f"one of {', '.join(sorted(allowed))}"


def _allowed_value_mismatch_context(allowed: Iterable[str], actual: Any) -> str:
    return _value_mismatch_context(_allowed_value_context(allowed), actual)


def _allowed_value_reason(prefix: str, allowed: Iterable[str], actual: Any) -> str:
    allowed_values = ", ".join(sorted(allowed))
    return (
        f"{prefix} must be one of {allowed_values} "
        f"({_value_mismatch_context(f'one of {allowed_values}', actual)})"
    )


def _source_frontend_supports_lexer_keyword(source_backend: str, keyword: str) -> bool:
    register_default_sources()
    discover_backend_plugins()
    spec = SOURCE_REGISTRY.get(source_backend)
    if spec is None:
        return False
    return spec.supports_lexer_keyword(keyword)


def _artifact_define_processing(
    source_backend: str,
    defines: Mapping[str, str],
    *,
    supports_defines: bool | None = None,
) -> dict[str, Any]:
    define_count = len(defines)
    if supports_defines is None:
        supports_defines = _source_frontend_supports_lexer_keyword(
            source_backend, "defines"
        )
    if define_count == 0:
        status = "not-requested"
    elif supports_defines:
        status = "forwarded"
    else:
        status = "not-supported"
    return {
        "status": status,
        "frontend": "lexer",
        "supportsDefines": supports_defines,
        "defineCount": define_count,
    }


def _artifact_include_path_processing(
    source_backend: str,
    include_paths: Sequence[str],
    *,
    supports_include_paths: bool | None = None,
) -> dict[str, Any]:
    include_path_count = len(include_paths)
    if supports_include_paths is None:
        supports_include_paths = _source_frontend_supports_lexer_keyword(
            source_backend, "include_paths"
        )
    if include_path_count == 0:
        status = "not-requested"
    elif supports_include_paths:
        status = "forwarded"
    else:
        status = "not-supported"
    return {
        "status": status,
        "frontend": "lexer",
        "supportsIncludePaths": supports_include_paths,
        "includePathCount": include_path_count,
    }


def _frontend_define_forwarding_diagnostic(
    unit: ProjectTranslationUnit, define_count: int
) -> ProjectDiagnostic:
    define_label = "define" if define_count == 1 else "defines"
    return ProjectDiagnostic(
        severity="warning",
        code="project.translate.defines-not-forwarded",
        message=(
            f"{define_count} configured {define_label} were not forwarded to "
            f"the {unit.source_backend} lexer frontend; translation used source "
            "text without project define preprocessing for this source backend."
        ),
        location=SourceLocation(file=unit.relative_path),
        source_backend=unit.source_backend,
        missing_capabilities=["macro.defines"],
    )


def _frontend_include_path_forwarding_diagnostic(
    unit: ProjectTranslationUnit, include_path_count: int
) -> ProjectDiagnostic:
    include_label = "include path" if include_path_count == 1 else "include paths"
    return ProjectDiagnostic(
        severity="warning",
        code="project.translate.include-paths-not-forwarded",
        message=(
            f"{include_path_count} configured {include_label} were not forwarded "
            f"to the {unit.source_backend} lexer frontend; translation used source "
            "text without project include path preprocessing for this source backend."
        ),
        location=SourceLocation(file=unit.relative_path),
        source_backend=unit.source_backend,
        missing_capabilities=["include.forwarding"],
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


def _advance_source_position(text: str, line: int, column: int) -> tuple[int, int]:
    for character in text:
        if character == "\n":
            line += 1
            column = 1
        else:
            column += 1
    return line, column


def _file_span(path: Path, report_path: str) -> SourceLocation:
    source_bytes = path.read_bytes()
    text = source_bytes.decode("utf-8", errors="replace")
    line, column = _advance_source_position(text, 1, 1)
    return SourceLocation(
        file=report_path,
        length=len(source_bytes),
        end_line=line,
        end_column=column,
        end_offset=len(source_bytes),
    )


def _line_spans(path: Path, report_path: str) -> list[SourceLocation]:
    source_bytes = path.read_bytes()
    spans = []
    offset = 0
    line = 1
    while offset < len(source_bytes):
        newline_index = source_bytes.find(b"\n", offset)
        end_offset = len(source_bytes) if newline_index == -1 else newline_index + 1
        line_bytes = source_bytes[offset:end_offset]
        text = line_bytes.decode("utf-8", errors="replace")
        end_line, end_column = _advance_source_position(text, line, 1)
        spans.append(
            SourceLocation(
                file=report_path,
                line=line,
                column=1,
                offset=offset,
                length=len(line_bytes),
                end_line=end_line,
                end_column=end_column,
                end_offset=end_offset,
            )
        )
        offset = end_offset
        line = end_line
    return spans


def _normalized_line_text(source_bytes: bytes) -> str:
    text = source_bytes.decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _line_preserving_source_map_mappings(
    source_path: Path,
    source_report_path: str,
    generated_path: Path,
    generated_report_path: str,
) -> list[dict[str, Any]]:
    source_text = _normalized_line_text(source_path.read_bytes())
    generated_text = _normalized_line_text(generated_path.read_bytes())
    if source_text.splitlines() != generated_text.splitlines():
        return []

    source_line_spans = _line_spans(source_path, source_report_path)
    generated_line_spans = _line_spans(generated_path, generated_report_path)
    if not source_line_spans or len(source_line_spans) != len(generated_line_spans):
        return []
    return [
        {
            "source": source_line_span.to_json(),
            "generated": generated_line_span.to_json(),
        }
        for source_line_span, generated_line_span in zip(
            source_line_spans, generated_line_spans
        )
    ]


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


def _diagnostic_counts_by_optional_value(
    diagnostics: Sequence[ProjectDiagnostic],
    field_name: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for diagnostic in diagnostics:
        value = getattr(diagnostic, field_name)
        if value:
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _diagnostic_counts_by_target(
    diagnostics: Sequence[ProjectDiagnostic],
) -> dict[str, int]:
    return _diagnostic_counts_by_optional_value(diagnostics, "target")


def _diagnostic_counts_by_source_backend(
    diagnostics: Sequence[ProjectDiagnostic],
) -> dict[str, int]:
    return _diagnostic_counts_by_optional_value(diagnostics, "source_backend")


def _diagnostic_counts_by_variant(
    diagnostics: Sequence[ProjectDiagnostic],
) -> dict[str, int]:
    return _diagnostic_counts_by_optional_value(diagnostics, "variant")


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


def _extension_rollup_key(extension: Any) -> str:
    if isinstance(extension, str) and extension:
        return extension.lower()
    return "extensionless"


def _unit_counts_by_extension(
    units: Sequence[ProjectTranslationUnit],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for unit in units:
        extension = _extension_rollup_key(unit.extension)
        counts[extension] = counts.get(extension, 0) + 1
    return dict(sorted(counts.items()))


def _unit_counts_by_source_override(
    units: Sequence[ProjectTranslationUnit],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for unit in units:
        source_override = unit.source_override
        if not _is_non_empty_string(source_override):
            continue
        counts[source_override] = counts.get(source_override, 0) + 1
    return dict(sorted(counts.items()))


def _skipped_counts_by_reason(
    skipped: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in skipped:
        reason = record.get("reason")
        if not _is_non_empty_string(reason):
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _skipped_counts_by_extension(
    skipped: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in skipped:
        path = record.get("path")
        if not _is_non_empty_string(path):
            continue
        extension = _extension_rollup_key(Path(path).suffix.lower())
        counts[extension] = counts.get(extension, 0) + 1
    return dict(sorted(counts.items()))


def _skipped_counts_by_source_override(
    skipped: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in skipped:
        source_override = record.get("sourceOverride")
        if not _is_non_empty_string(source_override):
            continue
        counts[source_override] = counts.get(source_override, 0) + 1
    return dict(sorted(counts.items()))


def _include_dependency_records(
    units: Sequence[ProjectTranslationUnit],
) -> list[Mapping[str, Any]]:
    return [
        dependency
        for unit in units
        for dependency in unit.include_dependencies
        if isinstance(dependency, Mapping)
    ]


def _include_dependency_counts_by_field(
    records: Sequence[Mapping[str, Any]],
    field_name: str,
    allowed_values: frozenset[str],
) -> dict[str, int]:
    counts = {value: 0 for value in sorted(allowed_values)}
    for record in records:
        value = record.get(field_name)
        if isinstance(value, str) and value in counts:
            counts[value] += 1
    return {value: count for value, count in counts.items() if count}


def _include_dependency_count(units: Sequence[ProjectTranslationUnit]) -> int:
    return len(_include_dependency_records(units))


def _include_dependency_counts_by_kind(
    units: Sequence[ProjectTranslationUnit],
) -> dict[str, int]:
    return _include_dependency_counts_by_field(
        _include_dependency_records(units),
        "kind",
        INCLUDE_DEPENDENCY_KINDS,
    )


def _include_dependency_counts_by_status(
    units: Sequence[ProjectTranslationUnit],
) -> dict[str, int]:
    return _include_dependency_counts_by_field(
        _include_dependency_records(units),
        "status",
        INCLUDE_DEPENDENCY_STATUSES,
    )


def _include_dependency_counts_by_resolved_from(
    units: Sequence[ProjectTranslationUnit],
) -> dict[str, int]:
    return _include_dependency_counts_by_field(
        _include_dependency_records(units),
        "resolvedFrom",
        INCLUDE_DEPENDENCY_RESOLUTION_SOURCES,
    )


def _include_dependency_counts_by_source_backend(
    units: Sequence[ProjectTranslationUnit],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for unit in units:
        if not _is_non_empty_string(unit.source_backend):
            continue
        dependency_count = sum(
            1
            for dependency in unit.include_dependencies
            if isinstance(dependency, Mapping)
        )
        if dependency_count:
            counts[unit.source_backend] = (
                counts.get(unit.source_backend, 0) + dependency_count
            )
    return dict(sorted(counts.items()))


def _include_dependency_counts_by_source_backend_status(
    units: Sequence[ProjectTranslationUnit],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for unit in units:
        if not _is_non_empty_string(unit.source_backend):
            continue
        for dependency in unit.include_dependencies:
            if not isinstance(dependency, Mapping):
                continue
            status = dependency.get("status")
            if not isinstance(status, str) or status not in INCLUDE_DEPENDENCY_STATUSES:
                continue
            row = counts.setdefault(unit.source_backend, {})
            row[status] = row.get(status, 0) + 1
    return {
        backend: dict(sorted(row.items())) for backend, row in sorted(counts.items())
    }


def _include_dependency_counts_by_variant(
    units: Sequence[ProjectTranslationUnit],
) -> dict[str, int]:
    return _include_dependency_counts_by_field(
        _include_dependency_records(units),
        "variant",
        frozenset(
            variant
            for unit in units
            for dependency in unit.include_dependencies
            if isinstance(dependency, Mapping)
            for variant in (dependency.get("variant"),)
            if _is_non_empty_string(variant)
        ),
    )


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


def _artifact_counts_by_source_backend(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for artifact in artifacts:
        source_backend = str(artifact.get("sourceBackend", "unknown"))
        row = counts.setdefault(
            source_backend,
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
    return {source_backend: counts[source_backend] for source_backend in sorted(counts)}


def _artifact_counts_by_variant(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for artifact in artifacts:
        variant = artifact.get("variant")
        if not _is_non_empty_string(variant):
            continue
        row = counts.setdefault(
            variant,
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
    return {variant: counts[variant] for variant in sorted(counts)}


def _source_map_counts(artifacts: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    source_map_count = 0
    fine_grained_source_map_count = 0
    for artifact in artifacts:
        source_map = artifact.get("sourceMap")
        if not isinstance(source_map, Mapping):
            continue
        source_map_count += 1
        granularity = source_map.get("mappingGranularity")
        if _is_non_empty_string(granularity) and granularity != "file":
            fine_grained_source_map_count += 1
    return {
        "sourceMapCount": source_map_count,
        "fineGrainedSourceMapCount": fine_grained_source_map_count,
    }


def _source_remap_count(artifacts: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for artifact in artifacts if artifact.get("sourceRemap"))


def _source_remap_mapping_count(artifacts: Sequence[Mapping[str, Any]]) -> int:
    mapping_count = 0
    for artifact in artifacts:
        source_remap = artifact.get("sourceRemap")
        if not isinstance(source_remap, Mapping):
            continue
        artifact_mapping_count = source_remap.get("mappingCount")
        if _is_non_negative_int(artifact_mapping_count):
            mapping_count += artifact_mapping_count
    return mapping_count


def _source_remap_counts_by_granularity(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for artifact in artifacts:
        source_remap = artifact.get("sourceRemap")
        if not isinstance(source_remap, Mapping):
            continue
        granularity = source_remap.get("mappingGranularity")
        key = granularity if _is_non_empty_string(granularity) else "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _source_map_counts_by_granularity(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for artifact in artifacts:
        source_map = artifact.get("sourceMap")
        if not isinstance(source_map, Mapping):
            continue
        granularity = source_map.get("mappingGranularity")
        key = granularity if _is_non_empty_string(granularity) else "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _source_map_counts_by_target(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for artifact in artifacts:
        if not isinstance(artifact.get("sourceMap"), Mapping):
            continue
        target = artifact.get("target")
        key = target if _is_non_empty_string(target) else "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _source_map_counts_by_source_backend(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for artifact in artifacts:
        if not isinstance(artifact.get("sourceMap"), Mapping):
            continue
        source_backend = artifact.get("sourceBackend")
        key = source_backend if _is_non_empty_string(source_backend) else "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _source_map_counts_by_variant(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for artifact in artifacts:
        if not isinstance(artifact.get("sourceMap"), Mapping):
            continue
        variant = artifact.get("variant")
        if not _is_non_empty_string(variant):
            continue
        counts[variant] = counts.get(variant, 0) + 1
    return dict(sorted(counts.items()))


def _source_remap_counts_by_target(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for artifact in artifacts:
        if not isinstance(artifact.get("sourceRemap"), Mapping):
            continue
        target = artifact.get("target")
        key = target if _is_non_empty_string(target) else "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _source_remap_counts_by_source_backend(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for artifact in artifacts:
        if not isinstance(artifact.get("sourceRemap"), Mapping):
            continue
        source_backend = artifact.get("sourceBackend")
        key = source_backend if _is_non_empty_string(source_backend) else "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _source_remap_counts_by_variant(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for artifact in artifacts:
        if not isinstance(artifact.get("sourceRemap"), Mapping):
            continue
        variant = artifact.get("variant")
        if not _is_non_empty_string(variant):
            continue
        counts[variant] = counts.get(variant, 0) + 1
    return dict(sorted(counts.items()))


def _artifact_provenance_counts_by_pipeline(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for artifact in artifacts:
        provenance = artifact.get("provenance")
        pipeline = (
            provenance.get("pipeline") if isinstance(provenance, Mapping) else None
        )
        key = pipeline if _is_non_empty_string(pipeline) else "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _artifact_provenance_counts_by_intermediate(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for artifact in artifacts:
        provenance = artifact.get("provenance")
        if not isinstance(provenance, Mapping):
            key = "unknown"
        elif provenance.get("intermediate") is None:
            key = "none"
        else:
            intermediate = provenance.get("intermediate")
            key = intermediate if _is_non_empty_string(intermediate) else "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _artifact_provenance_intermediate_counts_by_source_backend(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for artifact in artifacts:
        source_backend = artifact.get("sourceBackend")
        source_key = (
            source_backend if _is_non_empty_string(source_backend) else "unknown"
        )
        provenance = artifact.get("provenance")
        if not isinstance(provenance, Mapping):
            intermediate_key = "unknown"
        elif provenance.get("intermediate") is None:
            intermediate_key = "none"
        else:
            intermediate = provenance.get("intermediate")
            intermediate_key = (
                intermediate if _is_non_empty_string(intermediate) else "unknown"
            )
        row = counts.setdefault(source_key, {})
        row[intermediate_key] = row.get(intermediate_key, 0) + 1
    return {
        source_backend: dict(sorted(row.items()))
        for source_backend, row in sorted(counts.items())
    }


def _artifact_provenance_intermediate_counts_by_target(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for artifact in artifacts:
        target = artifact.get("target")
        target_key = target if _is_non_empty_string(target) else "unknown"
        provenance = artifact.get("provenance")
        if not isinstance(provenance, Mapping):
            intermediate_key = "unknown"
        elif provenance.get("intermediate") is None:
            intermediate_key = "none"
        else:
            intermediate = provenance.get("intermediate")
            intermediate_key = (
                intermediate if _is_non_empty_string(intermediate) else "unknown"
            )
        row = counts.setdefault(target_key, {})
        row[intermediate_key] = row.get(intermediate_key, 0) + 1
    return {target: dict(sorted(row.items())) for target, row in sorted(counts.items())}


def _artifact_provenance_intermediate_counts_by_variant(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for artifact in artifacts:
        variant = artifact.get("variant")
        if not _is_non_empty_string(variant):
            continue
        provenance = artifact.get("provenance")
        if not isinstance(provenance, Mapping):
            intermediate_key = "unknown"
        elif provenance.get("intermediate") is None:
            intermediate_key = "none"
        else:
            intermediate = provenance.get("intermediate")
            intermediate_key = (
                intermediate if _is_non_empty_string(intermediate) else "unknown"
            )
        row = counts.setdefault(variant, {})
        row[intermediate_key] = row.get(intermediate_key, 0) + 1
    return {
        variant: dict(sorted(row.items())) for variant, row in sorted(counts.items())
    }


def _artifact_provenance_rollups(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "artifactProvenanceByPipeline": _artifact_provenance_counts_by_pipeline(
            artifacts
        ),
        "artifactProvenanceByIntermediate": _artifact_provenance_counts_by_intermediate(
            artifacts
        ),
        "artifactProvenanceIntermediateBySourceBackend": (
            _artifact_provenance_intermediate_counts_by_source_backend(artifacts)
        ),
        "artifactProvenanceIntermediateByTarget": (
            _artifact_provenance_intermediate_counts_by_target(artifacts)
        ),
        "artifactProvenanceIntermediateByVariant": (
            _artifact_provenance_intermediate_counts_by_variant(artifacts)
        ),
    }


def _define_processing_status_counts(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts = {status: 0 for status in sorted(DEFINE_PROCESSING_STATUSES)}
    counts["unknown"] = 0
    for artifact in artifacts:
        define_processing = artifact.get("defineProcessing")
        status = (
            define_processing.get("status")
            if isinstance(define_processing, Mapping)
            else None
        )
        if isinstance(status, str) and status in counts:
            counts[status] += 1
        else:
            counts["unknown"] += 1
    return {status: count for status, count in counts.items() if count}


def _define_processing_counts_by_source_backend(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for artifact in artifacts:
        source_backend = artifact.get("sourceBackend")
        key = source_backend if _is_non_empty_string(source_backend) else "unknown"
        define_processing = artifact.get("defineProcessing")
        status = (
            define_processing.get("status")
            if isinstance(define_processing, Mapping)
            else "unknown"
        )
        if not isinstance(status, str) or status not in DEFINE_PROCESSING_STATUSES:
            status = "unknown"
        row = counts.setdefault(key, {})
        row[status] = row.get(status, 0) + 1
    return {source: dict(sorted(row.items())) for source, row in sorted(counts.items())}


def _define_processing_counts_by_target(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for artifact in artifacts:
        target = artifact.get("target")
        key = target if _is_non_empty_string(target) else "unknown"
        define_processing = artifact.get("defineProcessing")
        status = (
            define_processing.get("status")
            if isinstance(define_processing, Mapping)
            else "unknown"
        )
        if not isinstance(status, str) or status not in DEFINE_PROCESSING_STATUSES:
            status = "unknown"
        row = counts.setdefault(key, {})
        row[status] = row.get(status, 0) + 1
    return {target: dict(sorted(row.items())) for target, row in sorted(counts.items())}


def _define_processing_counts_by_variant(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for artifact in artifacts:
        variant = artifact.get("variant")
        if not _is_non_empty_string(variant):
            continue
        define_processing = artifact.get("defineProcessing")
        status = (
            define_processing.get("status")
            if isinstance(define_processing, Mapping)
            else "unknown"
        )
        if not isinstance(status, str) or status not in DEFINE_PROCESSING_STATUSES:
            status = "unknown"
        row = counts.setdefault(variant, {})
        row[status] = row.get(status, 0) + 1
    return {
        variant: dict(sorted(row.items())) for variant, row in sorted(counts.items())
    }


def _define_processing_rollups(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "defineProcessingByStatus": _define_processing_status_counts(artifacts),
        "defineProcessingBySourceBackend": _define_processing_counts_by_source_backend(
            artifacts
        ),
        "defineProcessingByTarget": _define_processing_counts_by_target(artifacts),
        "defineProcessingByVariant": _define_processing_counts_by_variant(artifacts),
    }


def _include_path_processing_status_counts(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts = {status: 0 for status in sorted(INCLUDE_PATH_PROCESSING_STATUSES)}
    counts["unknown"] = 0
    for artifact in artifacts:
        include_path_processing = artifact.get("includePathProcessing")
        status = (
            include_path_processing.get("status")
            if isinstance(include_path_processing, Mapping)
            else None
        )
        if isinstance(status, str) and status in counts:
            counts[status] += 1
        else:
            counts["unknown"] += 1
    return {status: count for status, count in counts.items() if count}


def _include_path_processing_counts_by_source_backend(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for artifact in artifacts:
        source_backend = artifact.get("sourceBackend")
        key = source_backend if _is_non_empty_string(source_backend) else "unknown"
        include_path_processing = artifact.get("includePathProcessing")
        status = (
            include_path_processing.get("status")
            if isinstance(include_path_processing, Mapping)
            else "unknown"
        )
        if (
            not isinstance(status, str)
            or status not in INCLUDE_PATH_PROCESSING_STATUSES
        ):
            status = "unknown"
        row = counts.setdefault(key, {})
        row[status] = row.get(status, 0) + 1
    return {source: dict(sorted(row.items())) for source, row in sorted(counts.items())}


def _include_path_processing_counts_by_target(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for artifact in artifacts:
        target = artifact.get("target")
        key = target if _is_non_empty_string(target) else "unknown"
        include_path_processing = artifact.get("includePathProcessing")
        status = (
            include_path_processing.get("status")
            if isinstance(include_path_processing, Mapping)
            else "unknown"
        )
        if (
            not isinstance(status, str)
            or status not in INCLUDE_PATH_PROCESSING_STATUSES
        ):
            status = "unknown"
        row = counts.setdefault(key, {})
        row[status] = row.get(status, 0) + 1
    return {target: dict(sorted(row.items())) for target, row in sorted(counts.items())}


def _include_path_processing_counts_by_variant(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for artifact in artifacts:
        variant = artifact.get("variant")
        if not _is_non_empty_string(variant):
            continue
        include_path_processing = artifact.get("includePathProcessing")
        status = (
            include_path_processing.get("status")
            if isinstance(include_path_processing, Mapping)
            else "unknown"
        )
        if (
            not isinstance(status, str)
            or status not in INCLUDE_PATH_PROCESSING_STATUSES
        ):
            status = "unknown"
        row = counts.setdefault(variant, {})
        row[status] = row.get(status, 0) + 1
    return {
        variant: dict(sorted(row.items())) for variant, row in sorted(counts.items())
    }


def _include_path_processing_rollups(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "includePathProcessingByStatus": _include_path_processing_status_counts(
            artifacts
        ),
        "includePathProcessingBySourceBackend": (
            _include_path_processing_counts_by_source_backend(artifacts)
        ),
        "includePathProcessingByTarget": _include_path_processing_counts_by_target(
            artifacts
        ),
        "includePathProcessingByVariant": _include_path_processing_counts_by_variant(
            artifacts
        ),
    }


def _migration_action_counts_by_kind(actions: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        kind = action.get("kind")
        if not _is_non_empty_string(kind):
            continue
        counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


def _migration_action_counts_by_severity(actions: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        severity = action.get("severity")
        if not _is_non_empty_string(severity):
            continue
        counts[severity] = counts.get(severity, 0) + 1
    return dict(sorted(counts.items()))


def _migration_action_counts_by_target(actions: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        targets = action.get("targets")
        if not isinstance(targets, Sequence) or isinstance(
            targets, (str, bytes, bytearray)
        ):
            continue
        for target in targets:
            if not _is_non_empty_string(target):
                continue
            counts[target] = counts.get(target, 0) + 1
    return dict(sorted(counts.items()))


def _migration_action_rollups(actions: Sequence[Any]) -> dict[str, Any]:
    return {
        "actionCount": len(actions),
        "actionsByKind": _migration_action_counts_by_kind(actions),
        "actionsBySeverity": _migration_action_counts_by_severity(actions),
        "actionsByTarget": _migration_action_counts_by_target(actions),
    }


def _source_map_rollups(artifacts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        **_source_map_counts(artifacts),
        "sourceMapsByGranularity": _source_map_counts_by_granularity(artifacts),
        "sourceMapsByTarget": _source_map_counts_by_target(artifacts),
        "sourceMapsBySourceBackend": _source_map_counts_by_source_backend(artifacts),
        "sourceMapsByVariant": _source_map_counts_by_variant(artifacts),
        "sourceRemapCount": _source_remap_count(artifacts),
        "sourceRemapMappingCount": _source_remap_mapping_count(artifacts),
        "sourceRemapsByGranularity": _source_remap_counts_by_granularity(artifacts),
        "sourceRemapsByTarget": _source_remap_counts_by_target(artifacts),
        "sourceRemapsBySourceBackend": _source_remap_counts_by_source_backend(
            artifacts
        ),
        "sourceRemapsByVariant": _source_remap_counts_by_variant(artifacts),
    }


def _external_corpus_empty_summary() -> dict[str, Any]:
    return {
        "manifestEntryCount": 0,
        "validEntryCount": 0,
        "invalidEntryCount": 0,
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
    return _project_config_path(config.root, manifest).resolve()


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

    if "id" in entry and not _is_non_empty_string(entry.get("id")):
        reasons.append("id must be a non-empty string")

    source_backend = entry.get("sourceBackend")
    if source_backend is not None and not _is_non_empty_string(source_backend):
        reasons.append("sourceBackend must be a non-empty string")

    targets = entry.get("targets")
    if targets is not None:
        valid_targets = _is_non_empty_string(targets) or (
            isinstance(targets, Sequence)
            and not isinstance(targets, (bytes, bytearray))
            and len(targets) > 0
            and all(_is_non_empty_string(target) for target in targets)
        )
        if not valid_targets:
            reasons.append(
                "targets must be a non-empty string or list of non-empty strings"
            )

    for field_name in ("repository", "commit", "sourceUrl"):
        if field_name in entry and not _is_non_empty_string(entry.get(field_name)):
            reasons.append(f"{field_name} must be a string")
    commit = entry.get("commit")
    if _is_non_empty_string(commit) and not _is_lowercase_hex_digest(
        commit,
        GIT_COMMIT_HEX_LENGTH,
    ):
        reasons.append("commit must be a lowercase 40-character hex digest")
    repository = entry.get("repository")
    source_url = entry.get("sourceUrl")
    if (
        _is_non_empty_string(repository)
        and _is_non_empty_string(source_url)
        and not _source_url_matches_repository(source_url, repository)
    ):
        reasons.append("sourceUrl must start with repository")

    return reasons


def _external_corpus_manifest_entry_identity(
    entry: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    entry_id = entry.get("id")
    normalized_id = entry_id if _is_non_empty_string(entry_id) else None
    path = entry.get("path")
    normalized_path = path.replace("\\", "/") if _is_non_empty_string(path) else None
    return normalized_id, normalized_path


def _external_corpus_manifest_entry_duplicate_reasons(
    entry: Mapping[str, Any],
    *,
    seen_ids: Mapping[str, int],
    seen_paths: Mapping[str, int],
) -> list[str]:
    entry_id, path = _external_corpus_manifest_entry_identity(entry)
    reasons = []
    if entry_id is not None and entry_id in seen_ids:
        reasons.append(f"id duplicates entry {seen_ids[entry_id] + 1}")
    if path is not None and path in seen_paths:
        reasons.append(f"path duplicates entry {seen_paths[path] + 1}")
    return reasons


def _valid_external_corpus_manifest_entries(
    manifest: Mapping[str, Any],
) -> list[tuple[int, Mapping[str, Any]]]:
    entries = []
    seen_ids: dict[str, int] = {}
    seen_paths: dict[str, int] = {}
    for index, entry in enumerate(manifest.get("entries", [])):
        if _external_corpus_manifest_entry_reasons(entry):
            continue
        duplicate_reasons = _external_corpus_manifest_entry_duplicate_reasons(
            entry,
            seen_ids=seen_ids,
            seen_paths=seen_paths,
        )
        if duplicate_reasons:
            continue
        entry_id, path = _external_corpus_manifest_entry_identity(entry)
        if entry_id is not None:
            seen_ids[entry_id] = index
        if path is not None:
            seen_paths[path] = index
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


def _resolve_external_corpus_source_backend(value: str) -> str | None:
    register_default_sources()
    discover_backend_plugins()
    return SOURCE_REGISTRY.resolve_name(value)


def _external_corpus_source_backend(value: Any) -> str:
    if not _is_non_empty_string(value):
        return "unknown"
    canonical_name = _resolve_external_corpus_source_backend(value)
    return canonical_name or value.strip()


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

    valid_manifest_entries = _valid_external_corpus_manifest_entries(manifest)
    manifest_entry_count = len(manifest.get("entries", []))

    entries = []
    for index, raw_entry in valid_manifest_entries:
        path = str(raw_entry.get("path", "")).replace("\\", "/")
        entry_targets = _manifest_entry_targets(raw_entry, targets)
        unit = units_by_path.get(path)
        manifest_source_backend = raw_entry.get("sourceBackend")
        if unit is not None:
            source_backend = unit.source_backend
        elif _is_non_empty_string(manifest_source_backend):
            source_backend = _external_corpus_source_backend(manifest_source_backend)
        else:
            source_backend = "unknown"
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
        discovered = unit is not None
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
        "manifestEntryCount": manifest_entry_count,
        "validEntryCount": len(entries),
        "invalidEntryCount": manifest_entry_count - len(entries),
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
    return _project_config_path(config.root, include_dir).resolve()


def _resolved_include_dirs(config: ProjectConfig) -> list[str]:
    include_dirs = []
    for include_dir in config.include_dirs:
        include_dirs.append(str(_resolved_include_dir(config, include_dir)))
    return include_dirs


def _include_dir_status_for_path(config: ProjectConfig, path: Path) -> str:
    if not _is_relative_to(path, config.root):
        return "outside-project"
    if not path.exists():
        return "missing"
    if not path.is_dir():
        return "not-directory"
    return "active"


def _include_dir_status_records(config: ProjectConfig) -> list[dict[str, Any]]:
    records = []
    for include_dir in config.include_dirs:
        absolute_dir = _resolved_include_dir(config, include_dir)
        status = _include_dir_status_for_path(config, absolute_dir)
        records.append(
            {
                "path": include_dir,
                "resolvedPath": str(absolute_dir),
                "status": status,
                "frontendVisible": status == "active",
            }
        )
    return records


def _include_dir_status_counts(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts = {status: 0 for status in sorted(INCLUDE_DIR_STATUSES)}
    for record in records:
        status = record.get("status")
        if isinstance(status, str) and status in counts:
            counts[status] += 1
    return {status: count for status, count in counts.items() if count}


def _frontend_include_dirs(config: ProjectConfig) -> list[str]:
    include_dirs = []
    for include_dir in config.include_dirs:
        absolute_dir = _resolved_include_dir(config, include_dir)
        if _include_dir_status_for_path(config, absolute_dir) == "active":
            include_dirs.append(str(absolute_dir))
    return include_dirs


def _strip_include_line_comment(value: str) -> str:
    return value.split("//", 1)[0].strip()


def _mask_preprocessor_block_comments(lines: Sequence[str]) -> list[str]:
    masked: list[str] = []
    in_block_comment = False
    for line in lines:
        chars = list(line)
        index = 0
        while index < len(chars):
            if in_block_comment:
                end = line.find("*/", index)
                if end == -1:
                    for offset in range(index, len(chars)):
                        chars[offset] = " "
                    index = len(chars)
                    continue
                for offset in range(index, end + 2):
                    chars[offset] = " "
                in_block_comment = False
                index = end + 2
                continue

            start = line.find("/*", index)
            line_comment = line.find("//", index)
            if line_comment != -1 and (start == -1 or line_comment < start):
                break
            if start == -1:
                break
            end = line.find("*/", start + 2)
            if end == -1:
                for offset in range(start, len(chars)):
                    chars[offset] = " "
                in_block_comment = True
                index = len(chars)
                continue
            for offset in range(start, end + 2):
                chars[offset] = " "
            index = end + 2
        masked.append("".join(chars))
    return masked


def _include_literal(value: str) -> tuple[str, str] | None:
    match = INCLUDE_LITERAL_RE.match(value)
    if not match:
        return None
    delimiter = match.group("open")
    close = match.group("close")
    if delimiter == '"' and close != '"':
        return None
    if delimiter == "<" and close != ">":
        return None
    kind = "local" if delimiter == '"' else "system"
    include_path = match.group("path").strip()
    if not include_path:
        return None
    return kind, include_path


def _include_literal_from_define(
    config: ProjectConfig, expression: str
) -> tuple[str, str, str] | None:
    define_name = expression.strip()
    if not define_name or define_name not in config.defines:
        return None

    literal = _include_literal(config.defines[define_name].strip())
    if literal is None:
        return None
    kind, include_path = literal
    return define_name, kind, include_path


@dataclass
class _IncludeConditionalFrame:
    parent_active: bool
    branch_active: bool
    branch_taken: bool


def _strip_preprocessor_line_comment(value: str) -> str:
    return value.split("//", 1)[0].strip()


def _include_condition_tokens(expression: str) -> list[str] | None:
    tokens = INCLUDE_CONDITION_TOKEN_RE.findall(
        _strip_preprocessor_line_comment(expression)
    )
    if not tokens:
        return None
    allowed_operators = {
        "defined",
        "&&",
        "||",
        "!",
        "(",
        ")",
        "==",
        "!=",
        "<",
        "<=",
        ">",
        ">=",
    }
    for token in tokens:
        if token in allowed_operators:
            continue
        if re.match(r"^[+-]?[0-9]+$", token):
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", token):
            continue
        return None
    return tokens


def _define_condition_scalar(value: str) -> int | bool | None:
    stripped = value.strip()
    if not stripped:
        return False
    if re.match(r"^[+-]?[0-9]+$", stripped):
        return int(stripped, 10)
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    return None


def _define_condition_value(value: str) -> bool | None:
    scalar = _define_condition_scalar(value)
    return None if scalar is None else bool(scalar)


def _include_if_expression_value(config: ProjectConfig, expression: str) -> bool | None:
    tokens = _include_condition_tokens(expression)
    if tokens is None:
        return None
    index = 0
    comparison_operators = {"==", "!=", "<", "<=", ">", ">="}

    def condition_bool(value: int | bool | None) -> bool | None:
        return None if value is None else bool(value)

    def condition_int(value: int | bool | None) -> int | None:
        if value is None:
            return None
        return int(value)

    def parse_or() -> bool | None:
        nonlocal index
        value = parse_and()
        while index < len(tokens) and tokens[index] == "||":
            index += 1
            right = parse_and()
            if value is None or right is None:
                value = None
            else:
                value = value or right
        return value

    def parse_and() -> bool | None:
        nonlocal index
        value = parse_comparison()
        while index < len(tokens) and tokens[index] == "&&":
            index += 1
            right = parse_comparison()
            if value is None or right is None:
                value = None
            else:
                value = value and right
        return value

    def parse_comparison() -> bool | None:
        nonlocal index
        left = parse_not()
        if index >= len(tokens) or tokens[index] not in comparison_operators:
            return condition_bool(left)
        operator = tokens[index]
        index += 1
        right = parse_not()
        left_value = condition_int(left)
        right_value = condition_int(right)
        if left_value is None or right_value is None:
            return None
        if operator == "==":
            return left_value == right_value
        if operator == "!=":
            return left_value != right_value
        if operator == "<":
            return left_value < right_value
        if operator == "<=":
            return left_value <= right_value
        if operator == ">":
            return left_value > right_value
        if operator == ">=":
            return left_value >= right_value
        return None

    def parse_not() -> int | bool | None:
        nonlocal index
        if index < len(tokens) and tokens[index] == "!":
            index += 1
            value = parse_not()
            truth_value = condition_bool(value)
            return None if truth_value is None else not truth_value
        return parse_atom()

    def parse_atom() -> int | bool | None:
        nonlocal index
        if index >= len(tokens):
            return None
        token = tokens[index]
        if token == "(":
            index += 1
            value = parse_or()
            if index >= len(tokens) or tokens[index] != ")":
                return None
            index += 1
            return value
        if token == "defined":
            index += 1
            if index < len(tokens) and tokens[index] == "(":
                index += 1
                if index >= len(tokens):
                    return None
                name = tokens[index]
                index += 1
                if index >= len(tokens) or tokens[index] != ")":
                    return None
                index += 1
            elif index < len(tokens):
                name = tokens[index]
                index += 1
            else:
                return None
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                return None
            return name in config.defines
        index += 1
        if re.match(r"^[+-]?[0-9]+$", token):
            return int(token, 10)
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", token):
            if token not in config.defines:
                return False
            return _define_condition_scalar(config.defines[token])
        return None

    value = parse_or()
    if index != len(tokens):
        return None
    return value


def _include_condition_value(
    config: ProjectConfig, directive: str, expression: str
) -> bool | None:
    expression = _strip_preprocessor_line_comment(expression)
    if directive == "ifdef":
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", expression):
            return None
        return expression in config.defines
    if directive == "ifndef":
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", expression):
            return None
        return expression not in config.defines
    return _include_if_expression_value(config, expression)


def _configured_define_origin_label(config: ProjectConfig, name: str) -> str:
    origins = []
    if name in config.defines:
        origins.append("project define")
    variants = sorted(
        variant for variant, defines in config.variants.items() if name in defines
    )
    if variants:
        label = "variant define" if len(variants) == 1 else "variant defines"
        origins.append(f"{label}: {', '.join(variants)}")
    return "; ".join(origins) or "configured define"


def _define_shadowing_diagnostic(
    *,
    config: ProjectConfig,
    relative_path: str,
    line_number: int,
    column: int,
    directive: str,
    name: str,
) -> ProjectDiagnostic:
    verb = "redefines" if directive == "define" else "undefines"
    origin = _configured_define_origin_label(config, name)
    return ProjectDiagnostic(
        severity="warning",
        code="project.scan.define-shadowed",
        message=(
            f"Preprocessor directive in {relative_path}:{line_number} {verb} "
            f"configured define '{name}' ({origin}); project define "
            "preprocessing may not match source-local macro state."
        ),
        location=SourceLocation(
            file=relative_path,
            line=line_number,
            column=column,
            end_line=line_number,
            end_column=column,
        ),
        missing_capabilities=["macro.defines"],
    )


def _scan_define_shadowing_lines(
    config: ProjectConfig,
    lines: Sequence[str],
    relative_path: str,
    *,
    seen: set[tuple[str, int, str, str]],
    diagnostic_config: ProjectConfig | None = None,
) -> list[ProjectDiagnostic]:
    if not config.defines:
        return []

    diagnostic_config = diagnostic_config or config
    diagnostics: list[ProjectDiagnostic] = []
    conditional_stack: list[_IncludeConditionalFrame] = []
    for line_number, line in enumerate(lines, start=1):
        conditional = INCLUDE_CONDITIONAL_DIRECTIVE_RE.match(line)
        if conditional:
            _apply_include_conditional_directive(
                config,
                conditional_stack,
                conditional.group("directive"),
                conditional.group("body"),
            )
            continue
        if not _include_conditionals_active(conditional_stack):
            continue
        define_directive = DEFINE_DIRECTIVE_RE.match(line)
        if not define_directive:
            continue
        directive = define_directive.group("directive")
        name = define_directive.group("name")
        if name not in config.defines:
            continue
        key = (relative_path, line_number, directive, name)
        if key in seen:
            continue
        seen.add(key)
        diagnostics.append(
            _define_shadowing_diagnostic(
                config=diagnostic_config,
                relative_path=relative_path,
                line_number=line_number,
                column=max(1, line.find("#") + 1),
                directive=directive,
                name=name,
            )
        )
    return diagnostics


def _preprocessor_diagnostic(
    *,
    relative_path: str,
    line_number: int,
    column: int,
    directive: str,
    message: str,
    variant: str | None = None,
) -> ProjectDiagnostic:
    severity = "error" if directive == "error" else "warning"
    code = f"project.scan.preprocessor-{directive}"
    rendered_message = message or "(no message)"
    return ProjectDiagnostic(
        severity=severity,
        code=code,
        message=(
            f"Active preprocessor #{directive} directive in "
            f"{relative_path}:{line_number}: {rendered_message}"
        ),
        location=SourceLocation(
            file=relative_path,
            line=line_number,
            column=column,
            end_line=line_number,
            end_column=column,
        ),
        variant=variant,
    )


def _scan_preprocessor_diagnostic_lines(
    config: ProjectConfig,
    lines: Sequence[str],
    relative_path: str,
    *,
    seen: set[tuple[str, int, str, str]],
    variant: str | None = None,
) -> list[ProjectDiagnostic]:
    diagnostics: list[ProjectDiagnostic] = []
    conditional_stack: list[_IncludeConditionalFrame] = []
    for line_number, line in enumerate(lines, start=1):
        conditional = INCLUDE_CONDITIONAL_DIRECTIVE_RE.match(line)
        if conditional:
            _apply_include_conditional_directive(
                config,
                conditional_stack,
                conditional.group("directive"),
                conditional.group("body"),
            )
            continue
        if not _include_conditionals_active(conditional_stack):
            continue
        directive_match = PREPROCESSOR_DIAGNOSTIC_DIRECTIVE_RE.match(line)
        if not directive_match:
            continue
        directive = directive_match.group("directive")
        message = _strip_preprocessor_line_comment(directive_match.group("body"))
        key = (relative_path, line_number, directive, message)
        if key in seen:
            continue
        seen.add(key)
        diagnostics.append(
            _preprocessor_diagnostic(
                relative_path=relative_path,
                line_number=line_number,
                column=max(1, line.find("#") + 1),
                directive=directive,
                message=message,
                variant=variant,
            )
        )
    return diagnostics


def _include_conditionals_active(stack: Sequence[_IncludeConditionalFrame]) -> bool:
    return all(frame.branch_active for frame in stack)


def _apply_include_conditional_directive(
    config: ProjectConfig,
    stack: list[_IncludeConditionalFrame],
    directive: str,
    expression: str,
) -> None:
    if directive in {"if", "ifdef", "ifndef"}:
        parent_active = _include_conditionals_active(stack)
        condition = _include_condition_value(config, directive, expression)
        branch_active = (
            parent_active if condition is None else parent_active and condition
        )
        stack.append(
            _IncludeConditionalFrame(
                parent_active=parent_active,
                branch_active=branch_active,
                branch_taken=parent_active and condition is True,
            )
        )
        return

    if directive == "elif":
        if not stack:
            return
        frame = stack[-1]
        if frame.branch_taken:
            frame.branch_active = False
            return
        condition = _include_condition_value(config, "if", expression)
        frame.branch_active = (
            frame.parent_active
            if condition is None
            else frame.parent_active and condition
        )
        frame.branch_taken = frame.parent_active and condition is True
        return

    if directive == "else":
        if not stack:
            return
        frame = stack[-1]
        frame.branch_active = frame.parent_active and not frame.branch_taken
        frame.branch_taken = frame.branch_taken or frame.branch_active
        return

    if directive == "endif" and stack:
        stack.pop()


def _include_search_roots(
    config: ProjectConfig, source_path: Path, kind: str
) -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = []
    if kind == "local":
        roots.append(("source", source_path.parent.resolve()))
    roots.extend(
        ("include-dir", Path(include_dir).resolve())
        for include_dir in _frontend_include_dirs(config)
    )
    return roots


def _include_target_is_absolute(include_path: str) -> bool:
    return (
        Path(include_path).is_absolute()
        or PureWindowsPath(include_path).is_absolute()
        or bool(PureWindowsPath(include_path).drive)
    )


def _resolve_include_dependency(
    config: ProjectConfig,
    source_path: Path,
    kind: str,
    include_path: str,
) -> tuple[str, str | None, str | None]:
    if _include_target_is_absolute(include_path):
        return "outside-project", None, None

    outside_project = False
    for source, root in _include_search_roots(config, source_path, kind):
        candidate = (root / include_path).resolve()
        if not _is_relative_to(candidate, config.root):
            outside_project = True
            continue
        if candidate.is_file():
            return "resolved", _relpath(candidate, config.root), source

    if outside_project:
        return "outside-project", None, None
    if kind == "system":
        return "system", None, None
    return "missing", None, None


def _include_resolution_diagnostic(
    *,
    relative_path: str,
    line_number: int,
    status: str,
    include_path: str,
    location: SourceLocation,
    define_name: str | None = None,
    variant: str | None = None,
) -> ProjectDiagnostic | None:
    context_suffix = _include_resolution_context_suffix(
        define_name=define_name,
        variant=variant,
    )
    if status == "missing":
        return ProjectDiagnostic(
            severity="warning",
            code="project.scan.missing-include",
            message=(
                f"Include directive in {relative_path}:{line_number} "
                f"could not be resolved{context_suffix}: {include_path}"
            ),
            location=location,
            variant=variant,
            missing_capabilities=["include.resolution"],
        )
    if status == "outside-project":
        return ProjectDiagnostic(
            severity="warning",
            code="project.scan.include-outside-project",
            message=(
                f"Include directive in {relative_path}:{line_number} "
                "resolves outside the repository or uses an absolute "
                f"path{context_suffix}: {include_path}"
            ),
            location=location,
            variant=variant,
            missing_capabilities=["include.resolution"],
        )
    return None


def _include_resolution_context_suffix(
    *, define_name: str | None = None, variant: str | None = None
) -> str:
    if _is_non_empty_string(define_name):
        define_source = (
            f"variant {variant} define"
            if _is_non_empty_string(variant)
            else "project define"
        )
        return f" (from {define_source} {define_name})"
    if _is_non_empty_string(variant):
        return f" (for variant {variant})"
    return ""


def _include_cycle_diagnostic(
    *,
    relative_path: str,
    line_number: int,
    include_path: str,
    location: SourceLocation,
    variant: str | None = None,
) -> ProjectDiagnostic:
    context_suffix = _include_resolution_context_suffix(variant=variant)
    return ProjectDiagnostic(
        severity="warning",
        code="project.scan.include-cycle",
        message=(
            f"Include directive in {relative_path}:{line_number} creates a "
            f"cycle and was not scanned recursively{context_suffix}: {include_path}"
        ),
        location=location,
        variant=variant,
        missing_capabilities=["include.resolution"],
    )


def _scan_resolved_include_dependencies(
    config: ProjectConfig,
    unit_relative_path: str,
    resolved_path: str,
    *,
    line_number: int,
    include_path: str,
    location: SourceLocation,
    include_stack: tuple[str, ...],
    scanned_sources: set[str],
    define_shadowing_seen: set[tuple[str, int, str, str]],
    preprocessor_seen: set[tuple[str, int, str, str]],
    diagnostic_config: ProjectConfig,
    variant: str | None = None,
) -> tuple[list[dict[str, Any]], list[ProjectDiagnostic]]:
    if resolved_path in include_stack:
        return [], [
            _include_cycle_diagnostic(
                relative_path=location.file,
                line_number=line_number,
                include_path=include_path,
                location=location,
                variant=variant,
            )
        ]
    if resolved_path in scanned_sources:
        return [], []
    return _scan_include_dependencies_for_source(
        config,
        unit_relative_path,
        (config.root / resolved_path).resolve(),
        resolved_path,
        include_stack=(*include_stack, resolved_path),
        scanned_sources=scanned_sources,
        define_shadowing_seen=define_shadowing_seen,
        preprocessor_seen=preprocessor_seen,
        diagnostic_config=diagnostic_config,
        variant=variant,
    )


def _scan_include_dependencies_for_source(
    config: ProjectConfig,
    unit_relative_path: str,
    source_path: Path,
    source_relative_path: str,
    *,
    include_stack: tuple[str, ...],
    scanned_sources: set[str],
    define_shadowing_seen: set[tuple[str, int, str, str]],
    preprocessor_seen: set[tuple[str, int, str, str]],
    diagnostic_config: ProjectConfig,
    variant: str | None = None,
) -> tuple[list[dict[str, Any]], list[ProjectDiagnostic]]:
    dependencies: list[dict[str, Any]] = []
    diagnostics: list[ProjectDiagnostic] = []
    scanned_sources.add(source_relative_path)

    try:
        lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        diagnostics.append(
            ProjectDiagnostic(
                severity="warning",
                code="project.scan.include-read-failed",
                message=(
                    f"Could not scan include directives in {source_relative_path}: "
                    f"{exc}"
                ),
                location=SourceLocation(file=source_relative_path),
                missing_capabilities=["include.resolution"],
            )
        )
        return dependencies, diagnostics

    scan_lines = _mask_preprocessor_block_comments(lines)
    diagnostics.extend(
        _scan_define_shadowing_lines(
            config,
            scan_lines,
            source_relative_path,
            seen=define_shadowing_seen,
            diagnostic_config=diagnostic_config,
        )
    )
    diagnostics.extend(
        _scan_preprocessor_diagnostic_lines(
            config,
            scan_lines,
            source_relative_path,
            seen=preprocessor_seen,
            variant=variant,
        )
    )

    conditional_stack: list[_IncludeConditionalFrame] = []
    for line_number, line in enumerate(scan_lines, start=1):
        conditional = INCLUDE_CONDITIONAL_DIRECTIVE_RE.match(line)
        if conditional:
            _apply_include_conditional_directive(
                config,
                conditional_stack,
                conditional.group("directive"),
                conditional.group("body"),
            )
            continue
        if not _include_conditionals_active(conditional_stack):
            continue
        directive = INCLUDE_DIRECTIVE_RE.match(line)
        if not directive:
            continue
        raw_body = directive.group("body").strip()
        column = max(1, line.find("#") + 1)
        location = SourceLocation(
            file=source_relative_path,
            line=line_number,
            column=column,
            end_line=line_number,
            end_column=column,
        )
        literal = _include_literal(raw_body)
        if literal is None:
            body = _strip_include_line_comment(raw_body) or raw_body
            define_literal = _include_literal_from_define(config, body)
            if define_literal is not None:
                define_name, kind, include_path = define_literal
                status, resolved_path, resolved_from = _resolve_include_dependency(
                    config, source_path, kind, include_path
                )
                dependency = {
                    "include": include_path,
                    "kind": kind,
                    "status": status,
                    "line": line_number,
                    "column": column,
                    "resolvedFromDefine": define_name,
                }
                if variant is not None:
                    dependency["variant"] = variant
                if source_relative_path != unit_relative_path:
                    dependency["source"] = source_relative_path
                if resolved_path:
                    resolved_file = config.root / resolved_path
                    dependency["resolvedPath"] = resolved_path
                    dependency["resolvedHash"] = _source_hash(resolved_file)
                    dependency["resolvedSizeBytes"] = resolved_file.stat().st_size
                if resolved_from:
                    dependency["resolvedFrom"] = resolved_from
                dependencies.append(dependency)
                diagnostic = _include_resolution_diagnostic(
                    relative_path=source_relative_path,
                    line_number=line_number,
                    status=status,
                    include_path=include_path,
                    location=location,
                    define_name=define_name,
                    variant=variant,
                )
                if diagnostic is not None:
                    diagnostics.append(diagnostic)
                if resolved_path:
                    nested_dependencies, nested_diagnostics = (
                        _scan_resolved_include_dependencies(
                            config,
                            unit_relative_path,
                            resolved_path,
                            line_number=line_number,
                            include_path=include_path,
                            location=location,
                            include_stack=include_stack,
                            scanned_sources=scanned_sources,
                            define_shadowing_seen=define_shadowing_seen,
                            preprocessor_seen=preprocessor_seen,
                            diagnostic_config=diagnostic_config,
                            variant=variant,
                        )
                    )
                    dependencies.extend(nested_dependencies)
                    diagnostics.extend(nested_diagnostics)
                continue

            dependency = {
                "include": body,
                "kind": "dynamic",
                "status": "dynamic",
                "line": line_number,
                "column": column,
            }
            if variant is not None:
                dependency["variant"] = variant
            if source_relative_path != unit_relative_path:
                dependency["source"] = source_relative_path
            dependencies.append(dependency)
            diagnostics.append(
                ProjectDiagnostic(
                    severity="warning",
                    code="project.scan.dynamic-include",
                    message=(
                        "Include directive uses a dynamic target that cannot be "
                        "resolved during project scan"
                        f"{_include_resolution_context_suffix(variant=variant)}: "
                        f"{body}"
                    ),
                    location=location,
                    variant=variant,
                    missing_capabilities=["include.resolution"],
                )
            )
            continue

        kind, include_path = literal
        status, resolved_path, resolved_from = _resolve_include_dependency(
            config, source_path, kind, include_path
        )
        dependency = {
            "include": include_path,
            "kind": kind,
            "status": status,
            "line": line_number,
            "column": column,
        }
        if variant is not None:
            dependency["variant"] = variant
        if source_relative_path != unit_relative_path:
            dependency["source"] = source_relative_path
        if resolved_path:
            resolved_file = config.root / resolved_path
            dependency["resolvedPath"] = resolved_path
            dependency["resolvedHash"] = _source_hash(resolved_file)
            dependency["resolvedSizeBytes"] = resolved_file.stat().st_size
        if resolved_from:
            dependency["resolvedFrom"] = resolved_from
        dependencies.append(dependency)
        diagnostic = _include_resolution_diagnostic(
            relative_path=source_relative_path,
            line_number=line_number,
            status=status,
            include_path=include_path,
            location=location,
            variant=variant,
        )
        if diagnostic is not None:
            diagnostics.append(diagnostic)
        if resolved_path:
            nested_dependencies, nested_diagnostics = (
                _scan_resolved_include_dependencies(
                    config,
                    unit_relative_path,
                    resolved_path,
                    line_number=line_number,
                    include_path=include_path,
                    location=location,
                    include_stack=include_stack,
                    scanned_sources=scanned_sources,
                    define_shadowing_seen=define_shadowing_seen,
                    preprocessor_seen=preprocessor_seen,
                    diagnostic_config=diagnostic_config,
                    variant=variant,
                )
            )
            dependencies.extend(nested_dependencies)
            diagnostics.extend(nested_diagnostics)

    return dependencies, diagnostics


def _scan_include_dependencies(
    config: ProjectConfig, unit_path: Path, relative_path: str
) -> tuple[list[dict[str, Any]], list[ProjectDiagnostic]]:
    dependencies: list[dict[str, Any]] = []
    diagnostics: list[ProjectDiagnostic] = []
    define_shadowing_seen: set[tuple[str, int, str, str]] = set()
    preprocessor_seen: set[tuple[str, int, str, str]] = set()
    for variant, defines in _variant_jobs(config):
        scan_config = (
            replace(config, defines=defines) if variant is not None else config
        )
        variant_dependencies, variant_diagnostics = (
            _scan_include_dependencies_for_source(
                scan_config,
                relative_path,
                unit_path,
                relative_path,
                include_stack=(relative_path,),
                scanned_sources=set(),
                define_shadowing_seen=define_shadowing_seen,
                preprocessor_seen=preprocessor_seen,
                diagnostic_config=config,
                variant=variant,
            )
        )
        dependencies.extend(variant_dependencies)
        diagnostics.extend(variant_diagnostics)
    return dependencies, diagnostics


def _resolved_source_root(config: ProjectConfig, source_root: str) -> Path:
    return _project_config_path(config.root, source_root).resolve()


def _source_root_status_for_path(config: ProjectConfig, path: Path) -> str:
    if not _is_relative_to(path, config.root):
        return "outside-project"
    if not path.exists():
        return "missing"
    if not path.is_dir():
        return "not-directory"
    return "active"


def _source_root_status_records(config: ProjectConfig) -> list[dict[str, Any]]:
    records = []
    for source_root in config.source_roots:
        absolute_root = _resolved_source_root(config, source_root)
        status = _source_root_status_for_path(config, absolute_root)
        records.append(
            {
                "path": source_root,
                "resolvedPath": str(absolute_root),
                "status": status,
                "scanVisible": status == "active",
            }
        )
    return records


def _source_root_status_counts(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts = {status: 0 for status in sorted(SOURCE_ROOT_STATUSES)}
    for record in records:
        status = record.get("status")
        if isinstance(status, str) and status in counts:
            counts[status] += 1
    return {status: count for status, count in counts.items() if count}


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
    selected_variants: Sequence[str] = ()
    external_corpus_manifest: str | None = None

    def normalized_targets(self) -> list[str]:
        return _normalized_targets(self.targets)

    @property
    def output_path(self) -> Path:
        return _project_output_path(self.root, self.output_dir)


def _project_config_hash(config: ProjectConfig) -> dict[str, str] | None:
    if config.config_path is None or not config.config_path.is_file():
        return None
    return _source_hash(config.config_path)


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
    source_backend: str | None = None
    variant: str | None = None
    missing_capabilities: Sequence[str] = ()
    original_location: SourceLocation | None = None

    def to_json(self) -> dict[str, Any]:
        payload = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "location": self.location.to_json(),
        }
        if self.original_location:
            payload["originalLocation"] = self.original_location.to_json()
        if self.target:
            payload["target"] = self.target
        if self.source_backend:
            payload["sourceBackend"] = self.source_backend
        if self.variant:
            payload["variant"] = self.variant
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
    seen_ids: dict[str, int] = {}
    seen_paths: dict[str, int] = {}
    for index, entry in enumerate(manifest.get("entries", [])):
        reasons = _external_corpus_manifest_entry_reasons(entry)
        if not reasons and isinstance(entry, Mapping):
            reasons = _external_corpus_manifest_entry_duplicate_reasons(
                entry,
                seen_ids=seen_ids,
                seen_paths=seen_paths,
            )
        if not reasons:
            entry_id, path = _external_corpus_manifest_entry_identity(entry)
            if entry_id is not None:
                seen_ids[entry_id] = index
            if path is not None:
                seen_paths[path] = index
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


def _external_corpus_source_backend_mismatch_diagnostics(
    config: ProjectConfig,
    units: Sequence[ProjectTranslationUnit],
) -> list[ProjectDiagnostic]:
    manifest_path = _external_corpus_manifest_path(config)
    if manifest_path is None or not _is_relative_to(manifest_path, config.root):
        return []

    manifest, _status = _load_external_corpus_manifest(config)
    if manifest is None:
        return []

    units_by_path = {unit.relative_path: unit for unit in units}
    diagnostics: list[ProjectDiagnostic] = []
    for index, entry in _valid_external_corpus_manifest_entries(manifest):
        path = str(entry.get("path", "")).replace("\\", "/")
        unit = units_by_path.get(path)
        declared_source_backend = entry.get("sourceBackend")
        if unit is None or not _is_non_empty_string(declared_source_backend):
            continue
        declared_backend = _external_corpus_source_backend(declared_source_backend)
        if declared_backend == unit.source_backend:
            continue
        diagnostics.append(
            ProjectDiagnostic(
                severity="warning",
                code="project.config.external-corpus-source-backend-mismatch",
                message=(
                    f"External corpus manifest entry {index + 1} declares "
                    f"sourceBackend {declared_backend} for {path}, but project "
                    f"discovery resolved {unit.source_backend}; the report uses "
                    "the discovered source backend."
                ),
                location=_config_location(config),
                missing_capabilities=["external.corpus"],
            )
        )
    return diagnostics


def _source_root_diagnostics(config: ProjectConfig) -> list[ProjectDiagnostic]:
    diagnostics: list[ProjectDiagnostic] = []
    location = _config_location(config)
    for source_root in config.source_roots:
        absolute_root = _resolved_source_root(config, source_root)
        status = _source_root_status_for_path(config, absolute_root)
        if status == "outside-project":
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
        if status == "missing":
            diagnostics.append(
                ProjectDiagnostic(
                    severity="warning",
                    code="project.scan.missing-source-root",
                    message=f"Configured source root does not exist: {source_root}",
                    location=location,
                    missing_capabilities=["repo.scan"],
                )
            )
            continue
        if status == "not-directory":
            diagnostics.append(
                ProjectDiagnostic(
                    severity="warning",
                    code="project.config.source-root-not-directory",
                    message=(
                        "Configured source root resolves to a file or "
                        f"non-directory path: {source_root}"
                    ),
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
        status = _include_dir_status_for_path(config, absolute_dir)
        if status == "outside-project":
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
        if status == "missing":
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
            continue
        if status == "not-directory":
            diagnostics.append(
                ProjectDiagnostic(
                    severity="warning",
                    code="project.config.include-dir-not-directory",
                    message=(
                        "Configured include directory resolves to a file or "
                        f"non-directory path: {include_dir}"
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
    for pattern in config.exclude_patterns:
        if _is_repository_relative_glob(pattern):
            continue
        diagnostics.append(
            ProjectDiagnostic(
                severity="error",
                code="project.config.exclude-pattern-outside-project",
                message=(
                    f"Configured exclude pattern '{pattern}' is not "
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
    if config.source_overrides:
        register_default_sources()
        discover_backend_plugins()
    supported_sources = ", ".join(SOURCE_REGISTRY.names())
    for pattern, backend in config.source_overrides.items():
        if _is_non_empty_string(backend) and SOURCE_REGISTRY.get(backend):
            continue
        diagnostics.append(
            ProjectDiagnostic(
                severity="error",
                code="project.config.unsupported-source-override",
                message=(
                    f"Source override '{pattern}' references unsupported backend "
                    f"'{backend}'. Supported source backends: {supported_sources}"
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
    source_hash: Mapping[str, str]
    source_size_bytes: int
    source_override: str | None = None
    include_dependencies: Sequence[Mapping[str, Any]] = ()

    def to_json(self) -> dict[str, Any]:
        payload = {
            "id": self.relative_path,
            "path": self.relative_path,
            "sourceBackend": self.source_backend,
            "extension": self.extension,
            "sourceHash": dict(self.source_hash),
            "sourceSizeBytes": self.source_size_bytes,
        }
        if self.source_override:
            payload["sourceOverride"] = self.source_override
        if self.include_dependencies:
            payload["includeDependencies"] = [
                dict(dependency) for dependency in self.include_dependencies
            ]
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
            migration_actions=_project_migration_actions(self.units, report_targets),
            artifact_matrix=_artifact_matrix_report(
                self.config, self.units, report_targets, self.config.variants, []
            ),
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
    artifact_matrix: Mapping[str, Any] | None = None
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
        source_map_rollups = _source_map_rollups(self.artifacts)
        define_processing_rollups = _define_processing_rollups(self.artifacts)
        include_path_processing_rollups = _include_path_processing_rollups(
            self.artifacts
        )
        source_root_status = _source_root_status_records(self.config)
        include_dir_status = _include_dir_status_records(self.config)
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
                "configHash": _project_config_hash(self.config),
                "sourceRoots": list(self.config.source_roots),
                "sourceRootCount": len(self.config.source_roots),
                "sourceRootStatus": source_root_status,
                "sourceRootStatusCounts": _source_root_status_counts(
                    source_root_status
                ),
                "includePatterns": list(self.config.include_patterns),
                "includePatternCount": len(self.config.include_patterns),
                "excludePatterns": list(self.config.exclude_patterns),
                "excludePatternCount": len(self.config.exclude_patterns),
                "targets": list(self.targets),
                "outputDir": str(self.config.output_path),
                "sourceOverrides": dict(sorted(self.config.source_overrides.items())),
                "sourceOverrideCount": len(self.config.source_overrides),
                "includeDirs": list(self.config.include_dirs),
                "includeDirCount": len(self.config.include_dirs),
                "includeDirStatus": include_dir_status,
                "includeDirStatusCounts": _include_dir_status_counts(
                    include_dir_status
                ),
                "defines": dict(sorted(self.config.defines.items())),
                "defineCount": len(self.config.defines),
                "variants": {
                    name: dict(sorted(defines.items()))
                    for name, defines in sorted(self.config.variants.items())
                },
                "variantCount": len(self.config.variants),
                "variantDefineCounts": _variant_define_counts(self.config.variants),
                "selectedVariants": list(self.config.selected_variants),
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
                "diagnosticsByTarget": _diagnostic_counts_by_target(self.diagnostics),
                "diagnosticsBySourceBackend": _diagnostic_counts_by_source_backend(
                    self.diagnostics
                ),
                "diagnosticsByVariant": _diagnostic_counts_by_variant(self.diagnostics),
                "missingCapabilityCounts": _missing_capability_counts(self.diagnostics),
                "unitsBySourceBackend": _unit_counts_by_source_backend(self.units),
                "unitsByExtension": _unit_counts_by_extension(self.units),
                "unitsBySourceOverride": _unit_counts_by_source_override(self.units),
                "includeDependencyCount": _include_dependency_count(self.units),
                "includeDependenciesByKind": _include_dependency_counts_by_kind(
                    self.units
                ),
                "includeDependenciesByStatus": _include_dependency_counts_by_status(
                    self.units
                ),
                "includeDependenciesByResolvedFrom": (
                    _include_dependency_counts_by_resolved_from(self.units)
                ),
                "includeDependenciesBySourceBackend": (
                    _include_dependency_counts_by_source_backend(self.units)
                ),
                "includeDependenciesBySourceBackendStatus": (
                    _include_dependency_counts_by_source_backend_status(self.units)
                ),
                "includeDependenciesByVariant": _include_dependency_counts_by_variant(
                    self.units
                ),
                "skippedByReason": _skipped_counts_by_reason(self.skipped),
                "skippedByExtension": _skipped_counts_by_extension(self.skipped),
                "skippedBySourceOverride": _skipped_counts_by_source_override(
                    self.skipped
                ),
                "artifactsBySourceBackend": _artifact_counts_by_source_backend(
                    self.artifacts
                ),
                "artifactsByVariant": _artifact_counts_by_variant(self.artifacts),
                "artifactsByTarget": _artifact_counts_by_target(self.artifacts),
                **_artifact_provenance_rollups(self.artifacts),
                **source_map_rollups,
                **define_processing_rollups,
                **include_path_processing_rollups,
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
                **_migration_action_rollups(self.migration_actions),
                "actions": list(self.migration_actions),
            },
        }
        if self.artifact_matrix is not None:
            payload["artifactMatrix"] = dict(self.artifact_matrix)
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
    explicit_config = config is not None
    if explicit_config:
        config_value = os.fspath(config)
        if not config_value.strip():
            raise ValueError("Project config path must be non-empty")
        config_path = _project_config_path(root_path, config_value).resolve()
    else:
        config_path = root_path / DEFAULT_CONFIG_NAME
    if not config_path.exists():
        if explicit_config:
            raise ValueError(f"Project config not found: {config_path}")
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
    external_corpus_manifest = _as_optional_non_empty_str(
        project.get("external_corpus_manifest"),
        field_name="crosstl.toml project.external_corpus_manifest",
    )
    if external_corpus_manifest is not None:
        external_corpus_manifest = _normalize_project_relative_path(
            external_corpus_manifest
        )

    excludes = _as_str_list(project.get("exclude"), field_name="project.exclude")
    if not excludes:
        excludes = list(DEFAULT_EXCLUDE_PATTERNS)
    return ProjectConfig(
        root=root_path,
        config_path=config_path,
        source_roots=(
            _normalize_project_relative_paths(
                _as_str_list(
                    project.get("source_roots", "."),
                    field_name="project.source_roots",
                )
            )
            or (".",)
        ),
        include_patterns=_normalize_project_relative_paths(
            _as_str_list(project.get("include"), field_name="project.include")
        ),
        exclude_patterns=_normalize_project_relative_paths(excludes),
        targets=_as_str_list(project.get("targets"), field_name="project.targets"),
        output_dir=_normalize_project_relative_path(output_dir or DEFAULT_OUTPUT_DIR),
        source_overrides=_normalize_project_relative_path_mapping(sources),
        include_dirs=_normalize_project_relative_paths(
            _as_str_list(project.get("include_dirs"), field_name="project.include_dirs")
        ),
        defines=defines,
        variants=_variant_defines(variants),
        selected_variants=_as_str_list(
            project.get("selected_variants"),
            field_name="project.selected_variants",
        ),
        external_corpus_manifest=external_corpus_manifest,
    )


def _override_for_path(relative_path: str, config: ProjectConfig) -> str | None:
    for pattern, backend in config.source_overrides.items():
        if fnmatch.fnmatch(relative_path, _normalize_project_relative_path(pattern)):
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
        if not absolute_root.is_dir():
            continue
        for pattern in include_patterns:
            if explicit_include_patterns or pattern in source_override_patterns:
                candidates.update(
                    path
                    for path in config.root.glob(pattern)
                    if path.is_file() and _is_relative_to(path, absolute_root)
                )
            else:
                candidates.update(
                    path for path in absolute_root.glob(pattern) if path.is_file()
                )
    return sorted(candidates)


def scan_project(
    config_or_root: ProjectConfig | str | os.PathLike[str],
    *,
    variants: Sequence[str] | str | None = None,
) -> ProjectScan:
    """Discover supported shader/GPU source files in a repository."""
    config = (
        config_or_root
        if isinstance(config_or_root, ProjectConfig)
        else load_project_config(config_or_root)
    )
    config = _config_with_selected_variants(config, variants)
    units: list[ProjectTranslationUnit] = []
    skipped: list[dict[str, Any]] = []
    diagnostics: list[ProjectDiagnostic] = _configuration_diagnostics(config)
    diagnostics.extend(_source_root_diagnostics(config))
    diagnostics.extend(_include_dir_diagnostics(config))
    diagnostics.extend(_scan_pattern_diagnostics(config))
    exclude_patterns = _repository_relative_globs(config.exclude_patterns)
    internal_exclude_patterns = _internal_exclude_patterns(config)

    for path in _iter_scan_candidates(config):
        try:
            relative_path = _relpath(path, config.root)
        except ValueError:
            continue
        if _path_matches(relative_path, exclude_patterns):
            continue
        if _path_matches(relative_path, internal_exclude_patterns):
            continue

        override = _override_for_path(relative_path, config)
        unsupported_extension_message = (
            None
            if override
            else SOURCE_REGISTRY.unsupported_extension_message(str(path))
        )
        if unsupported_extension_message:
            skipped.append({"path": relative_path, "reason": "unsupported-extension"})
            diagnostics.append(
                ProjectDiagnostic(
                    severity="warning",
                    code="project.scan.unsupported-source",
                    message=f"{relative_path}: {unsupported_extension_message}",
                    location=SourceLocation(file=relative_path),
                    missing_capabilities=["source.discovery"],
                )
            )
            continue

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

        include_dependencies, include_diagnostics = _scan_include_dependencies(
            config, path, relative_path
        )
        diagnostics.extend(include_diagnostics)
        units.append(
            ProjectTranslationUnit(
                path=path,
                relative_path=relative_path,
                source_backend=source_spec.name,
                extension=path.suffix.lower(),
                source_hash=_source_hash(path),
                source_size_bytes=path.stat().st_size,
                source_override=override,
                include_dependencies=include_dependencies,
            )
        )

    diagnostics.extend(
        _external_corpus_source_backend_mismatch_diagnostics(config, units)
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


def _selected_variant_names(variants: Sequence[str] | str | None) -> list[str] | None:
    if variants is None:
        return None
    names = [variants] if isinstance(variants, str) else list(variants)
    if not names:
        raise ValueError(
            "selected project variants must include at least one variant name"
        )

    selected: list[str] = []
    seen: set[str] = set()
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("selected project variants must be non-empty strings")
        if name not in seen:
            selected.append(name)
            seen.add(name)
    return selected


def _config_with_selected_variants(
    config: ProjectConfig, variants: Sequence[str] | str | None
) -> ProjectConfig:
    selected = _selected_variant_names(variants)
    if selected is None and config.selected_variants:
        selected = _selected_variant_names(config.selected_variants)
    if selected is None:
        return config

    unknown = [name for name in selected if name not in config.variants]
    if unknown:
        available = ", ".join(sorted(config.variants)) or "(none)"
        raise ValueError(
            "selected project variant is not declared in project config: "
            f"{', '.join(unknown)} (available: {available})"
        )
    return replace(
        config,
        variants={name: config.variants[name] for name in selected},
        selected_variants=tuple(selected),
    )


def _artifact_matrix_report(
    config: ProjectConfig,
    units: Sequence[ProjectTranslationUnit],
    targets: Sequence[str],
    variants: Mapping[str, Mapping[str, str]],
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    variant_count = len(variants)
    variant_factor = variant_count if variant_count else 1
    normalized_targets = _normalized_targets(targets)
    payload = {
        "unitCount": len(units),
        "targetCount": len(normalized_targets),
        "variantCount": variant_count,
        "variantMode": "named" if variant_count else "none",
        "expectedArtifactCount": len(units) * len(normalized_targets) * variant_factor,
    }
    artifact_identities = {
        identity
        for artifact in artifacts
        if isinstance(artifact, Mapping)
        for identity in (_artifact_identity(artifact),)
        if identity is not None
    }
    variant_names: list[str | None] = sorted(variants) if variants else [None]
    expected_identities = {
        identity
        for unit in units
        for target in normalized_targets
        for variant in variant_names
        for identity in (
            _expected_artifact_identity(
                config.root,
                config.output_path,
                unit.relative_path,
                target,
                variant,
            ),
        )
        if identity is not None
    }
    missing_identities = expected_identities - artifact_identities
    extra_identities = artifact_identities - expected_identities
    source_backend_by_identity = _artifact_matrix_source_backend_by_identity(
        expected_identities,
        artifacts,
        _artifact_matrix_unit_source_backends(units),
    )
    payload.update(
        {
            "emittedArtifactCount": len(artifacts),
            "translatedCount": sum(
                1 for artifact in artifacts if artifact.get("status") == "translated"
            ),
            "failedCount": sum(
                1 for artifact in artifacts if artifact.get("status") == "failed"
            ),
            "identityCoverageAvailable": True,
            "missingArtifactCount": len(missing_identities),
            "extraArtifactCount": len(extra_identities),
            "complete": not missing_identities and not extra_identities,
            "statusByTarget": _artifact_matrix_status_by_field(
                expected_identities,
                artifacts,
                missing_identities,
                extra_identities,
                "target",
            ),
            "statusBySourceBackend": _artifact_matrix_status_by_field(
                expected_identities,
                artifacts,
                missing_identities,
                extra_identities,
                "sourceBackend",
                source_backend_by_identity=source_backend_by_identity,
            ),
            "statusByVariant": _artifact_matrix_status_by_field(
                expected_identities,
                artifacts,
                missing_identities,
                extra_identities,
                "variant",
            ),
        }
    )
    return payload


def _artifact_matrix_status_row() -> dict[str, Any]:
    return {
        "expectedArtifactCount": 0,
        "emittedArtifactCount": 0,
        "translatedCount": 0,
        "failedCount": 0,
        "missingArtifactCount": 0,
        "extraArtifactCount": 0,
        "complete": True,
    }


def _artifact_matrix_identity_field(
    identity: ArtifactIdentity,
    field_name: str,
    source_backend_by_identity: Mapping[ArtifactIdentity, str] | None = None,
) -> str | None:
    if field_name == "target":
        return identity[1]
    if field_name == "sourceBackend" and source_backend_by_identity is not None:
        return source_backend_by_identity.get(identity)
    if field_name == "variant":
        return identity[3]
    return None


def _artifact_matrix_status_by_field(
    expected_identities: set[ArtifactIdentity],
    artifacts: Sequence[Mapping[str, Any]],
    missing_identities: set[ArtifactIdentity],
    extra_identities: set[ArtifactIdentity],
    field_name: str,
    *,
    source_backend_by_identity: Mapping[ArtifactIdentity, str] | None = None,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for identity in expected_identities:
        value = _artifact_matrix_identity_field(
            identity,
            field_name,
            source_backend_by_identity,
        )
        if value is None:
            continue
        rows.setdefault(value, _artifact_matrix_status_row())[
            "expectedArtifactCount"
        ] += 1
    for artifact in artifacts:
        identity = _artifact_identity(artifact)
        if identity is None:
            continue
        value = _artifact_matrix_identity_field(
            identity,
            field_name,
            source_backend_by_identity,
        )
        if value is None:
            continue
        row = rows.setdefault(value, _artifact_matrix_status_row())
        row["emittedArtifactCount"] += 1
        status = artifact.get("status")
        if status == "translated":
            row["translatedCount"] += 1
        elif status == "failed":
            row["failedCount"] += 1
    for identity in missing_identities:
        value = _artifact_matrix_identity_field(
            identity,
            field_name,
            source_backend_by_identity,
        )
        if value is not None:
            rows.setdefault(value, _artifact_matrix_status_row())[
                "missingArtifactCount"
            ] += 1
    for identity in extra_identities:
        value = _artifact_matrix_identity_field(
            identity,
            field_name,
            source_backend_by_identity,
        )
        if value is not None:
            rows.setdefault(value, _artifact_matrix_status_row())[
                "extraArtifactCount"
            ] += 1
    for row in rows.values():
        row["complete"] = (
            row["missingArtifactCount"] == 0 and row["extraArtifactCount"] == 0
        )
    return {name: rows[name] for name in sorted(rows)}


def _artifact_matrix_unit_source_backends(units: Sequence[Any]) -> dict[str, str]:
    source_backends: dict[str, str] = {}
    for unit in units:
        if isinstance(unit, ProjectTranslationUnit):
            source = unit.relative_path
            source_backend = unit.source_backend
        elif isinstance(unit, Mapping):
            source = unit.get("path")
            source_backend = unit.get("sourceBackend")
        else:
            continue
        if (
            _is_non_empty_string(source)
            and _is_report_identity_path(source)
            and _is_non_empty_string(source_backend)
        ):
            source_backends.setdefault(source, source_backend)
    return source_backends


def _artifact_matrix_source_backend_by_identity(
    expected_identities: set[ArtifactIdentity],
    artifacts: Sequence[Mapping[str, Any]],
    source_backend_by_source: Mapping[str, str],
) -> dict[ArtifactIdentity, str]:
    source_backend_by_identity = {
        identity: source_backend_by_source[identity[0]]
        for identity in expected_identities
        if identity[0] in source_backend_by_source
    }
    for artifact in artifacts:
        identity = _artifact_identity(artifact)
        if identity is None:
            continue
        source_backend = artifact.get("sourceBackend")
        if not _is_non_empty_string(source_backend):
            source_backend = source_backend_by_source.get(identity[0])
        if _is_non_empty_string(source_backend):
            source_backend_by_identity[identity] = source_backend
    return source_backend_by_identity


def _artifact_source_map(
    config: ProjectConfig,
    unit: ProjectTranslationUnit,
    target: str,
    output_path: Path,
) -> dict[str, Any]:
    artifact_path = _artifact_report_path(output_path, config)
    source_span = _file_span(unit.path, unit.relative_path)
    generated_span = _file_span(output_path, artifact_path)
    line_mappings = _line_preserving_source_map_mappings(
        unit.path,
        unit.relative_path,
        output_path,
        artifact_path,
    )
    mapping_granularity = "line" if line_mappings else "file"
    mappings = (
        line_mappings
        if line_mappings
        else [
            {
                "source": source_span.to_json(),
                "generated": generated_span.to_json(),
            }
        ]
    )
    return {
        "schemaVersion": 1,
        "kind": "crosstl-artifact-source-map",
        "mappingGranularity": mapping_granularity,
        "target": target,
        "source": source_span.to_json(),
        "generated": generated_span.to_json(),
        "mappings": mappings,
    }


def _is_crossgl_target(target: str) -> bool:
    return _normalized_targets([target])[0] in CROSSL_TARGETS


def _source_remap_report_path(artifact_path: str) -> str:
    path = PurePosixPath(artifact_path.replace("\\", "/"))
    return path.with_name(f"{path.stem}.source-remap.json").as_posix()


def _source_remap_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.source-remap.json")


def _source_remap_payload(source_map: Mapping[str, Any]) -> dict[str, Any]:
    source_mappings = source_map.get("mappings")
    if not isinstance(source_mappings, list) or not source_mappings:
        source_mappings = [
            {
                "source": source_map["source"],
                "generated": source_map["generated"],
            }
        ]
    return {
        "schemaVersion": SOURCE_REMAP_SCHEMA_VERSION,
        "generatedFile": source_map["generated"]["file"],
        "mappings": [
            {
                "generated": dict(mapping["generated"]),
                "original": dict(mapping["source"]),
            }
            for mapping in source_mappings
        ],
    }


def _write_source_remap_sidecar(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _artifact_source_remap(
    config: ProjectConfig,
    target: str,
    artifact_path: str,
    remap_path: Path,
    mapping_granularity: str,
    mapping_count: int,
) -> dict[str, Any]:
    return {
        "schemaVersion": SOURCE_REMAP_SCHEMA_VERSION,
        "path": _artifact_report_path(remap_path, config),
        "target": target,
        "generatedFile": artifact_path,
        "mappingGranularity": mapping_granularity,
        "mappingCount": mapping_count,
        "sizeBytes": remap_path.stat().st_size,
        "hash": _source_hash(remap_path),
    }


def _unsupported_mapping_field_reasons(
    prefix: str, value: Mapping[str, Any], allowed_fields: frozenset[str]
) -> list[str]:
    reasons = []
    for field_name in sorted(value, key=str):
        if field_name not in allowed_fields:
            reasons.append(f"{prefix}.{field_name} is not allowed")
    return reasons


def _compiler_source_remap_span_reasons(prefix: str, value: Any) -> list[str]:
    reasons = _source_map_span_reasons(prefix, value)
    if reasons:
        return reasons

    if not isinstance(value, Mapping):
        return reasons
    reasons.extend(
        _unsupported_mapping_field_reasons(
            prefix, value, COMPILER_SOURCE_REMAP_SPAN_FIELDS
        )
    )

    file_path = value["file"]
    if not _is_stable_relative_posix_path(file_path):
        reasons.append(f"{prefix}.file must be a stable relative POSIX source path")
    for field_name in ("line", "column", "length", "endLine", "endColumn"):
        if value[field_name] <= 0:
            reasons.append(f"{prefix}.{field_name} must be greater than zero")
    if (
        value["length"] > 0
        and value["endLine"] == value["line"]
        and value["endColumn"] <= value["column"]
    ):
        reasons.append(f"{prefix}.endColumn must be greater than column")
    return reasons


def _compiler_source_remap_payload_reasons(payload: Any) -> list[str]:
    if not isinstance(payload, Mapping):
        return ["$ must be an object"]

    reasons = _unsupported_mapping_field_reasons(
        "$", payload, COMPILER_SOURCE_REMAP_PAYLOAD_FIELDS
    )
    if payload.get("schemaVersion") != SOURCE_REMAP_SCHEMA_VERSION:
        reasons.append("$.schemaVersion must be 1")

    generated_file = payload.get("generatedFile")
    if not _is_non_empty_string(generated_file):
        reasons.append("$.generatedFile must be a string")
    elif not _is_stable_relative_posix_path(generated_file):
        reasons.append("$.generatedFile must be a stable relative POSIX source path")

    mappings = payload.get("mappings")
    if not isinstance(mappings, list):
        reasons.append("$.mappings must be a list")
        return reasons
    if not mappings:
        reasons.append("$.mappings must not be empty")

    for mapping_index, mapping in enumerate(mappings):
        mapping_prefix = f"$.mappings[{mapping_index}]"
        if not isinstance(mapping, Mapping):
            reasons.append(f"{mapping_prefix} must be an object")
            continue
        reasons.extend(
            _unsupported_mapping_field_reasons(
                mapping_prefix, mapping, COMPILER_SOURCE_REMAP_MAPPING_FIELDS
            )
        )
        generated = mapping.get("generated")
        original = mapping.get("original")
        reasons.extend(
            _compiler_source_remap_span_reasons(
                f"{mapping_prefix}.generated", generated
            )
        )
        reasons.extend(
            _compiler_source_remap_span_reasons(f"{mapping_prefix}.original", original)
        )
        if (
            isinstance(generated, Mapping)
            and _is_non_empty_string(generated.get("file"))
            and _is_non_empty_string(generated_file)
            and generated["file"] != generated_file
        ):
            reasons.append(
                f"{mapping_prefix}.generated.file must match $.generatedFile"
            )
    return reasons


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


def _has_system_include_dependencies(units: Sequence[ProjectTranslationUnit]) -> bool:
    for unit in units:
        for dependency in unit.include_dependencies:
            if (
                isinstance(dependency, Mapping)
                and dependency.get("status") == "system"
                and dependency.get("kind") == "system"
            ):
                return True
    return False


def _include_migration_actions(
    units: Sequence[ProjectTranslationUnit],
    targets: Sequence[str],
    artifacts: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not _has_system_include_dependencies(units):
        return []

    action_targets = _runtime_migration_targets(units, targets, artifacts)
    if not action_targets:
        return []

    return [
        {
            "kind": "manual-include-resolution",
            "severity": "note",
            "message": (
                "Review unresolved system include dependencies against target "
                "SDK or toolchain headers; CrossTL records them for portability "
                "but does not rewrite SDK or framework headers automatically."
            ),
            "targets": action_targets,
        }
    ]


def _project_migration_actions(
    units: Sequence[ProjectTranslationUnit],
    targets: Sequence[str],
    artifacts: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    return [
        *_runtime_migration_actions(units, targets, artifacts),
        *_include_migration_actions(units, targets, artifacts),
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
    variants: Sequence[str] | str | None = None,
    format_output: bool = False,
    validate: bool = False,
    run_toolchains: bool = False,
) -> ProjectPortabilityReport:
    """Translate all discovered project units to one or more target backends."""
    config = (
        config_or_root
        if isinstance(config_or_root, ProjectConfig)
        else load_project_config(config_or_root)
    )
    if output_dir is not None:
        output_dir = str(output_dir)
        if not output_dir.strip():
            raise ValueError("output_dir must be a non-empty string")
        config = ProjectConfig(
            root=config.root,
            config_path=config.config_path,
            source_roots=config.source_roots,
            include_patterns=config.include_patterns,
            exclude_patterns=config.exclude_patterns,
            targets=config.targets,
            output_dir=output_dir,
            source_overrides=config.source_overrides,
            include_dirs=config.include_dirs,
            defines=config.defines,
            variants=config.variants,
            selected_variants=config.selected_variants,
            external_corpus_manifest=config.external_corpus_manifest,
        )
    config = _config_with_selected_variants(config, variants)

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
    max_define_count = max(
        (len(defines) for _variant, defines in variant_jobs), default=0
    )
    define_forwarding_diagnostics: set[str] = set()
    include_path_forwarding_diagnostics: set[str] = set()

    for unit in scan.units:
        source_supports_defines = _source_frontend_supports_lexer_keyword(
            unit.source_backend, "defines"
        )
        source_supports_include_paths = _source_frontend_supports_lexer_keyword(
            unit.source_backend, "include_paths"
        )
        if (
            max_define_count > 0
            and not source_supports_defines
            and unit.source_backend not in define_forwarding_diagnostics
        ):
            diagnostics.append(
                _frontend_define_forwarding_diagnostic(unit, max_define_count)
            )
            define_forwarding_diagnostics.add(unit.source_backend)
        if (
            include_paths
            and not source_supports_include_paths
            and unit.source_backend not in include_path_forwarding_diagnostics
        ):
            diagnostics.append(
                _frontend_include_path_forwarding_diagnostic(
                    unit,
                    len(include_paths),
                )
            )
            include_path_forwarding_diagnostics.add(unit.source_backend)
        for target in selected_targets:
            for variant, defines in variant_jobs:
                output_path = _artifact_path(config, unit, target, variant)
                artifact = {
                    "source": unit.relative_path,
                    "sourceBackend": unit.source_backend,
                    "target": target,
                    "path": _artifact_report_path(output_path, config),
                    "status": "translated",
                    "defines": dict(sorted(defines.items())),
                    "defineProcessing": _artifact_define_processing(
                        unit.source_backend,
                        defines,
                        supports_defines=source_supports_defines,
                    ),
                    "includePathProcessing": _artifact_include_path_processing(
                        unit.source_backend,
                        include_paths,
                        supports_include_paths=source_supports_include_paths,
                    ),
                    "sourceHash": dict(unit.source_hash),
                    "sourceSizeBytes": unit.source_size_bytes,
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
                    artifact["generatedSizeBytes"] = output_path.stat().st_size
                    artifact["sourceMap"] = _artifact_source_map(
                        config, unit, target, output_path
                    )
                    if _is_crossgl_target(target):
                        remap_path = _source_remap_path(output_path)
                        remap_payload = _source_remap_payload(artifact["sourceMap"])
                        _write_source_remap_sidecar(remap_path, remap_payload)
                        artifact["sourceRemap"] = _artifact_source_remap(
                            config,
                            target,
                            artifact["path"],
                            remap_path,
                            str(artifact["sourceMap"]["mappingGranularity"]),
                            len(remap_payload["mappings"]),
                        )
                except Exception as exc:  # noqa: BLE001
                    # Project translation reports per-artifact failures so one bad
                    # unit does not hide the rest of the repository's migration state.
                    artifact["status"] = "failed"
                    artifact["error"] = str(exc)
                    artifact.pop("generatedHash", None)
                    artifact.pop("generatedSizeBytes", None)
                    artifact.pop("sourceMap", None)
                    artifact.pop("sourceRemap", None)
                    diagnostics.append(
                        ProjectDiagnostic(
                            severity="error",
                            code="project.translate.failed",
                            message=str(exc),
                            location=SourceLocation(file=unit.relative_path),
                            target=target,
                            source_backend=unit.source_backend,
                            variant=variant,
                            missing_capabilities=["batch.translation"],
                        )
                    )
                artifacts.append(artifact)

    validation = (
        _validate_artifacts(artifacts, selected_targets, config)
        if validate or run_toolchains
        else {"toolchains": [], "artifacts": []}
    )
    if run_toolchains:
        toolchain_runs = _run_toolchain_smoke(artifacts, config.root)
        validation["toolchainRuns"] = toolchain_runs
        validation["_diagnostics"].extend(_toolchain_run_diagnostics(toolchain_runs))
    diagnostics.extend(validation.pop("_diagnostics", []))
    return ProjectPortabilityReport(
        config=config,
        targets=selected_targets,
        units=scan.units,
        skipped=scan.skipped,
        artifacts=artifacts,
        diagnostics=diagnostics,
        validation=validation,
        migration_actions=_project_migration_actions(
            scan.units, selected_targets, artifacts
        ),
        artifact_matrix=_artifact_matrix_report(
            config, scan.units, selected_targets, config.variants, artifacts
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
        "sourceSizeStatusCounts": _status_counts(
            artifact_checks, "sourceSizeStatus", SOURCE_SIZE_VALIDATION_STATUSES
        ),
        "generatedHashStatusCounts": _status_counts(
            artifact_checks,
            "generatedHashStatus",
            GENERATED_HASH_VALIDATION_STATUSES,
        ),
        "generatedSizeStatusCounts": _status_counts(
            artifact_checks,
            "generatedSizeStatus",
            GENERATED_SIZE_VALIDATION_STATUSES,
        ),
        "sourceMapStatusCounts": _status_counts(
            artifact_checks,
            "sourceMapStatus",
            SOURCE_MAP_VALIDATION_STATUSES,
        ),
        "sourceRemapStatusCounts": _status_counts(
            artifact_checks,
            "sourceRemapStatus",
            SOURCE_REMAP_VALIDATION_STATUSES,
        ),
    }


def _validation_toolchain_status_counts(toolchains: Sequence[Any]) -> dict[str, int]:
    return _status_counts(
        toolchains,
        "status",
        frozenset(("available", "not-configured", "unavailable")),
    )


def _validation_toolchain_run_status_counts(runs: Sequence[Any]) -> dict[str, int]:
    return _status_counts(runs, "status", VALIDATION_TOOLCHAIN_RUN_STATUSES)


def _validation_run_status_rollup(
    records: Sequence[Any], field_name: str
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        value = record.get(field_name)
        if not isinstance(value, str) or not value.strip():
            continue
        name = value.strip()
        row = counts.setdefault(
            name,
            {
                "runCount": 0,
                "okCount": 0,
                "failedCount": 0,
            },
        )
        row["runCount"] += 1
        status = record.get("status")
        if status == "ok":
            row["okCount"] += 1
        elif status == "failed":
            row["failedCount"] += 1
    return {name: counts[name] for name in sorted(counts)}


def _validation_toolchain_run_status_by_target(
    runs: Sequence[Any],
) -> dict[str, dict[str, int]]:
    return _validation_run_status_rollup(runs, "target")


def _validation_toolchain_run_status_by_source_backend(
    runs: Sequence[Any],
) -> dict[str, dict[str, int]]:
    return _validation_run_status_rollup(runs, "sourceBackend")


def _validation_toolchain_run_status_by_check_kind(
    runs: Sequence[Any],
) -> dict[str, dict[str, int]]:
    return _validation_run_status_rollup(runs, "checkKind")


def _validation_toolchain_run_tool_name(run: Mapping[str, Any]) -> str | None:
    command = run.get("command")
    if (
        not isinstance(command, Sequence)
        or isinstance(command, (str, bytes, bytearray))
        or not command
    ):
        return None
    tool_name = command[0]
    if isinstance(tool_name, str) and tool_name.strip():
        return tool_name.strip()
    return None


def _validation_toolchain_run_status_by_tool(
    runs: Sequence[Any],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        tool_name = _validation_toolchain_run_tool_name(run)
        if tool_name is None:
            continue
        row = counts.setdefault(
            tool_name,
            {
                "runCount": 0,
                "okCount": 0,
                "failedCount": 0,
            },
        )
        row["runCount"] += 1
        status = run.get("status")
        if status == "ok":
            row["okCount"] += 1
        elif status == "failed":
            row["failedCount"] += 1
    return {tool_name: counts[tool_name] for tool_name in sorted(counts)}


def _validation_toolchain_run_status_by_variant(
    runs: Sequence[Any],
) -> dict[str, dict[str, int]]:
    return _validation_run_status_rollup(runs, "variant")


def _validation_artifact_status_by_target(
    artifact_checks: Sequence[Any],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for artifact_check in artifact_checks:
        if not isinstance(artifact_check, Mapping):
            continue
        target = artifact_check.get("target")
        target_name = (
            target.strip() if isinstance(target, str) and target.strip() else "unknown"
        )
        row = counts.setdefault(
            target_name,
            {
                "artifactCount": 0,
                "okCount": 0,
                "failedCount": 0,
            },
        )
        row["artifactCount"] += 1
        status = artifact_check.get("status")
        if status == "ok":
            row["okCount"] += 1
        elif status == "failed":
            row["failedCount"] += 1
    return {target: counts[target] for target in sorted(counts)}


def _validation_artifact_status_by_source_backend(
    artifact_checks: Sequence[Any],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for artifact_check in artifact_checks:
        if not isinstance(artifact_check, Mapping):
            continue
        source_backend = artifact_check.get("sourceBackend")
        if not isinstance(source_backend, str) or not source_backend.strip():
            continue
        source_backend_name = source_backend.strip()
        row = counts.setdefault(
            source_backend_name,
            {
                "artifactCount": 0,
                "okCount": 0,
                "failedCount": 0,
            },
        )
        row["artifactCount"] += 1
        status = artifact_check.get("status")
        if status == "ok":
            row["okCount"] += 1
        elif status == "failed":
            row["failedCount"] += 1
    return {source_backend: counts[source_backend] for source_backend in sorted(counts)}


def _validation_artifact_status_by_variant(
    artifact_checks: Sequence[Any],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for artifact_check in artifact_checks:
        if not isinstance(artifact_check, Mapping):
            continue
        variant = artifact_check.get("variant")
        if not isinstance(variant, str) or not variant.strip():
            continue
        variant_name = variant.strip()
        row = counts.setdefault(
            variant_name,
            {
                "artifactCount": 0,
                "okCount": 0,
                "failedCount": 0,
            },
        )
        row["artifactCount"] += 1
        status = artifact_check.get("status")
        if status == "ok":
            row["okCount"] += 1
        elif status == "failed":
            row["failedCount"] += 1
    return {variant: counts[variant] for variant in sorted(counts)}


def _source_map_file_span_mismatch_reasons(
    prefix: str,
    span: Any,
    expected_span: Mapping[str, Any],
    expected_name: str,
) -> list[str]:
    if not isinstance(span, Mapping):
        return []
    reasons = []
    for field_name in SOURCE_MAP_SPAN_FIELDS:
        if span.get(field_name) != expected_span[field_name]:
            expected_field_name = "path" if field_name == "file" else field_name
            reasons.append(
                f"{prefix}.{field_name} must match "
                f"{expected_name} {expected_field_name} "
                f"({_value_mismatch_context(expected_span[field_name], span.get(field_name))})"
            )
    return reasons


def _source_map_line_preserving_mapping_reasons(
    prefix: str,
    source_map: Mapping[str, Any],
    source_path: Path,
    source_report_path: str,
    artifact_path: Path,
    artifact_report_path: str,
) -> list[str]:
    if source_map.get("mappingGranularity") != "line":
        return []
    mappings = source_map.get("mappings")
    if not isinstance(mappings, list):
        return []
    expected_mappings = _line_preserving_source_map_mappings(
        source_path,
        source_report_path,
        artifact_path,
        artifact_report_path,
    )
    if not expected_mappings:
        return []
    if mappings == expected_mappings:
        return []
    if len(mappings) != len(expected_mappings):
        return [
            f"{prefix}.mappings count must match current line-preserving line count "
            f"({_value_mismatch_context(len(expected_mappings), len(mappings))})"
        ]
    for mapping_index, expected_mapping in enumerate(expected_mappings):
        mapping = mappings[mapping_index]
        if not isinstance(mapping, Mapping) or dict(mapping) != expected_mapping:
            return [
                f"{prefix}.mappings[{mapping_index}] must match current "
                "line-preserving line span "
                f"({_value_mismatch_context(expected_mapping, mapping)})"
            ]
    return [
        f"{prefix}.mappings must match current line-preserving line spans "
        f"({_value_mismatch_context(expected_mappings, mappings)})"
    ]


def _source_map_file_span_validation_diagnostics(
    artifact: Mapping[str, Any],
    config: ProjectConfig,
    artifact_path: Path,
    *,
    source_hash_status: str,
    generated_hash_status: str,
) -> list[ProjectDiagnostic]:
    if artifact.get("status") != "translated":
        return []

    source_map = artifact.get("sourceMap")
    if not isinstance(source_map, Mapping):
        return []

    reasons = []
    source_path: Path | None = None
    artifact_report_path = artifact.get("path")
    source = artifact.get("source")
    if source_hash_status in {"ok", "not-recorded"} and _is_non_empty_string(source):
        source_path = _resolve_report_path(config, source)
        if (
            _is_relative_to(source_path, config.root)
            and source_path.exists()
            and source_path.is_file()
        ):
            reasons.extend(
                _source_map_file_span_mismatch_reasons(
                    "sourceMap.source",
                    source_map.get("source"),
                    _file_span(source_path, source).to_json(),
                    "source file",
                )
            )

    if (
        generated_hash_status in {"ok", "not-recorded"}
        and _is_non_empty_string(artifact_report_path)
        and _is_relative_to(artifact_path, config.root)
        and artifact_path.exists()
        and artifact_path.is_file()
    ):
        reasons.extend(
            _source_map_file_span_mismatch_reasons(
                "sourceMap.generated",
                source_map.get("generated"),
                _file_span(artifact_path, artifact_report_path).to_json(),
                "generated artifact",
            )
        )

    if (
        source_path is not None
        and _is_non_empty_string(source)
        and _is_non_empty_string(artifact_report_path)
        and source_hash_status in {"ok", "not-recorded"}
        and generated_hash_status in {"ok", "not-recorded"}
        and _is_relative_to(source_path, config.root)
        and _is_relative_to(artifact_path, config.root)
        and source_path.exists()
        and source_path.is_file()
        and artifact_path.exists()
        and artifact_path.is_file()
    ):
        reasons.extend(
            _source_map_line_preserving_mapping_reasons(
                "sourceMap",
                source_map,
                source_path,
                source,
                artifact_path,
                artifact_report_path,
            )
        )

    if not reasons:
        return []
    return [
        ProjectDiagnostic(
            severity="error",
            code=(
                "project.validate.source-map-line-span-mismatch"
                if any("line-preserving line" in reason for reason in reasons)
                else "project.validate.source-map-file-span-mismatch"
            ),
            message=(
                "Source map spans do not match current files: "
                f"{artifact.get('path')}: {'; '.join(reasons)}"
            ),
            location=SourceLocation(file=str(artifact.get("source", ""))),
            **_artifact_diagnostic_context(artifact),
            missing_capabilities=["source.provenance"],
        )
    ]


def _source_remap_validation_diagnostics(
    artifact: Mapping[str, Any], config: ProjectConfig
) -> list[ProjectDiagnostic]:
    if artifact.get("status") != "translated":
        return []

    source_remap = artifact.get("sourceRemap")
    if not isinstance(source_remap, Mapping):
        return []

    remap_path = _resolve_report_path(config, source_remap["path"])
    if not _is_relative_to(remap_path, config.root):
        return [
            ProjectDiagnostic(
                severity="error",
                code="project.validate.source-remap-outside-project",
                message=(
                    "Source remap path resolves outside the repository: "
                    f"{source_remap['path']}"
                ),
                location=SourceLocation(file=str(artifact["source"])),
                **_artifact_diagnostic_context(artifact),
                missing_capabilities=["source.provenance"],
            )
        ]

    if not remap_path.exists():
        return [
            ProjectDiagnostic(
                severity="error",
                code="project.validate.missing-source-remap",
                message=(
                    "Expected source remap sidecar is missing: "
                    f"{source_remap['path']}"
                ),
                location=SourceLocation(file=str(artifact["source"])),
                **_artifact_diagnostic_context(artifact),
                missing_capabilities=["source.provenance"],
            )
        ]

    diagnostics = []
    expected_size = source_remap.get("sizeBytes")
    if not isinstance(expected_size, int) or isinstance(expected_size, bool):
        expected_size = None
    actual_remap_size = remap_path.stat().st_size
    if expected_size is not None and expected_size != actual_remap_size:
        diagnostics.append(
            ProjectDiagnostic(
                severity="error",
                code="project.validate.source-remap-size-mismatch",
                message=(
                    "Source remap sidecar size does not match report: "
                    f"{source_remap['path']} "
                    f"({_size_mismatch_context(expected_size, actual_remap_size)})"
                ),
                location=SourceLocation(file=str(artifact["source"])),
                **_artifact_diagnostic_context(artifact),
                missing_capabilities=["source.provenance"],
            )
        )

    actual_remap_hash = _source_hash(remap_path)
    source_remap_hash_matches = _hash_matches_report(
        actual_remap_hash, source_remap["hash"]
    )
    if not source_remap_hash_matches:
        diagnostics.append(
            ProjectDiagnostic(
                severity="error",
                code="project.validate.source-remap-hash-mismatch",
                message=(
                    "Source remap sidecar hash does not match report: "
                    f"{source_remap['path']} "
                    f"({_hash_mismatch_context(actual_remap_hash, source_remap['hash'])})"
                ),
                location=SourceLocation(file=str(artifact["source"])),
                **_artifact_diagnostic_context(artifact),
                missing_capabilities=["source.provenance"],
            )
        )

    try:
        remap_payload = json.loads(remap_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        diagnostics.append(
            ProjectDiagnostic(
                severity="error",
                code="project.validate.source-remap-invalid",
                message=(
                    "Source remap sidecar is not valid JSON: "
                    f"{source_remap['path']}: {exc}"
                ),
                location=SourceLocation(file=str(artifact["source"])),
                **_artifact_diagnostic_context(artifact),
                missing_capabilities=["source.provenance"],
            )
        )
        return diagnostics

    semantic_reasons = _compiler_source_remap_payload_reasons(remap_payload)
    if semantic_reasons:
        diagnostics.append(
            ProjectDiagnostic(
                severity="error",
                code="project.validate.source-remap-invalid",
                message=(
                    "Source remap sidecar is not compiler-compatible: "
                    f"{source_remap['path']}: {'; '.join(semantic_reasons)}"
                ),
                location=SourceLocation(file=str(artifact["source"])),
                **_artifact_diagnostic_context(artifact),
                missing_capabilities=["source.provenance"],
            )
        )

    source_map = artifact.get("sourceMap")
    if isinstance(source_map, Mapping):
        expected_payload = _source_remap_payload(source_map)
        if remap_payload != expected_payload:
            diagnostics.append(
                ProjectDiagnostic(
                    severity="error",
                    code="project.validate.source-remap-mismatch",
                    message=(
                        "Source remap sidecar does not match artifact source map: "
                        f"{source_remap['path']}"
                    ),
                    location=SourceLocation(file=str(artifact["source"])),
                    **_artifact_diagnostic_context(artifact),
                    missing_capabilities=["source.provenance"],
                )
            )
    return diagnostics


def _source_map_validation_status(
    artifact: Mapping[str, Any],
    *,
    source_hash_status: str,
    generated_hash_status: str,
    source_map_diagnostics: Sequence[ProjectDiagnostic],
) -> str:
    if artifact.get("status") != "translated":
        return "not-applicable"
    if not isinstance(artifact.get("sourceMap"), Mapping):
        return "not-recorded"
    if source_map_diagnostics:
        return "mismatch"
    if source_hash_status not in {"ok", "not-recorded"}:
        return "not-checked"
    if generated_hash_status not in {"ok", "not-recorded"}:
        return "not-checked"
    return "ok"


def _source_remap_validation_status(
    artifact: Mapping[str, Any],
    diagnostics: Sequence[ProjectDiagnostic],
) -> str:
    if artifact.get("status") != "translated":
        return "not-applicable"
    if not isinstance(artifact.get("sourceRemap"), Mapping):
        return "not-recorded"
    if not diagnostics:
        return "ok"

    codes = {diagnostic.code for diagnostic in diagnostics}
    if "project.validate.source-remap-outside-project" in codes:
        return "outside-project"
    if "project.validate.missing-source-remap" in codes:
        return "missing"
    if "project.validate.source-remap-invalid" in codes:
        return "invalid"
    if "project.validate.source-remap-size-mismatch" in codes:
        return "mismatch"
    if "project.validate.source-remap-mismatch" in codes:
        return "mismatch"
    if "project.validate.source-remap-hash-mismatch" in codes:
        return "hash-mismatch"
    return "invalid"


def _artifact_diagnostic_context(artifact: Mapping[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {"target": str(artifact.get("target", ""))}
    source_backend = artifact.get("sourceBackend")
    if _is_non_empty_string(source_backend):
        context["source_backend"] = source_backend
    variant = artifact.get("variant")
    if _is_non_empty_string(variant):
        context["variant"] = variant
    return context


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
                    **_artifact_diagnostic_context(artifact),
                    missing_capabilities=["artifact.manifest"],
                )
            )
        exists = artifact_path.exists() if artifact_inside_project else False
        source_hash_status = "not-recorded"
        source_size_status = "not-recorded"
        source_hash = artifact.get("sourceHash")
        source_size = artifact.get("sourceSizeBytes")
        validate_source_hash = isinstance(source_hash, Mapping)
        validate_source_size = _is_non_negative_int(source_size)
        if validate_source_hash or validate_source_size:
            source_path = _resolve_report_path(config, artifact["source"])
            source_inside_project = _is_relative_to(source_path, config.root)
            if not source_inside_project:
                if validate_source_hash:
                    source_hash_status = "outside-project"
                if validate_source_size:
                    source_size_status = "outside-project"
                diagnostics.append(
                    ProjectDiagnostic(
                        severity="error",
                        code="project.validate.source-outside-project",
                        message=(
                            "Source path resolves outside the repository: "
                            f"{artifact['source']}"
                        ),
                        location=SourceLocation(file=str(artifact["source"])),
                        **_artifact_diagnostic_context(artifact),
                        missing_capabilities=["source.provenance"],
                    )
                )
            elif not source_path.exists():
                if validate_source_hash:
                    source_hash_status = "missing"
                if validate_source_size:
                    source_size_status = "missing"
                diagnostics.append(
                    ProjectDiagnostic(
                        severity="error",
                        code="project.validate.missing-source",
                        message=(
                            "Expected source artifact is missing: "
                            f"{artifact['source']}"
                        ),
                        location=SourceLocation(file=str(artifact["source"])),
                        **_artifact_diagnostic_context(artifact),
                        missing_capabilities=["source.provenance"],
                    )
                )
            else:
                if validate_source_hash:
                    actual_source_hash = _source_hash(source_path)
                    if not _hash_matches_report(actual_source_hash, source_hash):
                        source_hash_status = "mismatch"
                        diagnostics.append(
                            ProjectDiagnostic(
                                severity="error",
                                code="project.validate.source-hash-mismatch",
                                message=(
                                    "Source artifact hash does not match report: "
                                    f"{artifact['source']} "
                                    f"({_hash_mismatch_context(actual_source_hash, source_hash)})"
                                ),
                                location=SourceLocation(file=str(artifact["source"])),
                                **_artifact_diagnostic_context(artifact),
                                missing_capabilities=["source.provenance"],
                            )
                        )
                    else:
                        source_hash_status = "ok"
                if validate_source_size:
                    actual_source_size = source_path.stat().st_size
                    if source_size != actual_source_size:
                        source_size_status = "mismatch"
                        diagnostics.append(
                            ProjectDiagnostic(
                                severity="error",
                                code="project.validate.source-size-mismatch",
                                message=(
                                    "Source artifact size does not match report: "
                                    f"{artifact['source']} "
                                    f"({_size_mismatch_context(source_size, actual_source_size)})"
                                ),
                                location=SourceLocation(file=str(artifact["source"])),
                                **_artifact_diagnostic_context(artifact),
                                missing_capabilities=["source.provenance"],
                            )
                        )
                    else:
                        source_size_status = "ok"
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
                    **_artifact_diagnostic_context(artifact),
                    missing_capabilities=["artifact.manifest"],
                )
            )
        generated_hash_status = (
            "not-recorded"
            if artifact.get("status") == "translated"
            else "not-applicable"
        )
        generated_size_status = generated_hash_status
        generated_hash = artifact.get("generatedHash")
        generated_size = artifact.get("generatedSizeBytes")
        source_remap_diagnostics: list[ProjectDiagnostic] = []
        source_map_diagnostics: list[ProjectDiagnostic] = []
        if artifact.get("status") == "translated":
            if not artifact_inside_project:
                generated_hash_status = "outside-project"
                generated_size_status = "outside-project"
            elif not exists:
                generated_hash_status = "missing"
                generated_size_status = "missing"
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
                                f"{artifact['path']} "
                                f"({_hash_mismatch_context(actual_hash, generated_hash)})"
                            ),
                            location=SourceLocation(file=str(artifact["source"])),
                            **_artifact_diagnostic_context(artifact),
                            missing_capabilities=["artifact.manifest"],
                        )
                    )
                else:
                    generated_hash_status = "ok"
            if exists and _is_non_negative_int(generated_size):
                actual_size = artifact_path.stat().st_size
                if generated_size != actual_size:
                    generated_size_status = "mismatch"
                    diagnostics.append(
                        ProjectDiagnostic(
                            severity="error",
                            code="project.validate.generated-size-mismatch",
                            message=(
                                "Generated artifact size does not match report: "
                                f"{artifact['path']} "
                                f"({_size_mismatch_context(generated_size, actual_size)})"
                            ),
                            location=SourceLocation(file=str(artifact["source"])),
                            **_artifact_diagnostic_context(artifact),
                            missing_capabilities=["artifact.manifest"],
                        )
                    )
                else:
                    generated_size_status = "ok"
            source_remap_diagnostics = _source_remap_validation_diagnostics(
                artifact, config
            )
            diagnostics.extend(source_remap_diagnostics)
            source_map_diagnostics = _source_map_file_span_validation_diagnostics(
                artifact,
                config,
                artifact_path,
                source_hash_status=source_hash_status,
                generated_hash_status=generated_hash_status,
            )
            diagnostics.extend(source_map_diagnostics)
        source_map_status = _source_map_validation_status(
            artifact,
            source_hash_status=source_hash_status,
            generated_hash_status=generated_hash_status,
            source_map_diagnostics=source_map_diagnostics,
        )
        source_remap_status = _source_remap_validation_status(
            artifact,
            source_remap_diagnostics,
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
                    **_artifact_diagnostic_context(artifact),
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
                and source_size_status in {"ok", "not-recorded"}
                and generated_hash_status in {"ok", "not-recorded"}
                and generated_size_status in {"ok", "not-recorded"}
                and not source_remap_diagnostics
                and not source_map_diagnostics
                else "failed"
            ),
            "sourceHashStatus": source_hash_status,
            "sourceSizeStatus": source_size_status,
            "generatedHashStatus": generated_hash_status,
            "generatedSizeStatus": generated_size_status,
            "sourceMapStatus": source_map_status,
            "sourceRemapStatus": source_remap_status,
        }
        if artifact.get("variant") is not None:
            artifact_check["variant"] = artifact["variant"]
        if artifact.get("sourceBackend") is not None:
            artifact_check["sourceBackend"] = artifact["sourceBackend"]
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
            project=report.get("project") if isinstance(report, Mapping) else None,
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
    diagnostics = _validation_diagnostics(
        source_diagnostics,
        [diagnostic.to_json() for diagnostic in diagnostic_objects],
    )
    return _validation_report_payload(
        path, diagnostics, validation, project=report.get("project")
    )


def _diagnostic_identity_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _diagnostic_identity_value(value[key]))
            for key in sorted(value, key=str)
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_diagnostic_identity_value(item) for item in value)
    return value


def _diagnostic_identity(diagnostic: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        (field_name, _diagnostic_identity_value(diagnostic.get(field_name)))
        for field_name in (
            "severity",
            "code",
            "message",
            "location",
            "target",
            "sourceBackend",
            "variant",
            "missingCapabilities",
        )
    )


def _validation_diagnostics(
    source_diagnostics: Sequence[Mapping[str, Any]],
    generated_diagnostics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    diagnostics = [dict(diagnostic) for diagnostic in source_diagnostics]
    seen_identities = {
        _diagnostic_identity(diagnostic) for diagnostic in source_diagnostics
    }
    for diagnostic in generated_diagnostics:
        identity = _diagnostic_identity(diagnostic)
        if identity in seen_identities:
            continue
        diagnostics.append(dict(diagnostic))
        seen_identities.add(identity)
    return diagnostics


def _inspection_external_corpus_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field_name in (
        "id",
        "path",
        "sourceBackend",
        "repository",
        "commit",
        "sourceUrl",
    ):
        value = entry.get(field_name)
        if isinstance(value, str):
            payload[field_name] = value
    targets = entry.get("targets")
    if isinstance(targets, list):
        payload["targets"] = [target for target in targets if isinstance(target, str)]
    return payload


def _inspection_external_corpus_sample(
    entries: Sequence[Any],
    *,
    present: bool,
    discovered: bool,
    limit: int = EXTERNAL_CORPUS_INSPECTION_SAMPLE_LIMIT,
) -> tuple[list[dict[str, Any]], int, int]:
    sampled_entries = [
        _inspection_external_corpus_entry(entry)
        for entry in entries
        if isinstance(entry, Mapping)
        and entry.get("present") is present
        and entry.get("discovered") is discovered
    ]
    return (
        sampled_entries[:limit],
        len(sampled_entries),
        max(0, len(sampled_entries) - limit),
    )


def _inspection_migration_action(action: Any) -> dict[str, Any] | None:
    if not isinstance(action, Mapping):
        return None

    payload: dict[str, Any] = {}
    for field_name in ("kind", "severity", "message"):
        value = action.get(field_name)
        if isinstance(value, str):
            payload[field_name] = value

    targets = action.get("targets")
    if isinstance(targets, list):
        payload["targets"] = [target for target in targets if isinstance(target, str)]
    return payload


def _empty_inspection_migration_summary() -> dict[str, Any]:
    return {
        "scope": None,
        "nonGoals": [],
        "actionCount": 0,
        "actionsByKind": {},
        "actionsBySeverity": {},
        "actionsByTarget": {},
        "truncatedActionCount": 0,
        "actions": [],
    }


def inspect_project_report(
    report_path: str | os.PathLike[str],
    *,
    run_toolchains: bool = False,
    max_diagnostics: int = 20,
    max_failed_artifacts: int = 20,
    max_source_map_artifacts: int = SOURCE_MAP_INSPECTION_SAMPLE_LIMIT,
    max_artifact_matrix_artifacts: int = ARTIFACT_MATRIX_INSPECTION_SAMPLE_LIMIT,
    max_artifact_provenance_artifacts: int = (
        ARTIFACT_PROVENANCE_INSPECTION_SAMPLE_LIMIT
    ),
    max_define_processing_artifacts: int = DEFINE_PROCESSING_INSPECTION_SAMPLE_LIMIT,
    max_skipped_sources: int = SKIPPED_SOURCE_INSPECTION_SAMPLE_LIMIT,
    max_include_path_processing_artifacts: int = (
        INCLUDE_PATH_PROCESSING_INSPECTION_SAMPLE_LIMIT
    ),
    max_include_dependencies: int = INCLUDE_DEPENDENCY_INSPECTION_SAMPLE_LIMIT,
    max_validation_artifacts: int = VALIDATION_INSPECTION_SAMPLE_LIMIT,
    max_toolchain_runs: int = VALIDATION_INSPECTION_SAMPLE_LIMIT,
    max_migration_actions: int = MIGRATION_ACTION_INSPECTION_SAMPLE_LIMIT,
    max_external_corpus_entries: int = EXTERNAL_CORPUS_INSPECTION_SAMPLE_LIMIT,
) -> dict[str, Any]:
    """Build a concise inspection summary for a project portability report."""
    path = Path(report_path)
    validation_report = validate_project_report(path, run_toolchains=run_toolchains)
    diagnostic_limit = max(0, max_diagnostics)
    failed_artifact_limit = max(0, max_failed_artifacts)
    source_map_artifact_limit = max(0, max_source_map_artifacts)
    artifact_matrix_artifact_limit = max(0, max_artifact_matrix_artifacts)
    artifact_provenance_artifact_limit = max(0, max_artifact_provenance_artifacts)
    define_processing_artifact_limit = max(0, max_define_processing_artifacts)
    skipped_source_limit = max(0, max_skipped_sources)
    include_path_processing_artifact_limit = max(
        0,
        max_include_path_processing_artifacts,
    )
    include_dependency_limit = max(0, max_include_dependencies)
    validation_artifact_limit = max(0, max_validation_artifacts)
    toolchain_run_limit = max(0, max_toolchain_runs)
    migration_action_limit = max(0, max_migration_actions)
    external_corpus_entry_limit = max(0, max_external_corpus_entries)
    diagnostics = list(validation_report.get("diagnostics", []))
    validation_result = validation_report.get("validation", {})
    validation_toolchains = (
        validation_result.get("toolchains")
        if isinstance(validation_result, Mapping)
        else []
    )
    validation_artifacts = (
        validation_result.get("artifacts")
        if isinstance(validation_result, Mapping)
        else []
    )
    validation_toolchain_runs = (
        validation_result.get("toolchainRuns")
        if isinstance(validation_result, Mapping)
        else []
    )
    artifact_status_by_target = validation_report.get("artifactStatusByTarget")
    artifact_status_by_source_backend = validation_report.get(
        "artifactStatusBySourceBackend"
    )
    artifact_status_by_variant = validation_report.get("artifactStatusByVariant")
    toolchain_status_counts = validation_report.get("toolchainStatusCounts")
    toolchain_run_status_counts = validation_report.get("toolchainRunStatusCounts")
    toolchain_run_status_by_target = validation_report.get("toolchainRunStatusByTarget")
    toolchain_run_status_by_source_backend = validation_report.get(
        "toolchainRunStatusBySourceBackend"
    )
    toolchain_run_status_by_check_kind = validation_report.get(
        "toolchainRunStatusByCheckKind"
    )
    toolchain_run_status_by_tool = validation_report.get("toolchainRunStatusByTool")
    toolchain_run_status_by_variant = validation_report.get(
        "toolchainRunStatusByVariant"
    )
    validation_artifact_samples = [
        sample
        for artifact in _record_sequence(validation_artifacts)
        for sample in (_inspection_validation_artifact(artifact),)
        if sample is not None
    ]
    validation_toolchain_run_samples = [
        sample
        for run in _record_sequence(validation_toolchain_runs)
        for sample in (_inspection_validation_toolchain_run(run),)
        if sample is not None
    ]
    payload: dict[str, Any] = {
        "schemaVersion": REPORT_SCHEMA_VERSION,
        "kind": REPORT_INSPECTION_KIND,
        "sourceReport": str(path),
        "sourceReportHash": _optional_source_hash(path),
        "generatedAt": int(time.time()),
        "success": bool(validation_report.get("success")),
        "report": {"available": False, "valid": False},
        "sourceMaps": {"available": False},
        "artifactProvenance": {"available": False},
        "defineProcessing": {"available": False},
        "skippedSources": {"available": False},
        "includeDependencies": {"available": False},
        "includePathProcessing": {"available": False},
        "artifactMatrix": {"available": False},
        "migration": _empty_inspection_migration_summary(),
        "externalCorpus": {"available": False},
        "diagnosticCount": len(diagnostics),
        "truncatedDiagnosticCount": max(0, len(diagnostics) - diagnostic_limit),
        "failedArtifactCount": 0,
        "truncatedFailedArtifactCount": 0,
        "failedArtifacts": [],
        "diagnostics": diagnostics[:diagnostic_limit],
        "validation": {
            "success": bool(validation_report.get("success")),
            "diagnosticCounts": dict(validation_report.get("diagnosticCounts", {})),
            "diagnosticsByCode": dict(validation_report.get("diagnosticsByCode", {})),
            "diagnosticsByTarget": dict(
                validation_report.get("diagnosticsByTarget", {})
            ),
            "diagnosticsBySourceBackend": dict(
                validation_report.get("diagnosticsBySourceBackend", {})
            ),
            "diagnosticsByVariant": dict(
                validation_report.get("diagnosticsByVariant", {})
            ),
            "missingCapabilityCounts": dict(
                validation_report.get("missingCapabilityCounts", {})
            ),
            "artifactStatusByTarget": (
                dict(artifact_status_by_target)
                if isinstance(artifact_status_by_target, Mapping)
                else _validation_artifact_status_by_target(
                    _record_sequence(validation_artifacts)
                )
            ),
            "artifactStatusBySourceBackend": (
                dict(artifact_status_by_source_backend)
                if isinstance(artifact_status_by_source_backend, Mapping)
                else _validation_artifact_status_by_source_backend(
                    _record_sequence(validation_artifacts)
                )
            ),
            "artifactStatusByVariant": (
                dict(artifact_status_by_variant)
                if isinstance(artifact_status_by_variant, Mapping)
                else _validation_artifact_status_by_variant(
                    _record_sequence(validation_artifacts)
                )
            ),
            "sourceHashStatusCounts": dict(
                validation_report.get("sourceHashStatusCounts", {})
            ),
            "sourceSizeStatusCounts": dict(
                validation_report.get("sourceSizeStatusCounts", {})
            ),
            "generatedHashStatusCounts": dict(
                validation_report.get("generatedHashStatusCounts", {})
            ),
            "generatedSizeStatusCounts": dict(
                validation_report.get("generatedSizeStatusCounts", {})
            ),
            "sourceMapStatusCounts": dict(
                validation_report.get("sourceMapStatusCounts", {})
            ),
            "sourceRemapStatusCounts": dict(
                validation_report.get("sourceRemapStatusCounts", {})
            ),
            "toolchainStatusCounts": (
                dict(toolchain_status_counts)
                if isinstance(toolchain_status_counts, Mapping)
                else _validation_toolchain_status_counts(
                    _record_sequence(validation_toolchains)
                )
            ),
            "toolchainRunStatusCounts": (
                dict(toolchain_run_status_counts)
                if isinstance(toolchain_run_status_counts, Mapping)
                else _validation_toolchain_run_status_counts(
                    _record_sequence(validation_toolchain_runs)
                )
            ),
            "toolchainRunStatusByTarget": (
                dict(toolchain_run_status_by_target)
                if isinstance(toolchain_run_status_by_target, Mapping)
                else _validation_toolchain_run_status_by_target(
                    _record_sequence(validation_toolchain_runs)
                )
            ),
            "toolchainRunStatusBySourceBackend": (
                dict(toolchain_run_status_by_source_backend)
                if isinstance(toolchain_run_status_by_source_backend, Mapping)
                else _validation_toolchain_run_status_by_source_backend(
                    _record_sequence(validation_toolchain_runs)
                )
            ),
            "toolchainRunStatusByCheckKind": (
                dict(toolchain_run_status_by_check_kind)
                if isinstance(toolchain_run_status_by_check_kind, Mapping)
                else _validation_toolchain_run_status_by_check_kind(
                    _record_sequence(validation_toolchain_runs)
                )
            ),
            "toolchainRunStatusByTool": (
                dict(toolchain_run_status_by_tool)
                if isinstance(toolchain_run_status_by_tool, Mapping)
                else _validation_toolchain_run_status_by_tool(
                    _record_sequence(validation_toolchain_runs)
                )
            ),
            "toolchainRunStatusByVariant": (
                dict(toolchain_run_status_by_variant)
                if isinstance(toolchain_run_status_by_variant, Mapping)
                else _validation_toolchain_run_status_by_variant(
                    _record_sequence(validation_toolchain_runs)
                )
            ),
            "artifactCount": len(validation_artifact_samples),
            "truncatedArtifactCount": max(
                0,
                len(validation_artifact_samples) - validation_artifact_limit,
            ),
            "artifacts": validation_artifact_samples[:validation_artifact_limit],
            "toolchainRunCount": len(validation_toolchain_run_samples),
            "truncatedToolchainRunCount": max(
                0,
                len(validation_toolchain_run_samples) - toolchain_run_limit,
            ),
            "toolchainRuns": validation_toolchain_run_samples[:toolchain_run_limit],
            "result": (
                dict(validation_result)
                if isinstance(validation_result, Mapping)
                else {}
            ),
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
    validation_codes = validation_report.get("diagnosticsByCode", {})
    report_is_valid = not (
        isinstance(validation_codes, Mapping)
        and validation_codes.get("project.validate.invalid-report", 0)
    )
    payload["report"] = {
        "available": True,
        "valid": report_is_valid,
        "schemaVersion": report.get("schemaVersion"),
        "kind": report.get("kind"),
        "generatedAt": report.get("generatedAt"),
        "generator": dict(generator) if isinstance(generator, Mapping) else {},
        "project": _inspection_project_summary(project),
        "summary": dict(summary) if isinstance(summary, Mapping) else {},
    }
    payload["sourceMaps"] = _inspection_source_map_summary(
        summary,
        report.get("artifacts"),
        validation_artifacts=validation_artifacts,
        sample_limit=source_map_artifact_limit,
    )
    payload["artifactProvenance"] = _inspection_artifact_provenance_summary(
        summary,
        report.get("artifacts"),
        validation_artifacts=validation_artifacts,
        sample_limit=artifact_provenance_artifact_limit,
    )
    payload["defineProcessing"] = _inspection_define_processing_summary(
        summary,
        report.get("artifacts"),
        project=project,
        sample_limit=define_processing_artifact_limit,
    )
    payload["skippedSources"] = _inspection_skipped_source_summary(
        summary,
        report.get("skipped"),
        sample_limit=skipped_source_limit,
    )
    payload["includeDependencies"] = _inspection_include_dependency_summary(
        summary,
        report.get("units"),
        project=project,
        sample_limit=include_dependency_limit,
    )
    payload["includePathProcessing"] = _inspection_include_path_processing_summary(
        summary,
        report.get("artifacts"),
        project=project,
        sample_limit=include_path_processing_artifact_limit,
    )
    payload["artifactMatrix"] = _inspection_artifact_matrix_summary(
        report.get("artifactMatrix"),
        report.get("artifacts"),
        project=project,
        units=report.get("units"),
        sample_limit=artifact_matrix_artifact_limit,
    )

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
                            "sourceBackend",
                            "sourceHashStatus",
                            "sourceSizeStatus",
                            "generatedHashStatus",
                            "generatedSizeStatus",
                            "sourceMapStatus",
                            "sourceRemapStatus",
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
        actions = (
            list(migration.get("actions", []))
            if isinstance(migration.get("actions"), list)
            else []
        )
        action_samples = [
            sample
            for action in actions
            if (sample := _inspection_migration_action(action)) is not None
        ]
        payload["migration"] = {
            "scope": migration.get("scope"),
            "nonGoals": (
                list(migration.get("nonGoals", []))
                if isinstance(migration.get("nonGoals"), list)
                else []
            ),
            "actionCount": (
                migration.get("actionCount")
                if _is_non_negative_int(migration.get("actionCount"))
                else len(actions)
            ),
            "actionsByKind": (
                dict(migration.get("actionsByKind", {}))
                if isinstance(migration.get("actionsByKind"), Mapping)
                else _migration_action_counts_by_kind(actions)
            ),
            "actionsBySeverity": (
                dict(migration.get("actionsBySeverity", {}))
                if isinstance(migration.get("actionsBySeverity"), Mapping)
                else _migration_action_counts_by_severity(actions)
            ),
            "actionsByTarget": (
                dict(migration.get("actionsByTarget", {}))
                if isinstance(migration.get("actionsByTarget"), Mapping)
                else _migration_action_counts_by_target(actions)
            ),
            "truncatedActionCount": max(
                0,
                len(action_samples) - migration_action_limit,
            ),
            "actions": action_samples[:migration_action_limit],
        }

    external_corpus = report.get("externalCorpus")
    if isinstance(external_corpus, Mapping):
        external_summary = external_corpus.get("summary")
        external_entries = _record_sequence(external_corpus.get("entries"))
        missing_entries, missing_entry_count, truncated_missing_entry_count = (
            _inspection_external_corpus_sample(
                external_entries,
                present=False,
                discovered=False,
                limit=external_corpus_entry_limit,
            )
        )
        undiscovered_entries, undiscovered_entry_count, truncated_undiscovered_count = (
            _inspection_external_corpus_sample(
                external_entries,
                present=True,
                discovered=False,
                limit=external_corpus_entry_limit,
            )
        )
        external_corpus_payload = {
            "available": True,
            "status": external_corpus.get("status"),
            "summary": (
                dict(external_summary) if isinstance(external_summary, Mapping) else {}
            ),
            "missingEntryCount": missing_entry_count,
            "truncatedMissingEntryCount": truncated_missing_entry_count,
            "missingEntries": missing_entries,
            "undiscoveredPresentEntryCount": undiscovered_entry_count,
            "truncatedUndiscoveredPresentEntryCount": truncated_undiscovered_count,
            "undiscoveredPresentEntries": undiscovered_entries,
        }
        for field_name in ("manifest", "name"):
            value = external_corpus.get(field_name)
            if isinstance(value, str):
                external_corpus_payload[field_name] = value
        payload["externalCorpus"] = external_corpus_payload
    return payload


def _inspection_artifact_matrix_summary(
    artifact_matrix: Any,
    artifacts: Any,
    *,
    project: Any = None,
    units: Any = None,
    sample_limit: int = ARTIFACT_MATRIX_INSPECTION_SAMPLE_LIMIT,
) -> dict[str, Any]:
    sample_limit = max(0, sample_limit)
    artifact_records = _record_sequence(artifacts)
    if not isinstance(artifact_matrix, Mapping) and not artifact_records:
        return {"available": False}

    matrix_source = "report"
    for _attempt in range(2):
        if isinstance(artifact_matrix, Mapping):
            fields = {
                field_name: artifact_matrix.get(field_name)
                for field_name in (
                    "unitCount",
                    "targetCount",
                    "variantCount",
                    "expectedArtifactCount",
                )
            }
            variant_mode = artifact_matrix.get("variantMode")
            if all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in fields.values()
            ) and variant_mode in {"none", "named"}:
                break
        if matrix_source == "derived" or not isinstance(project, Mapping):
            return {"available": False}
        artifact_matrix = _expected_artifact_matrix_metadata(project, units)
        matrix_source = "derived"
    else:  # pragma: no cover - loop always returns or breaks within two attempts
        return {"available": False}

    emitted_count = len(artifact_records)
    translated_count = sum(
        1 for artifact in artifact_records if artifact.get("status") == "translated"
    )
    failed_count = sum(
        1 for artifact in artifact_records if artifact.get("status") == "failed"
    )
    expected_count = fields["expectedArtifactCount"]
    (
        expected_identities,
        emitted_identities,
    ) = _inspection_artifact_matrix_identity_sets(
        project,
        units,
        artifact_records,
    )
    identity_coverage_available = (
        expected_identities is not None and emitted_identities is not None
    )
    if expected_identities is None or emitted_identities is None:
        missing_identities = None
        extra_identities = None
        missing_count = max(0, expected_count - emitted_count)
        extra_count = max(0, emitted_count - expected_count)
        status_by_target = {}
        status_by_source_backend = {}
        status_by_variant = {}
    else:
        missing_identities = expected_identities - emitted_identities
        extra_identities = emitted_identities - expected_identities
        missing_count = len(missing_identities)
        extra_count = len(extra_identities)
        source_backend_by_identity = _artifact_matrix_source_backend_by_identity(
            expected_identities,
            artifact_records,
            _artifact_matrix_unit_source_backends(_record_sequence(units)),
        )
        status_by_target = _artifact_matrix_status_by_field(
            expected_identities,
            artifact_records,
            missing_identities,
            extra_identities,
            "target",
        )
        status_by_source_backend = _artifact_matrix_status_by_field(
            expected_identities,
            artifact_records,
            missing_identities,
            extra_identities,
            "sourceBackend",
            source_backend_by_identity=source_backend_by_identity,
        )
        status_by_variant = _artifact_matrix_status_by_field(
            expected_identities,
            artifact_records,
            missing_identities,
            extra_identities,
            "variant",
        )
    missing_identity_list = sorted(missing_identities or ())
    extra_identity_list = sorted(extra_identities or ())
    missing_samples = [
        _artifact_identity_payload(identity)
        for identity in missing_identity_list[:sample_limit]
    ]
    extra_samples = [
        _artifact_identity_payload(identity)
        for identity in extra_identity_list[:sample_limit]
    ]
    return {
        "available": True,
        "source": matrix_source,
        **fields,
        "variantMode": variant_mode,
        "emittedArtifactCount": emitted_count,
        "translatedCount": translated_count,
        "failedCount": failed_count,
        "identityCoverageAvailable": identity_coverage_available,
        "missingArtifactCount": missing_count,
        "extraArtifactCount": extra_count,
        "missingArtifacts": missing_samples,
        "extraArtifacts": extra_samples,
        "truncatedMissingArtifactCount": max(0, missing_count - len(missing_samples)),
        "truncatedExtraArtifactCount": max(0, extra_count - len(extra_samples)),
        "complete": missing_count == 0 and extra_count == 0,
        "statusByTarget": status_by_target,
        "statusBySourceBackend": status_by_source_backend,
        "statusByVariant": status_by_variant,
    }


def _inspection_validation_artifact(artifact: Any) -> dict[str, Any] | None:
    if not isinstance(artifact, Mapping):
        return None

    sample = {
        "source": artifact.get("source"),
        "sourceBackend": artifact.get("sourceBackend"),
        "target": artifact.get("target"),
        "path": artifact.get("path"),
        "status": artifact.get("status"),
        "exists": artifact.get("exists"),
        "sourceHashStatus": artifact.get("sourceHashStatus"),
        "sourceSizeStatus": artifact.get("sourceSizeStatus"),
        "generatedHashStatus": artifact.get("generatedHashStatus"),
        "generatedSizeStatus": artifact.get("generatedSizeStatus"),
        "sourceMapStatus": artifact.get("sourceMapStatus"),
        "sourceRemapStatus": artifact.get("sourceRemapStatus"),
    }
    if "variant" in artifact:
        sample["variant"] = artifact.get("variant")
    return {key: value for key, value in sample.items() if value is not None}


def _inspection_validation_toolchain_run(run: Any) -> dict[str, Any] | None:
    if not isinstance(run, Mapping):
        return None

    sample = {
        "source": run.get("source"),
        "sourceBackend": run.get("sourceBackend"),
        "target": run.get("target"),
        "path": run.get("path"),
        "checkKind": run.get("checkKind"),
        "status": run.get("status"),
        "returncode": run.get("returncode"),
    }
    returncode = run.get("returncode")
    if run.get("status") == "failed" or (
        isinstance(returncode, int) and not isinstance(returncode, bool) and returncode
    ):
        failure_reason = _toolchain_run_failure_reason(run)
        if failure_reason:
            sample["failureReason"] = failure_reason
    command = run.get("command")
    if isinstance(command, list) and all(
        _is_non_empty_string(part) for part in command
    ):
        sample["command"] = list(command)
    for field_name in ("stdout", "stderr"):
        value = run.get(field_name)
        if isinstance(value, str):
            sample[f"{field_name}Length"] = len(value)
    if "variant" in run:
        sample["variant"] = run.get("variant")
    return {key: value for key, value in sample.items() if value is not None}


def _inspection_artifact_matrix_identity_sets(
    project: Any,
    units: Any,
    artifact_records: Sequence[Mapping[str, Any]],
) -> tuple[set[ArtifactIdentity] | None, set[ArtifactIdentity] | None]:
    if not isinstance(project, Mapping) or not isinstance(units, list):
        return None, None

    root = project.get("root")
    output_dir = project.get("outputDir")
    targets = project.get("targets")
    variants = project.get("variants", {})
    if not (
        _is_non_empty_string(root)
        and _is_non_empty_string(output_dir)
        and isinstance(targets, list)
        and all(_is_non_empty_string(target) for target in targets)
        and isinstance(variants, Mapping)
        and all(_is_non_empty_string(name) for name in variants)
    ):
        return None, None

    root_path = Path(root).resolve()
    project_output_path = _project_output_path(root_path, output_dir).resolve()
    variant_names: list[str | None] = sorted(variants) if variants else [None]
    expected_identities = set()
    for unit in units:
        if not isinstance(unit, Mapping):
            continue
        source = unit.get("path")
        if not (_is_non_empty_string(source) and _is_report_identity_path(source)):
            continue
        for target in _normalized_targets(targets):
            for variant in variant_names:
                identity = _expected_artifact_identity(
                    root_path,
                    project_output_path,
                    source,
                    target,
                    variant,
                )
                if identity is not None:
                    expected_identities.add(identity)

    emitted_identities = {
        identity
        for artifact in artifact_records
        for identity in (_artifact_identity(artifact),)
        if identity is not None
    }
    return expected_identities, emitted_identities


def _artifact_identity_payload(identity: ArtifactIdentity) -> dict[str, Any]:
    source, target, path, variant = identity
    payload = {
        "source": source,
        "target": target,
        "path": path,
    }
    if variant is not None:
        payload["variant"] = variant
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
    if "config" in project:
        summary["config"] = project.get("config")
    config_hash = project.get("configHash")
    if isinstance(config_hash, Mapping):
        summary["configHash"] = dict(config_hash)
    for field_name in (
        "sourceRoots",
        "includePatterns",
        "excludePatterns",
        "includeDirs",
    ):
        values = project.get(field_name)
        if isinstance(values, list):
            summary[field_name] = [value for value in values if isinstance(value, str)]
    defines = project.get("defines")
    if isinstance(defines, Mapping):
        define_names = _inspection_define_names(defines)
        summary["defineNames"] = define_names
        if define_names:
            summary["defineFingerprint"] = _inspection_define_fingerprint(defines)
    for field_name in (
        "sourceRootCount",
        "sourceRootStatus",
        "sourceRootStatusCounts",
        "includePatternCount",
        "excludePatternCount",
        "sourceOverrideCount",
        "includeDirCount",
        "includeDirStatus",
        "includeDirStatusCounts",
        "defineCount",
        "variantCount",
        "variantDefineCounts",
        "externalCorpusManifest",
    ):
        if field_name in project:
            summary[field_name] = project[field_name]
    source_overrides = project.get("sourceOverrides")
    if isinstance(source_overrides, Mapping):
        summary["sourceOverrides"] = {
            path: backend
            for path, backend in sorted(source_overrides.items())
            if isinstance(path, str) and isinstance(backend, str)
        }
    variants = project.get("variants")
    if isinstance(variants, Mapping):
        summary["variantNames"] = sorted(
            name for name in variants if isinstance(name, str) and name
        )
        variant_define_names = {}
        variant_define_fingerprints = {}
        for name, defines in sorted(variants.items()):
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(defines, Mapping)
            ):
                continue
            define_names = _inspection_define_names(defines)
            variant_define_names[name] = define_names
            if define_names:
                variant_define_fingerprints[name] = _inspection_define_fingerprint(
                    defines
                )
        summary["variantDefineNames"] = variant_define_names
        summary["variantDefineFingerprints"] = variant_define_fingerprints
        if "variantDefineCounts" not in summary:
            summary["variantDefineCounts"] = _variant_define_counts(variants)
    selected_variants = project.get("selectedVariants")
    if isinstance(selected_variants, list):
        summary["selectedVariants"] = [
            name for name in selected_variants if isinstance(name, str) and name
        ]
    return summary


def _inspection_define_names(defines: Mapping[Any, Any]) -> list[str]:
    return sorted(name for name in defines if _is_non_empty_string(name))


def _inspection_define_fingerprint(defines: Mapping[Any, Any]) -> dict[str, str]:
    normalized = {
        name: value
        for name, value in sorted(defines.items())
        if _is_non_empty_string(name) and isinstance(value, str)
    }
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "algorithm": "sha256",
        "value": hashlib.sha256(encoded).hexdigest(),
    }


def _inspection_define_processing_config_context(project: Any) -> dict[str, Any]:
    context: dict[str, Any] = {
        "projectDefineCount": 0,
        "projectDefineNames": [],
        "variantCount": 0,
        "selectedVariantCount": 0,
        "selectedVariants": [],
        "variantDefineRecords": [],
    }
    if not isinstance(project, Mapping):
        return context

    defines = project.get("defines")
    if isinstance(defines, Mapping):
        define_names = _inspection_define_names(defines)
        context["projectDefineCount"] = len(define_names)
        context["projectDefineNames"] = define_names
        if define_names:
            context["projectDefineFingerprint"] = _inspection_define_fingerprint(
                defines
            )

    selected_variant_values = project.get("selectedVariants")
    selected_variants = (
        [name for name in selected_variant_values if _is_non_empty_string(name)]
        if isinstance(selected_variant_values, list)
        else []
    )
    context["selectedVariantCount"] = len(selected_variants)
    context["selectedVariants"] = selected_variants
    selected_variant_names = set(selected_variants)

    variant_records = []
    variants = project.get("variants")
    if isinstance(variants, Mapping):
        for name, variant_defines in sorted(variants.items()):
            if not _is_non_empty_string(name) or not isinstance(
                variant_defines, Mapping
            ):
                continue
            define_names = _inspection_define_names(variant_defines)
            record = {
                "name": name,
                "defineCount": len(define_names),
                "defineNames": define_names,
                "selected": name in selected_variant_names,
            }
            if define_names:
                record["defineFingerprint"] = _inspection_define_fingerprint(
                    variant_defines
                )
            variant_records.append(record)
    context["variantCount"] = len(variant_records)
    context["variantDefineRecords"] = variant_records
    return context


def _inspection_skipped_source_sample(
    record: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = record.get("path")
    reason = record.get("reason")
    if not (_is_non_empty_string(path) and _is_non_empty_string(reason)):
        return None
    sample = {
        "path": path,
        "reason": reason,
        "extension": _extension_rollup_key(Path(path).suffix.lower()),
    }
    source_override = record.get("sourceOverride")
    if _is_non_empty_string(source_override):
        sample["sourceOverride"] = source_override
    return sample


def _inspection_skipped_source_summary(
    summary: Any,
    skipped: Any = None,
    *,
    sample_limit: int = SKIPPED_SOURCE_INSPECTION_SAMPLE_LIMIT,
) -> dict[str, Any]:
    sample_limit = max(0, sample_limit)
    if not isinstance(summary, Mapping):
        return {"available": False}

    by_reason = summary.get("skippedByReason")
    by_extension = summary.get("skippedByExtension")
    by_source_override = summary.get("skippedBySourceOverride")
    if not isinstance(by_reason, Mapping) or not isinstance(by_extension, Mapping):
        return {"available": False}

    skipped_sources = [
        sample
        for record in _record_sequence(skipped)
        if isinstance(record, Mapping)
        for sample in (_inspection_skipped_source_sample(record),)
        if sample is not None
    ]
    return {
        "available": True,
        "skippedCount": len(skipped_sources),
        "truncatedSkippedCount": max(0, len(skipped_sources) - sample_limit),
        "byReason": dict(by_reason),
        "byExtension": dict(by_extension),
        "bySourceOverride": (
            dict(by_source_override) if isinstance(by_source_override, Mapping) else {}
        ),
        "sources": skipped_sources[:sample_limit],
    }


def _inspection_define_processing_artifact(
    artifact: Mapping[str, Any],
) -> dict[str, Any] | None:
    define_processing = artifact.get("defineProcessing")
    if not isinstance(define_processing, Mapping):
        return None
    define_count = define_processing.get("defineCount")

    sample = {
        "source": artifact.get("source"),
        "sourceBackend": artifact.get("sourceBackend"),
        "target": artifact.get("target"),
        "path": artifact.get("path"),
        "status": define_processing.get("status"),
        "frontend": define_processing.get("frontend"),
        "supportsDefines": define_processing.get("supportsDefines"),
    }
    if (
        isinstance(define_count, int)
        and not isinstance(define_count, bool)
        and define_count >= 0
    ):
        sample["defineCount"] = define_count
    defines = artifact.get("defines")
    if isinstance(defines, Mapping):
        define_names = sorted(name for name in defines if _is_non_empty_string(name))
        if define_names:
            sample["defineNames"] = define_names
            sample["defineFingerprint"] = _inspection_define_fingerprint(defines)
    if "variant" in artifact:
        sample["variant"] = artifact.get("variant")
    return {key: value for key, value in sample.items() if value is not None}


def _inspection_unsupported_define_processing_artifact(
    artifact: Mapping[str, Any],
) -> dict[str, Any] | None:
    sample = _inspection_define_processing_artifact(artifact)
    if sample is None or sample.get("status") != "not-supported":
        return None
    define_count = sample.get("defineCount")
    if (
        not isinstance(define_count, int)
        or isinstance(define_count, bool)
        or define_count <= 0
    ):
        return None
    return sample


def _inspection_define_processing_summary(
    summary: Any,
    artifacts: Any = None,
    *,
    project: Any = None,
    sample_limit: int = DEFINE_PROCESSING_INSPECTION_SAMPLE_LIMIT,
) -> dict[str, Any]:
    sample_limit = max(0, sample_limit)
    if not isinstance(summary, Mapping):
        return {"available": False}

    by_status = summary.get("defineProcessingByStatus")
    by_source_backend = summary.get("defineProcessingBySourceBackend")
    by_target = summary.get("defineProcessingByTarget")
    by_variant = summary.get("defineProcessingByVariant")
    if not isinstance(by_status, Mapping) or not isinstance(by_source_backend, Mapping):
        return {"available": False}

    define_processing_artifacts = []
    not_supported_artifacts = []
    for artifact in _record_sequence(artifacts):
        if not isinstance(artifact, Mapping):
            continue
        sample = _inspection_define_processing_artifact(artifact)
        if sample:
            define_processing_artifacts.append(sample)
        sample = _inspection_unsupported_define_processing_artifact(artifact)
        if sample:
            not_supported_artifacts.append(sample)

    config_context = _inspection_define_processing_config_context(project)

    return {
        "available": True,
        "byStatus": dict(by_status),
        "bySourceBackend": {
            source_backend: dict(counts)
            for source_backend, counts in by_source_backend.items()
            if isinstance(source_backend, str) and isinstance(counts, Mapping)
        },
        "byTarget": (
            {
                target: dict(counts)
                for target, counts in by_target.items()
                if isinstance(target, str) and isinstance(counts, Mapping)
            }
            if isinstance(by_target, Mapping)
            else {}
        ),
        "byVariant": (
            {
                variant: dict(counts)
                for variant, counts in by_variant.items()
                if isinstance(variant, str) and isinstance(counts, Mapping)
            }
            if isinstance(by_variant, Mapping)
            else {}
        ),
        **config_context,
        "artifactCount": len(define_processing_artifacts),
        "truncatedArtifactCount": max(
            0,
            len(define_processing_artifacts) - sample_limit,
        ),
        "artifacts": define_processing_artifacts[:sample_limit],
        "notSupportedArtifactCount": len(not_supported_artifacts),
        "truncatedNotSupportedArtifactCount": max(
            0,
            len(not_supported_artifacts) - sample_limit,
        ),
        "notSupportedArtifacts": not_supported_artifacts[:sample_limit],
    }


def _inspection_include_path_processing_artifact(
    artifact: Mapping[str, Any],
) -> dict[str, Any] | None:
    include_path_processing = artifact.get("includePathProcessing")
    if not isinstance(include_path_processing, Mapping):
        return None
    include_path_count = include_path_processing.get("includePathCount")

    sample = {
        "source": artifact.get("source"),
        "sourceBackend": artifact.get("sourceBackend"),
        "target": artifact.get("target"),
        "path": artifact.get("path"),
        "status": include_path_processing.get("status"),
        "frontend": include_path_processing.get("frontend"),
        "supportsIncludePaths": include_path_processing.get("supportsIncludePaths"),
    }
    if (
        isinstance(include_path_count, int)
        and not isinstance(include_path_count, bool)
        and include_path_count >= 0
    ):
        sample["includePathCount"] = include_path_count
    if "variant" in artifact:
        sample["variant"] = artifact.get("variant")
    return {key: value for key, value in sample.items() if value is not None}


def _inspection_unsupported_include_path_processing_artifact(
    artifact: Mapping[str, Any],
) -> dict[str, Any] | None:
    sample = _inspection_include_path_processing_artifact(artifact)
    if sample is None or sample.get("status") != "not-supported":
        return None
    include_path_count = sample.get("includePathCount")
    if (
        not isinstance(include_path_count, int)
        or isinstance(include_path_count, bool)
        or include_path_count <= 0
    ):
        return None
    return sample


def _inspection_include_dir_status_record(record: Any) -> dict[str, Any] | None:
    if not isinstance(record, Mapping):
        return None

    path = record.get("path")
    status = record.get("status")
    if not _is_non_empty_string(path) or not _is_non_empty_string(status):
        return None

    sample = {
        "path": path,
        "status": status,
    }
    resolved_path = record.get("resolvedPath")
    if _is_non_empty_string(resolved_path):
        sample["resolvedPath"] = resolved_path
    frontend_visible = record.get("frontendVisible")
    if isinstance(frontend_visible, bool):
        sample["frontendVisible"] = frontend_visible
    return sample


def _inspection_include_dir_status_records(project: Any) -> list[dict[str, Any]]:
    if not isinstance(project, Mapping):
        return []
    records = project.get("includeDirStatus")
    if not isinstance(records, list):
        return []

    samples = []
    for record in records:
        sample = _inspection_include_dir_status_record(record)
        if sample is not None:
            samples.append(sample)
    return samples


def _inspection_include_path_processing_summary(
    summary: Any,
    artifacts: Any = None,
    *,
    project: Any = None,
    sample_limit: int = INCLUDE_PATH_PROCESSING_INSPECTION_SAMPLE_LIMIT,
) -> dict[str, Any]:
    sample_limit = max(0, sample_limit)
    if not isinstance(summary, Mapping):
        return {"available": False}

    by_status = summary.get("includePathProcessingByStatus")
    by_source_backend = summary.get("includePathProcessingBySourceBackend")
    by_target = summary.get("includePathProcessingByTarget")
    by_variant = summary.get("includePathProcessingByVariant")
    if not isinstance(by_status, Mapping) or not isinstance(by_source_backend, Mapping):
        return {"available": False}

    include_path_processing_artifacts = []
    not_supported_artifacts = []
    for artifact in _record_sequence(artifacts):
        if not isinstance(artifact, Mapping):
            continue
        sample = _inspection_include_path_processing_artifact(artifact)
        if sample:
            include_path_processing_artifacts.append(sample)
        sample = _inspection_unsupported_include_path_processing_artifact(artifact)
        if sample:
            not_supported_artifacts.append(sample)

    include_dir_records = _inspection_include_dir_status_records(project)
    frontend_visible_include_dirs = [
        record
        for record in include_dir_records
        if record.get("frontendVisible") is True
    ]
    inactive_include_dirs = [
        record
        for record in include_dir_records
        if record.get("frontendVisible") is not True
    ]

    return {
        "available": True,
        "byStatus": dict(by_status),
        "bySourceBackend": {
            source_backend: dict(counts)
            for source_backend, counts in by_source_backend.items()
            if isinstance(source_backend, str) and isinstance(counts, Mapping)
        },
        "byTarget": (
            {
                target: dict(counts)
                for target, counts in by_target.items()
                if isinstance(target, str) and isinstance(counts, Mapping)
            }
            if isinstance(by_target, Mapping)
            else {}
        ),
        "byVariant": (
            {
                variant: dict(counts)
                for variant, counts in by_variant.items()
                if isinstance(variant, str) and isinstance(counts, Mapping)
            }
            if isinstance(by_variant, Mapping)
            else {}
        ),
        "artifactCount": len(include_path_processing_artifacts),
        "truncatedArtifactCount": max(
            0,
            len(include_path_processing_artifacts) - sample_limit,
        ),
        "artifacts": include_path_processing_artifacts[:sample_limit],
        "notSupportedArtifactCount": len(not_supported_artifacts),
        "truncatedNotSupportedArtifactCount": max(
            0,
            len(not_supported_artifacts) - sample_limit,
        ),
        "notSupportedArtifacts": not_supported_artifacts[:sample_limit],
        "includeDirCount": len(include_dir_records),
        "frontendVisibleIncludeDirCount": len(frontend_visible_include_dirs),
        "inactiveIncludeDirCount": len(inactive_include_dirs),
        "includeDirs": include_dir_records,
        "frontendVisibleIncludeDirs": frontend_visible_include_dirs,
        "inactiveIncludeDirs": inactive_include_dirs,
    }


def _inspection_include_dependency_sample(
    unit_path: Any,
    source_backend: Any,
    unit_source_hash: Any,
    unit_source_size: Any,
    dependency: Mapping[str, Any],
    *,
    root_path: Path | None = None,
) -> dict[str, Any] | None:
    include = dependency.get("include")
    status = dependency.get("status")
    if not _is_non_empty_string(include) or not _is_non_empty_string(status):
        return None

    dependency_source = dependency.get("source")
    sample_source = (
        dependency_source if _is_non_empty_string(dependency_source) else unit_path
    )
    sample: dict[str, Any] = {
        "source": sample_source if _is_non_empty_string(sample_source) else None,
        "sourceBackend": (
            source_backend if _is_non_empty_string(source_backend) else None
        ),
        "include": include,
        "status": status,
    }
    kind = dependency.get("kind")
    if _is_non_empty_string(kind):
        sample["kind"] = kind
    for field_name in ("line", "column"):
        value = dependency.get(field_name)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            sample[field_name] = value
    for field_name in ("resolvedPath", "resolvedFrom", "resolvedFromDefine", "variant"):
        value = dependency.get(field_name)
        if _is_non_empty_string(value):
            sample[field_name] = value
    if isinstance(unit_source_hash, Mapping):
        hash_algorithm = unit_source_hash.get("algorithm")
        hash_value = unit_source_hash.get("value")
        if _is_non_empty_string(hash_algorithm):
            sample["unitSourceHashAlgorithm"] = hash_algorithm
        if _is_non_empty_string(hash_value):
            sample["unitSourceHash"] = hash_value
    if _is_non_negative_int(unit_source_size):
        sample["unitSourceSizeBytes"] = unit_source_size
    resolved_hash = dependency.get("resolvedHash")
    if isinstance(resolved_hash, Mapping):
        hash_algorithm = resolved_hash.get("algorithm")
        hash_value = resolved_hash.get("value")
        if _is_non_empty_string(hash_algorithm):
            sample["resolvedHashAlgorithm"] = hash_algorithm
        if _is_non_empty_string(hash_value):
            sample["resolvedHash"] = hash_value
    resolved_size = _inspection_resolved_include_size(root_path, dependency)
    if resolved_size is not None:
        sample["resolvedSizeBytes"] = resolved_size
    return {key: value for key, value in sample.items() if value is not None}


def _inspection_resolved_include_size(
    root_path: Path | None, dependency: Mapping[str, Any]
) -> int | None:
    resolved_size = dependency.get("resolvedSizeBytes")
    if _is_non_negative_int(resolved_size):
        return resolved_size
    if root_path is None:
        return None
    resolved_path = dependency.get("resolvedPath")
    if not _is_non_empty_string(resolved_path) or not _is_report_identity_path(
        resolved_path
    ):
        return None
    include_path = (root_path / resolved_path).resolve()
    if not _is_relative_to(include_path, root_path) or not include_path.is_file():
        return None
    try:
        return include_path.stat().st_size
    except OSError:
        return None


def _inspection_include_dependency_summary(
    summary: Any,
    units: Any,
    *,
    project: Any = None,
    sample_limit: int = INCLUDE_DEPENDENCY_INSPECTION_SAMPLE_LIMIT,
) -> dict[str, Any]:
    sample_limit = max(0, sample_limit)
    if not isinstance(summary, Mapping):
        return {"available": False}

    dependency_count = summary.get("includeDependencyCount")
    by_status = summary.get("includeDependenciesByStatus")
    by_kind = summary.get("includeDependenciesByKind")
    by_resolved_from = summary.get("includeDependenciesByResolvedFrom")
    by_source_backend = summary.get("includeDependenciesBySourceBackend")
    by_source_backend_status = summary.get("includeDependenciesBySourceBackendStatus")
    by_variant = summary.get("includeDependenciesByVariant")
    if (
        not isinstance(dependency_count, int)
        or isinstance(dependency_count, bool)
        or dependency_count < 0
        or not isinstance(by_status, Mapping)
        or not isinstance(by_kind, Mapping)
    ):
        return {"available": False}

    resolved_dependencies = []
    system_dependencies = []
    unresolved_dependencies = []
    root_path = _project_root_path(project) if isinstance(project, Mapping) else None
    for unit in _record_sequence(units):
        if not isinstance(unit, Mapping):
            continue
        dependencies = unit.get("includeDependencies")
        if not isinstance(dependencies, list):
            continue
        for dependency in dependencies:
            if not isinstance(dependency, Mapping):
                continue
            status = dependency.get("status")
            if status == "resolved":
                sample = _inspection_include_dependency_sample(
                    unit.get("path"),
                    unit.get("sourceBackend"),
                    unit.get("sourceHash"),
                    unit.get("sourceSizeBytes"),
                    dependency,
                    root_path=root_path,
                )
                if sample:
                    resolved_dependencies.append(sample)
                continue
            if status == "system":
                sample = _inspection_include_dependency_sample(
                    unit.get("path"),
                    unit.get("sourceBackend"),
                    unit.get("sourceHash"),
                    unit.get("sourceSizeBytes"),
                    dependency,
                )
                if sample:
                    system_dependencies.append(sample)
                continue
            if status not in {
                "dynamic",
                "missing",
                "outside-project",
            }:
                continue
            sample = _inspection_include_dependency_sample(
                unit.get("path"),
                unit.get("sourceBackend"),
                unit.get("sourceHash"),
                unit.get("sourceSizeBytes"),
                dependency,
            )
            if sample:
                unresolved_dependencies.append(sample)

    return {
        "available": True,
        "dependencyCount": dependency_count,
        "byStatus": dict(by_status),
        "byKind": dict(by_kind),
        "byResolvedFrom": (
            dict(by_resolved_from) if isinstance(by_resolved_from, Mapping) else {}
        ),
        "byVariant": dict(by_variant) if isinstance(by_variant, Mapping) else {},
        "bySourceBackend": (
            dict(by_source_backend) if isinstance(by_source_backend, Mapping) else {}
        ),
        "bySourceBackendStatus": (
            {
                source_backend: dict(counts)
                for source_backend, counts in by_source_backend_status.items()
                if isinstance(source_backend, str) and isinstance(counts, Mapping)
            }
            if isinstance(by_source_backend_status, Mapping)
            else {}
        ),
        "resolvedDependencyCount": len(resolved_dependencies),
        "truncatedResolvedDependencyCount": max(
            0,
            len(resolved_dependencies) - sample_limit,
        ),
        "resolvedDependencies": resolved_dependencies[:sample_limit],
        "systemDependencyCount": len(system_dependencies),
        "truncatedSystemDependencyCount": max(
            0,
            len(system_dependencies) - sample_limit,
        ),
        "systemDependencies": system_dependencies[:sample_limit],
        "unresolvedDependencyCount": len(unresolved_dependencies),
        "truncatedUnresolvedDependencyCount": max(
            0,
            len(unresolved_dependencies) - sample_limit,
        ),
        "unresolvedDependencies": unresolved_dependencies[:sample_limit],
    }


def _inspection_source_map_span(span: Any) -> dict[str, int] | None:
    if not isinstance(span, Mapping):
        return None

    fields = (
        "line",
        "column",
        "endLine",
        "endColumn",
        "offset",
        "endOffset",
        "length",
    )
    sample = {}
    for field_name in fields:
        value = span.get(field_name)
        if not _is_non_negative_int(value):
            return None
        sample[field_name] = value
    return sample


def _inspection_source_map_artifact(
    artifact: Mapping[str, Any],
) -> dict[str, Any] | None:
    source_map = artifact.get("sourceMap")
    if not isinstance(source_map, Mapping):
        return None

    sample: dict[str, Any] = {
        "source": artifact.get("source"),
        "sourceBackend": artifact.get("sourceBackend"),
        "target": artifact.get("target"),
        "path": artifact.get("path"),
        "mappingGranularity": source_map.get("mappingGranularity"),
    }
    if "variant" in artifact:
        sample["variant"] = artifact.get("variant")

    source = source_map.get("source")
    if isinstance(source, Mapping) and _is_non_empty_string(source.get("file")):
        sample["sourceFile"] = source.get("file")
        source_span = _inspection_source_map_span(source)
        if source_span:
            sample["sourceSpan"] = source_span

    source_map_target = source_map.get("target")
    if _is_non_empty_string(source_map_target):
        sample["sourceMapTarget"] = source_map_target

    generated = source_map.get("generated")
    if isinstance(generated, Mapping) and _is_non_empty_string(generated.get("file")):
        sample["generatedFile"] = generated.get("file")
        generated_span = _inspection_source_map_span(generated)
        if generated_span:
            sample["generatedSpan"] = generated_span

    mappings = source_map.get("mappings")
    if isinstance(mappings, list):
        sample["mappingCount"] = len(mappings)

    source_hash = artifact.get("sourceHash")
    if isinstance(source_hash, Mapping):
        hash_algorithm = source_hash.get("algorithm")
        hash_value = source_hash.get("value")
        if _is_non_empty_string(hash_algorithm):
            sample["sourceHashAlgorithm"] = hash_algorithm
        if _is_non_empty_string(hash_value):
            sample["sourceHash"] = hash_value
    source_size = artifact.get("sourceSizeBytes")
    if _is_non_negative_int(source_size):
        sample["sourceSizeBytes"] = source_size
    generated_hash = artifact.get("generatedHash")
    if isinstance(generated_hash, Mapping):
        hash_algorithm = generated_hash.get("algorithm")
        hash_value = generated_hash.get("value")
        if _is_non_empty_string(hash_algorithm):
            sample["generatedHashAlgorithm"] = hash_algorithm
        if _is_non_empty_string(hash_value):
            sample["generatedHash"] = hash_value
    generated_size = artifact.get("generatedSizeBytes")
    if _is_non_negative_int(generated_size):
        sample["generatedSizeBytes"] = generated_size

    return {key: value for key, value in sample.items() if value is not None}


def _inspection_source_remap_artifact(
    artifact: Mapping[str, Any],
) -> dict[str, Any] | None:
    source_remap = artifact.get("sourceRemap")
    if not isinstance(source_remap, Mapping):
        return None

    sample: dict[str, Any] = {
        "source": artifact.get("source"),
        "sourceBackend": artifact.get("sourceBackend"),
        "target": artifact.get("target"),
        "path": artifact.get("path"),
        "sourceRemapPath": source_remap.get("path"),
        "sourceRemapTarget": source_remap.get("target"),
        "generatedFile": source_remap.get("generatedFile"),
        "mappingGranularity": source_remap.get("mappingGranularity"),
    }
    if "variant" in artifact:
        sample["variant"] = artifact.get("variant")
    source_hash = artifact.get("sourceHash")
    if isinstance(source_hash, Mapping):
        hash_algorithm = source_hash.get("algorithm")
        hash_value = source_hash.get("value")
        if _is_non_empty_string(hash_algorithm):
            sample["sourceHashAlgorithm"] = hash_algorithm
        if _is_non_empty_string(hash_value):
            sample["sourceHash"] = hash_value
    source_size = artifact.get("sourceSizeBytes")
    if _is_non_negative_int(source_size):
        sample["sourceSizeBytes"] = source_size
    generated_hash = artifact.get("generatedHash")
    if isinstance(generated_hash, Mapping):
        hash_algorithm = generated_hash.get("algorithm")
        hash_value = generated_hash.get("value")
        if _is_non_empty_string(hash_algorithm):
            sample["generatedHashAlgorithm"] = hash_algorithm
        if _is_non_empty_string(hash_value):
            sample["generatedHash"] = hash_value
    generated_size = artifact.get("generatedSizeBytes")
    if _is_non_negative_int(generated_size):
        sample["generatedSizeBytes"] = generated_size
    source_remap_hash = source_remap.get("hash")
    if isinstance(source_remap_hash, Mapping):
        hash_algorithm = source_remap_hash.get("algorithm")
        hash_value = source_remap_hash.get("value")
        if _is_non_empty_string(hash_algorithm):
            sample["sourceRemapHashAlgorithm"] = hash_algorithm
        if _is_non_empty_string(hash_value):
            sample["sourceRemapHash"] = hash_value
    mapping_count = source_remap.get("mappingCount")
    if _is_non_negative_int(mapping_count):
        sample["mappingCount"] = mapping_count
    size_bytes = source_remap.get("sizeBytes")
    if _is_non_negative_int(size_bytes):
        sample["sourceRemapSizeBytes"] = size_bytes
    return {key: value for key, value in sample.items() if value is not None}


def _inspection_source_map_summary(
    summary: Any,
    artifacts: Any = None,
    *,
    validation_artifacts: Any = None,
    sample_limit: int = SOURCE_MAP_INSPECTION_SAMPLE_LIMIT,
) -> dict[str, Any]:
    sample_limit = max(0, sample_limit)
    if not isinstance(summary, Mapping):
        return {"available": False}

    source_map_count = summary.get("sourceMapCount")
    fine_grained_count = summary.get("fineGrainedSourceMapCount")
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in (source_map_count, fine_grained_count)
    ):
        return {"available": False}
    if fine_grained_count > source_map_count:
        return {"available": False}

    source_map_artifacts = []
    source_remap_artifacts = []
    validation_artifacts_by_key = {
        _inspection_failed_artifact_key(artifact): artifact
        for artifact in _record_sequence(validation_artifacts)
        if isinstance(artifact, Mapping)
    }
    for artifact in _record_sequence(artifacts):
        if not isinstance(artifact, Mapping):
            continue
        source_map_artifact = _inspection_source_map_artifact(artifact)
        if source_map_artifact:
            source_map_artifact.update(
                _inspection_artifact_validation_metadata(
                    artifact,
                    validation_artifacts_by_key,
                )
            )
            source_map_artifacts.append(source_map_artifact)
        source_remap_artifact = _inspection_source_remap_artifact(artifact)
        if source_remap_artifact:
            source_remap_artifact.update(
                _inspection_artifact_validation_metadata(
                    artifact,
                    validation_artifacts_by_key,
                )
            )
            source_remap_artifacts.append(source_remap_artifact)

    file_level_count = source_map_count - fine_grained_count
    payload = {
        "available": True,
        "sourceMapCount": source_map_count,
        "fileLevelSourceMapCount": file_level_count,
        "fineGrainedSourceMapCount": fine_grained_count,
        "sourceMapArtifactCount": len(source_map_artifacts),
        "truncatedSourceMapArtifactCount": max(
            0,
            len(source_map_artifacts) - sample_limit,
        ),
        "sourceMapArtifacts": source_map_artifacts[:sample_limit],
        "sourceRemapArtifactCount": len(source_remap_artifacts),
        "truncatedSourceRemapArtifactCount": max(
            0,
            len(source_remap_artifacts) - sample_limit,
        ),
        "sourceRemapArtifacts": source_remap_artifacts[:sample_limit],
    }
    source_remap_count = summary.get("sourceRemapCount")
    if (
        isinstance(source_remap_count, int)
        and not isinstance(source_remap_count, bool)
        and source_remap_count >= 0
    ):
        payload["sourceRemapCount"] = source_remap_count
    source_remap_mapping_count = summary.get("sourceRemapMappingCount")
    if (
        isinstance(source_remap_mapping_count, int)
        and not isinstance(source_remap_mapping_count, bool)
        and source_remap_mapping_count >= 0
    ):
        payload["sourceRemapMappingCount"] = source_remap_mapping_count
    for field_name in (
        "sourceMapsByGranularity",
        "sourceMapsByTarget",
        "sourceMapsBySourceBackend",
        "sourceMapsByVariant",
        "sourceRemapsByGranularity",
        "sourceRemapsByTarget",
        "sourceRemapsBySourceBackend",
        "sourceRemapsByVariant",
    ):
        value = summary.get(field_name)
        if isinstance(value, Mapping):
            payload[field_name] = dict(value)
    return payload


def _inspection_artifact_validation_metadata(
    artifact: Mapping[str, Any],
    validation_artifacts_by_key: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> dict[str, Any]:
    validation_artifact = validation_artifacts_by_key.get(
        _inspection_failed_artifact_key(artifact)
    )
    if (
        not isinstance(validation_artifact, Mapping)
        or validation_artifact.get("status") != "failed"
    ):
        return {}

    metadata: dict[str, Any] = {"validationStatus": "failed"}
    for field_name in (
        "exists",
        "sourceHashStatus",
        "sourceSizeStatus",
        "generatedHashStatus",
        "generatedSizeStatus",
        "sourceMapStatus",
        "sourceRemapStatus",
    ):
        if field_name in validation_artifact:
            metadata[field_name] = validation_artifact.get(field_name)
    return metadata


def _inspection_artifact_provenance_artifact(
    artifact: Mapping[str, Any],
    validation_artifacts_by_key: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> dict[str, Any] | None:
    provenance = artifact.get("provenance")
    if not isinstance(provenance, Mapping):
        return None

    intermediate = provenance.get("intermediate")
    sample = {
        "source": artifact.get("source"),
        "sourceBackend": artifact.get("sourceBackend"),
        "target": artifact.get("target"),
        "path": artifact.get("path"),
        "pipeline": provenance.get("pipeline"),
        "intermediate": (
            intermediate
            if _is_non_empty_string(intermediate)
            else "none" if intermediate is None else "unknown"
        ),
    }
    if "variant" in artifact:
        sample["variant"] = artifact.get("variant")
    source_hash = artifact.get("sourceHash")
    if isinstance(source_hash, Mapping):
        hash_algorithm = source_hash.get("algorithm")
        hash_value = source_hash.get("value")
        if _is_non_empty_string(hash_algorithm):
            sample["sourceHashAlgorithm"] = hash_algorithm
        if _is_non_empty_string(hash_value):
            sample["sourceHash"] = hash_value
    source_size = artifact.get("sourceSizeBytes")
    if _is_non_negative_int(source_size):
        sample["sourceSizeBytes"] = source_size
    generated_hash = artifact.get("generatedHash")
    if isinstance(generated_hash, Mapping):
        hash_algorithm = generated_hash.get("algorithm")
        hash_value = generated_hash.get("value")
        if _is_non_empty_string(hash_algorithm):
            sample["generatedHashAlgorithm"] = hash_algorithm
        if _is_non_empty_string(hash_value):
            sample["generatedHash"] = hash_value
    generated_size = artifact.get("generatedSizeBytes")
    if _is_non_negative_int(generated_size):
        sample["generatedSizeBytes"] = generated_size
    sample.update(
        _inspection_artifact_validation_metadata(
            artifact,
            validation_artifacts_by_key,
        )
    )
    return {key: value for key, value in sample.items() if value is not None}


def _is_direct_artifact_provenance_sample(sample: Mapping[str, Any]) -> bool:
    return sample.get("intermediate") == "none"


def _is_bridged_artifact_provenance_sample(sample: Mapping[str, Any]) -> bool:
    intermediate = sample.get("intermediate")
    return (
        isinstance(intermediate, str)
        and intermediate != "none"
        and intermediate != "unknown"
    )


def _inspection_artifact_provenance_summary(
    summary: Any,
    artifacts: Any = None,
    *,
    validation_artifacts: Any = None,
    sample_limit: int = ARTIFACT_PROVENANCE_INSPECTION_SAMPLE_LIMIT,
) -> dict[str, Any]:
    sample_limit = max(0, sample_limit)
    if not isinstance(summary, Mapping):
        return {"available": False}

    by_pipeline = summary.get("artifactProvenanceByPipeline")
    by_intermediate = summary.get("artifactProvenanceByIntermediate")
    intermediate_by_source_backend = summary.get(
        "artifactProvenanceIntermediateBySourceBackend"
    )
    intermediate_by_target = summary.get("artifactProvenanceIntermediateByTarget")
    intermediate_by_variant = summary.get("artifactProvenanceIntermediateByVariant")
    if not isinstance(by_pipeline, Mapping) or not isinstance(by_intermediate, Mapping):
        return {"available": False}

    validation_artifacts_by_key = {
        _inspection_failed_artifact_key(artifact): artifact
        for artifact in _record_sequence(validation_artifacts)
        if isinstance(artifact, Mapping)
    }
    provenance_artifacts = []
    for artifact in _record_sequence(artifacts):
        if not isinstance(artifact, Mapping):
            continue
        sample = _inspection_artifact_provenance_artifact(
            artifact,
            validation_artifacts_by_key,
        )
        if sample:
            provenance_artifacts.append(sample)

    direct_artifacts = [
        artifact
        for artifact in provenance_artifacts
        if _is_direct_artifact_provenance_sample(artifact)
    ]
    bridged_artifacts = [
        artifact
        for artifact in provenance_artifacts
        if _is_bridged_artifact_provenance_sample(artifact)
    ]

    return {
        "available": True,
        "byPipeline": dict(by_pipeline),
        "byIntermediate": dict(by_intermediate),
        "intermediateBySourceBackend": (
            {
                source_backend: dict(counts)
                for source_backend, counts in intermediate_by_source_backend.items()
                if isinstance(source_backend, str) and isinstance(counts, Mapping)
            }
            if isinstance(intermediate_by_source_backend, Mapping)
            else {}
        ),
        "intermediateByTarget": (
            {
                target: dict(counts)
                for target, counts in intermediate_by_target.items()
                if isinstance(target, str) and isinstance(counts, Mapping)
            }
            if isinstance(intermediate_by_target, Mapping)
            else {}
        ),
        "intermediateByVariant": (
            {
                variant: dict(counts)
                for variant, counts in intermediate_by_variant.items()
                if isinstance(variant, str) and isinstance(counts, Mapping)
            }
            if isinstance(intermediate_by_variant, Mapping)
            else {}
        ),
        "artifactCount": len(provenance_artifacts),
        "truncatedArtifactCount": max(
            0,
            len(provenance_artifacts) - sample_limit,
        ),
        "artifacts": provenance_artifacts[:sample_limit],
        "directArtifactCount": len(direct_artifacts),
        "truncatedDirectArtifactCount": max(
            0,
            len(direct_artifacts) - sample_limit,
        ),
        "directArtifacts": direct_artifacts[:sample_limit],
        "bridgedArtifactCount": len(bridged_artifacts),
        "truncatedBridgedArtifactCount": max(
            0,
            len(bridged_artifacts) - sample_limit,
        ),
        "bridgedArtifacts": bridged_artifacts[:sample_limit],
    }


def _inspection_failed_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    failed = {
        "source": artifact.get("source"),
        "target": artifact.get("target"),
        "path": artifact.get("path"),
    }
    if "sourceBackend" in artifact:
        failed["sourceBackend"] = artifact.get("sourceBackend")
    if "variant" in artifact:
        failed["variant"] = artifact.get("variant")
    if "error" in artifact:
        failed["error"] = artifact.get("error")
    for field_name in (
        "exists",
        "sourceHashStatus",
        "sourceSizeStatus",
        "generatedHashStatus",
        "generatedSizeStatus",
        "sourceMapStatus",
        "sourceRemapStatus",
    ):
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
        reason = _toolchain_run_failure_reason(run)
        if run.get("checkKind") == "tool-availability":
            message_base = (
                f"Validation toolchain availability check for target {target} "
                f"failed for {artifact_path}"
            )
        else:
            message_base = (
                f"Validation toolchain for target {target} rejected {artifact_path}"
            )
        message = f"{message_base}: {reason}" if reason else f"{message_base}."
        context: dict[str, Any] = {"target": target}
        source_backend = run.get("sourceBackend")
        if _is_non_empty_string(source_backend):
            context["source_backend"] = source_backend
        variant = run.get("variant")
        if _is_non_empty_string(variant):
            context["variant"] = variant
        diagnostics.append(
            ProjectDiagnostic(
                severity="error",
                code="project.validate.toolchain-failed",
                message=message,
                location=SourceLocation(file=artifact_path),
                **context,
                missing_capabilities=["toolchain.validation"],
            )
        )
    return diagnostics


def _toolchain_run_failure_reason(run: Mapping[str, Any]) -> str:
    lines = [
        line.strip()
        for line in _toolchain_output_text(run.get("stderr")).splitlines()
        if line.strip()
    ]
    for line in lines:
        if "timed out" in line.lower():
            return line
    if lines:
        return lines[0]
    stdout_lines = [
        line.strip()
        for line in _toolchain_output_text(run.get("stdout")).splitlines()
        if line.strip()
    ]
    return stdout_lines[0] if stdout_lines else ""


def _toolchain_output_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return str(output)


def _record_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _validation_report_payload(
    path: Path,
    diagnostics: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    *,
    project: Any = None,
) -> dict[str, Any]:
    validation_artifacts = _record_sequence(validation.get("artifacts"))
    validation_toolchains = _record_sequence(validation.get("toolchains"))
    validation_toolchain_runs = _record_sequence(validation.get("toolchainRuns"))
    validation_summary = validation.get("summary")
    validation_summary = (
        dict(validation_summary) if isinstance(validation_summary, Mapping) else {}
    )
    source_hash_status_counts = validation_summary.get("sourceHashStatusCounts")
    source_size_status_counts = validation_summary.get("sourceSizeStatusCounts")
    generated_hash_status_counts = validation_summary.get("generatedHashStatusCounts")
    generated_size_status_counts = validation_summary.get("generatedSizeStatusCounts")
    source_map_status_counts = validation_summary.get("sourceMapStatusCounts")
    source_remap_status_counts = validation_summary.get("sourceRemapStatusCounts")
    return {
        "schemaVersion": REPORT_SCHEMA_VERSION,
        "kind": VALIDATION_REPORT_KIND,
        "sourceReport": str(path),
        "sourceReportHash": _optional_source_hash(path),
        "generatedAt": int(time.time()),
        "success": not any(
            diagnostic.get("severity") == "error" for diagnostic in diagnostics
        ),
        "project": _inspection_project_summary(project),
        "diagnosticCounts": _diagnostic_payload_counts(diagnostics),
        "diagnosticsByCode": _payload_diagnostic_counts_by_code(diagnostics) or {},
        "diagnosticsByTarget": (
            _payload_diagnostic_counts_by_field(diagnostics, "target") or {}
        ),
        "diagnosticsBySourceBackend": (
            _payload_diagnostic_counts_by_field(diagnostics, "sourceBackend") or {}
        ),
        "diagnosticsByVariant": (
            _payload_diagnostic_counts_by_field(diagnostics, "variant") or {}
        ),
        "missingCapabilityCounts": (
            _payload_missing_capability_counts(diagnostics) or {}
        ),
        "artifactStatusByTarget": _validation_artifact_status_by_target(
            validation_artifacts
        ),
        "artifactStatusBySourceBackend": _validation_artifact_status_by_source_backend(
            validation_artifacts
        ),
        "artifactStatusByVariant": _validation_artifact_status_by_variant(
            validation_artifacts
        ),
        "toolchainStatusCounts": _validation_toolchain_status_counts(
            validation_toolchains
        ),
        "toolchainRunStatusCounts": _validation_toolchain_run_status_counts(
            validation_toolchain_runs
        ),
        "toolchainRunStatusByTarget": _validation_toolchain_run_status_by_target(
            validation_toolchain_runs
        ),
        "toolchainRunStatusBySourceBackend": (
            _validation_toolchain_run_status_by_source_backend(
                validation_toolchain_runs
            )
        ),
        "toolchainRunStatusByCheckKind": _validation_toolchain_run_status_by_check_kind(
            validation_toolchain_runs
        ),
        "toolchainRunStatusByTool": _validation_toolchain_run_status_by_tool(
            validation_toolchain_runs
        ),
        "toolchainRunStatusByVariant": _validation_toolchain_run_status_by_variant(
            validation_toolchain_runs
        ),
        "sourceHashStatusCounts": (
            dict(source_hash_status_counts)
            if isinstance(source_hash_status_counts, Mapping)
            else {}
        ),
        "sourceSizeStatusCounts": (
            dict(source_size_status_counts)
            if isinstance(source_size_status_counts, Mapping)
            else {}
        ),
        "generatedHashStatusCounts": (
            dict(generated_hash_status_counts)
            if isinstance(generated_hash_status_counts, Mapping)
            else {}
        ),
        "generatedSizeStatusCounts": (
            dict(generated_size_status_counts)
            if isinstance(generated_size_status_counts, Mapping)
            else {}
        ),
        "sourceMapStatusCounts": (
            dict(source_map_status_counts)
            if isinstance(source_map_status_counts, Mapping)
            else {}
        ),
        "sourceRemapStatusCounts": (
            dict(source_remap_status_counts)
            if isinstance(source_remap_status_counts, Mapping)
            else {}
        ),
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


def _is_lowercase_hex_digest(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in LOWERCASE_HEX_DIGITS for character in value)
    )


def _is_sha256_digest(value: Any) -> bool:
    return _is_lowercase_hex_digest(value, SHA256_HEX_LENGTH)


def _source_url_matches_repository(source_url: str, repository: str) -> bool:
    return source_url.startswith(f"{repository.rstrip('/')}/")


def _hash_contract_reasons(
    prefix: str, value: Any, *, require_closed_fields: bool = False
) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{prefix} must be an object"]

    reasons = (
        _unsupported_mapping_field_reasons(prefix, value, REPORT_HASH_FIELDS)
        if require_closed_fields
        else []
    )
    if value.get("algorithm") != "sha256":
        reasons.append(f"{prefix}.algorithm must be sha256")
    if not _is_sha256_digest(value.get("value")):
        reasons.append(f"{prefix}.value must be a lowercase 64-character hex digest")
    return reasons


def _source_hash_contract_reasons(
    index: int,
    artifact: Mapping[str, Any],
    *,
    required: bool = False,
    require_closed_fields: bool = False,
) -> list[str]:
    if "sourceHash" not in artifact and not required:
        return []
    return _hash_contract_reasons(
        f"artifacts[{index}].sourceHash",
        artifact.get("sourceHash"),
        require_closed_fields=require_closed_fields,
    )


def _artifact_unit_source_hash_contract_reasons(
    index: int,
    artifact: Mapping[str, Any],
    declared_units_by_path: Mapping[str, UnitDeclaration] | None,
) -> list[str]:
    if declared_units_by_path is None:
        return []

    source = artifact.get("source")
    if not _is_non_empty_string(source):
        return []

    declaration = declared_units_by_path.get(source)
    if declaration is None:
        return []

    unit_index, unit = declaration
    artifact_hash = artifact.get("sourceHash")
    unit_hash = unit.get("sourceHash")
    if _hash_contract_reasons(f"artifacts[{index}].sourceHash", artifact_hash):
        return []
    if _hash_contract_reasons(f"units[{unit_index}].sourceHash", unit_hash):
        return []

    if dict(artifact_hash) != dict(unit_hash):
        return [
            f"artifacts[{index}].sourceHash must match "
            f"units[{unit_index}].sourceHash "
            f"({_hash_mismatch_context(artifact_hash, unit_hash)})"
        ]
    return []


def _artifact_source_size_contract_reasons(
    index: int,
    artifact: Mapping[str, Any],
    *,
    required: bool = False,
) -> list[str]:
    if "sourceSizeBytes" not in artifact and not required:
        return []
    if not _is_non_negative_int(artifact.get("sourceSizeBytes")):
        return [f"artifacts[{index}].sourceSizeBytes must be a non-negative integer"]
    return []


def _artifact_unit_source_size_contract_reasons(
    index: int,
    artifact: Mapping[str, Any],
    declared_units_by_path: Mapping[str, UnitDeclaration] | None,
) -> list[str]:
    if declared_units_by_path is None:
        return []

    source = artifact.get("source")
    if not _is_non_empty_string(source):
        return []

    declaration = declared_units_by_path.get(source)
    if declaration is None:
        return []

    unit_index, unit = declaration
    artifact_size = artifact.get("sourceSizeBytes")
    unit_size = unit.get("sourceSizeBytes")
    if not _is_non_negative_int(artifact_size):
        return []
    if not _is_non_negative_int(unit_size):
        return []

    if artifact_size != unit_size:
        return [
            f"artifacts[{index}].sourceSizeBytes must match "
            f"units[{unit_index}].sourceSizeBytes "
            f"({_size_mismatch_context(unit_size, artifact_size)})"
        ]
    return []


def _generated_hash_contract_reasons(
    index: int,
    artifact: Mapping[str, Any],
    *,
    required: bool = False,
    require_closed_fields: bool = False,
) -> list[str]:
    if "generatedHash" not in artifact and not required:
        return []
    return _hash_contract_reasons(
        f"artifacts[{index}].generatedHash",
        artifact.get("generatedHash"),
        require_closed_fields=require_closed_fields,
    )


def _generated_size_contract_reasons(
    index: int,
    artifact: Mapping[str, Any],
    *,
    required: bool = False,
) -> list[str]:
    if "generatedSizeBytes" not in artifact and not required:
        return []
    if not _is_non_negative_int(artifact.get("generatedSizeBytes")):
        return [f"artifacts[{index}].generatedSizeBytes must be a non-negative integer"]
    return []


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
    index: int,
    artifact: Mapping[str, Any],
    *,
    required: bool = False,
    require_closed_fields: bool = False,
) -> list[str]:
    if "provenance" not in artifact:
        if required:
            return [f"artifacts[{index}].provenance must be an object"]
        return []

    prefix = f"artifacts[{index}].provenance"
    provenance = artifact.get("provenance")
    if not isinstance(provenance, Mapping):
        return [f"{prefix} must be an object"]

    reasons = (
        _unsupported_mapping_field_reasons(
            prefix, provenance, REPORT_ARTIFACT_PROVENANCE_FIELDS
        )
        if require_closed_fields
        else []
    )
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
        return [
            f"{prefix} must match {source_name} "
            f"({_value_mismatch_context(value, expected)})"
        ]
    return []


def _mapping_field_contract_reasons(
    prefix: str, value: Any, expected: Mapping[str, Any], source_name: str
) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{prefix} must be an object"]
    if dict(value) != dict(expected):
        return [
            f"{prefix} must match {source_name} "
            f"({_value_mismatch_context(dict(value), dict(expected))})"
        ]
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


def _payload_diagnostic_counts_by_field(
    diagnostics: Sequence[Any], field_name: str
) -> dict[str, int] | None:
    if not all(isinstance(diagnostic, Mapping) for diagnostic in diagnostics):
        return None

    counts: dict[str, int] = {}
    for diagnostic in diagnostics:
        value = diagnostic.get(field_name)
        if _is_non_empty_string(value):
            counts[value] = counts.get(value, 0) + 1
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


def _payload_unit_counts_by_extension(units: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for unit in units:
        if not isinstance(unit, Mapping):
            continue
        extension = unit.get("extension")
        if not isinstance(extension, str):
            continue
        key = _extension_rollup_key(extension)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _payload_unit_counts_by_source_override(units: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for unit in units:
        if not isinstance(unit, Mapping):
            continue
        source_override = unit.get("sourceOverride")
        if not _is_non_empty_string(source_override):
            continue
        counts[source_override] = counts.get(source_override, 0) + 1
    return dict(sorted(counts.items()))


def _payload_include_dependency_records(
    units: Sequence[Any],
) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for unit in units:
        if not isinstance(unit, Mapping):
            continue
        dependencies = unit.get("includeDependencies")
        if not isinstance(dependencies, list):
            continue
        records.extend(
            dependency for dependency in dependencies if isinstance(dependency, Mapping)
        )
    return records


def _payload_include_dependency_counts_by_kind(
    units: Sequence[Any],
) -> dict[str, int]:
    return _include_dependency_counts_by_field(
        _payload_include_dependency_records(units),
        "kind",
        INCLUDE_DEPENDENCY_KINDS,
    )


def _payload_include_dependency_counts_by_status(
    units: Sequence[Any],
) -> dict[str, int]:
    return _include_dependency_counts_by_field(
        _payload_include_dependency_records(units),
        "status",
        INCLUDE_DEPENDENCY_STATUSES,
    )


def _payload_include_dependency_counts_by_resolved_from(
    units: Sequence[Any],
) -> dict[str, int]:
    return _include_dependency_counts_by_field(
        _payload_include_dependency_records(units),
        "resolvedFrom",
        INCLUDE_DEPENDENCY_RESOLUTION_SOURCES,
    )


def _payload_include_dependency_counts_by_source_backend(
    units: Sequence[Any],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for unit in units:
        if not isinstance(unit, Mapping):
            continue
        source_backend = unit.get("sourceBackend")
        if not _is_non_empty_string(source_backend):
            continue
        dependencies = unit.get("includeDependencies")
        if not isinstance(dependencies, list):
            continue
        dependency_count = sum(
            1 for dependency in dependencies if isinstance(dependency, Mapping)
        )
        if dependency_count:
            counts[source_backend] = counts.get(source_backend, 0) + dependency_count
    return dict(sorted(counts.items()))


def _payload_include_dependency_counts_by_source_backend_status(
    units: Sequence[Any],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for unit in units:
        if not isinstance(unit, Mapping):
            continue
        source_backend = unit.get("sourceBackend")
        if not _is_non_empty_string(source_backend):
            continue
        dependencies = unit.get("includeDependencies")
        if not isinstance(dependencies, list):
            continue
        for dependency in dependencies:
            if not isinstance(dependency, Mapping):
                continue
            status = dependency.get("status")
            if not isinstance(status, str) or status not in INCLUDE_DEPENDENCY_STATUSES:
                continue
            row = counts.setdefault(source_backend, {})
            row[status] = row.get(status, 0) + 1
    return {
        backend: dict(sorted(row.items())) for backend, row in sorted(counts.items())
    }


def _payload_include_dependency_counts_by_variant(
    units: Sequence[Any],
) -> dict[str, int]:
    records = _payload_include_dependency_records(units)
    return _include_dependency_counts_by_field(
        records,
        "variant",
        frozenset(
            variant
            for dependency in records
            for variant in (dependency.get("variant"),)
            if _is_non_empty_string(variant)
        ),
    )


def _payload_skipped_counts_by_reason(skipped: Sequence[Any]) -> dict[str, int]:
    records = [record for record in skipped if isinstance(record, Mapping)]
    return _skipped_counts_by_reason(records)


def _payload_skipped_counts_by_extension(skipped: Sequence[Any]) -> dict[str, int]:
    records = [record for record in skipped if isinstance(record, Mapping)]
    return _skipped_counts_by_extension(records)


def _payload_skipped_counts_by_source_override(
    skipped: Sequence[Any],
) -> dict[str, int]:
    records = [record for record in skipped if isinstance(record, Mapping)]
    return _skipped_counts_by_source_override(records)


def _payload_artifact_records(artifacts: Sequence[Any]) -> list[Mapping[str, Any]]:
    return [artifact for artifact in artifacts if isinstance(artifact, Mapping)]


UnitDeclaration = Tuple[int, Mapping[str, Any]]


def _record_path(record: Any) -> str | None:
    if not isinstance(record, Mapping):
        return None
    path = record.get("path")
    if not _is_non_empty_string(path):
        return None
    if not _is_report_identity_path(path):
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


def _is_string_sequence(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _project_config_for_scan_validation(
    project: Mapping[str, Any] | None,
    root_path: Path | None,
) -> ProjectConfig | None:
    if project is None or root_path is None:
        return None

    source_roots = project.get("sourceRoots")
    include_patterns = project.get("includePatterns")
    exclude_patterns = project.get("excludePatterns")
    targets = project.get("targets")
    include_dirs = project.get("includeDirs")
    selected_variants = project.get("selectedVariants")
    if not all(
        _is_string_sequence(value)
        for value in (
            source_roots,
            include_patterns,
            exclude_patterns,
            targets,
            include_dirs,
            selected_variants,
        )
    ):
        return None

    source_overrides = project.get("sourceOverrides")
    defines = project.get("defines")
    if not _valid_string_mapping(source_overrides) or not _valid_string_mapping(
        defines
    ):
        return None

    raw_variants = project.get("variants")
    if not isinstance(raw_variants, Mapping):
        return None
    variants: dict[str, dict[str, str]] = {}
    for name, variant_defines in raw_variants.items():
        if not _is_non_empty_string(name) or not _valid_string_mapping(variant_defines):
            return None
        variants[name] = dict(variant_defines)

    output_dir = project.get("outputDir")
    if not isinstance(output_dir, str):
        return None

    config_path_value = project.get("config")
    if config_path_value is None:
        config_path = None
    elif isinstance(config_path_value, str) and config_path_value.strip():
        config_path = Path(config_path_value)
    else:
        return None

    external_corpus_manifest = project.get("externalCorpusManifest")
    if external_corpus_manifest is not None and not isinstance(
        external_corpus_manifest, str
    ):
        return None

    return ProjectConfig(
        root=root_path,
        config_path=config_path,
        source_roots=tuple(_normalize_project_relative_paths(source_roots)),
        include_patterns=tuple(_normalize_project_relative_paths(include_patterns)),
        exclude_patterns=tuple(_normalize_project_relative_paths(exclude_patterns)),
        targets=tuple(targets),
        output_dir=_normalize_project_relative_path(output_dir),
        source_overrides=_normalize_project_relative_path_mapping(source_overrides),
        include_dirs=tuple(_normalize_project_relative_paths(include_dirs)),
        defines=dict(defines),
        variants=variants,
        selected_variants=tuple(selected_variants),
        external_corpus_manifest=external_corpus_manifest,
    )


ProjectScanUnitIdentity = Tuple[str, str]
ProjectScanSkippedIdentity = Tuple[str, str, Optional[str]]


def _report_unit_scan_identity(unit: Any) -> ProjectScanUnitIdentity | None:
    if not isinstance(unit, Mapping):
        return None
    path = unit.get("path")
    source_backend = unit.get("sourceBackend")
    if not (
        _is_non_empty_string(path)
        and _is_report_identity_path(path)
        and _is_non_empty_string(source_backend)
    ):
        return None
    return (path, source_backend)


def _current_unit_scan_identity(
    unit: ProjectTranslationUnit,
) -> ProjectScanUnitIdentity:
    return (unit.relative_path, unit.source_backend)


def _report_skipped_scan_identity(
    skipped_record: Any,
) -> ProjectScanSkippedIdentity | None:
    if not isinstance(skipped_record, Mapping):
        return None
    path = skipped_record.get("path")
    reason = skipped_record.get("reason")
    if not (
        _is_non_empty_string(path)
        and _is_report_identity_path(path)
        and _is_non_empty_string(reason)
    ):
        return None
    source_override = skipped_record.get("sourceOverride")
    if source_override is not None and not _is_non_empty_string(source_override):
        return None
    return (path, reason, source_override)


def _current_skipped_scan_identity(
    skipped_record: Mapping[str, Any],
) -> ProjectScanSkippedIdentity | None:
    path = skipped_record.get("path")
    reason = skipped_record.get("reason")
    if not (_is_non_empty_string(path) and _is_non_empty_string(reason)):
        return None
    source_override = skipped_record.get("sourceOverride")
    if source_override is not None and not _is_non_empty_string(source_override):
        return None
    return (path, reason, source_override)


def _scan_unit_label(identity: ProjectScanUnitIdentity) -> str:
    path, source_backend = identity
    return f"{path} ({source_backend})"


def _scan_skipped_label(identity: ProjectScanSkippedIdentity) -> str:
    path, reason, source_override = identity
    label = f"{path} ({reason})"
    if source_override:
        label = f"{label} via {source_override}"
    return label


def _current_project_scan_contract_reasons(
    project: Mapping[str, Any] | None,
    units: Sequence[Any],
    skipped: Sequence[Any],
    *,
    root_path: Path | None,
    report_path: Path,
) -> list[str]:
    config = _project_config_for_scan_validation(project, root_path)
    if config is None:
        return []

    try:
        current_scan = scan_project(config)
    except (OSError, ValueError):
        return []

    ignored_report_path = None
    if root_path is not None:
        try:
            ignored_report_path = _relpath(report_path.resolve(), root_path)
        except ValueError:
            ignored_report_path = None

    current_unit_identities = Counter(
        _current_unit_scan_identity(unit)
        for unit in current_scan.units
        if unit.relative_path != ignored_report_path
    )
    reported_unit_identities: list[ProjectScanUnitIdentity] = []
    for unit in units:
        identity = _report_unit_scan_identity(unit)
        if identity is None:
            return []
        reported_unit_identities.append(identity)

    current_skipped_identities = Counter(
        identity
        for skipped_record in current_scan.skipped
        if skipped_record.get("path") != ignored_report_path
        for identity in (_current_skipped_scan_identity(skipped_record),)
        if identity is not None
    )
    reported_skipped_identities: list[ProjectScanSkippedIdentity] = []
    for skipped_record in skipped:
        identity = _report_skipped_scan_identity(skipped_record)
        if identity is None:
            return []
        reported_skipped_identities.append(identity)

    reported_unit_counts = Counter(reported_unit_identities)
    reported_skipped_counts = Counter(reported_skipped_identities)
    missing_units = current_unit_identities - reported_unit_counts
    missing_skipped = current_skipped_identities - reported_skipped_counts

    reasons = []
    for identity, count in sorted(missing_units.items()):
        suffix = f" ({count} records)" if count > 1 else ""
        reasons.append(
            f"units must include current project scan source "
            f"{_scan_unit_label(identity)}{suffix}"
        )
    for identity, count in sorted(missing_skipped.items()):
        suffix = f" ({count} records)" if count > 1 else ""
        reasons.append(
            f"skipped must include current project scan source "
            f"{_scan_skipped_label(identity)}{suffix}"
        )
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
        if not _is_non_empty_string(path) or not _is_report_identity_path(path):
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


def _expected_artifact_identity(
    root_path: Path,
    project_output_path: Path,
    source: str,
    target: str,
    variant: str | None,
) -> ArtifactIdentity | None:
    normalized_target = _normalized_targets([target])[0]
    output_base = project_output_path / normalized_target
    if variant is not None:
        output_base = output_base / _variant_output_segment(variant)
    expected_relative = Path(source.replace("\\", "/")).with_suffix(
        _artifact_target_extension(normalized_target)
    )
    expected_path = (output_base / expected_relative).resolve()
    if not _is_relative_to(expected_path, root_path):
        return None
    return source, normalized_target, _relpath(expected_path, root_path), variant


def _artifact_matrix_contract_reasons(
    project: Mapping[str, Any],
    units: Any,
    artifacts: Any,
    *,
    root_path: Path | None,
    project_output_path: Path | None,
) -> list[str]:
    if (
        not isinstance(units, list)
        or not isinstance(artifacts, list)
        or root_path is None
        or project_output_path is None
    ):
        return []

    targets = project.get("targets", [])
    if not isinstance(targets, list) or any(
        not _is_non_empty_string(target) for target in targets
    ):
        return []
    normalized_targets = _normalized_targets(targets)
    if not normalized_targets:
        return []

    variants = project.get("variants", {})
    if not isinstance(variants, Mapping) or any(
        not _is_non_empty_string(name) for name in variants
    ):
        return []
    variant_names: list[str | None] = sorted(variants) if variants else [None]

    artifact_identities = {
        identity
        for artifact in artifacts
        if isinstance(artifact, Mapping)
        for identity in (_artifact_identity(artifact),)
        if identity is not None
    }
    reasons = []
    for unit_index, unit in enumerate(units):
        if not isinstance(unit, Mapping):
            continue
        source = unit.get("path")
        if not (_is_non_empty_string(source) and _is_report_identity_path(source)):
            continue
        for target in normalized_targets:
            for variant in variant_names:
                identity = _expected_artifact_identity(
                    root_path, project_output_path, source, target, variant
                )
                if identity is not None and identity not in artifact_identities:
                    suffix = f" target {target}"
                    if variant is not None:
                        suffix = f"{suffix} variant {variant}"
                    reasons.append(
                        f"artifacts must include units[{unit_index}].path {source}"
                        f"{suffix}"
                    )
    return reasons


def _expected_artifact_matrix_metadata(
    project: Mapping[str, Any],
    units: Any,
) -> dict[str, Any] | None:
    if not isinstance(units, list):
        return None

    targets = project.get("targets", [])
    if not isinstance(targets, list) or any(
        not _is_non_empty_string(target) for target in targets
    ):
        return None

    variants = project.get("variants", {})
    if not isinstance(variants, Mapping) or any(
        not _is_non_empty_string(name) for name in variants
    ):
        return None

    variant_count = len(variants)
    variant_factor = variant_count if variant_count else 1
    normalized_targets = _normalized_targets(targets)
    return {
        "unitCount": len(units),
        "targetCount": len(normalized_targets),
        "variantCount": variant_count,
        "variantMode": "named" if variant_count else "none",
        "expectedArtifactCount": len(units) * len(normalized_targets) * variant_factor,
    }


def _artifact_matrix_metadata_contract_reasons(
    project: Mapping[str, Any],
    units: Any,
    artifacts: Any,
    artifact_matrix: Any,
    *,
    required: bool,
) -> list[str]:
    if artifact_matrix is None and not required:
        return []
    if not isinstance(artifact_matrix, Mapping):
        return ["artifactMatrix must be an object"]

    expected = _expected_artifact_matrix_metadata(project, units)
    if expected is None:
        return []

    reasons = _unsupported_mapping_field_reasons(
        "artifactMatrix", artifact_matrix, REPORT_ARTIFACT_MATRIX_FIELDS
    )
    for field_name in ("unitCount", "targetCount", "variantCount"):
        reasons.extend(
            _count_field_contract_reasons(
                f"artifactMatrix.{field_name}",
                artifact_matrix.get(field_name),
                expected[field_name],
                f"expected {field_name}",
            )
        )
    if artifact_matrix.get("variantMode") != expected["variantMode"]:
        reasons.append("artifactMatrix.variantMode must match project.variants")
    reasons.extend(
        _count_field_contract_reasons(
            "artifactMatrix.expectedArtifactCount",
            artifact_matrix.get("expectedArtifactCount"),
            expected["expectedArtifactCount"],
            "expected artifact matrix",
        )
    )
    rollup_fields = (
        "emittedArtifactCount",
        "translatedCount",
        "failedCount",
        "missingArtifactCount",
        "extraArtifactCount",
    )
    boolean_rollup_fields = ("identityCoverageAvailable", "complete")
    mapping_rollup_fields = (
        "statusByTarget",
        "statusBySourceBackend",
        "statusByVariant",
    )
    require_rollups = required or any(
        field in artifact_matrix
        for field in (*rollup_fields, *boolean_rollup_fields, *mapping_rollup_fields)
    )
    if require_rollups and isinstance(artifacts, list):
        actual = _inspection_artifact_matrix_summary(
            artifact_matrix,
            artifacts,
            project=project,
            units=units,
        )
        if actual.get("available"):
            for field_name in rollup_fields:
                reasons.extend(
                    _count_field_contract_reasons(
                        f"artifactMatrix.{field_name}",
                        artifact_matrix.get(field_name),
                        actual[field_name],
                        "artifact matrix artifacts",
                    )
                )
            for field_name in boolean_rollup_fields:
                value = artifact_matrix.get(field_name)
                if not isinstance(value, bool):
                    reasons.append(f"artifactMatrix.{field_name} must be a boolean")
                elif value != actual[field_name]:
                    reasons.append(
                        f"artifactMatrix.{field_name} must match artifact matrix "
                        f"({_value_mismatch_context(actual[field_name], value)})"
                    )
            for field_name in mapping_rollup_fields:
                reasons.extend(
                    _mapping_field_contract_reasons(
                        f"artifactMatrix.{field_name}",
                        artifact_matrix.get(field_name),
                        actual[field_name],
                        "artifact matrix artifacts",
                    )
                )
    return reasons


def _artifact_matrix_requires_artifact_records(artifact_matrix: Any) -> bool:
    if not isinstance(artifact_matrix, Mapping):
        return False
    for field_name in ("emittedArtifactCount", "translatedCount", "failedCount"):
        value = artifact_matrix.get(field_name)
        if not isinstance(value, int) or isinstance(value, bool):
            return True
        if value != 0:
            return True
    return False


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


def _diagnostic_field_counts_contract_reasons(
    prefix: str, value: Any, diagnostics: Sequence[Any], field_name: str
) -> list[str]:
    expected = _payload_diagnostic_counts_by_field(diagnostics, field_name)
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


def _target_name_contract_reasons(
    prefix: str,
    value: Any,
    *,
    declared_targets: set[str] | None = None,
    require_canonical: bool = False,
) -> list[str]:
    if not _is_non_empty_string(value):
        return [f"{prefix} must be a string"]

    normalized_target = _normalized_targets([value])[0]
    reasons = []
    if require_canonical and normalized_target != value:
        reasons.append(f"{prefix} must use normalized backend name {normalized_target}")
    if declared_targets is not None and normalized_target not in declared_targets:
        reasons.append(f"{prefix} must be listed in project.targets")
    return reasons


def _diagnostic_location_contract_reasons(
    prefix: str, value: Any, *, require_closed_fields: bool = False
) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{prefix} must be an object"]

    reasons = (
        _unsupported_mapping_field_reasons(prefix, value, SOURCE_MAP_SPAN_FIELD_SET)
        if require_closed_fields
        else []
    )
    file = value.get("file")
    if not _is_non_empty_string(file):
        reasons.append(f"{prefix}.file must be a string")
    elif not _is_diagnostic_location_path(file):
        reasons.append(f"{prefix}.file must be repository-relative")
    for field_name in SOURCE_MAP_SPAN_FIELDS[1:]:
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
    if not _is_report_identity_path(value):
        return [f"{prefix} must be repository-relative"]
    return []


def _source_backend_contract_reasons(
    prefix: str,
    value: Any,
    *,
    require_registered: bool,
) -> list[str]:
    if not _is_non_empty_string(value):
        return [f"{prefix} must be a string"]
    if not require_registered:
        return []

    register_default_sources()
    discover_backend_plugins()
    canonical_name = SOURCE_REGISTRY.resolve_name(value)
    if canonical_name is None:
        return [f"{prefix} must be a registered source backend"]
    if canonical_name != value:
        return [f"{prefix} must use canonical source backend name {canonical_name}"]
    return []


def _unit_source_hash_contract_reasons(
    index: int,
    unit: Mapping[str, Any],
    *,
    root_path: Path | None = None,
    required: bool = False,
    check_current_file: bool = False,
    require_closed_fields: bool = False,
) -> list[str]:
    if "sourceHash" not in unit and not required:
        return []

    prefix = f"units[{index}].sourceHash"
    reasons = _hash_contract_reasons(
        prefix,
        unit.get("sourceHash"),
        require_closed_fields=require_closed_fields,
    )
    if reasons:
        return reasons

    source_path = unit.get("path")
    if (
        not check_current_file
        or root_path is None
        or not _is_non_empty_string(source_path)
        or not _is_report_identity_path(source_path)
    ):
        return reasons

    absolute_source = root_path / source_path
    if not absolute_source.exists() or not absolute_source.is_file():
        reasons.append(
            f"{prefix} cannot be checked because units[{index}].path is missing"
        )
    else:
        actual_hash = _source_hash(absolute_source)
        if not _hash_matches_report(actual_hash, unit["sourceHash"]):
            reasons.append(
                f"{prefix} must match the current source file "
                f"({_hash_mismatch_context(actual_hash, unit['sourceHash'])})"
            )
    return reasons


def _unit_source_size_contract_reasons(
    index: int,
    unit: Mapping[str, Any],
    *,
    root_path: Path | None = None,
    required: bool = False,
    check_current_file: bool = False,
) -> list[str]:
    if "sourceSizeBytes" not in unit:
        if required:
            return [f"units[{index}].sourceSizeBytes must be a non-negative integer"]
        return []

    prefix = f"units[{index}].sourceSizeBytes"
    size_bytes = unit.get("sourceSizeBytes")
    reasons = []
    if not _is_non_negative_int(size_bytes):
        reasons.append(f"{prefix} must be a non-negative integer")
        return reasons

    source_path = unit.get("path")
    if (
        not check_current_file
        or root_path is None
        or not _is_non_empty_string(source_path)
        or not _is_report_identity_path(source_path)
    ):
        return reasons

    absolute_source = root_path / source_path
    if not absolute_source.exists() or not absolute_source.is_file():
        reasons.append(
            f"{prefix} cannot be checked because units[{index}].path is missing"
        )
    else:
        actual_size = absolute_source.stat().st_size
        if size_bytes != actual_size:
            reasons.append(
                f"{prefix} must match the current source file "
                f"({_size_mismatch_context(size_bytes, actual_size)})"
            )
    return reasons


def _unit_source_override_contract_reasons(
    index: int,
    unit: Mapping[str, Any],
    project: Mapping[str, Any] | None,
    *,
    require_declared_override: bool,
) -> list[str]:
    if "sourceOverride" not in unit:
        return []

    prefix = f"units[{index}].sourceOverride"
    source_override = unit.get("sourceOverride")
    if not _is_non_empty_string(source_override):
        return [f"{prefix} must be a string"]
    if not require_declared_override or not isinstance(project, Mapping):
        return []

    source_path = unit.get("path")
    source_backend = unit.get("sourceBackend")
    source_overrides = project.get("sourceOverrides")
    if (
        not _is_non_empty_string(source_path)
        or not _is_report_identity_path(source_path)
        or not _is_non_empty_string(source_backend)
        or not _valid_string_mapping(source_overrides)
    ):
        return []

    matching_override_backends = [
        backend
        for pattern, backend in source_overrides.items()
        if fnmatch.fnmatch(source_path, pattern)
    ]
    matching_override = any(
        source_override == backend for backend in matching_override_backends
    )
    if not matching_override:
        return [
            f"{prefix} must match project.sourceOverrides for units[{index}].path "
            f"({_value_mismatch_context(matching_override_backends, source_override)})"
        ]

    register_default_sources()
    discover_backend_plugins()
    canonical_name = SOURCE_REGISTRY.resolve_name(source_override)
    if canonical_name is None:
        return [f"{prefix} must be a registered source backend"]
    if canonical_name != source_backend:
        return [
            f"{prefix} must resolve to units[{index}].sourceBackend "
            f"({_value_mismatch_context(source_backend, canonical_name)})"
        ]
    return []


def _include_dependency_contract_reasons(
    unit_index: int,
    dependency_index: int,
    dependency: Any,
    *,
    declared_variants: frozenset[str] | None = None,
    require_closed_fields: bool = False,
) -> list[str]:
    prefix = f"units[{unit_index}].includeDependencies[{dependency_index}]"
    if not isinstance(dependency, Mapping):
        return [f"{prefix} must be an object"]

    reasons = (
        _unsupported_mapping_field_reasons(
            prefix, dependency, REPORT_INCLUDE_DEPENDENCY_FIELDS
        )
        if require_closed_fields
        else []
    )
    include_value = dependency.get("include")
    if not _is_non_empty_string(include_value):
        reasons.append(f"{prefix}.include must be a string")

    source = dependency.get("source")
    if "source" in dependency:
        reasons.extend(_repository_path_contract_reasons(f"{prefix}.source", source))

    kind = dependency.get("kind")
    if kind not in INCLUDE_DEPENDENCY_KINDS:
        reasons.append(
            "{}.kind must be one of {}".format(
                prefix, ", ".join(sorted(INCLUDE_DEPENDENCY_KINDS))
            )
        )

    status = dependency.get("status")
    if status not in INCLUDE_DEPENDENCY_STATUSES:
        reasons.append(
            "{}.status must be one of {}".format(
                prefix, ", ".join(sorted(INCLUDE_DEPENDENCY_STATUSES))
            )
        )

    for field_name in ("line", "column"):
        value = dependency.get(field_name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            reasons.append(f"{prefix}.{field_name} must be a positive integer")

    resolved_path = dependency.get("resolvedPath")
    if status == "resolved":
        reasons.extend(
            _repository_path_contract_reasons(f"{prefix}.resolvedPath", resolved_path)
        )
    elif "resolvedPath" in dependency:
        reasons.append(
            f"{prefix}.resolvedPath must be omitted unless status is resolved"
        )

    if status == "resolved":
        reasons.extend(
            _hash_contract_reasons(
                f"{prefix}.resolvedHash",
                dependency.get("resolvedHash"),
                require_closed_fields=require_closed_fields,
            )
        )
    elif "resolvedHash" in dependency:
        reasons.append(
            f"{prefix}.resolvedHash must be omitted unless status is resolved"
        )

    resolved_size = dependency.get("resolvedSizeBytes")
    if status == "resolved":
        if not _is_non_negative_int(resolved_size):
            reasons.append(f"{prefix}.resolvedSizeBytes must be a non-negative integer")
    elif "resolvedSizeBytes" in dependency:
        reasons.append(
            f"{prefix}.resolvedSizeBytes must be omitted unless status is resolved"
        )

    resolved_from = dependency.get("resolvedFrom")
    if "resolvedFrom" in dependency and resolved_from not in {"source", "include-dir"}:
        reasons.append(f"{prefix}.resolvedFrom must be source or include-dir")
    if status == "resolved" and "resolvedFrom" not in dependency:
        reasons.append(f"{prefix}.resolvedFrom must be recorded for resolved includes")
    elif status != "resolved" and "resolvedFrom" in dependency:
        reasons.append(
            f"{prefix}.resolvedFrom must be omitted unless status is resolved"
        )

    resolved_from_define = dependency.get("resolvedFromDefine")
    if "resolvedFromDefine" in dependency:
        if not _is_non_empty_string(resolved_from_define):
            reasons.append(f"{prefix}.resolvedFromDefine must be a string")
        if kind == "dynamic":
            reasons.append(
                f"{prefix}.resolvedFromDefine must be omitted for dynamic includes"
            )

    variant = dependency.get("variant")
    if "variant" in dependency:
        if not _is_non_empty_string(variant):
            reasons.append(f"{prefix}.variant must be a string")
        elif declared_variants is not None and variant not in declared_variants:
            reasons.append(f"{prefix}.variant must match a declared project variant")
    elif require_closed_fields and declared_variants:
        reasons.append(
            f"{prefix}.variant must be recorded when project.variants is non-empty"
        )

    return reasons


def _project_config_for_include_validation(
    project: Mapping[str, Any] | None,
    root_path: Path | None,
) -> ProjectConfig | None:
    if project is None or root_path is None:
        return None
    include_dirs = project.get("includeDirs", [])
    if not isinstance(include_dirs, list) or any(
        not isinstance(include_dir, str) for include_dir in include_dirs
    ):
        return None
    defines = project.get("defines", {})
    if not _valid_string_mapping(defines):
        defines = {}
    raw_variants = project.get("variants", {})
    variants = {}
    if isinstance(raw_variants, Mapping):
        variants = {
            name: dict(variant_defines)
            for name, variant_defines in raw_variants.items()
            if _is_non_empty_string(name) and _valid_string_mapping(variant_defines)
        }
    output_dir = project.get("outputDir", DEFAULT_OUTPUT_DIR)
    if not isinstance(output_dir, str):
        output_dir = DEFAULT_OUTPUT_DIR
    return ProjectConfig(
        root=root_path,
        include_dirs=tuple(_normalize_project_relative_paths(include_dirs)),
        defines=dict(defines),
        variants=variants,
        selected_variants=tuple(
            name
            for name in project.get("selectedVariants", [])
            if _is_non_empty_string(name)
        ),
        output_dir=output_dir,
    )


def _include_validation_config_for_dependency(
    include_config: ProjectConfig,
    dependency: Mapping[str, Any],
) -> ProjectConfig:
    variant = dependency.get("variant")
    if _is_non_empty_string(variant) and variant in include_config.variants:
        return replace(
            include_config,
            defines={
                **dict(include_config.defines),
                **dict(include_config.variants[variant]),
            },
        )
    return include_config


def _current_include_dependency_contract_reasons(
    unit_index: int,
    dependency_index: int,
    unit: Mapping[str, Any],
    dependency: Mapping[str, Any],
    *,
    root_path: Path | None,
    include_config: ProjectConfig | None,
) -> list[str]:
    if root_path is None or include_config is None:
        return []

    prefix = f"units[{unit_index}].includeDependencies[{dependency_index}]"
    include_value = dependency.get("include")
    kind = dependency.get("kind")
    status = dependency.get("status")
    source_path = dependency.get("source", unit.get("path"))
    if not (
        _is_non_empty_string(include_value)
        and kind in INCLUDE_DEPENDENCY_KINDS
        and status in INCLUDE_DEPENDENCY_STATUSES
        and _is_non_empty_string(source_path)
        and _is_report_identity_path(source_path)
    ):
        return []
    if kind == "dynamic":
        return []

    dependency_config = _include_validation_config_for_dependency(
        include_config, dependency
    )
    unit_path = (root_path / source_path).resolve()
    expected_status, expected_path, expected_from = _resolve_include_dependency(
        dependency_config,
        unit_path,
        kind,
        include_value,
    )

    reasons = []
    if status != expected_status:
        reasons.append(
            f"{prefix}.status must match current include resolution "
            f"({_value_mismatch_context(status, expected_status)})"
        )

    resolved_from_define = dependency.get("resolvedFromDefine")
    if _is_non_empty_string(resolved_from_define):
        define_literal = _include_literal_from_define(
            dependency_config, resolved_from_define
        )
        if define_literal is None:
            reasons.append(
                f"{prefix}.resolvedFromDefine must match a project define include"
            )
        else:
            _, expected_kind, expected_include = define_literal
            if kind != expected_kind:
                reasons.append(
                    f"{prefix}.kind must match current project define include"
                )
            if include_value != expected_include:
                reasons.append(
                    f"{prefix}.include must match current project define include"
                )
    if expected_status != "resolved":
        return reasons

    if dependency.get("resolvedPath") != expected_path:
        reasons.append(
            f"{prefix}.resolvedPath must match current include resolution "
            f"({_value_mismatch_context(dependency.get('resolvedPath'), expected_path)})"
        )
    if dependency.get("resolvedFrom") != expected_from:
        reasons.append(
            f"{prefix}.resolvedFrom must match current include resolution "
            f"({_value_mismatch_context(dependency.get('resolvedFrom'), expected_from)})"
        )
    resolved_hash = dependency.get("resolvedHash")
    include_path = (root_path / expected_path).resolve()
    if not _hash_contract_reasons(f"{prefix}.resolvedHash", resolved_hash):
        actual_hash = _source_hash(include_path)
        if not _hash_matches_report(actual_hash, resolved_hash):
            reasons.append(
                f"{prefix}.resolvedHash must match current include file "
                f"({_hash_mismatch_context(actual_hash, resolved_hash)})"
            )
    resolved_size = dependency.get("resolvedSizeBytes")
    actual_size = include_path.stat().st_size
    if _is_non_negative_int(resolved_size) and resolved_size != actual_size:
        reasons.append(
            f"{prefix}.resolvedSizeBytes must match current include file "
            f"({_size_mismatch_context(resolved_size, actual_size)})"
        )
    return reasons


def _include_dependency_scan_key(
    dependency: Mapping[str, Any],
) -> tuple[tuple[str, str | int], ...] | None:
    key: list[tuple[str, str | int]] = []
    for field_name in REPORT_INCLUDE_DEPENDENCY_SCAN_IDENTITY_FIELDS:
        if field_name not in dependency:
            continue
        value = dependency[field_name]
        if field_name in {"line", "column"}:
            if not isinstance(value, int) or isinstance(value, bool):
                return None
        elif not isinstance(value, str):
            return None
        key.append((field_name, value))
    return tuple(key)


def _include_dependency_scan_label(
    key: tuple[tuple[str, str | int], ...],
) -> str:
    values = dict(key)
    source = str(values.get("source", "unit source"))
    line = values.get("line", "?")
    column = values.get("column", "?")
    status = str(values.get("status", "unknown"))
    kind = str(values.get("kind", "include"))
    include_value = str(values.get("include", "<unknown>"))
    label = f"{source}:{line}:{column} {status} {kind} include {include_value}"
    resolved_path = values.get("resolvedPath")
    if isinstance(resolved_path, str) and resolved_path:
        label = f"{label} -> {resolved_path}"
    details = []
    variant = values.get("variant")
    if isinstance(variant, str) and variant:
        details.append(f"variant {variant}")
    resolved_from = values.get("resolvedFrom")
    if isinstance(resolved_from, str) and resolved_from:
        details.append(resolved_from)
    resolved_from_define = values.get("resolvedFromDefine")
    if isinstance(resolved_from_define, str) and resolved_from_define:
        details.append(f"define {resolved_from_define}")
    if details:
        label = f"{label} ({', '.join(details)})"
    return label


def _current_include_dependency_scan_contract_reasons(
    index: int,
    unit: Mapping[str, Any],
    dependencies: Sequence[Any],
    *,
    root_path: Path | None,
    include_config: ProjectConfig | None,
) -> list[str]:
    if root_path is None or include_config is None:
        return []

    unit_path_value = unit.get("path")
    if not (
        _is_non_empty_string(unit_path_value)
        and _is_report_identity_path(unit_path_value)
    ):
        return []

    unit_path = (root_path / unit_path_value).resolve()
    if not _is_relative_to(unit_path, root_path) or not unit_path.is_file():
        return []

    current_dependencies, _ = _scan_include_dependencies(
        include_config,
        unit_path,
        unit_path_value,
    )
    current_keys = [
        key
        for dependency in current_dependencies
        for key in (_include_dependency_scan_key(dependency),)
        if key is not None
    ]

    reported_keys = []
    for dependency in dependencies:
        if not isinstance(dependency, Mapping):
            return []
        key = _include_dependency_scan_key(dependency)
        if key is None:
            return []
        reported_keys.append(key)

    expected_counts = Counter(current_keys)
    reported_counts = Counter(reported_keys)
    missing = expected_counts - reported_counts
    extra = reported_counts - expected_counts
    if not missing and not extra:
        return []

    prefix = f"units[{index}].includeDependencies"
    reasons = []
    for key, count in sorted(missing.items()):
        suffix = f" ({count} records)" if count > 1 else ""
        reasons.append(
            f"{prefix} must include current include dependency "
            f"{_include_dependency_scan_label(key)}{suffix}"
        )
    for key, count in sorted(extra.items()):
        suffix = f" ({count} records)" if count > 1 else ""
        reasons.append(
            f"{prefix} contains include dependency not found in current source: "
            f"{_include_dependency_scan_label(key)}{suffix}"
        )
    return reasons


def _diagnostic_identity_key(diagnostic: Any) -> str | None:
    if isinstance(diagnostic, ProjectDiagnostic):
        payload = diagnostic.to_json()
    elif isinstance(diagnostic, Mapping):
        payload = {
            field_name: diagnostic[field_name]
            for field_name in REPORT_DIAGNOSTIC_FIELDS
            if field_name in diagnostic
        }
    else:
        return None

    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return None


def _diagnostic_identity_label(diagnostic: ProjectDiagnostic) -> str:
    payload = diagnostic.to_json()
    location = payload.get("location", {})
    if isinstance(location, Mapping):
        location_label = "{}:{}:{}".format(
            location.get("file", "unknown"),
            location.get("line", "?"),
            location.get("column", "?"),
        )
    else:
        location_label = "unknown:?:?"
    return "{} at {}: {}".format(
        payload.get("code", "unknown"),
        location_label,
        payload.get("message", ""),
    )


def _current_include_scan_diagnostic_contract_reasons(
    units: Sequence[Any],
    diagnostics: Sequence[Any],
    *,
    root_path: Path | None,
    project: Mapping[str, Any] | None,
) -> list[str]:
    include_config = _project_config_for_include_validation(project, root_path)
    if root_path is None or include_config is None:
        return []

    reported_diagnostic_counts = Counter(
        key
        for diagnostic in diagnostics
        for key in (_diagnostic_identity_key(diagnostic),)
        if key is not None
    )
    if not reported_diagnostic_counts and diagnostics:
        return []

    expected_diagnostics: list[ProjectDiagnostic] = []
    for unit in units:
        if not isinstance(unit, Mapping):
            return []
        unit_path_value = unit.get("path")
        if not (
            _is_non_empty_string(unit_path_value)
            and _is_report_identity_path(unit_path_value)
        ):
            return []
        unit_path = (root_path / unit_path_value).resolve()
        if not _is_relative_to(unit_path, root_path) or not unit_path.is_file():
            continue
        try:
            _, include_diagnostics = _scan_include_dependencies(
                include_config,
                unit_path,
                unit_path_value,
            )
        except (OSError, ValueError):
            continue
        expected_diagnostics.extend(include_diagnostics)

    expected_diagnostic_counts = Counter(
        key
        for diagnostic in expected_diagnostics
        for key in (_diagnostic_identity_key(diagnostic),)
        if key is not None
    )
    missing = expected_diagnostic_counts - reported_diagnostic_counts
    if not missing:
        return []

    labels_by_key = {
        key: _diagnostic_identity_label(diagnostic)
        for diagnostic in expected_diagnostics
        for key in (_diagnostic_identity_key(diagnostic),)
        if key is not None
    }
    reasons = []
    for key, count in sorted(missing.items(), key=lambda item: labels_by_key[item[0]]):
        suffix = f" ({count} records)" if count > 1 else ""
        reasons.append(
            "diagnostics must include current include scan diagnostic "
            f"{labels_by_key[key]}{suffix}"
        )
    return reasons


def _unit_include_dependencies_contract_reasons(
    index: int,
    unit: Mapping[str, Any],
    *,
    root_path: Path | None = None,
    project: Mapping[str, Any] | None = None,
    require_closed_fields: bool = False,
) -> list[str]:
    include_config = _project_config_for_include_validation(project, root_path)
    if "includeDependencies" not in unit:
        if require_closed_fields:
            return _current_include_dependency_scan_contract_reasons(
                index,
                unit,
                [],
                root_path=root_path,
                include_config=include_config,
            )
        return []

    dependencies = unit.get("includeDependencies")
    if not isinstance(dependencies, list):
        return [f"units[{index}].includeDependencies must be a list"]

    reasons = []
    declared_variants = (
        frozenset(include_config.variants) if include_config is not None else None
    )
    for dependency_index, dependency in enumerate(dependencies):
        dependency_reasons = _include_dependency_contract_reasons(
            index,
            dependency_index,
            dependency,
            declared_variants=declared_variants,
            require_closed_fields=require_closed_fields,
        )
        reasons.extend(dependency_reasons)
        if dependency_reasons or not isinstance(dependency, Mapping):
            continue
        reasons.extend(
            _current_include_dependency_contract_reasons(
                index,
                dependency_index,
                unit,
                dependency,
                root_path=root_path,
                include_config=include_config,
            )
        )
    if require_closed_fields:
        reasons.extend(
            _current_include_dependency_scan_contract_reasons(
                index,
                unit,
                dependencies,
                root_path=root_path,
                include_config=include_config,
            )
        )
    return reasons


def _unit_contract_reasons(
    index: int,
    unit: Any,
    *,
    root_path: Path | None = None,
    project: Mapping[str, Any] | None = None,
    require_source_hash: bool = False,
    check_current_source_hash: bool = False,
    require_registered_source_backend: bool = False,
    require_declared_source_override: bool = False,
    require_closed_fields: bool = False,
) -> list[str]:
    prefix = f"units[{index}]"
    if not isinstance(unit, Mapping):
        return [f"{prefix} must be an object"]

    reasons = (
        _unsupported_mapping_field_reasons(prefix, unit, REPORT_UNIT_FIELDS)
        if require_closed_fields
        else []
    )
    path = unit.get("path")
    reasons.extend(_repository_path_contract_reasons(f"{prefix}.path", path))

    unit_id = unit.get("id")
    if not _is_non_empty_string(unit_id):
        reasons.append(f"{prefix}.id must be a string")
    elif _is_non_empty_string(path) and unit_id != path:
        reasons.append(f"{prefix}.id must match {prefix}.path")

    reasons.extend(
        _source_backend_contract_reasons(
            f"{prefix}.sourceBackend",
            unit.get("sourceBackend"),
            require_registered=require_registered_source_backend,
        )
    )
    extension = unit.get("extension")
    if not isinstance(extension, str):
        reasons.append(f"{prefix}.extension must be a string")
    elif _is_non_empty_string(path) and _is_report_identity_path(path):
        expected_extension = Path(path).suffix.lower()
        if extension != expected_extension:
            reasons.append(f"{prefix}.extension must match {prefix}.path suffix")
    reasons.extend(
        _unit_source_override_contract_reasons(
            index,
            unit,
            project,
            require_declared_override=require_declared_source_override,
        )
    )
    reasons.extend(
        _unit_source_hash_contract_reasons(
            index,
            unit,
            root_path=root_path,
            required=require_source_hash,
            check_current_file=check_current_source_hash,
            require_closed_fields=require_source_hash,
        )
    )
    reasons.extend(
        _unit_source_size_contract_reasons(
            index,
            unit,
            root_path=root_path,
            required=require_source_hash,
            check_current_file=check_current_source_hash,
        )
    )
    reasons.extend(
        _unit_include_dependencies_contract_reasons(
            index,
            unit,
            root_path=root_path,
            project=project,
            require_closed_fields=require_closed_fields,
        )
    )
    return reasons


def _skipped_source_override_contract_reasons(
    index: int,
    skipped: Mapping[str, Any],
    project: Mapping[str, Any] | None,
    *,
    require_declared_override: bool,
) -> list[str]:
    prefix = f"skipped[{index}].sourceOverride"
    reason = skipped.get("reason")
    if "sourceOverride" not in skipped:
        if require_declared_override and reason == "unsupported-source-override":
            return [
                f"{prefix} must be recorded for unsupported-source-override records"
            ]
        return []

    source_override = skipped.get("sourceOverride")
    if not _is_non_empty_string(source_override):
        return [f"{prefix} must be a string"]
    if not require_declared_override or not isinstance(project, Mapping):
        return []

    skipped_path = skipped.get("path")
    source_overrides = project.get("sourceOverrides")
    if (
        not _is_non_empty_string(skipped_path)
        or not _is_report_identity_path(skipped_path)
        or not _valid_string_mapping(source_overrides)
    ):
        return []

    matching_override_backends = [
        backend
        for pattern, backend in source_overrides.items()
        if fnmatch.fnmatch(skipped_path, pattern)
    ]
    matching_override = any(
        source_override == backend for backend in matching_override_backends
    )
    if not matching_override:
        return [
            f"{prefix} must match project.sourceOverrides for skipped[{index}].path "
            f"({_value_mismatch_context(matching_override_backends, source_override)})"
        ]
    return []


def _skipped_contract_reasons(
    index: int,
    skipped: Any,
    *,
    project: Mapping[str, Any] | None = None,
    require_declared_source_override: bool = False,
    require_closed_fields: bool = False,
) -> list[str]:
    prefix = f"skipped[{index}]"
    if not isinstance(skipped, Mapping):
        return [f"{prefix} must be an object"]

    reasons = (
        _unsupported_mapping_field_reasons(prefix, skipped, REPORT_SKIPPED_FIELDS)
        if require_closed_fields
        else []
    )
    reasons.extend(
        _repository_path_contract_reasons(f"{prefix}.path", skipped.get("path"))
    )
    if not _is_non_empty_string(skipped.get("reason")):
        reasons.append(f"{prefix}.reason must be a string")
    reasons.extend(
        _skipped_source_override_contract_reasons(
            index,
            skipped,
            project,
            require_declared_override=require_declared_source_override,
        )
    )
    return reasons


def _config_string_list_contract_reasons(prefix: str, value: Any) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return [f"{prefix} must be a list of strings"]
    if any(not item.strip() for item in value):
        return [f"{prefix} entries must be non-empty strings"]
    return []


def _string_mapping_contract_reasons(prefix: str, value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{prefix} must be an object"]
    if any(not _is_non_empty_string(key) for key in value):
        return [f"{prefix} keys must be non-empty strings"]
    if any(not isinstance(item, str) for item in value.values()):
        return [f"{prefix} values must be strings"]
    return []


def _variant_mapping_contract_reasons(prefix: str, value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{prefix} must be an object"]

    reasons = []
    for variant_name, defines in value.items():
        if not _is_non_empty_string(variant_name):
            reasons.append(f"{prefix} keys must be non-empty strings")
            variant_prefix = prefix
        else:
            variant_prefix = _mapping_key_path(prefix, variant_name)
        reasons.extend(_string_mapping_contract_reasons(variant_prefix, defines))
    return reasons


def _valid_string_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and all(
        _is_non_empty_string(key) and isinstance(item, str)
        for key, item in value.items()
    )


def _expected_artifact_defines(
    project: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> dict[str, str] | None:
    project_defines = project.get("defines", {})
    project_variants = project.get("variants", {})
    if not _valid_string_mapping(project_defines) or not isinstance(
        project_variants, Mapping
    ):
        return None

    base_defines = dict(project_defines)
    if not project_variants:
        return base_defines

    variant = artifact.get("variant")
    if not _is_non_empty_string(variant):
        return None
    variant_defines = project_variants.get(variant)
    if not _valid_string_mapping(variant_defines):
        return None
    return {**base_defines, **dict(variant_defines)}


def _artifact_defines_contract_reasons(
    index: int,
    artifact: Mapping[str, Any],
    project: Mapping[str, Any],
    *,
    required: bool,
) -> list[str]:
    if not required and "defines" not in artifact:
        return []

    prefix = f"artifacts[{index}].defines"
    defines = artifact.get("defines")
    reasons = _string_mapping_contract_reasons(prefix, defines)
    if reasons:
        return reasons

    expected_defines = _expected_artifact_defines(project, artifact)
    if expected_defines is not None and dict(defines) != expected_defines:
        reasons.append(f"{prefix} must match project defines and artifact variant")
    return reasons


def _artifact_define_processing_contract_reasons(
    index: int,
    artifact: Mapping[str, Any],
    *,
    required: bool,
) -> list[str]:
    if not required and "defineProcessing" not in artifact:
        return []

    prefix = f"artifacts[{index}].defineProcessing"
    define_processing = artifact.get("defineProcessing")
    if not isinstance(define_processing, Mapping):
        return [f"{prefix} must be an object"]

    reasons = []
    if required:
        reasons.extend(
            _unsupported_mapping_field_reasons(
                prefix, define_processing, REPORT_ARTIFACT_DEFINE_PROCESSING_FIELDS
            )
        )
    status = define_processing.get("status")
    if not isinstance(status, str) or status not in DEFINE_PROCESSING_STATUSES:
        reasons.append(
            f"{prefix}.status must be one of "
            f"{', '.join(sorted(DEFINE_PROCESSING_STATUSES))}"
        )
    frontend = define_processing.get("frontend")
    if frontend != "lexer":
        reasons.append(f"{prefix}.frontend must be lexer")
    supports_defines = define_processing.get("supportsDefines")
    if not isinstance(supports_defines, bool):
        reasons.append(f"{prefix}.supportsDefines must be a boolean")
    define_count = define_processing.get("defineCount")
    if not _is_non_negative_int(define_count):
        reasons.append(f"{prefix}.defineCount must be a non-negative integer")

    defines = artifact.get("defines")
    if isinstance(defines, Mapping) and _valid_string_mapping(defines):
        if define_count != len(defines):
            reasons.append(
                f"{prefix}.defineCount must match artifacts[{index}].defines"
            )
        expected_status = (
            "not-requested"
            if not defines
            else "forwarded" if supports_defines is True else "not-supported"
        )
        if isinstance(status, str) and status != expected_status:
            reasons.append(
                f"{prefix}.status must match define count and source frontend support"
            )

    source_backend = artifact.get("sourceBackend")
    if _is_non_empty_string(source_backend) and isinstance(supports_defines, bool):
        expected_support = _source_frontend_supports_lexer_keyword(
            source_backend, "defines"
        )
        if supports_defines != expected_support:
            reasons.append(
                f"{prefix}.supportsDefines must match artifacts[{index}].sourceBackend"
            )
    return reasons


def _expected_include_path_count(project: Mapping[str, Any]) -> int | None:
    records = project.get("includeDirStatus")
    if not isinstance(records, list):
        return None
    active_count = 0
    for record in records:
        if not isinstance(record, Mapping):
            return None
        if record.get("status") == "active" and record.get("frontendVisible") is True:
            active_count += 1
    return active_count


def _artifact_include_path_processing_contract_reasons(
    index: int,
    artifact: Mapping[str, Any],
    project: Mapping[str, Any],
    *,
    required: bool,
) -> list[str]:
    if not required and "includePathProcessing" not in artifact:
        return []

    prefix = f"artifacts[{index}].includePathProcessing"
    include_path_processing = artifact.get("includePathProcessing")
    if not isinstance(include_path_processing, Mapping):
        return [f"{prefix} must be an object"]

    reasons = []
    if required:
        reasons.extend(
            _unsupported_mapping_field_reasons(
                prefix,
                include_path_processing,
                REPORT_ARTIFACT_INCLUDE_PATH_PROCESSING_FIELDS,
            )
        )
    status = include_path_processing.get("status")
    if not isinstance(status, str) or status not in INCLUDE_PATH_PROCESSING_STATUSES:
        reasons.append(
            f"{prefix}.status must be one of "
            f"{', '.join(sorted(INCLUDE_PATH_PROCESSING_STATUSES))}"
        )
    frontend = include_path_processing.get("frontend")
    if frontend != "lexer":
        reasons.append(f"{prefix}.frontend must be lexer")
    supports_include_paths = include_path_processing.get("supportsIncludePaths")
    if not isinstance(supports_include_paths, bool):
        reasons.append(f"{prefix}.supportsIncludePaths must be a boolean")
    include_path_count = include_path_processing.get("includePathCount")
    if not _is_non_negative_int(include_path_count):
        reasons.append(f"{prefix}.includePathCount must be a non-negative integer")

    expected_include_path_count = _expected_include_path_count(project)
    if expected_include_path_count is not None:
        if include_path_count != expected_include_path_count:
            reasons.append(f"{prefix}.includePathCount must match project include dirs")
        expected_status = (
            "not-requested"
            if expected_include_path_count == 0
            else "forwarded" if supports_include_paths is True else "not-supported"
        )
        if isinstance(status, str) and status != expected_status:
            reasons.append(
                f"{prefix}.status must match include path count and "
                "source frontend support"
            )

    source_backend = artifact.get("sourceBackend")
    if _is_non_empty_string(source_backend) and isinstance(
        supports_include_paths, bool
    ):
        expected_support = _source_frontend_supports_lexer_keyword(
            source_backend, "include_paths"
        )
        if supports_include_paths != expected_support:
            reasons.append(
                f"{prefix}.supportsIncludePaths must match "
                f"artifacts[{index}].sourceBackend"
            )
    return reasons


def _project_root_path(project: Mapping[str, Any]) -> Path | None:
    root = project.get("root")
    if not _is_non_empty_string(root):
        return None
    root_path = Path(root)
    if not root_path.is_absolute() or not root_path.exists() or not root_path.is_dir():
        return None
    return root_path


def _include_dir_status_contract_reasons(
    project: Mapping[str, Any],
    include_dirs: Any,
    *,
    require_counts: bool,
    require_closed_fields: bool = False,
) -> list[str]:
    records = project.get("includeDirStatus")
    if not isinstance(records, list):
        return ["project.includeDirStatus must be a list"]

    reasons = []
    include_dirs_is_list = isinstance(include_dirs, list) and all(
        isinstance(item, str) for item in include_dirs
    )
    if include_dirs_is_list and len(records) != len(include_dirs):
        reasons.append("project.includeDirStatus must match project.includeDirs")

    root_path = _project_root_path(project)
    valid_records = []
    for index, record in enumerate(records):
        prefix = f"project.includeDirStatus[{index}]"
        if not isinstance(record, Mapping):
            reasons.append(f"{prefix} must be an object")
            continue

        if require_closed_fields:
            reasons.extend(
                _unsupported_mapping_field_reasons(
                    prefix, record, REPORT_INCLUDE_DIR_STATUS_FIELDS
                )
            )
        valid_records.append(record)
        include_path = record.get("path")
        resolved_path = record.get("resolvedPath")
        status = record.get("status")
        frontend_visible = record.get("frontendVisible")

        if not isinstance(include_path, str):
            reasons.append(f"{prefix}.path must be a string")
        elif (
            include_dirs_is_list
            and index < len(include_dirs)
            and include_path != include_dirs[index]
        ):
            reasons.append(f"{prefix}.path must match project.includeDirs[{index}]")

        if not _is_non_empty_string(resolved_path):
            reasons.append(f"{prefix}.resolvedPath must be a string")
        elif not Path(resolved_path).is_absolute():
            reasons.append(f"{prefix}.resolvedPath must be an absolute path")

        if not isinstance(status, str) or status not in INCLUDE_DIR_STATUSES:
            reasons.append(f"{prefix}.status must be a known include directory status")
        if not isinstance(frontend_visible, bool):
            reasons.append(f"{prefix}.frontendVisible must be a boolean")

        if (
            root_path is None
            or not isinstance(include_path, str)
            or not _is_non_empty_string(resolved_path)
            or not Path(resolved_path).is_absolute()
            or not isinstance(status, str)
            or status not in INCLUDE_DIR_STATUSES
            or not isinstance(frontend_visible, bool)
        ):
            continue

        expected_path = Path(include_path)
        if not expected_path.is_absolute():
            expected_path = root_path / expected_path
        expected_path = expected_path.resolve()
        actual_resolved_path = Path(resolved_path).resolve()
        if actual_resolved_path != expected_path:
            reasons.append(
                f"{prefix}.resolvedPath must match the resolved include directory "
                f"({_value_mismatch_context(actual_resolved_path, expected_path)})"
            )
        expected_status = (
            "outside-project"
            if not _is_relative_to(expected_path, root_path)
            else (
                "missing"
                if not expected_path.exists()
                else "not-directory" if not expected_path.is_dir() else "active"
            )
        )
        if status != expected_status:
            reasons.append(
                f"{prefix}.status must match the resolved include directory "
                f"({_value_mismatch_context(status, expected_status)})"
            )
        expected_visible = expected_status == "active"
        if frontend_visible != expected_visible:
            reasons.append(
                f"{prefix}.frontendVisible must match status "
                f"({_value_mismatch_context(frontend_visible, expected_visible)})"
            )

    if require_counts or "includeDirStatusCounts" in project:
        counts = project.get("includeDirStatusCounts")
        expected_counts = _include_dir_status_counts(valid_records)
        reasons.extend(
            _mapping_field_contract_reasons(
                "project.includeDirStatusCounts",
                counts,
                expected_counts,
                "project.includeDirStatus",
            )
        )

    return reasons


def _source_root_status_contract_reasons(
    project: Mapping[str, Any],
    source_roots: Any,
    *,
    require_counts: bool,
    require_closed_fields: bool = False,
) -> list[str]:
    records = project.get("sourceRootStatus")
    if not isinstance(records, list):
        return ["project.sourceRootStatus must be a list"]

    reasons = []
    source_roots_is_list = isinstance(source_roots, list) and all(
        isinstance(item, str) for item in source_roots
    )
    if source_roots_is_list and len(records) != len(source_roots):
        reasons.append("project.sourceRootStatus must match project.sourceRoots")

    root_path = _project_root_path(project)
    valid_records = []
    for index, record in enumerate(records):
        prefix = f"project.sourceRootStatus[{index}]"
        if not isinstance(record, Mapping):
            reasons.append(f"{prefix} must be an object")
            continue

        if require_closed_fields:
            reasons.extend(
                _unsupported_mapping_field_reasons(
                    prefix, record, REPORT_SOURCE_ROOT_STATUS_FIELDS
                )
            )
        valid_records.append(record)
        source_root = record.get("path")
        resolved_path = record.get("resolvedPath")
        status = record.get("status")
        scan_visible = record.get("scanVisible")

        if not isinstance(source_root, str):
            reasons.append(f"{prefix}.path must be a string")
        elif (
            source_roots_is_list
            and index < len(source_roots)
            and source_root != source_roots[index]
        ):
            reasons.append(f"{prefix}.path must match project.sourceRoots[{index}]")

        if not _is_non_empty_string(resolved_path):
            reasons.append(f"{prefix}.resolvedPath must be a string")
        elif not Path(resolved_path).is_absolute():
            reasons.append(f"{prefix}.resolvedPath must be an absolute path")

        if not isinstance(status, str) or status not in SOURCE_ROOT_STATUSES:
            reasons.append(f"{prefix}.status must be a known source root status")
        if not isinstance(scan_visible, bool):
            reasons.append(f"{prefix}.scanVisible must be a boolean")

        if (
            root_path is None
            or not isinstance(source_root, str)
            or not _is_non_empty_string(resolved_path)
            or not Path(resolved_path).is_absolute()
            or not isinstance(status, str)
            or status not in SOURCE_ROOT_STATUSES
            or not isinstance(scan_visible, bool)
        ):
            continue

        expected_path = Path(source_root)
        if not expected_path.is_absolute():
            expected_path = root_path / expected_path
        expected_path = expected_path.resolve()
        actual_resolved_path = Path(resolved_path).resolve()
        if actual_resolved_path != expected_path:
            reasons.append(
                f"{prefix}.resolvedPath must match the resolved source root "
                f"({_value_mismatch_context(actual_resolved_path, expected_path)})"
            )
        expected_status = (
            "outside-project"
            if not _is_relative_to(expected_path, root_path)
            else (
                "missing"
                if not expected_path.exists()
                else "not-directory" if not expected_path.is_dir() else "active"
            )
        )
        if status != expected_status:
            reasons.append(
                f"{prefix}.status must match the resolved source root "
                f"({_value_mismatch_context(status, expected_status)})"
            )
        expected_visible = expected_status == "active"
        if scan_visible != expected_visible:
            reasons.append(
                f"{prefix}.scanVisible must match status "
                f"({_value_mismatch_context(scan_visible, expected_visible)})"
            )

    if require_counts or "sourceRootStatusCounts" in project:
        counts = project.get("sourceRootStatusCounts")
        expected_counts = _source_root_status_counts(valid_records)
        reasons.extend(
            _mapping_field_contract_reasons(
                "project.sourceRootStatusCounts",
                counts,
                expected_counts,
                "project.sourceRootStatus",
            )
        )

    return reasons


def _optional_project_field(
    project: Mapping[str, Any], key: str, *, required: bool
) -> bool:
    return required or key in project


def _project_metadata_contract_reasons(
    project: Mapping[str, Any], *, require_full_metadata: bool
) -> list[str]:
    reasons = (
        _unsupported_mapping_field_reasons("project", project, REPORT_PROJECT_FIELDS)
        if require_full_metadata
        else []
    )
    if _optional_project_field(project, "config", required=require_full_metadata):
        config_path = project.get("config")
        if config_path is not None and not isinstance(config_path, str):
            reasons.append("project.config must be a string or null")
        elif config_path is not None:
            if not config_path.strip():
                reasons.append("project.config must be a non-empty string or null")
            elif require_full_metadata:
                resolved_config_path = Path(config_path)
                if not resolved_config_path.is_absolute():
                    reasons.append("project.config must be an absolute path")
                elif not resolved_config_path.exists():
                    reasons.append("project.config must exist")
                elif not resolved_config_path.is_file():
                    reasons.append("project.config must be a file")

    if "configHash" in project:
        config_hash = project.get("configHash")
        config_path = project.get("config")
        if config_path is None:
            if config_hash is not None:
                reasons.append(
                    "project.configHash must be null when project.config is null"
                )
        elif not isinstance(config_hash, Mapping):
            reasons.append(
                "project.configHash must be an object when project.config is set"
            )
        else:
            hash_reasons = _hash_contract_reasons(
                "project.configHash",
                config_hash,
                require_closed_fields=require_full_metadata,
            )
            reasons.extend(hash_reasons)
            if (
                not hash_reasons
                and isinstance(config_path, str)
                and config_path.strip()
            ):
                config_file = Path(config_path)
                if config_file.is_file() and not _hash_matches_report(
                    _source_hash(config_file),
                    config_hash,
                ):
                    reasons.append(
                        "project.configHash must match the current config file"
                    )

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

    for field_name, list_name in (
        ("sourceRootCount", "sourceRoots"),
        ("includePatternCount", "includePatterns"),
        ("excludePatternCount", "excludePatterns"),
    ):
        list_value = project.get(list_name)
        list_is_valid = isinstance(list_value, list) and all(
            isinstance(item, str) for item in list_value
        )
        if _optional_project_field(project, field_name, required=require_full_metadata):
            count = project.get(field_name)
            if not _is_non_negative_int(count):
                reasons.append(f"project.{field_name} must be a non-negative integer")
            elif list_is_valid and count != len(list_value):
                reasons.append(
                    f"project.{field_name} must match project.{list_name} "
                    f"({_value_mismatch_context(count, len(list_value))})"
                )

    source_roots = project.get("sourceRoots")
    if _optional_project_field(
        project, "sourceRootStatus", required=require_full_metadata
    ):
        reasons.extend(
            _source_root_status_contract_reasons(
                project,
                source_roots,
                require_counts=require_full_metadata,
                require_closed_fields=require_full_metadata,
            )
        )
    elif "sourceRootStatusCounts" in project:
        reasons.append(
            "project.sourceRootStatusCounts requires project.sourceRootStatus"
        )

    include_dirs = project.get("includeDirs")
    include_dirs_is_list = isinstance(include_dirs, list) and all(
        isinstance(item, str) for item in include_dirs
    )
    if _optional_project_field(
        project, "includeDirCount", required=require_full_metadata
    ):
        include_dir_count = project.get("includeDirCount")
        if not _is_non_negative_int(include_dir_count):
            reasons.append("project.includeDirCount must be a non-negative integer")
        elif include_dirs_is_list and include_dir_count != len(include_dirs):
            reasons.append(
                "project.includeDirCount must match project.includeDirs "
                f"({_value_mismatch_context(include_dir_count, len(include_dirs))})"
            )

    if _optional_project_field(
        project, "includeDirStatus", required=require_full_metadata
    ):
        reasons.extend(
            _include_dir_status_contract_reasons(
                project,
                include_dirs,
                require_counts=require_full_metadata,
                require_closed_fields=require_full_metadata,
            )
        )
    elif "includeDirStatusCounts" in project:
        reasons.append(
            "project.includeDirStatusCounts requires project.includeDirStatus"
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
            reasons.append(
                "project.defineCount must match project.defines "
                f"({_value_mismatch_context(define_count, len(defines))})"
            )

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
                "project.sourceOverrideCount must match project.sourceOverrides "
                f"({_value_mismatch_context(source_override_count, len(source_overrides))})"
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
            reasons.append(
                "project.variantCount must match project.variants "
                f"({_value_mismatch_context(variant_count, len(variants))})"
            )

    if _optional_project_field(
        project, "variantDefineCounts", required=require_full_metadata
    ):
        variant_define_counts = project.get("variantDefineCounts")
        if variants_is_mapping:
            reasons.extend(
                _mapping_field_contract_reasons(
                    "project.variantDefineCounts",
                    variant_define_counts,
                    _variant_define_counts(variants),
                    "project.variants",
                )
            )
        elif not isinstance(variant_define_counts, Mapping):
            reasons.append("project.variantDefineCounts must be an object")
        elif any(
            not _is_non_empty_string(name) or not _is_non_negative_int(count)
            for name, count in variant_define_counts.items()
        ):
            reasons.append(
                "project.variantDefineCounts must map variant names to "
                "non-negative integers"
            )

    if _optional_project_field(
        project, "selectedVariants", required=require_full_metadata
    ):
        selected_variant_reasons = _config_string_list_contract_reasons(
            "project.selectedVariants", project.get("selectedVariants")
        )
        reasons.extend(selected_variant_reasons)
        selected_variants = project.get("selectedVariants")
        if (
            not selected_variant_reasons
            and isinstance(selected_variants, list)
            and variants_is_mapping
        ):
            if len(set(selected_variants)) != len(selected_variants):
                reasons.append(
                    "project.selectedVariants must not contain duplicate entries"
                )
            unknown_selected_variants = [
                name for name in selected_variants if name not in variants
            ]
            if unknown_selected_variants:
                reasons.append(
                    "project.selectedVariants must be listed in project.variants"
                )

    if _optional_project_field(
        project, "externalCorpusManifest", required=require_full_metadata
    ):
        external_corpus_manifest = project.get("externalCorpusManifest")
        if external_corpus_manifest is not None and not isinstance(
            external_corpus_manifest, str
        ):
            reasons.append("project.externalCorpusManifest must be a string or null")
        elif (
            external_corpus_manifest is not None
            and not external_corpus_manifest.strip()
        ):
            reasons.append(
                "project.externalCorpusManifest must be a non-empty string or null"
            )

    return reasons


def _generator_contract_reasons(
    report: Mapping[str, Any], *, require_generator: bool
) -> list[str]:
    if "generator" not in report:
        return ["generator must be an object"] if require_generator else []

    generator = report.get("generator")
    if not isinstance(generator, Mapping):
        return ["generator must be an object"]

    reasons = (
        _unsupported_mapping_field_reasons(
            "generator", generator, REPORT_GENERATOR_FIELDS
        )
        if require_generator
        else []
    )
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
    require_closed_fields: bool = False,
) -> list[str]:
    prefix = f"validation.toolchains[{index}]"
    if not isinstance(toolchain, Mapping):
        return [f"{prefix} must be an object"]

    reasons = (
        _unsupported_mapping_field_reasons(
            prefix, toolchain, VALIDATION_TOOLCHAIN_FIELDS
        )
        if require_closed_fields
        else []
    )
    status = toolchain.get("status")
    status_is_valid = status in {"available", "unavailable", "not-configured"}
    target = toolchain.get("target")
    target_reasons = _target_name_contract_reasons(
        f"{prefix}.target",
        target,
        declared_targets=declared_targets,
        require_canonical=require_closed_fields,
    )
    reasons.extend(target_reasons)
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
            if require_closed_fields:
                reasons.extend(
                    _unsupported_mapping_field_reasons(
                        tool_prefix, tool, VALIDATION_TOOL_FIELDS
                    )
                )
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
    if (
        tools_are_valid
        and not target_reasons
        and _is_non_empty_string(target)
        and isinstance(tools, list)
    ):
        normalized_target = _normalized_targets([target])[0]
        configured_tools = TOOLCHAIN_BY_BACKEND.get(normalized_target, ())
        if configured_tools:
            reported_tool_names = {
                tool.get("name")
                for tool in tools
                if isinstance(tool, Mapping) and _is_non_empty_string(tool.get("name"))
            }
            missing_tools = [
                tool_name
                for tool_name in configured_tools
                if tool_name not in reported_tool_names
            ]
            for tool_name in missing_tools:
                reasons.append(
                    f"{prefix}.tools must include configured validation tool "
                    f"{tool_name} for target {normalized_target}"
                )
            for tool_index, tool in enumerate(tools):
                name = tool.get("name") if isinstance(tool, Mapping) else None
                if _is_non_empty_string(name) and name not in configured_tools:
                    reasons.append(
                        f"{prefix}.tools[{tool_index}].name must match a configured "
                        f"validation tool for target {normalized_target}"
                    )
        elif tools:
            reasons.append(
                f"{prefix}.tools must be empty when no validation toolchain hook "
                f"is configured for target {normalized_target}"
            )
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


def _expected_validation_artifact_status(artifact: Mapping[str, Any]) -> str:
    if artifact.get("exists") is not True:
        return "failed"
    source_hash_status = artifact.get("sourceHashStatus", "not-recorded")
    source_size_status = artifact.get("sourceSizeStatus", "not-recorded")
    generated_hash_status = artifact.get("generatedHashStatus", "not-recorded")
    generated_size_status = artifact.get("generatedSizeStatus", "not-recorded")
    source_map_status = artifact.get("sourceMapStatus", "ok")
    source_remap_status = artifact.get("sourceRemapStatus", "ok")
    return (
        "ok"
        if source_hash_status in {"ok", "not-recorded"}
        and source_size_status in {"ok", "not-recorded"}
        and generated_hash_status in {"ok", "not-recorded"}
        and generated_size_status in {"ok", "not-recorded"}
        and source_map_status in {"ok", "not-applicable", "not-recorded"}
        and source_remap_status in {"ok", "not-applicable", "not-recorded"}
        else "failed"
    )


def _validation_artifact_status_matches_record(artifact: Mapping[str, Any]) -> bool:
    return artifact.get("status") == _expected_validation_artifact_status(artifact)


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
    require_status_fields: bool = False,
    require_closed_fields: bool = False,
) -> list[str]:
    prefix = f"validation.artifacts[{index}]"
    if not isinstance(artifact, Mapping):
        return [f"{prefix} must be an object"]

    reasons = (
        _unsupported_mapping_field_reasons(prefix, artifact, VALIDATION_ARTIFACT_FIELDS)
        if require_closed_fields
        else []
    )
    for field_name in ("source", "path"):
        if not _is_non_empty_string(artifact.get(field_name)):
            reasons.append(f"{prefix}.{field_name} must be a string")
    source = artifact.get("source")
    if _is_non_empty_string(source) and not _is_report_identity_path(source):
        reasons.append(f"{prefix}.source must be repository-relative")
    path = artifact.get("path")
    if _is_non_empty_string(path) and not _is_report_identity_path(path):
        reasons.append(f"{prefix}.path must be repository-relative")
    reasons.extend(
        _target_name_contract_reasons(
            f"{prefix}.target",
            artifact.get("target"),
            declared_targets=declared_targets,
            require_canonical=require_closed_fields,
        )
    )
    if not isinstance(artifact.get("exists"), bool):
        reasons.append(f"{prefix}.exists must be a boolean")
    status = artifact.get("status")
    status_is_valid = status in {"ok", "failed"}
    if not status_is_valid:
        reasons.append(
            f"{prefix}.status must be ok or failed "
            f"({_allowed_value_mismatch_context(['failed', 'ok'], status)})"
        )
    variant = artifact.get("variant")
    if "variant" in artifact:
        if not _is_non_empty_string(variant):
            reasons.append(f"{prefix}.variant must be a string")
        elif declared_variants is not None and variant not in declared_variants:
            reasons.append(f"{prefix}.variant must be listed in project.variants")
    if "sourceBackend" in artifact:
        reasons.extend(
            _source_backend_contract_reasons(
                f"{prefix}.sourceBackend",
                artifact.get("sourceBackend"),
                require_registered=True,
            )
        )
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
    if (
        "sourceBackend" not in artifact
        and referenced_artifact is not None
        and _is_non_empty_string(referenced_artifact[1].get("sourceBackend"))
    ):
        reasons.append(
            f"{prefix}.sourceBackend must be recorded when "
            f"report.artifacts[{referenced_artifact[0]}].sourceBackend is recorded"
        )
    if (
        "sourceBackend" in artifact
        and referenced_artifact is not None
        and _is_non_empty_string(artifact.get("sourceBackend"))
        and _is_non_empty_string(referenced_artifact[1].get("sourceBackend"))
        and artifact.get("sourceBackend") != referenced_artifact[1].get("sourceBackend")
    ):
        reasons.append(
            f"{prefix}.sourceBackend must match "
            f"report.artifacts[{referenced_artifact[0]}].sourceBackend"
        )
    source_hash_status = artifact.get("sourceHashStatus")
    if require_status_fields and "sourceHashStatus" not in artifact:
        reasons.append(
            f"{prefix}.sourceHashStatus must be recorded "
            "when validation.summary is present"
        )
    if (
        "sourceHashStatus" in artifact
        and source_hash_status not in SOURCE_HASH_VALIDATION_STATUSES
    ):
        reasons.append(
            _allowed_value_reason(
                f"{prefix}.sourceHashStatus",
                SOURCE_HASH_VALIDATION_STATUSES,
                source_hash_status,
            )
        )
    source_size_status = artifact.get("sourceSizeStatus")
    if require_status_fields and "sourceSizeStatus" not in artifact:
        reasons.append(
            f"{prefix}.sourceSizeStatus must be recorded "
            "when validation.summary is present"
        )
    if (
        "sourceSizeStatus" in artifact
        and source_size_status not in SOURCE_SIZE_VALIDATION_STATUSES
    ):
        reasons.append(
            _allowed_value_reason(
                f"{prefix}.sourceSizeStatus",
                SOURCE_SIZE_VALIDATION_STATUSES,
                source_size_status,
            )
        )
    generated_hash_status = artifact.get("generatedHashStatus")
    if require_status_fields and "generatedHashStatus" not in artifact:
        reasons.append(
            f"{prefix}.generatedHashStatus must be recorded "
            "when validation.summary is present"
        )
    if (
        "generatedHashStatus" in artifact
        and generated_hash_status not in GENERATED_HASH_VALIDATION_STATUSES
    ):
        reasons.append(
            _allowed_value_reason(
                f"{prefix}.generatedHashStatus",
                GENERATED_HASH_VALIDATION_STATUSES,
                generated_hash_status,
            )
        )
    generated_size_status = artifact.get("generatedSizeStatus")
    if require_status_fields and "generatedSizeStatus" not in artifact:
        reasons.append(
            f"{prefix}.generatedSizeStatus must be recorded "
            "when validation.summary is present"
        )
    if (
        "generatedSizeStatus" in artifact
        and generated_size_status not in GENERATED_SIZE_VALIDATION_STATUSES
    ):
        reasons.append(
            _allowed_value_reason(
                f"{prefix}.generatedSizeStatus",
                GENERATED_SIZE_VALIDATION_STATUSES,
                generated_size_status,
            )
        )
    source_map_status = artifact.get("sourceMapStatus")
    if require_status_fields and "sourceMapStatus" not in artifact:
        reasons.append(
            f"{prefix}.sourceMapStatus must be recorded "
            "when validation.summary is present"
        )
    if (
        "sourceMapStatus" in artifact
        and source_map_status not in SOURCE_MAP_VALIDATION_STATUSES
    ):
        reasons.append(
            _allowed_value_reason(
                f"{prefix}.sourceMapStatus",
                SOURCE_MAP_VALIDATION_STATUSES,
                source_map_status,
            )
        )
    source_remap_status = artifact.get("sourceRemapStatus")
    if require_status_fields and "sourceRemapStatus" not in artifact:
        reasons.append(
            f"{prefix}.sourceRemapStatus must be recorded "
            "when validation.summary is present"
        )
    if (
        "sourceRemapStatus" in artifact
        and source_remap_status not in SOURCE_REMAP_VALIDATION_STATUSES
    ):
        reasons.append(
            _allowed_value_reason(
                f"{prefix}.sourceRemapStatus",
                SOURCE_REMAP_VALIDATION_STATUSES,
                source_remap_status,
            )
        )
    if (
        referenced_artifact is not None
        and referenced_artifact[1].get("status") == "failed"
    ):
        for field_name, value in (
            ("generatedHashStatus", generated_hash_status),
            ("generatedSizeStatus", generated_size_status),
            ("sourceMapStatus", source_map_status),
            ("sourceRemapStatus", source_remap_status),
        ):
            if field_name in artifact and value != "not-applicable":
                reasons.append(
                    f"{prefix}.{field_name} must be not-applicable when "
                    f"report.artifacts[{referenced_artifact[0]}].status is failed "
                    f"({_value_mismatch_context('not-applicable', value)})"
                )
    if (
        status_is_valid
        and isinstance(artifact.get("exists"), bool)
        and (
            "sourceHashStatus" not in artifact
            or source_hash_status in SOURCE_HASH_VALIDATION_STATUSES
        )
        and (
            "sourceSizeStatus" not in artifact
            or source_size_status in SOURCE_SIZE_VALIDATION_STATUSES
        )
        and (
            "generatedHashStatus" not in artifact
            or generated_hash_status in GENERATED_HASH_VALIDATION_STATUSES
        )
        and (
            "generatedSizeStatus" not in artifact
            or generated_size_status in GENERATED_SIZE_VALIDATION_STATUSES
        )
        and (
            "sourceMapStatus" not in artifact
            or source_map_status in SOURCE_MAP_VALIDATION_STATUSES
        )
        and (
            "sourceRemapStatus" not in artifact
            or source_remap_status in SOURCE_REMAP_VALIDATION_STATUSES
        )
        and not _validation_artifact_status_matches_record(artifact)
    ):
        expected_status = _expected_validation_artifact_status(artifact)
        reasons.append(
            f"{prefix}.status must match exists, hash, size, and provenance statuses "
            f"({_value_mismatch_context(expected_status, status)})"
        )
    if (
        status == "ok"
        and referenced_artifact is not None
        and referenced_artifact[1].get("status") == "failed"
    ):
        reasons.append(
            f"{prefix}.status must match "
            f"report.artifacts[{referenced_artifact[0]}].status "
            f"({_value_mismatch_context('failed', status)})"
        )
    return reasons


def _validation_summary_contract_reasons(
    summary: Any,
    artifact_checks: Sequence[Any],
    *,
    require_closed_fields: bool = False,
) -> list[str]:
    if not isinstance(summary, Mapping):
        return ["validation.summary must be an object"]

    expected = _validation_summary(artifact_checks)
    reasons = (
        _unsupported_mapping_field_reasons(
            "validation.summary", summary, VALIDATION_SUMMARY_FIELDS
        )
        if require_closed_fields
        else []
    )
    for field_name in ("artifactCount", "okCount", "failedCount"):
        reasons.extend(
            _count_field_contract_reasons(
                f"validation.summary.{field_name}",
                summary.get(field_name),
                expected[field_name],
                "validation.artifacts",
            )
        )
    for field_name in (
        "sourceHashStatusCounts",
        "sourceSizeStatusCounts",
        "generatedHashStatusCounts",
        "generatedSizeStatusCounts",
        "sourceMapStatusCounts",
        "sourceRemapStatusCounts",
    ):
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
    root_path: Path | None = None,
    declared_targets: set[str] | None = None,
    declared_variants: set[str] | None = None,
    declared_artifact_identities: set[ArtifactIdentity] | None = None,
    declared_artifacts_by_identity: (
        Mapping[ArtifactIdentity, DeclaredArtifact] | None
    ) = None,
    require_closed_fields: bool = False,
) -> list[str]:
    prefix = f"validation.toolchainRuns[{index}]"
    if not isinstance(run, Mapping):
        return [f"{prefix} must be an object"]

    reasons = (
        _unsupported_mapping_field_reasons(prefix, run, VALIDATION_TOOLCHAIN_RUN_FIELDS)
        if require_closed_fields
        else []
    )
    for field_name in ("source", "path"):
        if not _is_non_empty_string(run.get(field_name)):
            reasons.append(f"{prefix}.{field_name} must be a string")
    source = run.get("source")
    if _is_non_empty_string(source) and not _is_report_identity_path(source):
        reasons.append(f"{prefix}.source must be repository-relative")
    path = run.get("path")
    if _is_non_empty_string(path) and not _is_report_identity_path(path):
        reasons.append(f"{prefix}.path must be repository-relative")
    reasons.extend(
        _target_name_contract_reasons(
            f"{prefix}.target",
            run.get("target"),
            declared_targets=declared_targets,
            require_canonical=require_closed_fields,
        )
    )
    variant = run.get("variant")
    if "variant" in run:
        if not _is_non_empty_string(variant):
            reasons.append(f"{prefix}.variant must be a string")
        elif declared_variants is not None and variant not in declared_variants:
            reasons.append(f"{prefix}.variant must be listed in project.variants")
    if "sourceBackend" in run:
        reasons.extend(
            _source_backend_contract_reasons(
                f"{prefix}.sourceBackend",
                run.get("sourceBackend"),
                require_registered=True,
            )
        )
    identity = _artifact_identity(run)
    if (
        identity is not None
        and declared_artifact_identities is not None
        and identity not in declared_artifact_identities
    ):
        reasons.append(f"{prefix} must reference an artifact in report.artifacts")
    referenced_artifact: DeclaredArtifact | None = None
    if identity is not None and declared_artifacts_by_identity is not None:
        referenced_artifact = declared_artifacts_by_identity.get(identity)
    if (
        "sourceBackend" not in run
        and referenced_artifact is not None
        and _is_non_empty_string(referenced_artifact[1].get("sourceBackend"))
    ):
        reasons.append(
            f"{prefix}.sourceBackend must be recorded when "
            f"report.artifacts[{referenced_artifact[0]}].sourceBackend is recorded"
        )
    if (
        "sourceBackend" in run
        and referenced_artifact is not None
        and _is_non_empty_string(run.get("sourceBackend"))
        and _is_non_empty_string(referenced_artifact[1].get("sourceBackend"))
        and run.get("sourceBackend") != referenced_artifact[1].get("sourceBackend")
    ):
        reasons.append(
            f"{prefix}.sourceBackend must match "
            f"report.artifacts[{referenced_artifact[0]}].sourceBackend"
        )
    if (
        referenced_artifact is not None
        and referenced_artifact[1].get("status") != "translated"
    ):
        reasons.append(
            f"{prefix} must reference a translated "
            f"report.artifacts[{referenced_artifact[0]}] record"
        )

    command = run.get("command")
    normalized_target = None
    if (
        not isinstance(command, list)
        or not command
        or any(not _is_non_empty_string(part) for part in command)
    ):
        reasons.append(f"{prefix}.command must be a non-empty list of strings")
    elif _is_non_empty_string(run.get("target")):
        normalized_target = _normalized_targets([str(run["target"])])[0]
        configured_tools = TOOLCHAIN_BY_BACKEND.get(normalized_target, ())
        if configured_tools and command[0] not in configured_tools:
            reasons.append(
                f"{prefix}.command[0] must match a configured validation tool "
                f"for target {normalized_target}"
            )

    check_kind = run.get("checkKind")
    if require_closed_fields and "checkKind" not in run:
        reasons.append(f"{prefix}.checkKind must be recorded")
    elif "checkKind" in run:
        if check_kind not in VALIDATION_TOOLCHAIN_RUN_CHECK_KINDS:
            allowed = ", ".join(sorted(VALIDATION_TOOLCHAIN_RUN_CHECK_KINDS))
            reasons.append(f"{prefix}.checkKind must be one of {allowed}")
        elif (
            normalized_target is not None
            and isinstance(command, list)
            and all(_is_non_empty_string(part) for part in command)
            and _is_non_empty_string(path)
        ):
            configured_tools = TOOLCHAIN_BY_BACKEND.get(normalized_target, ())
            artifact_path = Path(str(path))
            if root_path is not None and not artifact_path.is_absolute():
                artifact_path = (root_path / artifact_path).resolve()
            smoke_command = _toolchain_smoke_command(
                normalized_target, configured_tools, artifact_path
            )
            if smoke_command is not None:
                expected_command, expected_check_kind = smoke_command
                if check_kind != expected_check_kind:
                    reasons.append(
                        f"{prefix}.checkKind must be {expected_check_kind} "
                        f"for target {normalized_target}"
                    )
                elif command != expected_command:
                    check_label = (
                        "tool availability"
                        if expected_check_kind == "tool-availability"
                        else "artifact"
                    )
                    reasons.append(
                        f"{prefix}.command must match the configured {check_label} "
                        f"check for target {normalized_target}"
                    )

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
    report: Mapping[str, Any],
    *,
    root_path: Path | None = None,
    require_validation: bool,
) -> list[str]:
    if "validation" not in report:
        return ["validation must be an object"] if require_validation else []

    validation = report.get("validation")
    if not isinstance(validation, Mapping):
        return ["validation must be an object"]

    reasons = (
        _unsupported_mapping_field_reasons("validation", validation, VALIDATION_FIELDS)
        if require_validation
        else []
    )
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
                    require_closed_fields=require_validation,
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
                    require_status_fields=summarized_validation,
                    require_closed_fields=require_validation,
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
                require_closed_fields=require_validation,
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
                        root_path=root_path,
                        declared_targets=declared_targets,
                        declared_variants=declared_variants,
                        declared_artifact_identities=declared_artifact_identities,
                        declared_artifacts_by_identity=declared_artifacts_by_identity,
                        require_closed_fields=require_validation,
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
    require_closed_fields: bool = False,
) -> list[str]:
    prefix = f"externalCorpus.entries[{index}]"
    if not isinstance(entry, Mapping):
        return [f"{prefix} must be an object"]

    reasons = (
        _unsupported_mapping_field_reasons(
            prefix, entry, REPORT_EXTERNAL_CORPUS_ENTRY_FIELDS
        )
        if require_closed_fields
        else []
    )
    for field_name in ("id", "path", "sourceBackend"):
        if not _is_non_empty_string(entry.get(field_name)):
            reasons.append(f"{prefix}.{field_name} must be a string")
    source_backend = entry.get("sourceBackend")
    if require_closed_fields and _is_non_empty_string(source_backend):
        canonical_source_backend = _resolve_external_corpus_source_backend(
            source_backend
        )
        if (
            canonical_source_backend is not None
            and canonical_source_backend != source_backend
        ):
            reasons.append(
                f"{prefix}.sourceBackend must use canonical source backend name "
                f"{canonical_source_backend}"
            )
    path = entry.get("path")
    path_is_valid = False
    if _is_non_empty_string(path):
        if not _is_report_identity_path(path):
            reasons.append(f"{prefix}.path must be repository-relative")
        else:
            path_is_valid = True
    target_reasons = _string_list_contract_reasons(
        f"{prefix}.targets", entry.get("targets")
    )
    reasons.extend(target_reasons)
    if not target_reasons and _normalized_targets(entry["targets"]) != entry["targets"]:
        reasons.append(
            f"{prefix}.targets must use normalized backend names without duplicates"
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
    commit = entry.get("commit")
    if _is_non_empty_string(commit) and not _is_lowercase_hex_digest(
        commit,
        GIT_COMMIT_HEX_LENGTH,
    ):
        reasons.append(f"{prefix}.commit must be a lowercase 40-character hex digest")
    repository = entry.get("repository")
    source_url = entry.get("sourceUrl")
    if (
        _is_non_empty_string(repository)
        and _is_non_empty_string(source_url)
        and not _source_url_matches_repository(source_url, repository)
    ):
        reasons.append(f"{prefix}.sourceUrl must start with repository")
    return reasons


def _external_corpus_summary_contract_reasons(
    summary: Any,
    entries: Sequence[Any],
    artifacts: Any,
    *,
    require_closed_fields: bool = False,
) -> list[str]:
    if not isinstance(summary, Mapping):
        return ["externalCorpus.summary must be an object"]

    entry_records = [entry for entry in entries if isinstance(entry, Mapping)]
    reasons = (
        _unsupported_mapping_field_reasons(
            "externalCorpus.summary",
            summary,
            REPORT_EXTERNAL_CORPUS_SUMMARY_FIELDS,
        )
        if require_closed_fields
        else []
    )
    accounting_fields = (
        "manifestEntryCount",
        "validEntryCount",
        "invalidEntryCount",
    )
    for field_name in accounting_fields:
        if not _is_non_negative_int(summary.get(field_name)):
            reasons.append(
                f"externalCorpus.summary.{field_name} must be a non-negative integer"
            )
    if all(_is_non_negative_int(summary.get(field)) for field in accounting_fields):
        manifest_entry_count = summary["manifestEntryCount"]
        valid_entry_count = summary["validEntryCount"]
        invalid_entry_count = summary["invalidEntryCount"]
        if valid_entry_count != len(entries):
            reasons.append(
                "externalCorpus.summary.validEntryCount must match "
                "externalCorpus.entries"
            )
        if manifest_entry_count != valid_entry_count + invalid_entry_count:
            reasons.append(
                "externalCorpus.summary.manifestEntryCount must equal "
                "externalCorpus.summary.validEntryCount plus "
                "externalCorpus.summary.invalidEntryCount"
            )
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
    require_closed_fields: bool = False,
    root_path: Path | None = None,
    declared_units_by_path: Mapping[str, UnitDeclaration] | None = None,
) -> list[str]:
    if "externalCorpus" not in report:
        return ["externalCorpus must be an object"] if require_external_corpus else []

    external_corpus = report.get("externalCorpus")
    if not isinstance(external_corpus, Mapping):
        return ["externalCorpus must be an object"]

    reasons = (
        _unsupported_mapping_field_reasons(
            "externalCorpus", external_corpus, REPORT_EXTERNAL_CORPUS_FIELDS
        )
        if require_closed_fields
        else []
    )
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
                    require_closed_fields=require_closed_fields,
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
            external_corpus.get("summary"),
            entries,
            artifacts,
            require_closed_fields=require_closed_fields,
        )
    )
    summary = external_corpus.get("summary")
    if (
        status in {"missing", "invalid", "outside-project"}
        and isinstance(summary, Mapping)
        and any(
            summary.get(field_name) != expected
            for field_name, expected in _external_corpus_empty_summary().items()
        )
    ):
        reasons.append("externalCorpus.summary must be empty when status is not ok")
    return reasons


def _migration_contract_reasons(
    report: Mapping[str, Any], *, require_migration: bool
) -> list[str]:
    if "migration" not in report:
        return ["migration must be an object"] if require_migration else []

    migration = report.get("migration")
    if not isinstance(migration, Mapping):
        return ["migration must be an object"]

    reasons = (
        _unsupported_mapping_field_reasons(
            "migration", migration, REPORT_MIGRATION_FIELDS
        )
        if require_migration
        else []
    )
    project = report.get("project")
    project_targets = project.get("targets", []) if isinstance(project, Mapping) else []
    project_targets_valid = isinstance(project_targets, list) and all(
        _is_non_empty_string(target) for target in project_targets
    )
    declared_targets = (
        set(_normalized_targets(project_targets)) if project_targets_valid else set()
    )
    artifacts = report.get("artifacts")
    translated_artifact_targets = None
    if isinstance(artifacts, list) and artifacts:
        translated_artifact_targets = {
            artifact.get("target")
            for artifact in artifacts
            if (
                isinstance(artifact, Mapping)
                and artifact.get("status") == "translated"
                and _is_non_empty_string(artifact.get("target"))
            )
        }
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
            if require_migration:
                reasons.extend(
                    _unsupported_mapping_field_reasons(
                        prefix, action, REPORT_MIGRATION_ACTION_FIELDS
                    )
                )
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
                if not action_targets:
                    reasons.append(f"{prefix}.targets must not be empty")
                    continue
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
                if translated_artifact_targets is not None:
                    for target in normalized_action_targets:
                        if target not in translated_artifact_targets:
                            reasons.append(
                                f"{prefix}.targets must reference translated "
                                "artifact targets"
                            )
                            break
        rollups = _migration_action_rollups(actions)
        reasons.extend(
            _count_field_contract_reasons(
                "migration.actionCount",
                migration.get("actionCount"),
                rollups["actionCount"],
                "migration.actions",
            )
        )
        for field_name in ("actionsByKind", "actionsBySeverity", "actionsByTarget"):
            reasons.extend(
                _mapping_field_contract_reasons(
                    f"migration.{field_name}",
                    migration.get(field_name),
                    rollups[field_name],
                    "migration.actions",
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

    reasons = _unsupported_mapping_field_reasons(
        "summary", summary, REPORT_SUMMARY_FIELDS
    )
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
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.unitsByExtension",
                summary.get("unitsByExtension"),
                _payload_unit_counts_by_extension(units),
                "units",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.unitsBySourceOverride",
                summary.get("unitsBySourceOverride"),
                _payload_unit_counts_by_source_override(units),
                "units",
            )
        )
        reasons.extend(
            _count_field_contract_reasons(
                "summary.includeDependencyCount",
                summary.get("includeDependencyCount"),
                len(_payload_include_dependency_records(units)),
                "unit include dependencies",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.includeDependenciesByKind",
                summary.get("includeDependenciesByKind"),
                _payload_include_dependency_counts_by_kind(units),
                "unit include dependencies",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.includeDependenciesByStatus",
                summary.get("includeDependenciesByStatus"),
                _payload_include_dependency_counts_by_status(units),
                "unit include dependencies",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.includeDependenciesByResolvedFrom",
                summary.get("includeDependenciesByResolvedFrom"),
                _payload_include_dependency_counts_by_resolved_from(units),
                "unit include dependencies",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.includeDependenciesBySourceBackend",
                summary.get("includeDependenciesBySourceBackend"),
                _payload_include_dependency_counts_by_source_backend(units),
                "unit include dependencies",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.includeDependenciesBySourceBackendStatus",
                summary.get("includeDependenciesBySourceBackendStatus"),
                _payload_include_dependency_counts_by_source_backend_status(units),
                "unit include dependencies",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.includeDependenciesByVariant",
                summary.get("includeDependenciesByVariant"),
                _payload_include_dependency_counts_by_variant(units),
                "unit include dependencies",
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
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.skippedByReason",
                summary.get("skippedByReason"),
                _payload_skipped_counts_by_reason(skipped),
                "skipped",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.skippedByExtension",
                summary.get("skippedByExtension"),
                _payload_skipped_counts_by_extension(skipped),
                "skipped",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.skippedBySourceOverride",
                summary.get("skippedBySourceOverride"),
                _payload_skipped_counts_by_source_override(skipped),
                "skipped",
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
                "summary.artifactsBySourceBackend",
                summary.get("artifactsBySourceBackend"),
                _artifact_counts_by_source_backend(artifact_records),
                "artifacts",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.artifactsByVariant",
                summary.get("artifactsByVariant"),
                _artifact_counts_by_variant(artifact_records),
                "artifacts",
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
        artifact_provenance_rollups = _artifact_provenance_rollups(artifact_records)
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.artifactProvenanceByPipeline",
                summary.get("artifactProvenanceByPipeline"),
                artifact_provenance_rollups["artifactProvenanceByPipeline"],
                "artifact provenance",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.artifactProvenanceByIntermediate",
                summary.get("artifactProvenanceByIntermediate"),
                artifact_provenance_rollups["artifactProvenanceByIntermediate"],
                "artifact provenance",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.artifactProvenanceIntermediateBySourceBackend",
                summary.get("artifactProvenanceIntermediateBySourceBackend"),
                artifact_provenance_rollups[
                    "artifactProvenanceIntermediateBySourceBackend"
                ],
                "artifact provenance",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.artifactProvenanceIntermediateByTarget",
                summary.get("artifactProvenanceIntermediateByTarget"),
                artifact_provenance_rollups["artifactProvenanceIntermediateByTarget"],
                "artifact provenance",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.artifactProvenanceIntermediateByVariant",
                summary.get("artifactProvenanceIntermediateByVariant"),
                artifact_provenance_rollups["artifactProvenanceIntermediateByVariant"],
                "artifact provenance",
            )
        )
        define_processing_rollups = _define_processing_rollups(artifact_records)
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.defineProcessingByStatus",
                summary.get("defineProcessingByStatus"),
                define_processing_rollups["defineProcessingByStatus"],
                "artifact define processing",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.defineProcessingBySourceBackend",
                summary.get("defineProcessingBySourceBackend"),
                define_processing_rollups["defineProcessingBySourceBackend"],
                "artifact define processing",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.defineProcessingByTarget",
                summary.get("defineProcessingByTarget"),
                define_processing_rollups["defineProcessingByTarget"],
                "artifact define processing",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.defineProcessingByVariant",
                summary.get("defineProcessingByVariant"),
                define_processing_rollups["defineProcessingByVariant"],
                "artifact define processing",
            )
        )
        include_path_processing_rollups = _include_path_processing_rollups(
            artifact_records
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.includePathProcessingByStatus",
                summary.get("includePathProcessingByStatus"),
                include_path_processing_rollups["includePathProcessingByStatus"],
                "artifact include path processing",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.includePathProcessingBySourceBackend",
                summary.get("includePathProcessingBySourceBackend"),
                include_path_processing_rollups["includePathProcessingBySourceBackend"],
                "artifact include path processing",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.includePathProcessingByTarget",
                summary.get("includePathProcessingByTarget"),
                include_path_processing_rollups["includePathProcessingByTarget"],
                "artifact include path processing",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.includePathProcessingByVariant",
                summary.get("includePathProcessingByVariant"),
                include_path_processing_rollups["includePathProcessingByVariant"],
                "artifact include path processing",
            )
        )
        source_map_rollups = _source_map_rollups(artifact_records)
        reasons.extend(
            _count_field_contract_reasons(
                "summary.sourceMapCount",
                summary.get("sourceMapCount"),
                source_map_rollups["sourceMapCount"],
                "artifact source maps",
            )
        )
        reasons.extend(
            _count_field_contract_reasons(
                "summary.fineGrainedSourceMapCount",
                summary.get("fineGrainedSourceMapCount"),
                source_map_rollups["fineGrainedSourceMapCount"],
                "artifact source maps",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.sourceMapsByGranularity",
                summary.get("sourceMapsByGranularity"),
                source_map_rollups["sourceMapsByGranularity"],
                "artifact source maps",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.sourceMapsByTarget",
                summary.get("sourceMapsByTarget"),
                source_map_rollups["sourceMapsByTarget"],
                "artifact source maps",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.sourceMapsBySourceBackend",
                summary.get("sourceMapsBySourceBackend"),
                source_map_rollups["sourceMapsBySourceBackend"],
                "artifact source maps",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.sourceMapsByVariant",
                summary.get("sourceMapsByVariant"),
                source_map_rollups["sourceMapsByVariant"],
                "artifact source maps",
            )
        )
        reasons.extend(
            _count_field_contract_reasons(
                "summary.sourceRemapCount",
                summary.get("sourceRemapCount"),
                source_map_rollups["sourceRemapCount"],
                "artifact source remaps",
            )
        )
        reasons.extend(
            _count_field_contract_reasons(
                "summary.sourceRemapMappingCount",
                summary.get("sourceRemapMappingCount"),
                source_map_rollups["sourceRemapMappingCount"],
                "artifact source remap mappings",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.sourceRemapsByGranularity",
                summary.get("sourceRemapsByGranularity"),
                source_map_rollups["sourceRemapsByGranularity"],
                "artifact source remaps",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.sourceRemapsByTarget",
                summary.get("sourceRemapsByTarget"),
                source_map_rollups["sourceRemapsByTarget"],
                "artifact source remaps",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.sourceRemapsBySourceBackend",
                summary.get("sourceRemapsBySourceBackend"),
                source_map_rollups["sourceRemapsBySourceBackend"],
                "artifact source remaps",
            )
        )
        reasons.extend(
            _mapping_field_contract_reasons(
                "summary.sourceRemapsByVariant",
                summary.get("sourceRemapsByVariant"),
                source_map_rollups["sourceRemapsByVariant"],
                "artifact source remaps",
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
        reasons.extend(
            _diagnostic_code_counts_contract_reasons(
                "summary.diagnosticsByCode",
                summary.get("diagnosticsByCode"),
                diagnostics,
            )
        )
        reasons.extend(
            _diagnostic_field_counts_contract_reasons(
                "summary.diagnosticsByTarget",
                summary.get("diagnosticsByTarget"),
                diagnostics,
                "target",
            )
        )
        reasons.extend(
            _diagnostic_field_counts_contract_reasons(
                "summary.diagnosticsBySourceBackend",
                summary.get("diagnosticsBySourceBackend"),
                diagnostics,
                "sourceBackend",
            )
        )
        reasons.extend(
            _diagnostic_field_counts_contract_reasons(
                "summary.diagnosticsByVariant",
                summary.get("diagnosticsByVariant"),
                diagnostics,
                "variant",
            )
        )
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

    reasons = _unsupported_mapping_field_reasons(
        prefix, value, SOURCE_MAP_SPAN_FIELD_SET
    )
    if not _is_non_empty_string(value.get("file")):
        reasons.append(f"{prefix}.file must be a string")
    for field_name in SOURCE_MAP_SPAN_FIELDS[1:]:
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
    is_file_granularity = source_map.get("mappingGranularity") == "file"

    if isinstance(source, Mapping) and _is_non_empty_string(source.get("file")):
        if _is_non_empty_string(artifact.get("source")):
            if source.get("file") != artifact.get("source"):
                reasons.append(
                    f"{prefix}.source.file must match artifacts[{index}].source "
                    f"({_value_mismatch_context(source.get('file'), artifact.get('source'))})"
                )
    if isinstance(generated, Mapping) and _is_non_empty_string(generated.get("file")):
        if _is_non_empty_string(artifact.get("path")):
            if generated.get("file") != artifact.get("path"):
                reasons.append(
                    f"{prefix}.generated.file must match artifacts[{index}].path "
                    f"({_value_mismatch_context(generated.get('file'), artifact.get('path'))})"
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
            if is_file_granularity:
                if dict(mapping_source) != dict(source):
                    reasons.append(
                        f"{mapping_prefix}.source must match {prefix}.source "
                        f"({_value_mismatch_context(dict(mapping_source), dict(source))})"
                    )
            elif (
                _is_non_empty_string(source.get("file"))
                and _is_non_empty_string(mapping_source.get("file"))
                and mapping_source.get("file") != source.get("file")
            ):
                reasons.append(
                    f"{mapping_prefix}.source.file must match {prefix}.source.file "
                    f"({_value_mismatch_context(mapping_source.get('file'), source.get('file'))})"
                )
        mapping_generated = mapping.get("generated")
        if isinstance(generated, Mapping) and isinstance(mapping_generated, Mapping):
            if is_file_granularity:
                if dict(mapping_generated) != dict(generated):
                    reasons.append(
                        f"{mapping_prefix}.generated must match {prefix}.generated "
                        f"({_value_mismatch_context(dict(mapping_generated), dict(generated))})"
                    )
            elif (
                _is_non_empty_string(generated.get("file"))
                and _is_non_empty_string(mapping_generated.get("file"))
                and mapping_generated.get("file") != generated.get("file")
            ):
                reasons.append(
                    f"{mapping_prefix}.generated.file must match "
                    f"{prefix}.generated.file "
                    f"({_value_mismatch_context(mapping_generated.get('file'), generated.get('file'))})"
                )
    return reasons


def _fine_grained_source_map_span_reasons(prefix: str, value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    reasons = []
    if _is_non_negative_int(value.get("length")) and value["length"] <= 0:
        reasons.append(f"{prefix}.length must be greater than zero")
    return reasons


def _source_map_span_within_anchor_reasons(
    prefix: str,
    value: Any,
    anchor: Any,
    anchor_prefix: str,
) -> list[str]:
    if not isinstance(value, Mapping) or not isinstance(anchor, Mapping):
        return []
    offset = value.get("offset")
    end_offset = value.get("endOffset")
    anchor_offset = anchor.get("offset")
    anchor_end_offset = anchor.get("endOffset")
    if not all(
        _is_non_negative_int(candidate)
        for candidate in (offset, end_offset, anchor_offset, anchor_end_offset)
    ):
        return []
    out_of_bounds = offset < anchor_offset or end_offset > anchor_end_offset
    line = value.get("line")
    column = value.get("column")
    end_line = value.get("endLine")
    end_column = value.get("endColumn")
    anchor_line = anchor.get("line")
    anchor_column = anchor.get("column")
    anchor_end_line = anchor.get("endLine")
    anchor_end_column = anchor.get("endColumn")
    if all(
        _is_non_negative_int(candidate)
        for candidate in (
            line,
            column,
            end_line,
            end_column,
            anchor_line,
            anchor_column,
            anchor_end_line,
            anchor_end_column,
        )
    ):
        out_of_bounds = (
            out_of_bounds
            or (line, column) < (anchor_line, anchor_column)
            or (end_line, end_column) > (anchor_end_line, anchor_end_column)
        )
    if out_of_bounds:
        return [f"{prefix} must be within {anchor_prefix}"]
    return []


def _source_map_contract_reasons(
    index: int,
    artifact: Mapping[str, Any],
    *,
    required: bool = False,
    declared_targets: set[str] | None = None,
    require_canonical: bool = False,
) -> list[str]:
    if "sourceMap" not in artifact:
        if required:
            return [f"artifacts[{index}].sourceMap must be an object"]
        return []

    prefix = f"artifacts[{index}].sourceMap"
    source_map = artifact.get("sourceMap")
    if not isinstance(source_map, Mapping):
        return [f"{prefix} must be an object"]

    reasons = _unsupported_mapping_field_reasons(
        prefix, source_map, SOURCE_MAP_PAYLOAD_FIELDS
    )
    if source_map.get("schemaVersion") != 1:
        reasons.append(f"{prefix}.schemaVersion must be 1")
    if source_map.get("kind") != "crosstl-artifact-source-map":
        reasons.append(f"{prefix}.kind must be crosstl-artifact-source-map")
    mapping_granularity = source_map.get("mappingGranularity")
    if mapping_granularity not in SOURCE_MAP_GRANULARITIES:
        reasons.append(
            f"{prefix}.mappingGranularity must be one of "
            f"{', '.join(SOURCE_MAP_GRANULARITIES)}"
        )

    target = source_map.get("target")
    target_reasons = _target_name_contract_reasons(
        f"{prefix}.target",
        target,
        declared_targets=declared_targets,
        require_canonical=require_canonical,
    )
    reasons.extend(target_reasons)
    if (
        _is_non_empty_string(target)
        and _is_non_empty_string(artifact.get("target"))
        and target != artifact["target"]
    ):
        reasons.append(
            f"{prefix}.target must match artifacts[{index}].target "
            f"({_value_mismatch_context(target, artifact['target'])})"
        )

    reasons.extend(
        _source_map_span_reasons(f"{prefix}.source", source_map.get("source"))
    )
    reasons.extend(
        _source_map_span_reasons(f"{prefix}.generated", source_map.get("generated"))
    )
    source_anchor = source_map.get("source")
    generated_anchor = source_map.get("generated")

    mappings = source_map.get("mappings")
    if not isinstance(mappings, list):
        reasons.append(f"{prefix}.mappings must be a list")
    elif not mappings:
        reasons.append(f"{prefix}.mappings must not be empty")
    elif mapping_granularity == "file" and len(mappings) != 1:
        reasons.append(f"{prefix}.mappings must contain one file-level mapping")
    else:
        for mapping_index, mapping in enumerate(mappings):
            mapping_prefix = f"{prefix}.mappings[{mapping_index}]"
            if not isinstance(mapping, Mapping):
                reasons.append(f"{mapping_prefix} must be an object")
                continue
            reasons.extend(
                _unsupported_mapping_field_reasons(
                    mapping_prefix, mapping, SOURCE_MAP_MAPPING_FIELDS
                )
            )
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
            if mapping_granularity != "file":
                reasons.extend(
                    _fine_grained_source_map_span_reasons(
                        f"{mapping_prefix}.source", mapping.get("source")
                    )
                )
                reasons.extend(
                    _source_map_span_within_anchor_reasons(
                        f"{mapping_prefix}.source",
                        mapping.get("source"),
                        source_anchor,
                        f"{prefix}.source",
                    )
                )
                reasons.extend(
                    _fine_grained_source_map_span_reasons(
                        f"{mapping_prefix}.generated", mapping.get("generated")
                    )
                )
                reasons.extend(
                    _source_map_span_within_anchor_reasons(
                        f"{mapping_prefix}.generated",
                        mapping.get("generated"),
                        generated_anchor,
                        f"{prefix}.generated",
                    )
                )

    reasons.extend(_source_map_anchor_reasons(index, source_map, artifact))
    return reasons


def _source_remap_contract_reasons(
    index: int,
    artifact: Mapping[str, Any],
    *,
    required: bool = False,
    require_closed_fields: bool = False,
    declared_targets: set[str] | None = None,
    require_canonical: bool = False,
) -> list[str]:
    if "sourceRemap" not in artifact:
        if required:
            return [f"artifacts[{index}].sourceRemap must be an object"]
        return []

    prefix = f"artifacts[{index}].sourceRemap"
    source_remap = artifact.get("sourceRemap")
    if not isinstance(source_remap, Mapping):
        return [f"{prefix} must be an object"]

    reasons = (
        _unsupported_mapping_field_reasons(
            prefix, source_remap, REPORT_ARTIFACT_SOURCE_REMAP_FIELDS
        )
        if require_closed_fields
        else []
    )
    artifact_target = artifact.get("target")
    if _is_non_empty_string(artifact_target) and not _is_crossgl_target(
        artifact_target
    ):
        reasons.append(
            f"{prefix} must be omitted unless artifacts[{index}].target is CrossGL"
        )
    if source_remap.get("schemaVersion") != SOURCE_REMAP_SCHEMA_VERSION:
        reasons.append(f"{prefix}.schemaVersion must be 1")

    path = source_remap.get("path")
    if not _is_non_empty_string(path):
        reasons.append(f"{prefix}.path must be a string")
    elif not _is_report_identity_path(path):
        reasons.append(f"{prefix}.path must be repository-relative")

    artifact_path = artifact.get("path")
    if _is_non_empty_string(path) and _is_non_empty_string(artifact_path):
        expected_path = _source_remap_report_path(artifact_path)
        if path != expected_path:
            reasons.append(
                f"{prefix}.path must match artifacts[{index}].path "
                f"({_value_mismatch_context(path, expected_path)})"
            )

    target = source_remap.get("target")
    target_reasons = _target_name_contract_reasons(
        f"{prefix}.target",
        target,
        declared_targets=declared_targets,
        require_canonical=require_canonical,
    )
    reasons.extend(target_reasons)
    if (
        _is_non_empty_string(target)
        and _is_non_empty_string(artifact.get("target"))
        and target != artifact["target"]
    ):
        reasons.append(
            f"{prefix}.target must match artifacts[{index}].target "
            f"({_value_mismatch_context(target, artifact['target'])})"
        )

    generated_file = source_remap.get("generatedFile")
    if not _is_non_empty_string(generated_file):
        reasons.append(f"{prefix}.generatedFile must be a string")
    elif _is_non_empty_string(artifact_path) and generated_file != artifact_path:
        reasons.append(
            f"{prefix}.generatedFile must match artifacts[{index}].path "
            f"({_value_mismatch_context(generated_file, artifact_path)})"
        )

    mapping_granularity = source_remap.get("mappingGranularity")
    source_map = artifact.get("sourceMap")
    if mapping_granularity not in SOURCE_MAP_GRANULARITIES:
        reasons.append(
            f"{prefix}.mappingGranularity must be one of "
            f"{', '.join(SOURCE_MAP_GRANULARITIES)}"
        )
    else:
        if isinstance(source_map, Mapping):
            source_map_granularity = source_map.get("mappingGranularity")
            if (
                source_map_granularity in SOURCE_MAP_GRANULARITIES
                and mapping_granularity != source_map_granularity
            ):
                reasons.append(
                    f"{prefix}.mappingGranularity must match "
                    f"artifacts[{index}].sourceMap.mappingGranularity "
                    f"({_value_mismatch_context(mapping_granularity, source_map_granularity)})"
                )
    mapping_count = source_remap.get("mappingCount")
    if mapping_count is None:
        if required:
            reasons.append(f"{prefix}.mappingCount must be a non-negative integer")
    elif not _is_non_negative_int(mapping_count):
        reasons.append(f"{prefix}.mappingCount must be a non-negative integer")
    elif isinstance(source_map, Mapping):
        mappings = source_map.get("mappings")
        if isinstance(mappings, list) and mapping_count != len(mappings):
            reasons.append(
                f"{prefix}.mappingCount must match "
                f"artifacts[{index}].sourceMap.mappings "
                f"({_value_mismatch_context(mapping_count, len(mappings))})"
            )
    size_bytes = source_remap.get("sizeBytes")
    if size_bytes is None:
        if required:
            reasons.append(f"{prefix}.sizeBytes must be a non-negative integer")
    elif not _is_non_negative_int(size_bytes):
        reasons.append(f"{prefix}.sizeBytes must be a non-negative integer")
    reasons.extend(
        _hash_contract_reasons(
            f"{prefix}.hash",
            source_remap.get("hash"),
            require_closed_fields=require_closed_fields,
        )
    )
    return reasons


def _report_contract_diagnostics(path: Path, report: Any) -> list[ProjectDiagnostic]:
    if not isinstance(report, Mapping):
        return [_invalid_report_diagnostic(path, ["expected a JSON object"])]

    reasons = []
    has_summary = "summary" in report
    if has_summary:
        reasons.extend(
            _unsupported_mapping_field_reasons("report", report, REPORT_FIELDS)
        )
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
            output_path = _project_output_path(root_path, output_dir)
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
                reasons.extend(
                    _unit_contract_reasons(
                        index,
                        unit,
                        root_path=root_path,
                        project=project if isinstance(project, Mapping) else None,
                        require_source_hash=has_summary,
                        require_registered_source_backend=has_summary,
                        require_declared_source_override=has_summary,
                        require_closed_fields=has_summary,
                        check_current_source_hash=(
                            has_summary
                            and isinstance(report.get("artifacts", []), list)
                            and not report.get("artifacts", [])
                        ),
                    )
                )
            reasons.extend(_duplicate_path_contract_reasons("units", units))

    skipped = report.get("skipped", [])
    if has_summary and "skipped" not in report:
        reasons.append("skipped must be a list")
    if has_summary or "skipped" in report:
        if not isinstance(skipped, list):
            reasons.append("skipped must be a list")
        else:
            for index, skipped_record in enumerate(skipped):
                reasons.extend(
                    _skipped_contract_reasons(
                        index,
                        skipped_record,
                        project=project if isinstance(project, Mapping) else None,
                        require_declared_source_override=has_summary,
                        require_closed_fields=has_summary,
                    )
                )
            reasons.extend(_duplicate_path_contract_reasons("skipped", skipped))
            if isinstance(units, list):
                reasons.extend(_unit_skipped_path_contract_reasons(units, skipped))

    if has_summary and isinstance(units, list) and isinstance(skipped, list):
        reasons.extend(
            _current_project_scan_contract_reasons(
                project if isinstance(project, Mapping) else None,
                units,
                skipped,
                root_path=root_path,
                report_path=path,
            )
        )

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
        artifact_matrix = report.get("artifactMatrix")
        if has_summary and isinstance(project, Mapping):
            reasons.extend(
                _artifact_matrix_metadata_contract_reasons(
                    project,
                    units,
                    artifacts,
                    artifact_matrix,
                    required=bool(artifacts),
                )
            )
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
            if has_summary:
                reasons.extend(
                    _unsupported_mapping_field_reasons(
                        f"artifacts[{index}]", artifact, REPORT_ARTIFACT_FIELDS
                    )
                )
            for field_name in ("source", "path", "status"):
                if not _is_non_empty_string(artifact.get(field_name)):
                    reasons.append(f"artifacts[{index}].{field_name} must be a string")
            source = artifact.get("source")
            if _is_non_empty_string(source) and not _is_report_identity_path(source):
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
            reasons.extend(
                _target_name_contract_reasons(
                    f"artifacts[{index}].target",
                    target,
                    declared_targets=(
                        declared_targets if project_targets_valid else None
                    ),
                    require_canonical=has_summary,
                )
            )
            if _is_non_empty_string(path) and not _is_report_identity_path(path):
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
                    expected_extension = _artifact_target_extension(target)
                    reasons.append(
                        f"artifacts[{index}].path suffix must match "
                        f"artifacts[{index}].target "
                        f"({_value_mismatch_context(expected_extension, Path(path).suffix.lower())})"
                    )
                elif _is_non_empty_string(source) and _is_report_identity_path(source):
                    expected_relative = Path(source.replace("\\", "/")).with_suffix(
                        _artifact_target_extension(target)
                    )
                    expected_path = (expected_output_base / expected_relative).resolve()
                    if (root_path / path).resolve() != expected_path:
                        reasons.append(
                            f"artifacts[{index}].path must match "
                            "project.outputDir target/variant directory plus "
                            f"artifacts[{index}].source "
                            f"({_value_mismatch_context(_relpath(expected_path, root_path), path)})"
                        )
            source_backend = artifact.get("sourceBackend")
            if has_summary or "sourceBackend" in artifact:
                source_backend_reasons = _source_backend_contract_reasons(
                    f"artifacts[{index}].sourceBackend",
                    source_backend,
                    require_registered=has_summary,
                )
                reasons.extend(source_backend_reasons)
                if (
                    not source_backend_reasons
                    and declared_units_by_path is not None
                    and _is_non_empty_string(source)
                    and source in declared_units_by_path
                ):
                    unit_index, declared_unit = declared_units_by_path[source]
                    if source_backend != declared_unit.get("sourceBackend"):
                        reasons.append(
                            f"artifacts[{index}].sourceBackend must match "
                            f"units[{unit_index}].sourceBackend "
                            f"({_value_mismatch_context(declared_unit.get('sourceBackend'), source_backend)})"
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
                if "generatedSizeBytes" in artifact:
                    reasons.append(
                        f"artifacts[{index}].generatedSizeBytes must be omitted "
                        "for failed artifacts"
                    )
                if "sourceMap" in artifact:
                    reasons.append(
                        f"artifacts[{index}].sourceMap must be omitted "
                        "for failed artifacts"
                    )
                if "sourceRemap" in artifact:
                    reasons.append(
                        f"artifacts[{index}].sourceRemap must be omitted "
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
            elif has_summary and project_variants_valid and project_variants:
                reasons.append(
                    f"artifacts[{index}].variant must be recorded "
                    "when project.variants is non-empty"
                )
            if isinstance(project, Mapping):
                reasons.extend(
                    _artifact_defines_contract_reasons(
                        index,
                        artifact,
                        project,
                        required=has_summary,
                    )
                )
            reasons.extend(
                _artifact_define_processing_contract_reasons(
                    index,
                    artifact,
                    required=has_summary,
                )
            )
            if isinstance(project, Mapping):
                reasons.extend(
                    _artifact_include_path_processing_contract_reasons(
                        index,
                        artifact,
                        project,
                        required=has_summary,
                    )
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
                    require_closed_fields=has_summary,
                )
            )
            reasons.extend(
                _artifact_unit_source_hash_contract_reasons(
                    index,
                    artifact,
                    declared_units_by_path,
                )
            )
            reasons.extend(
                _artifact_source_size_contract_reasons(
                    index,
                    artifact,
                    required=has_summary,
                )
            )
            reasons.extend(
                _artifact_unit_source_size_contract_reasons(
                    index,
                    artifact,
                    declared_units_by_path,
                )
            )
            reasons.extend(
                _generated_hash_contract_reasons(
                    index,
                    artifact,
                    required=has_summary and status == "translated",
                    require_closed_fields=has_summary,
                )
            )
            reasons.extend(
                _generated_size_contract_reasons(
                    index,
                    artifact,
                    required=has_summary and status == "translated",
                )
            )
            reasons.extend(
                _provenance_contract_reasons(
                    index,
                    artifact,
                    required=has_summary,
                    require_closed_fields=has_summary,
                )
            )
            reasons.extend(
                _source_map_contract_reasons(
                    index,
                    artifact,
                    required=has_summary and status == "translated",
                    declared_targets=(
                        declared_targets
                        if has_summary and project_targets_valid
                        else None
                    ),
                    require_canonical=has_summary,
                )
            )
            reasons.extend(
                _source_remap_contract_reasons(
                    index,
                    artifact,
                    required=(
                        has_summary
                        and status == "translated"
                        and _is_non_empty_string(target)
                        and _is_crossgl_target(target)
                    ),
                    require_closed_fields=has_summary,
                    declared_targets=(
                        declared_targets
                        if has_summary and project_targets_valid
                        else None
                    ),
                    require_canonical=has_summary,
                )
            )
        if (
            has_summary
            and isinstance(project, Mapping)
            and (
                bool(artifacts)
                or _artifact_matrix_requires_artifact_records(artifact_matrix)
            )
        ):
            reasons.extend(
                _artifact_matrix_contract_reasons(
                    project,
                    units,
                    artifacts,
                    root_path=root_path,
                    project_output_path=project_output_path,
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
            if has_summary:
                reasons.extend(
                    _unsupported_mapping_field_reasons(
                        f"diagnostics[{index}]",
                        diagnostic,
                        REPORT_DIAGNOSTIC_FIELDS,
                    )
                )
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
                    f"diagnostics[{index}].location",
                    diagnostic.get("location"),
                    require_closed_fields=has_summary,
                )
            )
            if "originalLocation" in diagnostic:
                reasons.extend(
                    _diagnostic_location_contract_reasons(
                        f"diagnostics[{index}].originalLocation",
                        diagnostic.get("originalLocation"),
                        require_closed_fields=has_summary,
                    )
                )
            if "target" in diagnostic:
                reasons.extend(
                    _target_name_contract_reasons(
                        f"diagnostics[{index}].target",
                        diagnostic.get("target"),
                        declared_targets=(
                            declared_targets
                            if has_summary and project_targets_valid
                            else None
                        ),
                        require_canonical=has_summary,
                    )
                )
            if "sourceBackend" in diagnostic:
                reasons.extend(
                    _source_backend_contract_reasons(
                        f"diagnostics[{index}].sourceBackend",
                        diagnostic.get("sourceBackend"),
                        require_registered=has_summary,
                    )
                )
            if "variant" in diagnostic:
                variant = diagnostic.get("variant")
                if not _is_non_empty_string(variant):
                    reasons.append(f"diagnostics[{index}].variant must be a string")
                elif (
                    has_summary
                    and project_variants_valid
                    and project_variants
                    and variant not in declared_variants
                ):
                    reasons.append(
                        f"diagnostics[{index}].variant must be listed in "
                        "project.variants"
                    )
            missing_capabilities = diagnostic.get("missingCapabilities", [])
            if not isinstance(missing_capabilities, list) or any(
                not _is_non_empty_string(capability)
                for capability in missing_capabilities
            ):
                reasons.append(
                    f"diagnostics[{index}].missingCapabilities must be a list of strings"
                )
        if has_summary and isinstance(units, list) and isinstance(project, Mapping):
            reasons.extend(
                _current_include_scan_diagnostic_contract_reasons(
                    units,
                    diagnostics,
                    root_path=root_path,
                    project=project,
                )
            )

    reasons.extend(
        _validation_contract_reasons(
            report,
            root_path=root_path,
            require_validation=has_summary,
        )
    )
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
            require_closed_fields=has_summary,
            root_path=external_corpus_root_path,
            declared_units_by_path=external_corpus_units_by_path,
        )
    )
    if (has_summary or "diagnosticCounts" in report) and isinstance(diagnostics, list):
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
        if not tools:
            continue
        artifact_path = Path(str(artifact["path"]))
        if not artifact_path.is_absolute():
            artifact_path = root / artifact_path
        artifact_path = artifact_path.resolve()
        if not _is_relative_to(artifact_path, root):
            continue
        if not artifact_path.is_file():
            continue
        smoke_command = _toolchain_smoke_command(target, tools, artifact_path)
        if smoke_command is None:
            continue
        command, check_kind = smoke_command
        if not command or not shutil.which(command[0]):
            continue
        try:
            completed = subprocess.run(
                command,
                cwd=str(root),
                input=(
                    artifact_path.read_text(encoding="utf-8", errors="replace")
                    if "--stdin" in command
                    else None
                ),
                capture_output=True,
                text=True,
                check=False,
                timeout=TOOLCHAIN_SMOKE_TIMEOUT_SECONDS,
            )
            returncode = completed.returncode
            stdout = _toolchain_output_text(completed.stdout)
            stderr = _toolchain_output_text(completed.stderr)
        except subprocess.TimeoutExpired as exc:
            returncode = TOOLCHAIN_TIMEOUT_RETURNCODE
            stdout = _toolchain_output_text(getattr(exc, "stdout", None))
            stderr = _toolchain_output_text(getattr(exc, "stderr", None))
            timeout_message = (
                "Validation toolchain timed out after "
                f"{TOOLCHAIN_SMOKE_TIMEOUT_SECONDS} seconds."
            )
            stderr = f"{stderr.rstrip()}\n{timeout_message}".strip()
        run = {
            "source": str(artifact.get("source", "")),
            "target": target,
            "path": str(artifact["path"]),
            "command": command,
            "checkKind": check_kind,
            "returncode": returncode,
            "status": "ok" if returncode == 0 else "failed",
            "stdout": stdout[-4000:],
            "stderr": stderr[-4000:],
        }
        if _is_non_empty_string(artifact.get("sourceBackend")):
            run["sourceBackend"] = artifact["sourceBackend"]
        if artifact.get("variant") is not None:
            run["variant"] = artifact["variant"]
        runs.append(run)
    return runs


def _toolchain_smoke_command(
    target: str, tools: Sequence[str], artifact_path: Path
) -> tuple[list[str], str] | None:
    if target == "opengl":
        return [tools[0], "--stdin", "-S", _glslang_stage(artifact_path)], "artifact"
    if target == "vulkan":
        if artifact_path.suffix.lower() == ".spvasm" and len(tools) > 1:
            return [tools[1], str(artifact_path), "-o", os.devnull], "artifact"
        return [tools[0], str(artifact_path)], "artifact"
    availability_command = TOOLCHAIN_AVAILABILITY_COMMANDS.get(target)
    if availability_command is None:
        return None
    return list(availability_command), "tool-availability"


def _glslang_stage(artifact_path: Path) -> str:
    suffix_stage = {
        ".vert": "vert",
        ".vs": "vert",
        ".frag": "frag",
        ".fs": "frag",
        ".geom": "geom",
        ".tesc": "tesc",
        ".tese": "tese",
        ".comp": "comp",
        ".cs": "comp",
    }.get(artifact_path.suffix.lower())
    if suffix_stage:
        return suffix_stage

    try:
        source = artifact_path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return "comp"

    if "local_size_" in source or "gl_globalinvocationid" in source:
        return "comp"
    if "gl_position" in source:
        return "vert"
    if "gl_fragcoord" in source or "gl_fragcolor" in source:
        return "frag"
    return "comp"
