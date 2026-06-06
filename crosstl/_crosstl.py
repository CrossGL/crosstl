"""High-level translation API and command-line entry point for CrossGL Translator."""

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .translator.codegen import (
    backend_names,
    get_backend_extension,
    get_codegen,
    normalize_backend_name,
)
from .translator.plugin_loader import discover_backend_plugins
from .translator.source_registry import (
    BINARY_SPIRV_UNSUPPORTED_MESSAGE,
    SOURCE_REGISTRY,
    register_default_sources,
)

try:
    from .formatter import format_shader_code

    FORMATTER_AVAILABLE = True
except ImportError:
    FORMATTER_AVAILABLE = False


SPIRV_BINARY_MAGIC_PREFIXES = (b"\x03\x02\x23\x07", b"\x07\x23\x02\x03")


def _read_shader_source(file_path: str, source_name: str) -> str:
    with open(file_path, "rb") as file:
        shader_bytes = file.read()

    if source_name == "vulkan" and shader_bytes.startswith(SPIRV_BINARY_MAGIC_PREFIXES):
        raise ValueError(BINARY_SPIRV_UNSUPPORTED_MESSAGE)

    return shader_bytes.decode("utf-8", errors="replace")


def translate(
    file_path: str,
    backend: str = "cgl",
    save_shader: Optional[str] = None,
    format_output: bool = True,
    source_backend: Optional[str] = None,
    *,
    include_paths: Optional[Sequence[str]] = None,
    defines: Optional[Mapping[str, str]] = None,
) -> str:
    """Translate a shader file to another language.

    Args:
        file_path (str): The path to the shader file
        backend (str, optional): The target language to translate to. Defaults to "cgl".
        save_shader (str, optional): The path to save the translated shader. Defaults to None.
        format_output (bool, optional): Whether to format the generated code. Defaults to True.
        source_backend (str, optional): Override the source parser instead of inferring it
            from the file extension. Defaults to None.
        include_paths (Sequence[str], optional): Source parser include search paths.
            Defaults to None.
        defines (Mapping[str, str], optional): Source parser preprocessor defines.
            Defaults to None.

    Returns:
        str: The translated shader code
    """
    file_path = os.fspath(file_path)
    if not isinstance(file_path, str):
        raise TypeError(
            "Shader file path must be a string or path-like object returning str, "
            f"got {type(file_path)}"
        )

    register_default_sources()
    discover_backend_plugins()
    backend = (backend or "cgl").strip().lower()

    source_spec = (
        SOURCE_REGISTRY.get(source_backend)
        if source_backend
        else SOURCE_REGISTRY.get_by_extension(file_path)
    )
    if not source_spec:
        if source_backend:
            supported = ", ".join(SOURCE_REGISTRY.names())
            raise ValueError(
                f"Unsupported source backend: {source_backend}. Supported: {supported}"
            )
        supported = ", ".join(SOURCE_REGISTRY.extensions())
        raise ValueError(
            f"Unsupported shader file type: {file_path}. Supported: {supported}"
        )

    shader_code = _read_shader_source(file_path, source_spec.name)

    ast = source_spec.parse(
        shader_code,
        file_path=file_path,
        include_paths=include_paths,
        defines=defines,
    )

    requested_backend = backend
    normalized_backend = normalize_backend_name(requested_backend) or requested_backend

    if source_spec.name == "cgl":
        if normalized_backend in ["cgl", "crossgl"]:
            generated_code = shader_code
        else:
            codegen = get_codegen(normalized_backend)
            generated_code = codegen.generate(ast)
    else:
        if normalized_backend in ["cgl", "crossgl"]:
            if not source_spec.reverse_codegen_factory:
                raise ValueError(f"Reverse translation not supported for: {file_path}")
            codegen = source_spec.reverse_codegen_factory()
            generated_code = codegen.generate(ast)
        else:
            if not source_spec.reverse_codegen_factory:
                raise ValueError(
                    f"Unsupported translation scenario: {file_path} to {backend}"
                )
            # Translate to CrossGL first, then to target backend
            reverse_codegen = source_spec.reverse_codegen_factory()
            intermediate_code = reverse_codegen.generate(ast)
            cgl_spec = SOURCE_REGISTRY.get("cgl")
            if not cgl_spec:
                raise ValueError("CrossGL parser not available for intermediate step")
            cgl_ast = cgl_spec.parse(intermediate_code)
            codegen = get_codegen(normalized_backend)
            generated_code = codegen.generate(cgl_ast)

    if (
        format_output
        and FORMATTER_AVAILABLE
        and normalized_backend not in ["cgl", "crossgl"]
    ):
        generated_code = format_shader_code(
            generated_code, normalized_backend, save_shader
        )

    if save_shader is not None:
        with open(save_shader, "w", encoding="utf-8") as file:
            file.write(generated_code)

    return generated_code


def _derive_single_file_output(input_path, backend):
    base, _ = os.path.splitext(input_path)
    normalized_backend = normalize_backend_name(backend) or backend
    if normalized_backend in ["cgl", "crossgl"]:
        ext = ".cgl"
    else:
        ext = get_backend_extension(normalized_backend) or ".out"
    return base + ext


def _run_single_file(args):
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        return 1

    output_path = args.output or _derive_single_file_output(args.input, args.backend)
    defines = _parse_project_define_overrides(getattr(args, "define", None))
    translate(
        args.input,
        backend=args.backend,
        save_shader=output_path,
        format_output=not args.no_format,
        source_backend=getattr(args, "source_backend", None),
        include_paths=getattr(args, "include_dir", None),
        defines=defines or None,
    )
    print(f"Successfully translated to {output_path}")
    return 0


def _legacy_parser():
    parser = argparse.ArgumentParser(description="CrossGL Shader Translator")

    parser.add_argument("input", help="Input shader file path")
    supported_backends = ", ".join(backend_names() + ["cgl"])
    parser.add_argument(
        "--backend",
        "-b",
        default="cgl",
        help=f"Target backend ({supported_backends})",
    )
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument(
        "--no-format", action="store_true", help="Disable code formatting"
    )
    parser.add_argument("--source-backend", help="Override source parser backend")
    parser.add_argument(
        "--include-dir",
        action="append",
        help="Source parser include directory; repeatable",
    )
    parser.add_argument(
        "--define",
        action="append",
        help="Source parser preprocessor define as NAME or NAME=VALUE; repeatable",
    )
    parser.set_defaults(func=_run_single_file)
    return parser


def _write_json_payload(payload, output_path=None):
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"Wrote {path}")
    else:
        print(text, end="")


def _parse_project_define_overrides(values):
    defines = {}
    for value in values or []:
        name, separator, define_value = value.partition("=")
        name = name.strip()
        if not name:
            raise ValueError("--define entries must use NAME or NAME=VALUE")
        defines[name] = define_value.strip() if separator else "1"
    return defines


def _parse_project_source_overrides(values):
    overrides = {}
    for value in values or []:
        pattern, separator, backend = value.partition("=")
        pattern = pattern.strip()
        backend = backend.strip()
        if not pattern or not separator or not backend:
            raise ValueError("--source-override entries must use PATTERN=BACKEND")
        overrides[pattern] = backend
    return overrides


def _load_project_config_from_args(args):
    from .project import load_project_config

    config = load_project_config(args.root, args.config)
    include_dirs = tuple(config.include_dirs) + tuple(
        getattr(args, "include_dir", None) or ()
    )
    define_overrides = _parse_project_define_overrides(getattr(args, "define", None))
    source_overrides = _parse_project_source_overrides(
        getattr(args, "source_override", None)
    )
    if (
        include_dirs == tuple(config.include_dirs)
        and not define_overrides
        and not source_overrides
    ):
        return config
    return replace(
        config,
        include_dirs=include_dirs,
        defines={**dict(config.defines), **define_overrides},
        source_overrides={**dict(config.source_overrides), **source_overrides},
    )


def _add_project_override_args(parser):
    parser.add_argument(
        "--include-dir",
        action="append",
        help="Project include directory override; repeatable",
    )
    parser.add_argument(
        "--define",
        action="append",
        help="Project preprocessor define override as NAME or NAME=VALUE; repeatable",
    )
    parser.add_argument(
        "--source-override",
        action="append",
        help="Project source backend override as PATTERN=BACKEND; repeatable",
    )


def _add_project_scan_args(parser):
    parser.add_argument("root", help="Repository root to scan")
    parser.add_argument("--config", help="Path to crosstl.toml")
    _add_project_override_args(parser)


def _run_project_scan(args):
    from .project import scan_project

    config = _load_project_config_from_args(args)
    report = scan_project(config).to_report(targets=args.target)
    payload = report.to_json()
    _write_json_payload(payload, args.output)
    return 1 if payload["summary"]["diagnosticCounts"]["error"] else 0


def _run_translate_project(args):
    from .project import translate_project

    config = _load_project_config_from_args(args)
    report = translate_project(
        config,
        targets=args.target,
        output_dir=args.output_dir,
        variants=args.variant,
        format_output=not args.no_format,
        validate=args.validate,
    )
    payload = report.to_json()
    if args.report:
        report.write_json(Path(args.report))
        print(f"Wrote {args.report}")
    else:
        _write_json_payload(payload)
    summary = payload["summary"]
    return 1 if summary["failedCount"] or summary["diagnosticCounts"]["error"] else 0


def _run_validate_project(args):
    from .project import validate_project_report

    payload = validate_project_report(args.report, run_toolchains=args.run_toolchains)
    if args.format == "sarif":
        _write_json_payload(_format_project_diagnostics_sarif(payload), args.output)
    elif args.format == "text":
        text = _format_project_validation_report(payload)
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"Wrote {path}")
        else:
            print(text, end="")
    else:
        _write_json_payload(payload, args.output)
    return 0 if payload["success"] else 1


def _format_count_rollup(label, counts, *, include_zero=True):
    if not isinstance(counts, Mapping):
        return None

    entries = []
    for name, count in counts.items():
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            continue
        if not include_zero and count == 0:
            continue
        entries.append((name, count))
    if not entries:
        return None

    entries.sort(key=lambda item: (-item[1], item[0]))
    return f"{label}: " + ", ".join(f"{name}={count}" for name, count in entries)


def _format_nested_count_rollup(label, counts, *, include_zero=True):
    if not isinstance(counts, Mapping):
        return None

    entries = []
    for name, row in counts.items():
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(row, Mapping):
            continue
        row_entries = []
        for status, count in row.items():
            if not isinstance(status, str) or not status.strip():
                continue
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                continue
            if not include_zero and count == 0:
                continue
            row_entries.append((status, count))
        if not row_entries:
            continue
        row_entries.sort(key=lambda item: (-item[1], item[0]))
        entries.append((name, sum(count for _, count in row_entries), row_entries))
    if not entries:
        return None

    entries.sort(key=lambda item: (-item[1], item[0]))
    return f"{label}: " + ", ".join(
        (
            f"{name}=("
            + ", ".join(f"{status}={count}" for status, count in row_entries)
            + ")"
        )
        for name, _, row_entries in entries
    )


def _format_artifact_rollup(label, counts):
    if not isinstance(counts, Mapping):
        return None

    entries = []
    for target, row in counts.items():
        if not isinstance(target, str) or not target.strip():
            continue
        if not isinstance(row, Mapping):
            continue
        artifact_count = row.get("artifactCount")
        translated_count = row.get("translatedCount")
        failed_count = row.get("failedCount")
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (artifact_count, translated_count, failed_count)
        ):
            continue
        entries.append((target, artifact_count, translated_count, failed_count))
    if not entries:
        return None

    entries.sort(key=lambda item: (-item[1], item[0]))
    return f"{label}: " + ", ".join(
        (
            f"{target}={artifact_count} "
            f"{'artifact' if artifact_count == 1 else 'artifacts'} "
            f"({translated_count} translated, {failed_count} failed)"
        )
        for target, artifact_count, translated_count, failed_count in entries
    )


def _format_validation_artifact_rollup(label, counts):
    if not isinstance(counts, Mapping):
        return None

    entries = []
    for target, row in counts.items():
        if not isinstance(target, str) or not target.strip():
            continue
        if not isinstance(row, Mapping):
            continue
        artifact_count = row.get("artifactCount")
        ok_count = row.get("okCount")
        failed_count = row.get("failedCount")
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (artifact_count, ok_count, failed_count)
        ):
            continue
        entries.append((target, artifact_count, ok_count, failed_count))
    if not entries:
        return None

    entries.sort(key=lambda item: (-item[1], item[0]))
    return f"{label}: " + ", ".join(
        (
            f"{target}={artifact_count} "
            f"{'artifact' if artifact_count == 1 else 'artifacts'} "
            f"({ok_count} ok, {failed_count} failed)"
        )
        for target, artifact_count, ok_count, failed_count in entries
    )


def _format_validation_run_rollup(label, counts):
    if not isinstance(counts, Mapping):
        return None

    entries = []
    for name, row in counts.items():
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(row, Mapping):
            continue
        run_count = row.get("runCount")
        ok_count = row.get("okCount")
        failed_count = row.get("failedCount")
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (run_count, ok_count, failed_count)
        ):
            continue
        entries.append((name, run_count, ok_count, failed_count))
    if not entries:
        return None

    entries.sort(key=lambda item: (-item[1], item[0]))
    return f"{label}: " + ", ".join(
        (
            f"{name}={run_count} "
            f"{'run' if run_count == 1 else 'runs'} "
            f"({ok_count} ok, {failed_count} failed)"
        )
        for name, run_count, ok_count, failed_count in entries
    )


def _format_project_validation_report(payload):
    counts = payload.get("diagnosticCounts", {})
    counts = counts if isinstance(counts, Mapping) else {}
    lines = [
        f"Project validation report: {payload.get('sourceReport')}",
        f"Status: {'ok' if payload.get('success') else 'failed'}",
        (
            "Diagnostics: "
            f"{counts.get('error', 0)} errors, "
            f"{counts.get('warning', 0)} warnings, "
            f"{counts.get('note', 0)} notes"
        ),
    ]

    for line in (
        _format_count_rollup(
            "Diagnostic codes", payload.get("diagnosticsByCode"), include_zero=False
        ),
        _format_count_rollup(
            "Missing capabilities",
            payload.get("missingCapabilityCounts"),
            include_zero=False,
        ),
        _format_validation_artifact_rollup(
            "Validation artifacts by target",
            payload.get("artifactStatusByTarget"),
        ),
        _format_validation_artifact_rollup(
            "Validation artifacts by variant",
            payload.get("artifactStatusByVariant"),
        ),
        _format_count_rollup(
            "Validation source hashes",
            payload.get("sourceHashStatusCounts"),
            include_zero=False,
        ),
        _format_count_rollup(
            "Validation generated hashes",
            payload.get("generatedHashStatusCounts"),
            include_zero=False,
        ),
        _format_count_rollup(
            "Validation source maps",
            payload.get("sourceMapStatusCounts"),
            include_zero=False,
        ),
        _format_count_rollup(
            "Validation source remaps",
            payload.get("sourceRemapStatusCounts"),
            include_zero=False,
        ),
        _format_count_rollup(
            "Validation toolchains",
            payload.get("toolchainStatusCounts"),
            include_zero=False,
        ),
        _format_count_rollup(
            "Validation toolchain runs",
            payload.get("toolchainRunStatusCounts"),
            include_zero=False,
        ),
        _format_validation_run_rollup(
            "Validation toolchain runs by target",
            payload.get("toolchainRunStatusByTarget"),
        ),
        _format_validation_run_rollup(
            "Validation toolchain runs by source backend",
            payload.get("toolchainRunStatusBySourceBackend"),
        ),
        _format_validation_run_rollup(
            "Validation toolchain runs by variant",
            payload.get("toolchainRunStatusByVariant"),
        ),
    ):
        if line:
            lines.append(line)

    diagnostics = payload.get("diagnostics", [])
    if isinstance(diagnostics, list) and diagnostics:
        lines.append("Validation diagnostics:")
        for diagnostic in diagnostics:
            if not isinstance(diagnostic, Mapping):
                continue
            lines.append(
                "- "
                f"{diagnostic.get('severity', 'unknown')} "
                f"{diagnostic.get('code', 'unknown')}: "
                f"{diagnostic.get('message', '')}"
            )

    return "\n".join(lines) + "\n"


def _sarif_level(severity):
    return {
        "error": "error",
        "warning": "warning",
        "note": "note",
    }.get(severity, "none")


def _sarif_region(location):
    if not isinstance(location, Mapping):
        return {}

    region = {}
    for source_name, sarif_name in (
        ("line", "startLine"),
        ("column", "startColumn"),
        ("endLine", "endLine"),
        ("endColumn", "endColumn"),
    ):
        value = location.get(source_name)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            region[sarif_name] = value
    return region


def _sarif_location(diagnostic):
    location = diagnostic.get("location")
    if not isinstance(location, Mapping):
        return None

    file_name = location.get("file")
    if not isinstance(file_name, str) or not file_name:
        return None

    physical_location = {"artifactLocation": {"uri": file_name}}
    region = _sarif_region(location)
    if region:
        physical_location["region"] = region
    return {"physicalLocation": physical_location}


def _format_project_diagnostics_sarif(
    payload, *, tool_name="CrossTL project validation"
):
    diagnostics = payload.get("diagnostics", [])
    if not isinstance(diagnostics, list):
        diagnostics = []

    rules = {}
    results = []
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, Mapping):
            continue

        code = diagnostic.get("code")
        rule_id = code if isinstance(code, str) and code else "crosstl.project"
        message = diagnostic.get("message")
        message_text = message if isinstance(message, str) else ""
        rules.setdefault(rule_id, {"id": rule_id, "name": rule_id})

        result = {
            "ruleId": rule_id,
            "level": _sarif_level(diagnostic.get("severity")),
            "message": {"text": message_text},
        }
        location = _sarif_location(diagnostic)
        if location:
            result["locations"] = [location]

        properties = {}
        target = diagnostic.get("target")
        if isinstance(target, str) and target:
            properties["target"] = target
        missing_capabilities = diagnostic.get("missingCapabilities")
        if isinstance(missing_capabilities, list) and missing_capabilities:
            properties["missingCapabilities"] = list(missing_capabilities)
        if properties:
            result["properties"] = properties
        results.append(result)

    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": tool_name,
                        "informationUri": "https://github.com/CrossGL/crosstl",
                        "rules": [rules[key] for key in sorted(rules)],
                    }
                },
                "invocations": [
                    {
                        "executionSuccessful": bool(payload.get("success")),
                        "properties": {
                            "sourceReport": payload.get("sourceReport"),
                        },
                    }
                ],
                "results": results,
            }
        ],
    }


def _format_project_config_counts(project):
    if not isinstance(project, Mapping):
        return None

    entries = []
    for label, field_name in (
        ("sourceRoots", "sourceRootCount"),
        ("includePatterns", "includePatternCount"),
        ("excludePatterns", "excludePatternCount"),
        ("sourceOverrides", "sourceOverrideCount"),
        ("includeDirs", "includeDirCount"),
        ("defines", "defineCount"),
        ("variants", "variantCount"),
    ):
        count = project.get(field_name)
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            entries.append(f"{label}={count}")
    if not entries:
        return None
    return "Project config: " + ", ".join(entries)


def _format_project_variant_names(project):
    if not isinstance(project, Mapping):
        return None

    variant_names = project.get("variantNames")
    if not isinstance(variant_names, list):
        return None

    names = [name for name in variant_names if isinstance(name, str) and name]
    if not names:
        return None
    return "Project variants: " + ", ".join(names)


def _format_project_variant_define_counts(project):
    if not isinstance(project, Mapping):
        return None
    return _format_count_rollup(
        "Project variant define counts",
        project.get("variantDefineCounts"),
    )


def _format_project_config_path(project):
    if not isinstance(project, Mapping) or "config" not in project:
        return None

    config_path = project.get("config")
    if config_path is None:
        return "Config file: (none)"
    if isinstance(config_path, str):
        return f"Config file: {config_path or '(empty)'}"
    return None


def _format_inactive_status_records(label, records):
    if not isinstance(records, list):
        return None

    entries = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        path = record.get("path")
        status = record.get("status")
        if (
            not isinstance(path, str)
            or not isinstance(status, str)
            or status == "active"
        ):
            continue
        entries.append(f"{path or '(empty)'} ({status})")

    if not entries:
        return None
    return f"{label}: " + ", ".join(entries)


def _format_source_map_counts(summary):
    if not isinstance(summary, Mapping):
        return None

    source_map_count = summary.get("sourceMapCount")
    fine_grained_count = summary.get("fineGrainedSourceMapCount")
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in (source_map_count, fine_grained_count)
    ):
        return None
    return (
        "Source maps: "
        f"{source_map_count} file-level, "
        f"{fine_grained_count} fine-grained"
    )


def _format_source_remap_counts(summary):
    if not isinstance(summary, Mapping):
        return None

    source_remap_count = summary.get("sourceRemapCount")
    if (
        not isinstance(source_remap_count, int)
        or isinstance(source_remap_count, bool)
        or source_remap_count < 0
    ):
        return None
    return f"Source remaps: {source_remap_count}"


def _format_source_map_artifact_line(artifact):
    if not isinstance(artifact, Mapping):
        return None

    source = artifact.get("sourceFile") or artifact.get("source")
    generated = artifact.get("generatedFile") or artifact.get("path")
    if not isinstance(source, str) or not source:
        return None
    if not isinstance(generated, str) or not generated:
        return None

    details = []
    source_backend = artifact.get("sourceBackend")
    if isinstance(source_backend, str) and source_backend:
        details.append(f"sourceBackend={source_backend}")
    target = artifact.get("target")
    if isinstance(target, str) and target:
        details.append(f"target={target}")
    variant = artifact.get("variant")
    if isinstance(variant, str) and variant:
        details.append(f"variant={variant}")
    granularity = artifact.get("mappingGranularity")
    if isinstance(granularity, str) and granularity:
        details.append(f"granularity={granularity}")
    mapping_count = artifact.get("mappingCount")
    if (
        isinstance(mapping_count, int)
        and not isinstance(mapping_count, bool)
        and mapping_count >= 0
    ):
        details.append(f"mappings={mapping_count}")

    suffix = f" ({', '.join(details)})" if details else ""
    return f"- {source} -> {generated}{suffix}"


def _format_source_map_artifact_lines(source_maps):
    if not isinstance(source_maps, Mapping):
        return []
    artifacts = source_maps.get("sourceMapArtifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return []

    lines = ["Source map artifacts:"]
    for artifact in artifacts:
        line = _format_source_map_artifact_line(artifact)
        if line:
            lines.append(line)
    if len(lines) == 1:
        return []

    truncated_count = source_maps.get("truncatedSourceMapArtifactCount")
    if (
        isinstance(truncated_count, int)
        and not isinstance(truncated_count, bool)
        and truncated_count > 0
    ):
        lines.append(f"- +{truncated_count} more")
    return lines


def _format_source_remap_artifact_line(artifact):
    if not isinstance(artifact, Mapping):
        return None

    remap_path = artifact.get("sourceRemapPath")
    generated = artifact.get("generatedFile") or artifact.get("path")
    if not isinstance(remap_path, str) or not remap_path:
        return None
    if not isinstance(generated, str) or not generated:
        return None

    details = []
    source_backend = artifact.get("sourceBackend")
    if isinstance(source_backend, str) and source_backend:
        details.append(f"sourceBackend={source_backend}")
    target = artifact.get("target")
    if isinstance(target, str) and target:
        details.append(f"target={target}")
    variant = artifact.get("variant")
    if isinstance(variant, str) and variant:
        details.append(f"variant={variant}")
    granularity = artifact.get("mappingGranularity")
    if isinstance(granularity, str) and granularity:
        details.append(f"granularity={granularity}")

    suffix = f" ({', '.join(details)})" if details else ""
    return f"- {remap_path} -> {generated}{suffix}"


def _format_source_remap_artifact_lines(source_maps):
    if not isinstance(source_maps, Mapping):
        return []
    artifacts = source_maps.get("sourceRemapArtifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return []

    lines = ["Source remap artifacts:"]
    for artifact in artifacts:
        line = _format_source_remap_artifact_line(artifact)
        if line:
            lines.append(line)
    if len(lines) == 1:
        return []

    truncated_count = source_maps.get("truncatedSourceRemapArtifactCount")
    if (
        isinstance(truncated_count, int)
        and not isinstance(truncated_count, bool)
        and truncated_count > 0
    ):
        lines.append(f"- +{truncated_count} more")
    return lines


def _format_artifact_provenance_line(artifact):
    if not isinstance(artifact, Mapping):
        return None

    source = artifact.get("source")
    path = artifact.get("path")
    if not isinstance(source, str) or not source:
        return None
    if not isinstance(path, str) or not path:
        return None

    details = []
    source_backend = artifact.get("sourceBackend")
    if isinstance(source_backend, str) and source_backend:
        details.append(f"sourceBackend={source_backend}")
    target = artifact.get("target")
    if isinstance(target, str) and target:
        details.append(f"target={target}")
    variant = artifact.get("variant")
    if isinstance(variant, str) and variant:
        details.append(f"variant={variant}")
    pipeline = artifact.get("pipeline")
    if isinstance(pipeline, str) and pipeline:
        details.append(f"pipeline={pipeline}")
    intermediate = artifact.get("intermediate")
    if isinstance(intermediate, str) and intermediate:
        details.append(f"intermediate={intermediate}")

    suffix = f" ({', '.join(details)})" if details else ""
    return f"- {source} -> {path}{suffix}"


def _format_artifact_provenance_lines(artifact_provenance):
    if not isinstance(artifact_provenance, Mapping):
        return []
    artifacts = artifact_provenance.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return []

    lines = ["Artifact provenance samples:"]
    for artifact in artifacts:
        line = _format_artifact_provenance_line(artifact)
        if line:
            lines.append(line)
    if len(lines) == 1:
        return []

    truncated_count = artifact_provenance.get("truncatedArtifactCount")
    if (
        isinstance(truncated_count, int)
        and not isinstance(truncated_count, bool)
        and truncated_count > 0
    ):
        lines.append(f"- +{truncated_count} more")
    return lines


def _format_artifact_matrix_summary(artifact_matrix):
    if not isinstance(artifact_matrix, Mapping):
        return None
    if not artifact_matrix.get("available"):
        return None

    expected_count = artifact_matrix.get("expectedArtifactCount")
    emitted_count = artifact_matrix.get("emittedArtifactCount")
    translated_count = artifact_matrix.get("translatedCount")
    failed_count = artifact_matrix.get("failedCount")
    missing_count = artifact_matrix.get("missingArtifactCount")
    extra_count = artifact_matrix.get("extraArtifactCount")
    variant_mode = artifact_matrix.get("variantMode")
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in (
            expected_count,
            emitted_count,
            translated_count,
            failed_count,
            missing_count,
            extra_count,
        )
    ):
        return None
    if variant_mode not in {"none", "named"}:
        return None

    return (
        "Artifact matrix: "
        f"{emitted_count} emitted of {expected_count} expected "
        f"({translated_count} translated, {failed_count} failed, "
        f"{missing_count} missing, {extra_count} extra; "
        f"variants={variant_mode})"
    )


def _format_artifact_matrix_rollup(label, counts):
    if not isinstance(counts, Mapping):
        return None

    entries = []
    for name, row in counts.items():
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(row, Mapping):
            continue
        expected_count = row.get("expectedArtifactCount")
        emitted_count = row.get("emittedArtifactCount")
        translated_count = row.get("translatedCount")
        failed_count = row.get("failedCount")
        missing_count = row.get("missingArtifactCount")
        extra_count = row.get("extraArtifactCount")
        complete = row.get("complete")
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (
                expected_count,
                emitted_count,
                translated_count,
                failed_count,
                missing_count,
                extra_count,
            )
        ):
            continue
        if not isinstance(complete, bool):
            continue
        entries.append(
            (
                name,
                expected_count,
                emitted_count,
                translated_count,
                failed_count,
                missing_count,
                extra_count,
                complete,
            )
        )
    if not entries:
        return None

    entries.sort(key=lambda item: (-item[1], item[0]))
    return f"{label}: " + ", ".join(
        (
            f"{name}={emitted_count}/{expected_count} emitted "
            f"({translated_count} translated, {failed_count} failed, "
            f"{missing_count} missing, {extra_count} extra, "
            f"{'complete' if complete else 'incomplete'})"
        )
        for (
            name,
            expected_count,
            emitted_count,
            translated_count,
            failed_count,
            missing_count,
            extra_count,
            complete,
        ) in entries
    )


def _format_artifact_identity_line(artifact):
    if not isinstance(artifact, Mapping):
        return None

    source = artifact.get("source")
    target = artifact.get("target")
    path = artifact.get("path")
    if not all(isinstance(value, str) and value for value in (source, target, path)):
        return None

    variant = artifact.get("variant")
    variant_label = (
        f"(variant: {variant}) " if isinstance(variant, str) and variant else ""
    )
    return f"- {source} -> {target} {variant_label}at {path}"


def _format_artifact_sample_lines(label, artifacts, truncated_count):
    if not isinstance(artifacts, list) or not artifacts:
        return []

    lines = [f"{label}:"]
    for artifact in artifacts:
        line = _format_artifact_identity_line(artifact)
        if line:
            lines.append(line)
    if len(lines) == 1:
        return []

    if (
        isinstance(truncated_count, int)
        and not isinstance(truncated_count, bool)
        and truncated_count > 0
    ):
        lines.append(f"- +{truncated_count} more")
    return lines


def _format_validation_artifact_sample_line(artifact):
    line = _format_artifact_identity_line(artifact)
    if not line:
        return None

    details = []
    status = artifact.get("status")
    if isinstance(status, str) and status:
        details.append(f"status={status}")
    exists = artifact.get("exists")
    if isinstance(exists, bool):
        details.append(f"exists={'true' if exists else 'false'}")
    source_hash = artifact.get("sourceHashStatus")
    if isinstance(source_hash, str) and source_hash:
        details.append(f"sourceHash={source_hash}")
    generated_hash = artifact.get("generatedHashStatus")
    if isinstance(generated_hash, str) and generated_hash:
        details.append(f"generatedHash={generated_hash}")
    source_map = artifact.get("sourceMapStatus")
    if isinstance(source_map, str) and source_map:
        details.append(f"sourceMap={source_map}")
    source_remap = artifact.get("sourceRemapStatus")
    if isinstance(source_remap, str) and source_remap:
        details.append(f"sourceRemap={source_remap}")

    suffix = f" ({', '.join(details)})" if details else ""
    return f"{line}{suffix}"


def _format_validation_artifact_sample_lines(validation):
    if not isinstance(validation, Mapping):
        return []
    artifacts = validation.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return []

    lines = ["Validation artifact samples:"]
    for artifact in artifacts:
        line = _format_validation_artifact_sample_line(artifact)
        if line:
            lines.append(line)
    if len(lines) == 1:
        return []

    truncated_count = validation.get("truncatedArtifactCount")
    if (
        isinstance(truncated_count, int)
        and not isinstance(truncated_count, bool)
        and truncated_count > 0
    ):
        lines.append(f"- +{truncated_count} more")
    return lines


def _format_validation_toolchain_run_sample_line(run):
    line = _format_artifact_identity_line(run)
    if not line:
        return None

    details = []
    source_backend = run.get("sourceBackend")
    if isinstance(source_backend, str) and source_backend:
        details.append(f"sourceBackend={source_backend}")
    status = run.get("status")
    if isinstance(status, str) and status:
        details.append(f"status={status}")
    returncode = run.get("returncode")
    if isinstance(returncode, int) and not isinstance(returncode, bool):
        details.append(f"returncode={returncode}")
    command = run.get("command")
    if isinstance(command, list):
        command_parts = [part for part in command if isinstance(part, str) and part]
        if command_parts:
            details.append(f"command={' '.join(command_parts)}")
    for field_name, label in (("stdoutLength", "stdout"), ("stderrLength", "stderr")):
        length = run.get(field_name)
        if isinstance(length, int) and not isinstance(length, bool) and length >= 0:
            unit = "char" if length == 1 else "chars"
            details.append(f"{label}={length} {unit}")

    suffix = f" ({', '.join(details)})" if details else ""
    return f"{line}{suffix}"


def _format_validation_toolchain_run_sample_lines(validation):
    if not isinstance(validation, Mapping):
        return []
    runs = validation.get("toolchainRuns")
    if not isinstance(runs, list) or not runs:
        return []

    lines = ["Validation toolchain run samples:"]
    for run in runs:
        line = _format_validation_toolchain_run_sample_line(run)
        if line:
            lines.append(line)
    if len(lines) == 1:
        return []

    truncated_count = validation.get("truncatedToolchainRunCount")
    if (
        isinstance(truncated_count, int)
        and not isinstance(truncated_count, bool)
        and truncated_count > 0
    ):
        lines.append(f"- +{truncated_count} more")
    return lines


def _format_define_processing_issue_line(artifact):
    line = _format_artifact_identity_line(artifact)
    if not line:
        return None

    define_count = artifact.get("defineCount")
    if (
        not isinstance(define_count, int)
        or isinstance(define_count, bool)
        or define_count <= 0
    ):
        return None

    source_backend = artifact.get("sourceBackend")
    source_backend_label = (
        f" by {source_backend}"
        if isinstance(source_backend, str) and source_backend
        else ""
    )
    frontend = artifact.get("frontend")
    frontend_label = (
        f" {frontend} frontend" if isinstance(frontend, str) and frontend else ""
    )
    define_label = "define" if define_count == 1 else "defines"
    return (
        f"{line}: {define_count} {define_label} not forwarded"
        f"{source_backend_label}{frontend_label}"
    )


def _format_define_processing_issue_lines(define_processing):
    if not isinstance(define_processing, Mapping):
        return []
    artifacts = define_processing.get("notSupportedArtifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return []

    lines = ["Define processing issues:"]
    for artifact in artifacts:
        line = _format_define_processing_issue_line(artifact)
        if line:
            lines.append(line)
    if len(lines) == 1:
        return []

    truncated_count = define_processing.get("truncatedNotSupportedArtifactCount")
    if (
        isinstance(truncated_count, int)
        and not isinstance(truncated_count, bool)
        and truncated_count > 0
    ):
        lines.append(f"- +{truncated_count} more")
    return lines


def _format_define_processing_artifact_line(artifact):
    if not isinstance(artifact, Mapping):
        return None

    source = artifact.get("source")
    path = artifact.get("path")
    if not isinstance(source, str) or not source:
        return None
    if not isinstance(path, str) or not path:
        return None

    details = []
    source_backend = artifact.get("sourceBackend")
    if isinstance(source_backend, str) and source_backend:
        details.append(f"sourceBackend={source_backend}")
    target = artifact.get("target")
    if isinstance(target, str) and target:
        details.append(f"target={target}")
    variant = artifact.get("variant")
    if isinstance(variant, str) and variant:
        details.append(f"variant={variant}")
    status = artifact.get("status")
    if isinstance(status, str) and status:
        details.append(f"status={status}")
    frontend = artifact.get("frontend")
    if isinstance(frontend, str) and frontend:
        details.append(f"frontend={frontend}")
    supports_defines = artifact.get("supportsDefines")
    if isinstance(supports_defines, bool):
        details.append(f"supportsDefines={str(supports_defines).lower()}")
    define_count = artifact.get("defineCount")
    if (
        isinstance(define_count, int)
        and not isinstance(define_count, bool)
        and define_count >= 0
    ):
        details.append(f"defines={define_count}")

    suffix = f" ({', '.join(details)})" if details else ""
    return f"- {source} -> {path}{suffix}"


def _format_define_processing_artifact_lines(define_processing):
    if not isinstance(define_processing, Mapping):
        return []
    artifacts = define_processing.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return []

    lines = ["Define processing artifacts:"]
    for artifact in artifacts:
        line = _format_define_processing_artifact_line(artifact)
        if line:
            lines.append(line)
    if len(lines) == 1:
        return []

    truncated_count = define_processing.get("truncatedArtifactCount")
    if (
        isinstance(truncated_count, int)
        and not isinstance(truncated_count, bool)
        and truncated_count > 0
    ):
        lines.append(f"- +{truncated_count} more")
    return lines


def _format_include_path_processing_issue_line(artifact):
    line = _format_artifact_identity_line(artifact)
    if not line:
        return None

    include_path_count = artifact.get("includePathCount")
    if (
        not isinstance(include_path_count, int)
        or isinstance(include_path_count, bool)
        or include_path_count <= 0
    ):
        return None

    source_backend = artifact.get("sourceBackend")
    source_backend_label = (
        f" by {source_backend}"
        if isinstance(source_backend, str) and source_backend
        else ""
    )
    frontend = artifact.get("frontend")
    frontend_label = (
        f" {frontend} frontend" if isinstance(frontend, str) and frontend else ""
    )
    include_path_label = "include path" if include_path_count == 1 else "include paths"
    return (
        f"{line}: {include_path_count} {include_path_label} not forwarded"
        f"{source_backend_label}{frontend_label}"
    )


def _format_include_path_processing_issue_lines(include_path_processing):
    if not isinstance(include_path_processing, Mapping):
        return []
    artifacts = include_path_processing.get("notSupportedArtifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return []

    lines = ["Include path processing issues:"]
    for artifact in artifacts:
        line = _format_include_path_processing_issue_line(artifact)
        if line:
            lines.append(line)
    if len(lines) == 1:
        return []

    truncated_count = include_path_processing.get("truncatedNotSupportedArtifactCount")
    if (
        isinstance(truncated_count, int)
        and not isinstance(truncated_count, bool)
        and truncated_count > 0
    ):
        lines.append(f"- +{truncated_count} more")
    return lines


def _format_include_path_processing_artifact_line(artifact):
    if not isinstance(artifact, Mapping):
        return None

    source = artifact.get("source")
    path = artifact.get("path")
    if not isinstance(source, str) or not source:
        return None
    if not isinstance(path, str) or not path:
        return None

    details = []
    source_backend = artifact.get("sourceBackend")
    if isinstance(source_backend, str) and source_backend:
        details.append(f"sourceBackend={source_backend}")
    target = artifact.get("target")
    if isinstance(target, str) and target:
        details.append(f"target={target}")
    variant = artifact.get("variant")
    if isinstance(variant, str) and variant:
        details.append(f"variant={variant}")
    status = artifact.get("status")
    if isinstance(status, str) and status:
        details.append(f"status={status}")
    frontend = artifact.get("frontend")
    if isinstance(frontend, str) and frontend:
        details.append(f"frontend={frontend}")
    supports_include_paths = artifact.get("supportsIncludePaths")
    if isinstance(supports_include_paths, bool):
        details.append(f"supportsIncludePaths={str(supports_include_paths).lower()}")
    include_path_count = artifact.get("includePathCount")
    if (
        isinstance(include_path_count, int)
        and not isinstance(include_path_count, bool)
        and include_path_count >= 0
    ):
        details.append(f"includePaths={include_path_count}")

    suffix = f" ({', '.join(details)})" if details else ""
    return f"- {source} -> {path}{suffix}"


def _format_include_path_processing_artifact_lines(include_path_processing):
    if not isinstance(include_path_processing, Mapping):
        return []
    artifacts = include_path_processing.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return []

    lines = ["Include path processing artifacts:"]
    for artifact in artifacts:
        line = _format_include_path_processing_artifact_line(artifact)
        if line:
            lines.append(line)
    if len(lines) == 1:
        return []

    truncated_count = include_path_processing.get("truncatedArtifactCount")
    if (
        isinstance(truncated_count, int)
        and not isinstance(truncated_count, bool)
        and truncated_count > 0
    ):
        lines.append(f"- +{truncated_count} more")
    return lines


def _format_include_dependency_issue_line(dependency):
    if not isinstance(dependency, Mapping):
        return None

    include = dependency.get("include")
    status = dependency.get("status")
    if not isinstance(include, str) or not include:
        return None
    if not isinstance(status, str) or not status:
        return None

    source = dependency.get("source")
    location = source if isinstance(source, str) and source else "(unknown)"
    line = dependency.get("line")
    column = dependency.get("column")
    if isinstance(line, int) and not isinstance(line, bool) and line > 0:
        location += f":{line}"
        if isinstance(column, int) and not isinstance(column, bool) and column > 0:
            location += f":{column}"

    kind = dependency.get("kind")
    kind_label = f" {kind}" if isinstance(kind, str) and kind else ""
    return f"- {location}: {status}{kind_label} include {include}"


def _format_resolved_include_dependency_line(dependency):
    if not isinstance(dependency, Mapping):
        return None

    include = dependency.get("include")
    resolved_path = dependency.get("resolvedPath")
    if not isinstance(include, str) or not include:
        return None
    if not isinstance(resolved_path, str) or not resolved_path:
        return None

    source = dependency.get("source")
    location = source if isinstance(source, str) and source else "(unknown)"
    line = dependency.get("line")
    column = dependency.get("column")
    if isinstance(line, int) and not isinstance(line, bool) and line > 0:
        location += f":{line}"
        if isinstance(column, int) and not isinstance(column, bool) and column > 0:
            location += f":{column}"

    kind = dependency.get("kind")
    kind_label = f" {kind}" if isinstance(kind, str) and kind else ""
    resolved_from = dependency.get("resolvedFrom")
    source_label = (
        f" ({resolved_from})"
        if isinstance(resolved_from, str) and resolved_from
        else ""
    )
    return f"- {location}: resolved{kind_label} include {include} -> {resolved_path}{source_label}"


def _format_resolved_include_dependency_lines(include_dependencies):
    if not isinstance(include_dependencies, Mapping):
        return []
    dependencies = include_dependencies.get("resolvedDependencies")
    if not isinstance(dependencies, list) or not dependencies:
        return []

    lines = ["Resolved include dependencies:"]
    for dependency in dependencies:
        line = _format_resolved_include_dependency_line(dependency)
        if line:
            lines.append(line)
    if len(lines) == 1:
        return []

    truncated_count = include_dependencies.get("truncatedResolvedDependencyCount")
    if (
        isinstance(truncated_count, int)
        and not isinstance(truncated_count, bool)
        and truncated_count > 0
    ):
        lines.append(f"- +{truncated_count} more")
    return lines


def _format_include_dependency_issue_lines(include_dependencies):
    if not isinstance(include_dependencies, Mapping):
        return []
    dependencies = include_dependencies.get("unresolvedDependencies")
    if not isinstance(dependencies, list) or not dependencies:
        return []

    lines = ["Include dependency issues:"]
    for dependency in dependencies:
        line = _format_include_dependency_issue_line(dependency)
        if line:
            lines.append(line)
    if len(lines) == 1:
        return []

    truncated_count = include_dependencies.get("truncatedUnresolvedDependencyCount")
    if (
        isinstance(truncated_count, int)
        and not isinstance(truncated_count, bool)
        and truncated_count > 0
    ):
        lines.append(f"- +{truncated_count} more")
    return lines


def _format_external_corpus_accounting(summary):
    if not isinstance(summary, Mapping):
        return None

    values = {
        field_name: summary.get(field_name)
        for field_name in (
            "manifestEntryCount",
            "validEntryCount",
            "discoveredUnitCount",
            "undiscoveredPresentCount",
        )
    }
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in values.values()
    ):
        return None
    return (
        "External corpus coverage: "
        f"{values['discoveredUnitCount']} discovered, "
        f"{values['undiscoveredPresentCount']} present but undiscovered; "
        f"{values['manifestEntryCount']} manifest entries, "
        f"{values['validEntryCount']} valid"
    )


def _format_external_corpus_entry(entry):
    if not isinstance(entry, Mapping):
        return None

    path = entry.get("path")
    if not isinstance(path, str) or not path:
        path = "(unknown)"
    details = []
    entry_id = entry.get("id")
    if isinstance(entry_id, str) and entry_id:
        details.append(entry_id)
    source_backend = entry.get("sourceBackend")
    if isinstance(source_backend, str) and source_backend:
        details.append(source_backend)
    targets = entry.get("targets")
    if isinstance(targets, list):
        target_names = [target for target in targets if isinstance(target, str)]
        if target_names:
            details.append("targets=" + ",".join(target_names))
    if not details:
        return path
    return f"{path} ({'; '.join(details)})"


def _format_external_corpus_entry_samples(label, entries, truncated_count):
    if not isinstance(entries, list) or not entries:
        return None

    formatted_entries = [
        formatted_entry
        for formatted_entry in (
            _format_external_corpus_entry(entry) for entry in entries
        )
        if formatted_entry
    ]
    if not formatted_entries:
        return None

    suffix = ""
    if (
        isinstance(truncated_count, int)
        and not isinstance(truncated_count, bool)
        and truncated_count > 0
    ):
        suffix = f"; +{truncated_count} more"
    return f"{label}: " + ", ".join(formatted_entries) + suffix


def _format_report_status(report, validation_diagnostic_codes):
    if isinstance(report, Mapping) and report.get("valid") is False:
        return "Report: invalid" if report.get("available") else "Report: unavailable"
    if isinstance(validation_diagnostic_codes, Mapping) and (
        validation_diagnostic_codes.get("project.validate.invalid-report", 0)
    ):
        return "Report: invalid"
    if isinstance(report, Mapping) and report.get("available") is False:
        return "Report: unavailable"
    return None


def _format_project_report_inspection(payload):
    report = payload.get("report", {})
    summary = report.get("summary", {}) if isinstance(report, Mapping) else {}
    project = report.get("project", {}) if isinstance(report, Mapping) else {}
    diagnostic_counts = summary.get("diagnosticCounts", {})
    validation_counts = payload.get("validation", {}).get("diagnosticCounts", {})
    validation_diagnostic_codes = payload.get("validation", {}).get("diagnosticsByCode")
    validation_missing_capabilities = payload.get("validation", {}).get(
        "missingCapabilityCounts"
    )
    validation_toolchain_counts = payload.get("validation", {}).get(
        "toolchainStatusCounts"
    )
    validation_toolchain_run_counts = payload.get("validation", {}).get(
        "toolchainRunStatusCounts"
    )
    validation_toolchain_run_status_by_target = payload.get("validation", {}).get(
        "toolchainRunStatusByTarget"
    )
    validation_toolchain_run_status_by_source_backend = payload.get(
        "validation", {}
    ).get("toolchainRunStatusBySourceBackend")
    validation_toolchain_run_status_by_variant = payload.get("validation", {}).get(
        "toolchainRunStatusByVariant"
    )
    validation_artifact_status_by_target = payload.get("validation", {}).get(
        "artifactStatusByTarget"
    )
    validation_artifact_status_by_variant = payload.get("validation", {}).get(
        "artifactStatusByVariant"
    )
    validation_result = payload.get("validation", {}).get("result", {})
    validation_summary = (
        validation_result.get("summary")
        if isinstance(validation_result, Mapping)
        else {}
    )
    targets = project.get("targets", [])
    target_names = (
        [str(target) for target in targets] if isinstance(targets, list) else []
    )
    lines = [
        f"Project report: {payload.get('sourceReport')}",
        f"Status: {'ok' if payload.get('success') else 'failed'}",
        "Targets: " + (", ".join(target_names) if target_names else "(none)"),
        (
            "Units: "
            f"{summary.get('unitCount', 0)}; artifacts: "
            f"{summary.get('artifactCount', 0)} "
            f"({summary.get('translatedCount', 0)} translated, "
            f"{summary.get('failedCount', 0)} failed)"
        ),
        (
            "Diagnostics: "
            f"{diagnostic_counts.get('error', 0)} errors, "
            f"{diagnostic_counts.get('warning', 0)} warnings, "
            f"{diagnostic_counts.get('note', 0)} notes"
        ),
    ]
    report_status = _format_report_status(report, validation_diagnostic_codes)
    if report_status:
        lines.insert(2, report_status)
    project_insert_index = 3
    project_config_counts = _format_project_config_counts(project)
    if project_config_counts:
        lines.insert(project_insert_index, project_config_counts)
        project_insert_index += 1
    project_variant_names = _format_project_variant_names(project)
    if project_variant_names:
        lines.insert(project_insert_index, project_variant_names)
        project_insert_index += 1
    project_variant_define_counts = _format_project_variant_define_counts(project)
    if project_variant_define_counts:
        lines.insert(project_insert_index, project_variant_define_counts)
    project_status_lines = []
    source_root_status = _format_count_rollup(
        "Source roots by status",
        project.get("sourceRootStatusCounts"),
        include_zero=False,
    )
    if source_root_status:
        project_status_lines.append(source_root_status)
    source_root_issues = _format_inactive_status_records(
        "Source root issues",
        project.get("sourceRootStatus"),
    )
    if source_root_issues:
        project_status_lines.append(source_root_issues)
    include_dir_status = _format_count_rollup(
        "Include dirs by status",
        project.get("includeDirStatusCounts"),
        include_zero=False,
    )
    if include_dir_status:
        project_status_lines.append(include_dir_status)
    include_dir_issues = _format_inactive_status_records(
        "Include dir issues",
        project.get("includeDirStatus"),
    )
    if include_dir_issues:
        project_status_lines.append(include_dir_issues)
    for offset, line in enumerate(project_status_lines):
        lines.insert(4 + offset, line)
    project_config_path = _format_project_config_path(project)
    if project_config_path:
        lines.insert(1, project_config_path)
    source_maps = _format_source_map_counts(summary)
    if source_maps:
        lines.append(source_maps)
    source_remaps = _format_source_remap_counts(summary)
    if source_remaps:
        lines.append(source_remaps)
    define_processing = _format_count_rollup(
        "Define processing",
        summary.get("defineProcessingByStatus"),
        include_zero=False,
    )
    if define_processing:
        lines.append(define_processing)
    define_processing_by_source_backend = _format_nested_count_rollup(
        "Define processing by source backend",
        summary.get("defineProcessingBySourceBackend"),
        include_zero=False,
    )
    if define_processing_by_source_backend:
        lines.append(define_processing_by_source_backend)
    define_processing_by_variant = _format_nested_count_rollup(
        "Define processing by variant",
        summary.get("defineProcessingByVariant"),
        include_zero=False,
    )
    if define_processing_by_variant:
        lines.append(define_processing_by_variant)
    lines.extend(
        _format_define_processing_artifact_lines(payload.get("defineProcessing"))
    )
    lines.extend(_format_define_processing_issue_lines(payload.get("defineProcessing")))
    include_path_processing = _format_count_rollup(
        "Include path processing",
        summary.get("includePathProcessingByStatus"),
        include_zero=False,
    )
    if include_path_processing:
        lines.append(include_path_processing)
    include_path_processing_by_source_backend = _format_nested_count_rollup(
        "Include path processing by source backend",
        summary.get("includePathProcessingBySourceBackend"),
        include_zero=False,
    )
    if include_path_processing_by_source_backend:
        lines.append(include_path_processing_by_source_backend)
    include_path_processing_by_variant = _format_nested_count_rollup(
        "Include path processing by variant",
        summary.get("includePathProcessingByVariant"),
        include_zero=False,
    )
    if include_path_processing_by_variant:
        lines.append(include_path_processing_by_variant)
    lines.extend(
        _format_include_path_processing_artifact_lines(
            payload.get("includePathProcessing")
        )
    )
    lines.extend(
        _format_include_path_processing_issue_lines(
            payload.get("includePathProcessing")
        )
    )
    artifact_matrix_payload = payload.get("artifactMatrix")
    artifact_matrix = _format_artifact_matrix_summary(artifact_matrix_payload)
    if artifact_matrix:
        lines.append(artifact_matrix)
    if isinstance(artifact_matrix_payload, Mapping):
        artifact_matrix_by_target = _format_artifact_matrix_rollup(
            "Artifact matrix by target",
            artifact_matrix_payload.get("statusByTarget"),
        )
        if artifact_matrix_by_target:
            lines.append(artifact_matrix_by_target)
        artifact_matrix_by_variant = _format_artifact_matrix_rollup(
            "Artifact matrix by variant",
            artifact_matrix_payload.get("statusByVariant"),
        )
        if artifact_matrix_by_variant:
            lines.append(artifact_matrix_by_variant)
        lines.extend(
            _format_artifact_sample_lines(
                "Artifact matrix missing artifacts",
                artifact_matrix_payload.get("missingArtifacts"),
                artifact_matrix_payload.get("truncatedMissingArtifactCount"),
            )
        )
        lines.extend(
            _format_artifact_sample_lines(
                "Artifact matrix extra artifacts",
                artifact_matrix_payload.get("extraArtifacts"),
                artifact_matrix_payload.get("truncatedExtraArtifactCount"),
            )
        )
    for line in (
        _format_count_rollup(
            "Source maps by granularity",
            summary.get("sourceMapsByGranularity"),
            include_zero=False,
        ),
        _format_count_rollup(
            "Source maps by target",
            summary.get("sourceMapsByTarget"),
            include_zero=False,
        ),
        _format_count_rollup(
            "Source maps by source backend",
            summary.get("sourceMapsBySourceBackend"),
            include_zero=False,
        ),
        _format_count_rollup(
            "Source maps by variant",
            summary.get("sourceMapsByVariant"),
            include_zero=False,
        ),
        _format_count_rollup(
            "Source remaps by target",
            summary.get("sourceRemapsByTarget"),
            include_zero=False,
        ),
        _format_count_rollup(
            "Source remaps by source backend",
            summary.get("sourceRemapsBySourceBackend"),
            include_zero=False,
        ),
        _format_count_rollup(
            "Source remaps by variant",
            summary.get("sourceRemapsByVariant"),
            include_zero=False,
        ),
        _format_count_rollup(
            "Artifact provenance by pipeline",
            summary.get("artifactProvenanceByPipeline"),
            include_zero=False,
        ),
        _format_count_rollup(
            "Artifact provenance by intermediate",
            summary.get("artifactProvenanceByIntermediate"),
            include_zero=False,
        ),
        _format_nested_count_rollup(
            "Artifact provenance by source backend and intermediate",
            summary.get("artifactProvenanceIntermediateBySourceBackend"),
            include_zero=False,
        ),
    ):
        if line:
            lines.append(line)
    source_map_payload = payload.get("sourceMaps")
    lines.extend(_format_source_map_artifact_lines(source_map_payload))
    lines.extend(_format_source_remap_artifact_lines(source_map_payload))
    lines.extend(_format_artifact_provenance_lines(payload.get("artifactProvenance")))
    units_by_source_backend = _format_count_rollup(
        "Units by source backend",
        summary.get("unitsBySourceBackend"),
        include_zero=False,
    )
    if units_by_source_backend:
        lines.append(units_by_source_backend)
    units_by_extension = _format_count_rollup(
        "Units by extension",
        summary.get("unitsByExtension"),
        include_zero=False,
    )
    if units_by_extension:
        lines.append(units_by_extension)
    units_by_source_override = _format_count_rollup(
        "Units by source override",
        summary.get("unitsBySourceOverride"),
        include_zero=False,
    )
    if units_by_source_override:
        lines.append(units_by_source_override)
    include_dependencies_by_status = _format_count_rollup(
        "Include dependencies by status",
        summary.get("includeDependenciesByStatus"),
        include_zero=False,
    )
    if include_dependencies_by_status:
        lines.append(include_dependencies_by_status)
    include_dependencies_by_kind = _format_count_rollup(
        "Include dependencies by kind",
        summary.get("includeDependenciesByKind"),
        include_zero=False,
    )
    if include_dependencies_by_kind:
        lines.append(include_dependencies_by_kind)
    include_dependencies_by_source_backend = _format_count_rollup(
        "Include dependencies by source backend",
        summary.get("includeDependenciesBySourceBackend"),
        include_zero=False,
    )
    if include_dependencies_by_source_backend:
        lines.append(include_dependencies_by_source_backend)
    include_dependencies_by_source_backend_status = _format_nested_count_rollup(
        "Include dependencies by source backend status",
        summary.get("includeDependenciesBySourceBackendStatus"),
        include_zero=False,
    )
    if include_dependencies_by_source_backend_status:
        lines.append(include_dependencies_by_source_backend_status)
    include_dependencies_by_resolved_from = _format_count_rollup(
        "Include dependencies by resolution source",
        summary.get("includeDependenciesByResolvedFrom"),
        include_zero=False,
    )
    if include_dependencies_by_resolved_from:
        lines.append(include_dependencies_by_resolved_from)
    lines.extend(
        _format_resolved_include_dependency_lines(payload.get("includeDependencies"))
    )
    lines.extend(
        _format_include_dependency_issue_lines(payload.get("includeDependencies"))
    )
    skipped_by_reason = _format_count_rollup(
        "Skipped by reason",
        summary.get("skippedByReason"),
        include_zero=False,
    )
    if skipped_by_reason:
        lines.append(skipped_by_reason)
    skipped_by_extension = _format_count_rollup(
        "Skipped by extension",
        summary.get("skippedByExtension"),
        include_zero=False,
    )
    if skipped_by_extension:
        lines.append(skipped_by_extension)
    skipped_by_source_override = _format_count_rollup(
        "Skipped by source override",
        summary.get("skippedBySourceOverride"),
        include_zero=False,
    )
    if skipped_by_source_override:
        lines.append(skipped_by_source_override)
    artifacts_by_source_backend = _format_artifact_rollup(
        "Artifacts by source backend",
        summary.get("artifactsBySourceBackend"),
    )
    if artifacts_by_source_backend:
        lines.append(artifacts_by_source_backend)
    artifacts_by_variant = _format_artifact_rollup(
        "Artifacts by variant",
        summary.get("artifactsByVariant"),
    )
    if artifacts_by_variant:
        lines.append(artifacts_by_variant)
    artifacts_by_target = _format_artifact_rollup(
        "Artifacts by target",
        summary.get("artifactsByTarget"),
    )
    if artifacts_by_target:
        lines.append(artifacts_by_target)
    diagnostic_codes = _format_count_rollup(
        "Diagnostic codes", summary.get("diagnosticsByCode")
    )
    if diagnostic_codes:
        lines.append(diagnostic_codes)
    missing_capabilities = _format_count_rollup(
        "Missing capabilities", summary.get("missingCapabilityCounts")
    )
    if missing_capabilities:
        lines.append(missing_capabilities)
    lines.append(
        "Validation diagnostics: "
        f"{validation_counts.get('error', 0)} errors, "
        f"{validation_counts.get('warning', 0)} warnings, "
        f"{validation_counts.get('note', 0)} notes"
    )
    validation_codes = _format_count_rollup(
        "Validation diagnostic codes",
        validation_diagnostic_codes,
        include_zero=False,
    )
    if validation_codes:
        lines.append(validation_codes)
    validation_missing = _format_count_rollup(
        "Validation missing capabilities",
        validation_missing_capabilities,
        include_zero=False,
    )
    if validation_missing:
        lines.append(validation_missing)
    validation_toolchains = _format_count_rollup(
        "Validation toolchains",
        validation_toolchain_counts,
        include_zero=False,
    )
    if validation_toolchains:
        lines.append(validation_toolchains)
    validation_toolchain_runs = _format_count_rollup(
        "Validation toolchain runs",
        validation_toolchain_run_counts,
        include_zero=False,
    )
    if validation_toolchain_runs:
        lines.append(validation_toolchain_runs)
    validation_toolchain_runs_by_target = _format_validation_run_rollup(
        "Validation toolchain runs by target",
        validation_toolchain_run_status_by_target,
    )
    if validation_toolchain_runs_by_target:
        lines.append(validation_toolchain_runs_by_target)
    validation_toolchain_runs_by_source_backend = _format_validation_run_rollup(
        "Validation toolchain runs by source backend",
        validation_toolchain_run_status_by_source_backend,
    )
    if validation_toolchain_runs_by_source_backend:
        lines.append(validation_toolchain_runs_by_source_backend)
    validation_toolchain_runs_by_variant = _format_validation_run_rollup(
        "Validation toolchain runs by variant",
        validation_toolchain_run_status_by_variant,
    )
    if validation_toolchain_runs_by_variant:
        lines.append(validation_toolchain_runs_by_variant)
    lines.extend(
        _format_validation_toolchain_run_sample_lines(payload.get("validation"))
    )
    if isinstance(validation_summary, Mapping):
        lines.append(
            "Validation artifacts: "
            f"{validation_summary.get('okCount', 0)} ok, "
            f"{validation_summary.get('failedCount', 0)} failed"
        )
        validation_artifacts_by_target = _format_validation_artifact_rollup(
            "Validation artifacts by target",
            validation_artifact_status_by_target,
        )
        if validation_artifacts_by_target:
            lines.append(validation_artifacts_by_target)
        validation_artifacts_by_variant = _format_validation_artifact_rollup(
            "Validation artifacts by variant",
            validation_artifact_status_by_variant,
        )
        if validation_artifacts_by_variant:
            lines.append(validation_artifacts_by_variant)
        source_hashes = _format_count_rollup(
            "Validation source hashes",
            validation_summary.get("sourceHashStatusCounts"),
            include_zero=False,
        )
        if source_hashes:
            lines.append(source_hashes)
        generated_hashes = _format_count_rollup(
            "Validation generated hashes",
            validation_summary.get("generatedHashStatusCounts"),
            include_zero=False,
        )
        if generated_hashes:
            lines.append(generated_hashes)
        source_maps = _format_count_rollup(
            "Validation source maps",
            validation_summary.get("sourceMapStatusCounts"),
            include_zero=False,
        )
        if source_maps:
            lines.append(source_maps)
        source_remaps = _format_count_rollup(
            "Validation source remaps",
            validation_summary.get("sourceRemapStatusCounts"),
            include_zero=False,
        )
        if source_remaps:
            lines.append(source_remaps)
        lines.extend(
            _format_validation_artifact_sample_lines(payload.get("validation"))
        )

    external_corpus = payload.get("externalCorpus")
    if isinstance(external_corpus, Mapping):
        external_summary = external_corpus.get("summary", {})
        if isinstance(external_summary, Mapping):
            invalid_count = external_summary.get("invalidEntryCount", 0)
            invalid_entries = ""
            if (
                isinstance(invalid_count, int)
                and not isinstance(invalid_count, bool)
                and invalid_count > 0
            ):
                invalid_entries = f", {invalid_count} invalid"
            lines.append(
                "External corpus: "
                f"{external_corpus.get('status', 'unknown')}; "
                f"{external_summary.get('entryCount', 0)} entries, "
                f"{external_summary.get('presentCount', 0)} present, "
                f"{external_summary.get('missingCount', 0)} missing"
                f"{invalid_entries}"
            )
            corpus_accounting = _format_external_corpus_accounting(external_summary)
            if corpus_accounting:
                lines.append(corpus_accounting)
            missing_entries = _format_external_corpus_entry_samples(
                "External corpus missing entries",
                external_corpus.get("missingEntries"),
                external_corpus.get("truncatedMissingEntryCount"),
            )
            if missing_entries:
                lines.append(missing_entries)
            undiscovered_entries = _format_external_corpus_entry_samples(
                "External corpus undiscovered present entries",
                external_corpus.get("undiscoveredPresentEntries"),
                external_corpus.get("truncatedUndiscoveredPresentEntryCount"),
            )
            if undiscovered_entries:
                lines.append(undiscovered_entries)
            corpus_sources = _format_count_rollup(
                "External corpus sources",
                external_summary.get("entriesBySourceBackend"),
            )
            if corpus_sources:
                lines.append(corpus_sources)
            corpus_targets = _format_count_rollup(
                "External corpus targets",
                external_summary.get("entriesByTarget"),
            )
            if corpus_targets:
                lines.append(corpus_targets)
            corpus_artifacts = _format_artifact_rollup(
                "External corpus artifacts",
                external_summary.get("artifactsByTarget"),
            )
            if corpus_artifacts:
                lines.append(corpus_artifacts)

    failed_artifacts = payload.get("failedArtifacts", [])
    if failed_artifacts:
        lines.append("Failed artifacts:")
        for artifact in failed_artifacts:
            description = _format_artifact_identity_line(artifact)
            if description is None:
                continue
            if artifact.get("error"):
                description = f"{description}: {artifact.get('error')}"
            else:
                validation_details = []
                source_hash_status = artifact.get("sourceHashStatus")
                generated_hash_status = artifact.get("generatedHashStatus")
                if source_hash_status and source_hash_status != "ok":
                    validation_details.append(f"source hash: {source_hash_status}")
                if generated_hash_status and generated_hash_status != "ok":
                    validation_details.append(
                        f"generated hash: {generated_hash_status}"
                    )
                if validation_details:
                    description = (
                        f"{description}: validation failed "
                        f"({'; '.join(validation_details)})"
                    )
            lines.append(description)
    truncated_failed_artifacts = payload.get("truncatedFailedArtifactCount", 0)
    if (
        isinstance(truncated_failed_artifacts, int)
        and not isinstance(truncated_failed_artifacts, bool)
        and truncated_failed_artifacts > 0
    ):
        failed_artifact_count = payload.get("failedArtifactCount", 0)
        displayed_count = (
            len(failed_artifacts) if isinstance(failed_artifacts, list) else 0
        )
        lines.append(
            "Failed artifacts truncated: "
            f"showing {displayed_count} of {failed_artifact_count}"
        )

    diagnostics = payload.get("diagnostics", [])
    if diagnostics:
        lines.append("Diagnostics:")
        for diagnostic in diagnostics:
            if not isinstance(diagnostic, Mapping):
                continue
            lines.append(
                "- "
                f"{diagnostic.get('severity', 'note')} "
                f"{diagnostic.get('code', 'unknown')}: "
                f"{diagnostic.get('message', '')}"
            )
    truncated_diagnostics = payload.get("truncatedDiagnosticCount", 0)
    if (
        isinstance(truncated_diagnostics, int)
        and not isinstance(truncated_diagnostics, bool)
        and truncated_diagnostics > 0
    ):
        diagnostic_count = payload.get("diagnosticCount", 0)
        displayed_count = len(diagnostics) if isinstance(diagnostics, list) else 0
        lines.append(
            f"Diagnostics truncated: showing {displayed_count} of {diagnostic_count}"
        )

    migration = payload.get("migration")
    actions = migration.get("actions", []) if isinstance(migration, Mapping) else []
    if actions:
        for line in (
            _format_count_rollup(
                "Migration actions by kind",
                (
                    migration.get("actionsByKind")
                    if isinstance(migration, Mapping)
                    else {}
                ),
                include_zero=False,
            ),
            _format_count_rollup(
                "Migration actions by severity",
                (
                    migration.get("actionsBySeverity")
                    if isinstance(migration, Mapping)
                    else {}
                ),
                include_zero=False,
            ),
            _format_count_rollup(
                "Migration actions by target",
                (
                    migration.get("actionsByTarget")
                    if isinstance(migration, Mapping)
                    else {}
                ),
                include_zero=False,
            ),
        ):
            if line:
                lines.append(line)
        lines.append("Migration actions:")
        for action in actions:
            if not isinstance(action, Mapping):
                continue
            lines.append(
                "- " f"{action.get('kind', 'unknown')}: " f"{action.get('message', '')}"
            )
        truncated_actions = migration.get("truncatedActionCount", 0)
        if (
            isinstance(truncated_actions, int)
            and not isinstance(truncated_actions, bool)
            and truncated_actions > 0
        ):
            lines.append(f"- +{truncated_actions} more")
    return "\n".join(lines) + "\n"


def _run_inspect_report(args):
    from .project import inspect_project_report

    payload = inspect_project_report(
        args.report,
        run_toolchains=args.run_toolchains,
        max_diagnostics=args.max_diagnostics,
        max_failed_artifacts=args.max_failed_artifacts,
    )
    if args.format == "sarif":
        _write_json_payload(
            _format_project_diagnostics_sarif(
                payload, tool_name="CrossTL project report inspection"
            ),
            args.output,
        )
    elif args.format == "text":
        text = _format_project_report_inspection(payload)
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"Wrote {path}")
        else:
            print(text, end="")
    else:
        _write_json_payload(payload, args.output)
    return 0 if payload["success"] else 1


def _build_parser():
    parser = argparse.ArgumentParser(description="CrossGL Shader Translator")
    subparsers = parser.add_subparsers(dest="command")

    supported_backends = ", ".join(backend_names() + ["cgl"])

    translate_parser = subparsers.add_parser(
        "translate", help="Translate a single shader file"
    )
    translate_parser.add_argument("input", help="Input shader file path")
    translate_parser.add_argument(
        "--backend",
        "-b",
        default="cgl",
        help=f"Target backend ({supported_backends})",
    )
    translate_parser.add_argument("--output", "-o", help="Output file path")
    translate_parser.add_argument(
        "--no-format", action="store_true", help="Disable code formatting"
    )
    translate_parser.add_argument(
        "--source-backend", help="Override source parser backend"
    )
    translate_parser.add_argument(
        "--include-dir",
        action="append",
        help="Source parser include directory; repeatable",
    )
    translate_parser.add_argument(
        "--define",
        action="append",
        help="Source parser preprocessor define as NAME or NAME=VALUE; repeatable",
    )
    translate_parser.set_defaults(func=_run_single_file)

    scan_parser = subparsers.add_parser(
        "scan", help="Scan a repository for supported shader/GPU sources"
    )
    _add_project_scan_args(scan_parser)
    scan_parser.add_argument(
        "--target",
        "-b",
        action="append",
        help="Target backend to include in the scan report; repeatable",
    )
    scan_parser.add_argument("--output", "-o", help="Write JSON report to this path")
    scan_parser.set_defaults(func=_run_project_scan)

    translate_project_parser = subparsers.add_parser(
        "translate-project", help="Translate all discovered project units"
    )
    _add_project_scan_args(translate_project_parser)
    translate_project_parser.add_argument(
        "--target",
        "-b",
        action="append",
        help="Target backend; repeatable. Defaults to config targets or cgl.",
    )
    translate_project_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for translated artifacts",
    )
    translate_project_parser.add_argument(
        "--variant",
        action="append",
        help="Named project variant to translate; repeatable",
    )
    translate_project_parser.add_argument(
        "--report", help="Write project portability JSON report to this path"
    )
    translate_project_parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate emitted artifacts and record available toolchains",
    )
    translate_project_parser.add_argument(
        "--no-format", action="store_true", help="Disable code formatting"
    )
    translate_project_parser.set_defaults(func=_run_translate_project)

    validate_parser = subparsers.add_parser(
        "validate-project", help="Validate artifacts referenced by a project report"
    )
    validate_parser.add_argument("report", help="Project portability report JSON")
    validate_parser.add_argument(
        "--format",
        choices=("json", "text", "sarif"),
        default="json",
        help="Validation output format",
    )
    validate_parser.add_argument("--output", "-o", help="Write validation output")
    validate_parser.add_argument(
        "--run-toolchains",
        action="store_true",
        help="Run lightweight optional toolchain smoke checks when tools exist",
    )
    validate_parser.set_defaults(func=_run_validate_project)

    inspect_parser = subparsers.add_parser(
        "inspect-report", help="Inspect an existing project portability report"
    )
    inspect_parser.add_argument("report", help="Project portability report JSON")
    inspect_parser.add_argument(
        "--format",
        choices=("json", "text", "sarif"),
        default="json",
        help="Inspection output format",
    )
    inspect_parser.add_argument("--output", "-o", help="Write inspection output")
    inspect_parser.add_argument(
        "--max-diagnostics",
        type=int,
        default=20,
        help="Maximum diagnostics to include in the inspection summary",
    )
    inspect_parser.add_argument(
        "--max-failed-artifacts",
        type=int,
        default=20,
        help="Maximum failed artifacts to include in the inspection summary",
    )
    inspect_parser.add_argument(
        "--run-toolchains",
        action="store_true",
        help="Run lightweight optional toolchain smoke checks when tools exist",
    )
    inspect_parser.set_defaults(func=_run_inspect_report)

    report_parser = subparsers.add_parser(
        "report", help="Emit a scan-only project portability report"
    )
    _add_project_scan_args(report_parser)
    report_parser.add_argument(
        "--target",
        "-b",
        action="append",
        help="Target backend to include in the report; repeatable",
    )
    report_parser.add_argument("--output", "-o", help="Write JSON report to this path")
    report_parser.set_defaults(func=_run_project_scan)
    return parser


def _use_legacy_cli(argv):
    commands = {
        "translate",
        "scan",
        "translate-project",
        "validate-project",
        "inspect-report",
        "report",
    }
    return argv and argv[0] not in commands and not argv[0].startswith("-")


def main(argv=None):
    """Command-line entry point for CrossGL translation."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _legacy_parser() if _use_legacy_cli(argv) else _build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    try:
        return args.func(args)
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
