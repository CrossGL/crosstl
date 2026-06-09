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
STDOUT_OUTPUT_PATH = "-"


def _non_negative_int(value):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _project_define_arg(value):
    name, separator, define_value = value.partition("=")
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("--define entries must use NAME or NAME=VALUE")
    return f"{name}={define_value.strip()}" if separator else name


def _project_source_override_arg(value):
    pattern, separator, backend = value.partition("=")
    pattern = pattern.strip()
    backend = backend.strip()
    if not pattern or not separator or not backend:
        raise argparse.ArgumentTypeError(
            "--source-override entries must use PATTERN=BACKEND"
        )
    return f"{pattern}={backend}"


def _non_empty_project_arg(option):
    def parse(value):
        parsed = value.strip()
        if not parsed:
            raise argparse.ArgumentTypeError(f"{option} entries must be non-empty")
        return parsed

    return parse


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
        with open(save_shader, "w", encoding="utf-8", newline="") as file:
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


def _is_stdout_output(output_path):
    return str(output_path) == STDOUT_OUTPUT_PATH


def _run_single_file(args):
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        return 1

    output_path = args.output or _derive_single_file_output(args.input, args.backend)
    write_stdout = _is_stdout_output(output_path)
    defines = _parse_project_define_overrides(getattr(args, "define", None))
    generated = translate(
        args.input,
        backend=args.backend,
        save_shader=None if write_stdout else output_path,
        format_output=not args.no_format,
        source_backend=getattr(args, "source_backend", None),
        include_paths=getattr(args, "include_dir", None),
        defines=defines or None,
    )
    if write_stdout:
        print(generated, end="" if generated.endswith("\n") else "\n")
    else:
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
    parser.add_argument("--output", "-o", help="Output file path; use '-' for stdout")
    parser.add_argument(
        "--no-format", action="store_true", help="Disable code formatting"
    )
    parser.add_argument("--source-backend", help="Override source parser backend")
    parser.add_argument(
        "--include-dir",
        action="append",
        type=_non_empty_project_arg("--include-dir"),
        help="Source parser include directory; repeatable",
    )
    parser.add_argument(
        "--define",
        action="append",
        type=_project_define_arg,
        help="Source parser preprocessor define as NAME or NAME=VALUE; repeatable",
    )
    parser.set_defaults(func=_run_single_file)
    return parser


def _write_text_payload(text, output_path=None):
    if output_path and not _is_stdout_output(output_path):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"Wrote {path}")
    else:
        print(text, end="")


def _write_json_payload(payload, output_path=None):
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _write_text_payload(text, output_path)


def _parse_project_define_overrides(values):
    defines = {}
    for value in values or []:
        name, separator, define_value = value.partition("=")
        name = name.strip()
        if not name:
            raise ValueError("--define entries must use NAME or NAME=VALUE")
        defines[name] = define_value.strip() if separator else "1"
    return defines


def _normalize_project_cli_path(value):
    from .project.pipeline import _normalize_project_relative_path

    return _normalize_project_relative_path(value)


def _parse_project_source_overrides(values):
    overrides = {}
    for value in values or []:
        pattern, separator, backend = value.partition("=")
        pattern = pattern.strip()
        backend = backend.strip()
        if not pattern or not separator or not backend:
            raise ValueError("--source-override entries must use PATTERN=BACKEND")
        overrides[_normalize_project_cli_path(pattern)] = backend
    return overrides


def _parse_project_source_roots(values):
    source_roots = []
    for value in values or []:
        source_root = value.strip()
        if not source_root:
            raise ValueError("--source-root entries must be non-empty")
        source_roots.append(_normalize_project_cli_path(source_root))
    return tuple(source_roots)


def _parse_project_include_dirs(values):
    include_dirs = []
    for value in values or []:
        include_dir = value.strip()
        if not include_dir:
            raise ValueError("--include-dir entries must be non-empty")
        include_dirs.append(_normalize_project_cli_path(include_dir))
    return tuple(include_dirs)


def _parse_project_targets(values):
    if values is None:
        return None
    targets = []
    for value in values:
        target = value.strip()
        if not target:
            raise ValueError("--target entries must be non-empty")
        targets.append(target)
    return tuple(targets)


def _load_project_config_from_args(args):
    from .project import load_project_config

    config = load_project_config(args.root, args.config)
    source_roots = _parse_project_source_roots(
        getattr(args, "source_root", None),
    )
    include_dirs = tuple(config.include_dirs) + _parse_project_include_dirs(
        getattr(args, "include_dir", None)
    )
    define_overrides = _parse_project_define_overrides(getattr(args, "define", None))
    source_overrides = _parse_project_source_overrides(
        getattr(args, "source_override", None)
    )
    if (
        not source_roots
        and include_dirs == tuple(config.include_dirs)
        and not define_overrides
        and not source_overrides
    ):
        return config
    return replace(
        config,
        source_roots=source_roots or tuple(config.source_roots),
        include_dirs=include_dirs,
        defines={**dict(config.defines), **define_overrides},
        source_overrides={**dict(config.source_overrides), **source_overrides},
    )


def _add_project_override_args(parser):
    parser.add_argument(
        "--source-root",
        action="append",
        type=_non_empty_project_arg("--source-root"),
        help=(
            "Project source root override; repeatable. Replaces configured "
            "source roots for this command."
        ),
    )
    parser.add_argument(
        "--include-dir",
        action="append",
        type=_non_empty_project_arg("--include-dir"),
        help="Project include directory override; repeatable",
    )
    parser.add_argument(
        "--define",
        action="append",
        type=_project_define_arg,
        help="Project preprocessor define override as NAME or NAME=VALUE; repeatable",
    )
    parser.add_argument(
        "--source-override",
        action="append",
        type=_project_source_override_arg,
        help="Project source backend override as PATTERN=BACKEND; repeatable",
    )


def _add_project_scan_args(parser):
    parser.add_argument("root", help="Repository root to scan")
    parser.add_argument("--config", help="Path to crosstl.toml")
    _add_project_override_args(parser)


def _add_project_variant_args(parser, *, action_label):
    parser.add_argument(
        "--variant",
        action="append",
        type=_non_empty_project_arg("--variant"),
        help=f"Named project variant to {action_label}; repeatable",
    )


def _run_project_scan(args):
    from .project import scan_project

    config = _load_project_config_from_args(args)
    report = scan_project(
        config,
        variants=getattr(args, "variant", None),
    ).to_report(targets=_parse_project_targets(args.target))
    payload = report.to_json()
    _write_json_payload(payload, args.output)
    return 1 if payload["summary"]["diagnosticCounts"]["error"] else 0


def _run_translate_project(args):
    from .project import translate_project

    config = _load_project_config_from_args(args)
    report = translate_project(
        config,
        targets=_parse_project_targets(args.target),
        output_dir=args.output_dir,
        variants=args.variant,
        format_output=not args.no_format,
        validate=args.validate,
        run_toolchains=args.run_toolchains,
    )
    payload = report.to_json()
    if args.report:
        _write_json_payload(payload, args.report)
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
        _write_text_payload(_format_project_validation_report(payload), args.output)
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


def _format_payload_schema_version(payload, label):
    if not isinstance(payload, Mapping):
        return None

    schema_version = payload.get("schemaVersion")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version < 0
    ):
        return None
    return f"{label}: {schema_version}"


def _format_payload_kind(payload, label):
    if not isinstance(payload, Mapping):
        return None

    kind = payload.get("kind")
    if not isinstance(kind, str) or not kind:
        return None
    return f"{label}: {kind}"


def _format_payload_generated_at(payload, label):
    if not isinstance(payload, Mapping):
        return None

    generated_at = payload.get("generatedAt")
    if (
        not isinstance(generated_at, int)
        or isinstance(generated_at, bool)
        or generated_at < 0
    ):
        return None
    return f"{label}: {generated_at}"


def _format_payload_hash(payload, field_name, label):
    if not isinstance(payload, Mapping):
        return None

    hash_payload = payload.get(field_name)
    if not isinstance(hash_payload, Mapping):
        return None

    hash_preview = _format_hash_preview(
        hash_payload.get("algorithm"),
        hash_payload.get("value"),
    )
    if not hash_preview:
        return None
    return f"{label}: {hash_preview}"


def _format_project_validation_report(payload):
    counts = payload.get("diagnosticCounts", {})
    counts = counts if isinstance(counts, Mapping) else {}
    lines = [f"Project validation report: {payload.get('sourceReport')}"]
    for header_line in (
        _format_payload_schema_version(payload, "Validation schema version"),
        _format_payload_kind(payload, "Validation kind"),
        _format_payload_generated_at(payload, "Validation generated at"),
        _format_payload_hash(payload, "sourceReportHash", "Source report hash"),
    ):
        if header_line:
            lines.append(header_line)
    project = payload.get("project")
    for project_line in (
        _format_project_root_path(project),
        _format_project_output_dir(project),
        _format_project_string_list(project, "Project targets", "targets"),
        _format_project_source_roots(project),
        _format_project_include_dirs(project),
    ):
        if project_line:
            lines.append(project_line)
    lines.extend(
        [
            f"Status: {'ok' if payload.get('success') else 'failed'}",
            (
                "Diagnostics: "
                f"{counts.get('error', 0)} errors, "
                f"{counts.get('warning', 0)} warnings, "
                f"{counts.get('note', 0)} notes"
            ),
        ]
    )

    for line in (
        _format_count_rollup(
            "Diagnostic codes", payload.get("diagnosticsByCode"), include_zero=False
        ),
        _format_count_rollup(
            "Diagnostics by target",
            payload.get("diagnosticsByTarget"),
            include_zero=False,
        ),
        _format_count_rollup(
            "Diagnostics by source backend",
            payload.get("diagnosticsBySourceBackend"),
            include_zero=False,
        ),
        _format_count_rollup(
            "Diagnostics by variant",
            payload.get("diagnosticsByVariant"),
            include_zero=False,
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
            "Validation artifacts by source backend",
            payload.get("artifactStatusBySourceBackend"),
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
            "Validation source sizes",
            payload.get("sourceSizeStatusCounts"),
            include_zero=False,
        ),
        _format_count_rollup(
            "Validation generated hashes",
            payload.get("generatedHashStatusCounts"),
            include_zero=False,
        ),
        _format_count_rollup(
            "Validation generated sizes",
            payload.get("generatedSizeStatusCounts"),
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
            "Validation toolchain runs by check kind",
            payload.get("toolchainRunStatusByCheckKind"),
        ),
        _format_validation_run_rollup(
            "Validation toolchain runs by tool",
            payload.get("toolchainRunStatusByTool"),
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
            lines.append(_format_project_diagnostic_line(diagnostic))

    return "\n".join(lines) + "\n"


def _format_project_diagnostic_location(location):
    if not isinstance(location, Mapping):
        return None

    file_name = location.get("file")
    if not isinstance(file_name, str) or not file_name:
        return None

    text = file_name
    line = location.get("line")
    if isinstance(line, int) and not isinstance(line, bool) and line > 0:
        text = f"{text}:{line}"
        column = location.get("column")
        if isinstance(column, int) and not isinstance(column, bool) and column > 0:
            text = f"{text}:{column}"
    return text


def _format_project_diagnostic_line(diagnostic):
    line = (
        "- "
        f"{diagnostic.get('severity', 'note')} "
        f"{diagnostic.get('code', 'unknown')}: "
        f"{diagnostic.get('message', '')}"
    )

    details = []
    location = _format_project_diagnostic_location(diagnostic.get("location"))
    if location:
        details.append(f"location={location}")
    original_location = _format_project_diagnostic_location(
        diagnostic.get("originalLocation")
    )
    if original_location:
        details.append(f"originalLocation={original_location}")
    target = diagnostic.get("target")
    if isinstance(target, str) and target:
        details.append(f"target={target}")
    source_backend = diagnostic.get("sourceBackend")
    if isinstance(source_backend, str) and source_backend:
        details.append(f"sourceBackend={source_backend}")
    variant = diagnostic.get("variant")
    if isinstance(variant, str) and variant:
        details.append(f"variant={variant}")
    missing_capabilities = diagnostic.get("missingCapabilities")
    if isinstance(missing_capabilities, list):
        capabilities = [
            capability
            for capability in missing_capabilities
            if isinstance(capability, str) and capability
        ]
        if capabilities:
            details.append(f"missingCapabilities={','.join(capabilities)}")
    if details:
        line = f"{line} ({'; '.join(details)})"
    return line


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
    offset = location.get("offset")
    length = location.get("length")
    if (
        isinstance(offset, int)
        and not isinstance(offset, bool)
        and offset >= 0
        and isinstance(length, int)
        and not isinstance(length, bool)
        and length > 0
    ):
        region["charOffset"] = offset
        region["charLength"] = length
    return region


def _sarif_location_from_mapping(location):
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


def _sarif_location_property(location):
    if not isinstance(location, Mapping):
        return None

    file_name = location.get("file")
    if not isinstance(file_name, str) or not file_name:
        return None

    payload = {"file": file_name}
    for field_name in (
        "line",
        "column",
        "offset",
        "length",
        "endLine",
        "endColumn",
        "endOffset",
    ):
        value = location.get(field_name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            payload[field_name] = value
    return payload


def _sarif_location(diagnostic):
    return _sarif_location_from_mapping(diagnostic.get("location"))


def _sarif_non_negative_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _sarif_string_list(value):
    if not isinstance(value, list):
        return None
    return [item for item in value if isinstance(item, str)]


def _sarif_hash_payload(value):
    if not isinstance(value, Mapping):
        return None

    algorithm = value.get("algorithm")
    digest = value.get("value")
    if not isinstance(algorithm, str) or not algorithm:
        return None
    if not isinstance(digest, str) or not digest:
        return None
    return {"algorithm": algorithm, "value": digest}


def _add_sarif_project_properties(properties, project):
    if not isinstance(project, Mapping):
        return

    string_fields = {
        "root": "projectRoot",
        "outputDir": "projectOutputDir",
        "config": "projectConfig",
    }
    for source_name, property_name in string_fields.items():
        value = project.get(source_name)
        if isinstance(value, str) and value:
            properties[property_name] = value

    list_fields = {
        "targets": "projectTargets",
        "sourceRoots": "projectSourceRoots",
        "includePatterns": "projectIncludePatterns",
        "excludePatterns": "projectExcludePatterns",
        "includeDirs": "projectIncludeDirs",
        "selectedVariants": "projectSelectedVariants",
    }
    for source_name, property_name in list_fields.items():
        values = _sarif_string_list(project.get(source_name))
        if values is not None:
            properties[property_name] = values


def _sarif_invocation_properties(payload):
    properties = {}

    source_report = payload.get("sourceReport")
    if isinstance(source_report, str) and source_report:
        properties["sourceReport"] = source_report

    source_report_hash = _sarif_hash_payload(payload.get("sourceReportHash"))
    if source_report_hash is not None:
        properties["sourceReportHash"] = source_report_hash

    schema_version = payload.get("schemaVersion")
    if _sarif_non_negative_int(schema_version):
        properties["schemaVersion"] = schema_version

    kind = payload.get("kind")
    if isinstance(kind, str) and kind:
        properties["kind"] = kind

    generated_at = payload.get("generatedAt")
    if _sarif_non_negative_int(generated_at):
        properties["generatedAt"] = generated_at

    _add_sarif_project_properties(properties, payload.get("project"))

    report = payload.get("report")
    if isinstance(report, Mapping):
        source_schema_version = report.get("schemaVersion")
        if _sarif_non_negative_int(source_schema_version):
            properties["sourceReportSchemaVersion"] = source_schema_version

        source_kind = report.get("kind")
        if isinstance(source_kind, str) and source_kind:
            properties["sourceReportKind"] = source_kind

        source_generated_at = report.get("generatedAt")
        if _sarif_non_negative_int(source_generated_at):
            properties["sourceReportGeneratedAt"] = source_generated_at

        _add_sarif_project_properties(properties, report.get("project"))

    return properties


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
        original_location = _sarif_location_from_mapping(
            diagnostic.get("originalLocation")
        )
        if original_location:
            original_location["id"] = 1
            original_location["message"] = {"text": "Original source location"}
            result["relatedLocations"] = [original_location]

        properties = {}
        target = diagnostic.get("target")
        if isinstance(target, str) and target:
            properties["target"] = target
        source_backend = diagnostic.get("sourceBackend")
        if isinstance(source_backend, str) and source_backend:
            properties["sourceBackend"] = source_backend
        variant = diagnostic.get("variant")
        if isinstance(variant, str) and variant:
            properties["variant"] = variant
        missing_capabilities = diagnostic.get("missingCapabilities")
        if isinstance(missing_capabilities, list) and missing_capabilities:
            properties["missingCapabilities"] = list(missing_capabilities)
        if original_location:
            diagnostic_location = _sarif_location_property(diagnostic.get("location"))
            if diagnostic_location is not None:
                properties["diagnosticLocation"] = diagnostic_location
            original_location_property = _sarif_location_property(
                diagnostic.get("originalLocation")
            )
            if original_location_property is not None:
                properties["originalLocation"] = original_location_property
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
                        "properties": _sarif_invocation_properties(payload),
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


def _format_project_string_list(project, label, field_name):
    if not isinstance(project, Mapping):
        return None

    values = project.get(field_name)
    if not isinstance(values, list):
        return None

    entries = [value for value in values if isinstance(value, str) and value]
    if not entries:
        return None
    return f"{label}: " + ", ".join(entries)


def _format_project_source_roots(project):
    return _format_project_string_list(project, "Project source roots", "sourceRoots")


def _format_project_include_dirs(project):
    return _format_project_string_list(project, "Project include dirs", "includeDirs")


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


def _format_project_define_names(project):
    if not isinstance(project, Mapping):
        return None

    define_names = project.get("defineNames")
    if not isinstance(define_names, list):
        return None

    names = [name for name in define_names if isinstance(name, str) and name]
    if not names:
        return None
    return "Project define names: " + ", ".join(names)


def _format_project_define_fingerprint(project):
    if not isinstance(project, Mapping):
        return None

    define_fingerprint = project.get("defineFingerprint")
    if not isinstance(define_fingerprint, Mapping):
        return None

    hash_preview = _format_hash_preview(
        define_fingerprint.get("algorithm"),
        define_fingerprint.get("value"),
    )
    if not hash_preview:
        return None
    return f"Project define fingerprint: {hash_preview}"


def _format_project_source_overrides(project):
    if not isinstance(project, Mapping):
        return None

    source_overrides = project.get("sourceOverrides")
    if not isinstance(source_overrides, Mapping):
        return None

    entries = [
        f"{path}={backend}"
        for path, backend in sorted(source_overrides.items())
        if isinstance(path, str) and path and isinstance(backend, str) and backend
    ]
    if not entries:
        return None
    return "Project source overrides: " + ", ".join(entries)


def _format_project_selected_variants(project):
    if not isinstance(project, Mapping):
        return None

    selected_variants = project.get("selectedVariants")
    if not isinstance(selected_variants, list):
        return None

    names = [name for name in selected_variants if isinstance(name, str) and name]
    if not names:
        return None
    return "Selected variants: " + ", ".join(names)


def _format_project_variant_define_counts(project):
    if not isinstance(project, Mapping):
        return None
    return _format_count_rollup(
        "Project variant define counts",
        project.get("variantDefineCounts"),
    )


def _format_project_variant_define_names(project):
    if not isinstance(project, Mapping):
        return None

    variant_define_names = project.get("variantDefineNames")
    if not isinstance(variant_define_names, Mapping):
        return None

    entries = []
    for variant, define_names in variant_define_names.items():
        if not isinstance(variant, str) or not variant:
            continue
        if not isinstance(define_names, list):
            continue
        names = [name for name in define_names if isinstance(name, str) and name]
        if names:
            entries.append((variant, ",".join(names)))
    if not entries:
        return None
    entries.sort(key=lambda item: item[0])
    return "Project variant define names: " + ", ".join(
        f"{variant}=({names})" for variant, names in entries
    )


def _format_project_variant_define_fingerprints(project):
    if not isinstance(project, Mapping):
        return None

    variant_fingerprints = project.get("variantDefineFingerprints")
    if not isinstance(variant_fingerprints, Mapping):
        return None

    entries = []
    for variant, fingerprint in variant_fingerprints.items():
        if not isinstance(variant, str) or not variant:
            continue
        if not isinstance(fingerprint, Mapping):
            continue
        hash_preview = _format_hash_preview(
            fingerprint.get("algorithm"),
            fingerprint.get("value"),
        )
        if hash_preview:
            entries.append((variant, hash_preview))
    if not entries:
        return None
    entries.sort(key=lambda item: item[0])
    return "Project variant define fingerprints: " + ", ".join(
        f"{variant}={hash_preview}" for variant, hash_preview in entries
    )


def _format_project_config_path(project):
    if not isinstance(project, Mapping) or "config" not in project:
        return None

    config_path = project.get("config")
    config_hash = project.get("configHash")
    hash_preview = None
    if isinstance(config_hash, Mapping):
        hash_preview = _format_hash_preview(
            config_hash.get("algorithm"),
            config_hash.get("value"),
        )
    suffix = f" (hash={hash_preview})" if hash_preview else ""
    if config_path is None:
        return "Config file: (none)"
    if isinstance(config_path, str):
        return f"Config file: {config_path or '(empty)'}{suffix}"
    return None


def _format_project_root_path(project):
    if not isinstance(project, Mapping):
        return None

    root_path = project.get("root")
    if not isinstance(root_path, str) or not root_path:
        return None
    return f"Project root: {root_path}"


def _format_project_output_dir(project):
    if not isinstance(project, Mapping):
        return None

    output_dir = project.get("outputDir")
    if not isinstance(output_dir, str) or not output_dir:
        return None
    return f"Output directory: {output_dir}"


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
        details = [status]
        resolved_path = record.get("resolvedPath")
        if isinstance(resolved_path, str) and resolved_path:
            details.append(f"resolved={resolved_path}")
        for visibility_field in ("scanVisible", "frontendVisible"):
            visibility = record.get(visibility_field)
            if isinstance(visibility, bool):
                details.append(f"{visibility_field}={str(visibility).lower()}")
        entries.append(f"{path or '(empty)'} ({'; '.join(details)})")

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
    if fine_grained_count > source_map_count:
        return None
    file_level_count = source_map_count - fine_grained_count
    return (
        "Source maps: "
        f"{file_level_count} file-level, "
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
    source_remap_mapping_count = summary.get("sourceRemapMappingCount")
    if (
        isinstance(source_remap_mapping_count, int)
        and not isinstance(source_remap_mapping_count, bool)
        and source_remap_mapping_count >= 0
    ):
        return f"Source remaps: {source_remap_count} ({source_remap_mapping_count} mappings)"
    return f"Source remaps: {source_remap_count}"


def _format_source_map_span_preview(span):
    if not isinstance(span, Mapping):
        return None

    values = {}
    for field_name in (
        "line",
        "column",
        "endLine",
        "endColumn",
        "offset",
        "length",
    ):
        value = span.get(field_name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return None
        values[field_name] = value

    return (
        f"{values['line']}:{values['column']}-"
        f"{values['endLine']}:{values['endColumn']}"
        f"@{values['offset']}+{values['length']}"
    )


def _append_artifact_validation_details(details, artifact):
    validation_status = artifact.get("validationStatus")
    if isinstance(validation_status, str) and validation_status:
        details.append(f"validation={validation_status}")
    exists = artifact.get("exists")
    if isinstance(exists, bool):
        details.append(f"exists={'true' if exists else 'false'}")
    for field_name, label in (
        ("sourceHashStatus", "sourceHashStatus"),
        ("sourceSizeStatus", "sourceSizeStatus"),
        ("generatedHashStatus", "generatedHashStatus"),
        ("generatedSizeStatus", "generatedSizeStatus"),
        ("sourceMapStatus", "sourceMapStatus"),
        ("sourceRemapStatus", "sourceRemapStatus"),
    ):
        value = artifact.get(field_name)
        if isinstance(value, str) and value:
            details.append(f"{label}={value}")


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
    source_map_target = artifact.get("sourceMapTarget")
    if isinstance(source_map_target, str) and source_map_target:
        details.append(f"sourceMapTarget={source_map_target}")
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
    source_span = _format_source_map_span_preview(artifact.get("sourceSpan"))
    if source_span:
        details.append(f"sourceSpan={source_span}")
    generated_span = _format_source_map_span_preview(artifact.get("generatedSpan"))
    if generated_span:
        details.append(f"generatedSpan={generated_span}")
    _append_source_artifact_metadata(details, artifact)
    _append_generated_artifact_metadata(details, artifact)
    _append_artifact_validation_details(details, artifact)

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


def _format_hash_preview(algorithm, value):
    if not isinstance(value, str) or not value:
        return None
    prefix = f"{algorithm}:" if isinstance(algorithm, str) and algorithm else ""
    suffix = "..." if len(value) > 12 else ""
    return f"{prefix}{value[:12]}{suffix}"


def _format_byte_size(value):
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        size_label = "byte" if value == 1 else "bytes"
        return f"{value} {size_label}"
    return None


def _append_source_artifact_metadata(details, artifact):
    source_hash_preview = _format_hash_preview(
        artifact.get("sourceHashAlgorithm"),
        artifact.get("sourceHash"),
    )
    if source_hash_preview:
        details.append(f"sourceHash={source_hash_preview}")
    source_size = _format_byte_size(artifact.get("sourceSizeBytes"))
    if source_size:
        details.append(f"sourceSize={source_size}")


def _append_generated_artifact_metadata(details, artifact):
    generated_hash_preview = _format_hash_preview(
        artifact.get("generatedHashAlgorithm"),
        artifact.get("generatedHash"),
    )
    if generated_hash_preview:
        details.append(f"generatedHash={generated_hash_preview}")
    generated_size = _format_byte_size(artifact.get("generatedSizeBytes"))
    if generated_size:
        details.append(f"generatedSize={generated_size}")


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
    source_remap_target = artifact.get("sourceRemapTarget")
    if isinstance(source_remap_target, str) and source_remap_target:
        details.append(f"sourceRemapTarget={source_remap_target}")
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
    _append_source_artifact_metadata(details, artifact)
    _append_generated_artifact_metadata(details, artifact)
    hash_preview = _format_hash_preview(
        artifact.get("sourceRemapHashAlgorithm"),
        artifact.get("sourceRemapHash"),
    )
    if hash_preview:
        details.append(f"hash={hash_preview}")
    source_remap_size = _format_byte_size(artifact.get("sourceRemapSizeBytes"))
    if source_remap_size:
        details.append(f"sourceRemapSize={source_remap_size}")
    _append_artifact_validation_details(details, artifact)

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
    _append_source_artifact_metadata(details, artifact)
    _append_generated_artifact_metadata(details, artifact)
    _append_artifact_validation_details(details, artifact)

    suffix = f" ({', '.join(details)})" if details else ""
    return f"- {source} -> {path}{suffix}"


def _format_artifact_provenance_sample_lines(
    title,
    artifacts,
    truncated_count,
):
    if not isinstance(artifacts, list) or not artifacts:
        return []

    lines = [f"{title}:"]
    for artifact in artifacts:
        line = _format_artifact_provenance_line(artifact)
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


def _format_artifact_provenance_lines(artifact_provenance):
    if not isinstance(artifact_provenance, Mapping):
        return []

    lines = _format_artifact_provenance_sample_lines(
        "Artifact provenance samples",
        artifact_provenance.get("artifacts"),
        artifact_provenance.get("truncatedArtifactCount"),
    )

    direct_count = artifact_provenance.get("directArtifactCount")
    bridged_count = artifact_provenance.get("bridgedArtifactCount")
    if (
        isinstance(direct_count, int)
        and not isinstance(direct_count, bool)
        and direct_count > 0
        and isinstance(bridged_count, int)
        and not isinstance(bridged_count, bool)
        and bridged_count > 0
    ):
        lines.extend(
            _format_artifact_provenance_sample_lines(
                "Direct artifact provenance samples",
                artifact_provenance.get("directArtifacts"),
                artifact_provenance.get("truncatedDirectArtifactCount"),
            )
        )
        lines.extend(
            _format_artifact_provenance_sample_lines(
                "Bridged artifact provenance samples",
                artifact_provenance.get("bridgedArtifacts"),
                artifact_provenance.get("truncatedBridgedArtifactCount"),
            )
        )
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


def _format_artifact_matrix_source(artifact_matrix):
    if not isinstance(artifact_matrix, Mapping):
        return None
    if not artifact_matrix.get("available"):
        return None

    matrix_source = artifact_matrix.get("source")
    if matrix_source not in {"report", "derived"}:
        return None
    return f"Artifact matrix source: {matrix_source}"


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
    source_size = artifact.get("sourceSizeStatus")
    if isinstance(source_size, str) and source_size:
        details.append(f"sourceSize={source_size}")
    generated_hash = artifact.get("generatedHashStatus")
    if isinstance(generated_hash, str) and generated_hash:
        details.append(f"generatedHash={generated_hash}")
    generated_size = artifact.get("generatedSizeStatus")
    if isinstance(generated_size, str) and generated_size:
        details.append(f"generatedSize={generated_size}")
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
    check_kind = run.get("checkKind")
    if isinstance(check_kind, str) and check_kind:
        details.append(f"checkKind={check_kind}")
    returncode = run.get("returncode")
    if isinstance(returncode, int) and not isinstance(returncode, bool):
        details.append(f"returncode={returncode}")
    failure_reason = run.get("failureReason")
    if isinstance(failure_reason, str) and failure_reason:
        details.append(f"failureReason={failure_reason}")
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
    define_names = artifact.get("defineNames")
    if isinstance(define_names, list):
        names = [name for name in define_names if isinstance(name, str) and name]
        if names:
            details.append("defineNames=" + ",".join(names))
    define_fingerprint = artifact.get("defineFingerprint")
    if isinstance(define_fingerprint, Mapping):
        fingerprint_preview = _format_hash_preview(
            define_fingerprint.get("algorithm"),
            define_fingerprint.get("value"),
        )
        if fingerprint_preview:
            details.append(f"defineFingerprint={fingerprint_preview}")

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


def _include_path_processing_frontend_visible_dirs(include_path_processing):
    if not isinstance(include_path_processing, Mapping):
        return []

    records = include_path_processing.get("frontendVisibleIncludeDirs")
    if not isinstance(records, list):
        records = include_path_processing.get("includeDirs")
    if not isinstance(records, list):
        return []

    paths = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        if record.get("frontendVisible") is not True:
            continue
        path = record.get("path")
        if isinstance(path, str) and path:
            paths.append(path)
    return paths


def _format_include_path_processing_issue_line(artifact, include_dirs=None):
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
    suffix = ""
    if include_dirs:
        suffix = f" (includeDirs={','.join(include_dirs)})"
    return (
        f"{line}: {include_path_count} {include_path_label} not forwarded"
        f"{source_backend_label}{frontend_label}{suffix}"
    )


def _format_include_path_processing_issue_lines(include_path_processing):
    if not isinstance(include_path_processing, Mapping):
        return []
    artifacts = include_path_processing.get("notSupportedArtifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return []

    include_dirs = _include_path_processing_frontend_visible_dirs(
        include_path_processing
    )
    lines = ["Include path processing issues:"]
    for artifact in artifacts:
        line = _format_include_path_processing_issue_line(artifact, include_dirs)
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


def _format_include_path_processing_dir_summary(include_path_processing):
    if not isinstance(include_path_processing, Mapping):
        return None

    include_dir_count = include_path_processing.get("includeDirCount")
    frontend_visible_count = include_path_processing.get(
        "frontendVisibleIncludeDirCount"
    )
    inactive_count = include_path_processing.get("inactiveIncludeDirCount")
    counts = (
        include_dir_count,
        frontend_visible_count,
        inactive_count,
    )
    if not all(
        isinstance(count, int) and not isinstance(count, bool) and count >= 0
        for count in counts
    ):
        return None
    if include_dir_count == 0:
        return None
    return (
        "Include path processing dirs: "
        f"{include_dir_count} configured, "
        f"{frontend_visible_count} frontend-visible, "
        f"{inactive_count} inactive"
    )


def _format_include_path_processing_dir_line(record):
    if not isinstance(record, Mapping):
        return None

    path = record.get("path")
    status = record.get("status")
    if not isinstance(path, str) or not path:
        return None
    if not isinstance(status, str) or not status:
        return None

    details = [status]
    resolved_path = record.get("resolvedPath")
    if isinstance(resolved_path, str) and resolved_path:
        details.append(f"resolved={resolved_path}")
    frontend_visible = record.get("frontendVisible")
    if isinstance(frontend_visible, bool):
        details.append(f"frontendVisible={str(frontend_visible).lower()}")
    return f"- {path} ({'; '.join(details)})"


def _format_include_path_processing_dir_lines(include_path_processing):
    if not isinstance(include_path_processing, Mapping):
        return []

    records = include_path_processing.get("includeDirs")
    if not isinstance(records, list) or not records:
        return []

    lines = ["Include path processing dirs:"]
    for record in records:
        line = _format_include_path_processing_dir_line(record)
        if line:
            lines.append(line)
    if len(lines) == 1:
        return []
    return lines


def _format_skipped_source_line(source):
    if not isinstance(source, Mapping):
        return None
    path = source.get("path")
    reason = source.get("reason")
    if not isinstance(path, str) or not path:
        return None
    if not isinstance(reason, str) or not reason:
        return None

    details = [reason]
    extension = source.get("extension")
    if isinstance(extension, str) and extension:
        details.append(f"extension={extension}")
    source_override = source.get("sourceOverride")
    if isinstance(source_override, str) and source_override:
        details.append(f"sourceOverride={source_override}")
    return f"- {path} ({'; '.join(details)})"


def _format_skipped_source_lines(skipped_sources):
    if not isinstance(skipped_sources, Mapping):
        return []
    sources = skipped_sources.get("sources")
    if not isinstance(sources, list) or not sources:
        return []

    lines = ["Skipped sources:"]
    for source in sources:
        line = _format_skipped_source_line(source)
        if line:
            lines.append(line)
    if len(lines) == 1:
        return []

    truncated_count = skipped_sources.get("truncatedSkippedCount")
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
    source_backend = dependency.get("sourceBackend")
    if isinstance(source_backend, str) and source_backend:
        location += f" [{source_backend}]"

    kind = dependency.get("kind")
    kind_label = f" {kind}" if isinstance(kind, str) and kind else ""
    provenance_label = _format_include_dependency_provenance_label(dependency)
    return f"- {location}: {status}{kind_label} include {include}{provenance_label}"


def _format_include_dependency_provenance_label(dependency):
    parts = []
    variant = dependency.get("variant")
    if isinstance(variant, str) and variant:
        parts.append(f"variant {variant}")
    resolved_from = dependency.get("resolvedFrom")
    if isinstance(resolved_from, str) and resolved_from:
        parts.append(resolved_from)
    resolved_from_define = dependency.get("resolvedFromDefine")
    if isinstance(resolved_from_define, str) and resolved_from_define:
        parts.append(f"define {resolved_from_define}")
    unit_hash_preview = _format_hash_preview(
        dependency.get("unitSourceHashAlgorithm"),
        dependency.get("unitSourceHash"),
    )
    if unit_hash_preview:
        parts.append(f"unitHash={unit_hash_preview}")
    unit_source_size = _format_byte_size(dependency.get("unitSourceSizeBytes"))
    if unit_source_size:
        parts.append(f"unitSize={unit_source_size}")
    hash_preview = _format_hash_preview(
        dependency.get("resolvedHashAlgorithm"),
        dependency.get("resolvedHash"),
    )
    if hash_preview:
        parts.append(f"hash={hash_preview}")
    resolved_size = dependency.get("resolvedSizeBytes")
    if (
        isinstance(resolved_size, int)
        and not isinstance(resolved_size, bool)
        and resolved_size >= 0
    ):
        resolved_size_label = _format_byte_size(resolved_size)
        if resolved_size_label:
            parts.append(f"size={resolved_size_label}")
    if not parts:
        return ""
    return f" ({', '.join(parts)})"


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
    source_backend = dependency.get("sourceBackend")
    if isinstance(source_backend, str) and source_backend:
        location += f" [{source_backend}]"

    kind = dependency.get("kind")
    kind_label = f" {kind}" if isinstance(kind, str) and kind else ""
    source_label = _format_include_dependency_provenance_label(dependency)
    return (
        f"- {location}: resolved{kind_label} include {include} -> "
        f"{resolved_path}{source_label}"
    )


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


def _format_system_include_dependency_line(dependency):
    if not isinstance(dependency, Mapping):
        return None

    include = dependency.get("include")
    if not isinstance(include, str) or not include:
        return None

    source = dependency.get("source")
    location = source if isinstance(source, str) and source else "(unknown)"
    line = dependency.get("line")
    column = dependency.get("column")
    if isinstance(line, int) and not isinstance(line, bool) and line > 0:
        location += f":{line}"
        if isinstance(column, int) and not isinstance(column, bool) and column > 0:
            location += f":{column}"
    source_backend = dependency.get("sourceBackend")
    if isinstance(source_backend, str) and source_backend:
        location += f" [{source_backend}]"

    provenance_label = _format_include_dependency_provenance_label(dependency)
    return f"- {location}: system include {include}{provenance_label}"


def _format_system_include_dependency_lines(include_dependencies):
    if not isinstance(include_dependencies, Mapping):
        return []
    dependencies = include_dependencies.get("systemDependencies")
    if not isinstance(dependencies, list) or not dependencies:
        return []

    lines = ["System include dependencies:"]
    for dependency in dependencies:
        line = _format_system_include_dependency_line(dependency)
        if line:
            lines.append(line)
    if len(lines) == 1:
        return []

    truncated_count = include_dependencies.get("truncatedSystemDependencyCount")
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
    repository = entry.get("repository")
    if isinstance(repository, str) and repository:
        details.append(f"repository={repository}")
    commit = entry.get("commit")
    if isinstance(commit, str) and commit:
        details.append(f"commit={commit}")
    source_url = entry.get("sourceUrl")
    if isinstance(source_url, str) and source_url:
        details.append(f"sourceUrl={source_url}")
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


def _format_report_generated_at(report):
    return _format_payload_generated_at(report, "Report generated at")


def _format_inspection_schema_version(payload):
    return _format_payload_schema_version(payload, "Inspection schema version")


def _format_inspection_kind(payload):
    return _format_payload_kind(payload, "Inspection kind")


def _format_inspection_generated_at(payload):
    return _format_payload_generated_at(payload, "Inspection generated at")


def _format_source_report_schema_version(report):
    return _format_payload_schema_version(report, "Source report schema version")


def _format_source_report_kind(report):
    return _format_payload_kind(report, "Source report kind")


def _format_report_generator(report):
    if not isinstance(report, Mapping):
        return None

    generator = report.get("generator")
    if not isinstance(generator, Mapping):
        return None

    name = generator.get("name")
    pipeline = generator.get("pipeline")
    package_version = generator.get("packageVersion")
    if not isinstance(name, str) or not name:
        return None

    details = []
    if isinstance(pipeline, str) and pipeline:
        details.append(pipeline)
    if isinstance(package_version, str) and package_version:
        details.append(f"packageVersion={package_version}")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"Report generator: {name}{suffix}"


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
    validation_toolchain_run_status_by_check_kind = payload.get("validation", {}).get(
        "toolchainRunStatusByCheckKind"
    )
    validation_toolchain_run_status_by_tool = payload.get("validation", {}).get(
        "toolchainRunStatusByTool"
    )
    validation_toolchain_run_status_by_variant = payload.get("validation", {}).get(
        "toolchainRunStatusByVariant"
    )
    validation_artifact_status_by_target = payload.get("validation", {}).get(
        "artifactStatusByTarget"
    )
    validation_artifact_status_by_source_backend = payload.get("validation", {}).get(
        "artifactStatusBySourceBackend"
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
    lines = [f"Project report: {payload.get('sourceReport')}"]
    for header_line in (
        _format_inspection_schema_version(payload),
        _format_inspection_kind(payload),
        _format_inspection_generated_at(payload),
        _format_payload_hash(payload, "sourceReportHash", "Source report hash"),
        _format_source_report_schema_version(report),
        _format_source_report_kind(report),
        _format_project_config_path(project),
        _format_project_root_path(project),
        _format_project_output_dir(project),
        _format_report_generated_at(report),
        _format_report_generator(report),
    ):
        if header_line:
            lines.append(header_line)
    lines.append(f"Status: {'ok' if payload.get('success') else 'failed'}")
    report_status = _format_report_status(report, validation_diagnostic_codes)
    if report_status:
        lines.append(report_status)
    lines.append("Targets: " + (", ".join(target_names) if target_names else "(none)"))
    for project_line in (
        _format_project_config_counts(project),
        _format_project_source_roots(project),
        _format_project_include_dirs(project),
        _format_project_source_overrides(project),
        _format_project_define_names(project),
        _format_project_define_fingerprint(project),
        _format_project_variant_names(project),
        _format_project_selected_variants(project),
        _format_project_variant_define_counts(project),
        _format_project_variant_define_names(project),
        _format_project_variant_define_fingerprints(project),
    ):
        if project_line:
            lines.append(project_line)
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
    lines.extend(project_status_lines)
    lines.extend(
        [
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
    )
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
    define_processing_by_target = _format_nested_count_rollup(
        "Define processing by target",
        summary.get("defineProcessingByTarget"),
        include_zero=False,
    )
    if define_processing_by_target:
        lines.append(define_processing_by_target)
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
    include_path_processing_by_target = _format_nested_count_rollup(
        "Include path processing by target",
        summary.get("includePathProcessingByTarget"),
        include_zero=False,
    )
    if include_path_processing_by_target:
        lines.append(include_path_processing_by_target)
    include_path_processing_by_variant = _format_nested_count_rollup(
        "Include path processing by variant",
        summary.get("includePathProcessingByVariant"),
        include_zero=False,
    )
    if include_path_processing_by_variant:
        lines.append(include_path_processing_by_variant)
    include_path_processing_dir_summary = _format_include_path_processing_dir_summary(
        payload.get("includePathProcessing")
    )
    if include_path_processing_dir_summary:
        lines.append(include_path_processing_dir_summary)
    lines.extend(
        _format_include_path_processing_dir_lines(payload.get("includePathProcessing"))
    )
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
    artifact_matrix_source = _format_artifact_matrix_source(artifact_matrix_payload)
    if artifact_matrix_source:
        lines.append(artifact_matrix_source)
    if isinstance(artifact_matrix_payload, Mapping):
        artifact_matrix_by_target = _format_artifact_matrix_rollup(
            "Artifact matrix by target",
            artifact_matrix_payload.get("statusByTarget"),
        )
        if artifact_matrix_by_target:
            lines.append(artifact_matrix_by_target)
        artifact_matrix_by_source_backend = _format_artifact_matrix_rollup(
            "Artifact matrix by source backend",
            artifact_matrix_payload.get("statusBySourceBackend"),
        )
        if artifact_matrix_by_source_backend:
            lines.append(artifact_matrix_by_source_backend)
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
            "Source remaps by granularity",
            summary.get("sourceRemapsByGranularity"),
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
        _format_nested_count_rollup(
            "Artifact provenance by target and intermediate",
            summary.get("artifactProvenanceIntermediateByTarget"),
            include_zero=False,
        ),
        _format_nested_count_rollup(
            "Artifact provenance by variant and intermediate",
            summary.get("artifactProvenanceIntermediateByVariant"),
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
    include_dependencies_by_variant = _format_count_rollup(
        "Include dependencies by variant",
        summary.get("includeDependenciesByVariant"),
        include_zero=False,
    )
    if include_dependencies_by_variant:
        lines.append(include_dependencies_by_variant)
    lines.extend(
        _format_resolved_include_dependency_lines(payload.get("includeDependencies"))
    )
    lines.extend(
        _format_system_include_dependency_lines(payload.get("includeDependencies"))
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
    lines.extend(_format_skipped_source_lines(payload.get("skippedSources")))
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
    diagnostics_by_target = _format_count_rollup(
        "Diagnostics by target",
        summary.get("diagnosticsByTarget"),
        include_zero=False,
    )
    if diagnostics_by_target:
        lines.append(diagnostics_by_target)
    diagnostics_by_source_backend = _format_count_rollup(
        "Diagnostics by source backend",
        summary.get("diagnosticsBySourceBackend"),
        include_zero=False,
    )
    if diagnostics_by_source_backend:
        lines.append(diagnostics_by_source_backend)
    diagnostics_by_variant = _format_count_rollup(
        "Diagnostics by variant",
        summary.get("diagnosticsByVariant"),
        include_zero=False,
    )
    if diagnostics_by_variant:
        lines.append(diagnostics_by_variant)
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
    validation_diagnostics_by_target = _format_count_rollup(
        "Validation diagnostics by target",
        payload.get("validation", {}).get("diagnosticsByTarget"),
        include_zero=False,
    )
    if validation_diagnostics_by_target:
        lines.append(validation_diagnostics_by_target)
    validation_diagnostics_by_source_backend = _format_count_rollup(
        "Validation diagnostics by source backend",
        payload.get("validation", {}).get("diagnosticsBySourceBackend"),
        include_zero=False,
    )
    if validation_diagnostics_by_source_backend:
        lines.append(validation_diagnostics_by_source_backend)
    validation_diagnostics_by_variant = _format_count_rollup(
        "Validation diagnostics by variant",
        payload.get("validation", {}).get("diagnosticsByVariant"),
        include_zero=False,
    )
    if validation_diagnostics_by_variant:
        lines.append(validation_diagnostics_by_variant)
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
    validation_toolchain_runs_by_check_kind = _format_validation_run_rollup(
        "Validation toolchain runs by check kind",
        validation_toolchain_run_status_by_check_kind,
    )
    if validation_toolchain_runs_by_check_kind:
        lines.append(validation_toolchain_runs_by_check_kind)
    validation_toolchain_runs_by_tool = _format_validation_run_rollup(
        "Validation toolchain runs by tool",
        validation_toolchain_run_status_by_tool,
    )
    if validation_toolchain_runs_by_tool:
        lines.append(validation_toolchain_runs_by_tool)
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
        validation_artifacts_by_source_backend = _format_validation_artifact_rollup(
            "Validation artifacts by source backend",
            validation_artifact_status_by_source_backend,
        )
        if validation_artifacts_by_source_backend:
            lines.append(validation_artifacts_by_source_backend)
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
        source_sizes = _format_count_rollup(
            "Validation source sizes",
            validation_summary.get("sourceSizeStatusCounts"),
            include_zero=False,
        )
        if source_sizes:
            lines.append(source_sizes)
        generated_hashes = _format_count_rollup(
            "Validation generated hashes",
            validation_summary.get("generatedHashStatusCounts"),
            include_zero=False,
        )
        if generated_hashes:
            lines.append(generated_hashes)
        generated_sizes = _format_count_rollup(
            "Validation generated sizes",
            validation_summary.get("generatedSizeStatusCounts"),
            include_zero=False,
        )
        if generated_sizes:
            lines.append(generated_sizes)
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
    if (
        isinstance(external_corpus, Mapping)
        and external_corpus.get("available") is not False
    ):
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
            manifest = external_corpus.get("manifest")
            if isinstance(manifest, str) and manifest:
                lines.append(f"External corpus manifest: {manifest}")
            corpus_name = external_corpus.get("name")
            if isinstance(corpus_name, str) and corpus_name:
                lines.append(f"External corpus name: {corpus_name}")
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
                for field_name, label in (
                    ("sourceHashStatus", "source hash"),
                    ("sourceSizeStatus", "source size"),
                    ("generatedHashStatus", "generated hash"),
                    ("generatedSizeStatus", "generated size"),
                    ("sourceMapStatus", "source map"),
                    ("sourceRemapStatus", "source remap"),
                ):
                    value = artifact.get(field_name)
                    if value and value != "ok":
                        validation_details.append(f"{label}: {value}")
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
            lines.append(_format_project_diagnostic_line(diagnostic))
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
    if isinstance(migration, Mapping):
        migration_scope = migration.get("scope")
        if isinstance(migration_scope, str) and migration_scope:
            lines.append(f"Migration scope: {migration_scope}")
        migration_non_goals = migration.get("nonGoals")
        if isinstance(migration_non_goals, list):
            non_goal_labels = [
                non_goal
                for non_goal in migration_non_goals
                if isinstance(non_goal, str) and non_goal
            ]
            if non_goal_labels:
                lines.append(f"Migration non-goals: {', '.join(non_goal_labels)}")
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
            details = []
            severity = action.get("severity")
            if isinstance(severity, str) and severity:
                details.append(f"severity: {severity}")
            targets = action.get("targets")
            if isinstance(targets, list):
                target_names = [target for target in targets if isinstance(target, str)]
                if target_names:
                    details.append(f"targets: {', '.join(target_names)}")
            detail_suffix = f" [{'; '.join(details)}]" if details else ""
            lines.append(
                "- "
                f"{action.get('kind', 'unknown')}{detail_suffix}: "
                f"{action.get('message', '')}"
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
        max_source_map_artifacts=args.max_source_map_artifacts,
        max_artifact_matrix_artifacts=args.max_artifact_matrix_artifacts,
        max_artifact_provenance_artifacts=args.max_artifact_provenance_artifacts,
        max_define_processing_artifacts=args.max_define_processing_artifacts,
        max_skipped_sources=args.max_skipped_sources,
        max_include_path_processing_artifacts=(
            args.max_include_path_processing_artifacts
        ),
        max_include_dependencies=args.max_include_dependencies,
        max_validation_artifacts=args.max_validation_artifacts,
        max_toolchain_runs=args.max_toolchain_runs,
        max_migration_actions=args.max_migration_actions,
        max_external_corpus_entries=args.max_external_corpus_entries,
    )
    if args.format == "sarif":
        _write_json_payload(
            _format_project_diagnostics_sarif(
                payload, tool_name="CrossTL project report inspection"
            ),
            args.output,
        )
    elif args.format == "text":
        _write_text_payload(_format_project_report_inspection(payload), args.output)
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
    translate_parser.add_argument(
        "--output", "-o", help="Output file path; use '-' for stdout"
    )
    translate_parser.add_argument(
        "--no-format", action="store_true", help="Disable code formatting"
    )
    translate_parser.add_argument(
        "--source-backend", help="Override source parser backend"
    )
    translate_parser.add_argument(
        "--include-dir",
        action="append",
        type=_non_empty_project_arg("--include-dir"),
        help="Source parser include directory; repeatable",
    )
    translate_parser.add_argument(
        "--define",
        action="append",
        type=_project_define_arg,
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
        type=_non_empty_project_arg("--target"),
        help="Target backend to include in the scan report; repeatable",
    )
    _add_project_variant_args(scan_parser, action_label="scan")
    scan_parser.add_argument(
        "--output", "-o", help="Write JSON report to this path; use '-' for stdout"
    )
    scan_parser.set_defaults(func=_run_project_scan)

    translate_project_parser = subparsers.add_parser(
        "translate-project", help="Translate all discovered project units"
    )
    _add_project_scan_args(translate_project_parser)
    translate_project_parser.add_argument(
        "--target",
        "-b",
        action="append",
        type=_non_empty_project_arg("--target"),
        help="Target backend; repeatable. Defaults to config targets or cgl.",
    )
    translate_project_parser.add_argument(
        "--output-dir",
        default=None,
        type=_non_empty_project_arg("--output-dir"),
        help="Directory for translated artifacts",
    )
    translate_project_parser.add_argument(
        "--variant",
        action="append",
        type=_non_empty_project_arg("--variant"),
        help="Named project variant to translate; repeatable",
    )
    translate_project_parser.add_argument(
        "--report",
        help="Write project portability JSON report to this path; use '-' for stdout",
    )
    translate_project_parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate emitted artifacts and record available toolchains",
    )
    translate_project_parser.add_argument(
        "--run-toolchains",
        action="store_true",
        help=(
            "Run lightweight optional toolchain smoke checks when tools exist; "
            "implies artifact validation"
        ),
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
    validate_parser.add_argument(
        "--output", "-o", help="Write validation output; use '-' for stdout"
    )
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
    inspect_parser.add_argument(
        "--output", "-o", help="Write inspection output; use '-' for stdout"
    )
    inspect_parser.add_argument(
        "--max-diagnostics",
        type=_non_negative_int,
        default=20,
        help="Maximum diagnostics to include in the inspection summary",
    )
    inspect_parser.add_argument(
        "--max-failed-artifacts",
        type=_non_negative_int,
        default=20,
        help="Maximum failed artifacts to include in the inspection summary",
    )
    inspect_parser.add_argument(
        "--max-source-map-artifacts",
        type=_non_negative_int,
        default=20,
        help="Maximum source-map and source-remap artifact samples to include",
    )
    inspect_parser.add_argument(
        "--max-artifact-matrix-artifacts",
        type=_non_negative_int,
        default=20,
        help="Maximum missing or extra artifact-matrix samples to include",
    )
    inspect_parser.add_argument(
        "--max-artifact-provenance-artifacts",
        type=_non_negative_int,
        default=20,
        help="Maximum artifact provenance samples to include",
    )
    inspect_parser.add_argument(
        "--max-define-processing-artifacts",
        type=_non_negative_int,
        default=20,
        help="Maximum define-processing artifact samples to include",
    )
    inspect_parser.add_argument(
        "--max-skipped-sources",
        type=_non_negative_int,
        default=20,
        help="Maximum skipped source samples to include in the inspection summary",
    )
    inspect_parser.add_argument(
        "--max-include-path-processing-artifacts",
        type=_non_negative_int,
        default=20,
        help="Maximum include-path processing artifact samples to include",
    )
    inspect_parser.add_argument(
        "--max-include-dependencies",
        type=_non_negative_int,
        default=20,
        help="Maximum resolved or unresolved include dependency samples to include",
    )
    inspect_parser.add_argument(
        "--max-validation-artifacts",
        type=_non_negative_int,
        default=20,
        help="Maximum validation artifact samples to include in the inspection summary",
    )
    inspect_parser.add_argument(
        "--max-toolchain-runs",
        type=_non_negative_int,
        default=20,
        help="Maximum validation toolchain-run samples to include in the inspection summary",
    )
    inspect_parser.add_argument(
        "--max-migration-actions",
        type=_non_negative_int,
        default=20,
        help="Maximum migration action samples to include in the inspection summary",
    )
    inspect_parser.add_argument(
        "--max-external-corpus-entries",
        type=_non_negative_int,
        default=20,
        help="Maximum external corpus entry samples to include in the inspection summary",
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
        type=_non_empty_project_arg("--target"),
        help="Target backend to include in the report; repeatable",
    )
    _add_project_variant_args(report_parser, action_label="report")
    report_parser.add_argument(
        "--output", "-o", help="Write JSON report to this path; use '-' for stdout"
    )
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
    if not argv or argv[0] in {"-h", "--help"}:
        return False
    return argv[0] not in commands


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
