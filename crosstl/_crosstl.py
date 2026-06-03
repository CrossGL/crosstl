"""High-level translation API and command-line entry point for CrossGL Translator."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from .translator.codegen import (
    backend_names,
    get_backend_extension,
    get_codegen,
    normalize_backend_name,
)
from .translator.plugin_loader import discover_backend_plugins
from .translator.source_registry import SOURCE_REGISTRY, register_default_sources

try:
    from .formatter import format_shader_code

    FORMATTER_AVAILABLE = True
except ImportError:
    FORMATTER_AVAILABLE = False


def translate(
    file_path: str,
    backend: str = "cgl",
    save_shader: Optional[str] = None,
    format_output: bool = True,
) -> str:
    """Translate a shader file to another language.

    Args:
        file_path (str): The path to the shader file
        backend (str, optional): The target language to translate to. Defaults to "cgl".
        save_shader (str, optional): The path to save the translated shader. Defaults to None.
        format_output (bool, optional): Whether to format the generated code. Defaults to True.

    Returns:
        str: The translated shader code
    """
    register_default_sources()
    discover_backend_plugins()
    backend = (backend or "cgl").strip().lower()

    source_spec = SOURCE_REGISTRY.get_by_extension(file_path)
    if not source_spec:
        supported = ", ".join(SOURCE_REGISTRY.extensions())
        raise ValueError(
            f"Unsupported shader file type: {file_path}. Supported: {supported}"
        )

    with open(file_path, encoding="utf-8", errors="replace") as file:
        shader_code = file.read()

    ast = source_spec.parse(shader_code, file_path=file_path)

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
    translate(
        args.input,
        backend=args.backend,
        save_shader=output_path,
        format_output=not args.no_format,
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


def _load_project_config_from_args(args):
    from .project import load_project_config

    return load_project_config(args.root, args.config)


def _run_project_scan(args):
    from .project import scan_project

    config = _load_project_config_from_args(args)
    report = scan_project(config).to_report(targets=args.target)
    _write_json_payload(report.to_json(), args.output)
    return 0


def _run_translate_project(args):
    from .project import translate_project

    config = _load_project_config_from_args(args)
    report = translate_project(
        config,
        targets=args.target,
        output_dir=args.output_dir,
        format_output=not args.no_format,
        validate=args.validate,
    )
    if args.report:
        report.write_json(Path(args.report))
        print(f"Wrote {args.report}")
    else:
        _write_json_payload(report.to_json())
    return 1 if report.to_json()["summary"]["failedCount"] else 0


def _run_validate_project(args):
    from .project import validate_project_report

    payload = validate_project_report(args.report, run_toolchains=args.run_toolchains)
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
    translate_parser.set_defaults(func=_run_single_file)

    scan_parser = subparsers.add_parser(
        "scan", help="Scan a repository for supported shader/GPU sources"
    )
    scan_parser.add_argument("root", help="Repository root to scan")
    scan_parser.add_argument("--config", help="Path to crosstl.toml")
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
    translate_project_parser.add_argument("root", help="Repository root to scan")
    translate_project_parser.add_argument("--config", help="Path to crosstl.toml")
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
    validate_parser.add_argument("--output", "-o", help="Write JSON validation report")
    validate_parser.add_argument(
        "--run-toolchains",
        action="store_true",
        help="Run lightweight optional toolchain smoke checks when tools exist",
    )
    validate_parser.set_defaults(func=_run_validate_project)

    report_parser = subparsers.add_parser(
        "report", help="Emit a scan-only project portability report"
    )
    report_parser.add_argument("root", help="Repository root to scan")
    report_parser.add_argument("--config", help="Path to crosstl.toml")
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
    commands = {"translate", "scan", "translate-project", "validate-project", "report"}
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
