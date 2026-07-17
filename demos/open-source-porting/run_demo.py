#!/usr/bin/env python3
"""Regenerate and verify open-source project translation demos."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from crosstl.project import load_project_config
from crosstl.project.directx_toolchain import (
    dxc_compiler_arguments_for_source,
    dxc_profile_for_source,
)

DEMO_ROOT = Path(__file__).resolve().parent
CASE_ROOT = DEMO_ROOT / "cases"
OUTPUT_DIR_NAME = "crosstl-out"
REPORT_NAME = "portability-report.json"
CORPUS_MANIFEST_NAME = "corpus.json"
CORPUS_SCHEMA_VERSION = 1
CORPUS_ENTRY_ROLES = ("translation_unit", "support_file")
DEFAULT_CORPUS_ENTRY_ROLE = "translation_unit"
DIRECTX_DEFAULT_ENTRY_PROFILES = (
    ("VSMain", "vs_6_0"),
    ("PSMain", "ps_6_0"),
    ("CSMain", "cs_6_0"),
)
DIRECTX_COMPILE_OVERRIDES = {
    "demos/open-source-porting/cases/diligent-samples-vrs-cube/"
    "crosstl-out/directx/CubeFDM_fs.hlsl": (("PSMain", "ps_6_4"),),
}


def _case_dirs() -> list[Path]:
    return sorted(
        path for path in CASE_ROOT.iterdir() if (path / "crosstl.toml").is_file()
    )


def _case_targets(case_dir: Path) -> list[str]:
    targets = list(load_project_config(case_dir).targets)
    if not targets:
        raise ValueError(f"{case_dir}/crosstl.toml does not declare project targets")
    return targets


def _translation_unit_paths(case_dir: Path) -> set[str]:
    patterns = load_project_config(case_dir).include_patterns
    if isinstance(patterns, str):
        patterns = [patterns]
    return {str(pattern).replace("\\", "/") for pattern in patterns}


def _load_corpus_manifest(case_dir: Path) -> dict:
    manifest = json.loads((case_dir / CORPUS_MANIFEST_NAME).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{CORPUS_MANIFEST_NAME} must be a JSON object")
    return manifest


def _corpus_entry_role(entry: dict) -> object:
    return entry.get("role", DEFAULT_CORPUS_ENTRY_ROLE)


def _corpus_adjustment_problems(manifest: dict) -> list[str]:
    problems: list[str] = []
    adjustments = manifest.get("adjustments")
    if adjustments is not None:
        if not isinstance(adjustments, list) or not adjustments:
            problems.append("adjustments must be a non-empty list when present")
        else:
            for index, adjustment in enumerate(adjustments):
                label = f"adjustments[{index}]"
                if not isinstance(adjustment, dict):
                    problems.append(f"{label} must be an object")
                    continue
                for field_name in ("kind", "summary"):
                    value = adjustment.get(field_name)
                    if not isinstance(value, str) or not value.strip():
                        problems.append(
                            f"{label} {field_name} must be a non-empty string"
                        )
                detail = adjustment.get("detail")
                if detail is not None and (
                    not isinstance(detail, str) or not detail.strip()
                ):
                    problems.append(
                        f"{label} detail must be a non-empty string when present"
                    )
    out_of_scope = manifest.get("outOfScope")
    if out_of_scope is not None:
        if not isinstance(out_of_scope, list) or not out_of_scope:
            problems.append("outOfScope must be a non-empty list when present")
        elif not all(isinstance(item, str) and item.strip() for item in out_of_scope):
            problems.append("outOfScope entries must be non-empty strings")
    return problems


def _corpus_manifest_problems(case_dir: Path) -> list[str]:
    try:
        manifest = _load_corpus_manifest(case_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"could not read {CORPUS_MANIFEST_NAME}: {exc}"]

    problems: list[str] = []
    if manifest.get("schemaVersion") != CORPUS_SCHEMA_VERSION:
        problems.append(f"schemaVersion must be {CORPUS_SCHEMA_VERSION}")

    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        problems.append("entries must be a non-empty list")
        entries = []

    translation_units = _translation_unit_paths(case_dir)
    for index, entry in enumerate(entries):
        label = f"entries[{index}]"
        if not isinstance(entry, dict):
            problems.append(f"{label} must be an object")
            continue
        role = _corpus_entry_role(entry)
        if role not in CORPUS_ENTRY_ROLES:
            problems.append(
                f"{label} role must be one of {', '.join(CORPUS_ENTRY_ROLES)}"
            )
            continue
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            problems.append(f"{label} path must be a non-empty string")
            continue
        translated = path.replace("\\", "/") in translation_units
        if role == "translation_unit" and not translated:
            problems.append(
                f"{label} role=translation_unit but '{path}' is not a crosstl.toml "
                f"translation unit (include={sorted(translation_units)})"
            )
        elif role == "support_file" and translated:
            problems.append(
                f"{label} role=support_file but '{path}' is a crosstl.toml "
                f"translation unit; support files must not produce target output"
            )

    problems.extend(_corpus_adjustment_problems(manifest))
    return problems


def _validate_corpus_manifest(case_dir: Path) -> None:
    problems = _corpus_manifest_problems(case_dir)
    if problems:
        raise SystemExit(
            f"{case_dir.name}: invalid {CORPUS_MANIFEST_NAME}: " + "; ".join(problems)
        )


def _repo_relative(path: Path) -> str:
    return path.relative_to(DEMO_ROOT.parents[1]).as_posix()


def _case_names_for_target(target: str) -> list[str]:
    return [
        case_dir.name for case_dir in _case_dirs() if target in _case_targets(case_dir)
    ]


def _case_args_for_target(target: str) -> list[str]:
    args: list[str] = []
    for case_name in _case_names_for_target(target):
        args.extend(["--case", case_name])
    return args


def _artifact_paths_for_target(target: str, suffix: str = "") -> list[str]:
    paths: list[str] = []
    for case_dir in _case_dirs():
        if target not in _case_targets(case_dir):
            continue
        target_dir = case_dir / OUTPUT_DIR_NAME / target
        if not target_dir.is_dir():
            continue
        for path in sorted(target_dir.rglob("*")):
            if path.is_file() and (not suffix or path.name.endswith(suffix)):
                paths.append(_repo_relative(path))
    return paths


def _directx_compile_jobs() -> list[tuple[str, str, str]]:
    jobs: list[tuple[str, str, str]] = []
    for path in _artifact_paths_for_target("directx", ".hlsl"):
        source = (DEMO_ROOT.parents[1] / path).read_text(encoding="utf-8")
        override = DIRECTX_COMPILE_OVERRIDES.get(path)
        if override is not None:
            entry_profiles = override
        else:
            entry_profiles = tuple(
                (entry, profile)
                for entry, profile in DIRECTX_DEFAULT_ENTRY_PROFILES
                if re.search(rf"\b{re.escape(entry)}\s*\(", source)
            )
        jobs.extend(
            (path, entry, dxc_profile_for_source(profile, source))
            for entry, profile in entry_profiles
        )
    return jobs


def _directx_compile_command(
    path: str,
    entry: str,
    profile: str,
    *,
    output_dir: Path,
) -> list[str]:
    source = (DEMO_ROOT.parents[1] / path).read_text(encoding="utf-8")
    profile = dxc_profile_for_source(profile, source)
    output_name = re.sub(r"[/.:\\]", "_", f"{path}_{entry}_{profile}") + ".dxil"
    return [
        "dxc",
        "-T",
        profile,
        *dxc_compiler_arguments_for_source(source),
        "-E",
        entry,
        path,
        "-Fo",
        str(output_dir / output_name),
    ]


def _compile_directx_references(output_dir: Path) -> None:
    if shutil.which("dxc") is None:
        raise SystemExit("dxc is required to compile DirectX reference artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, entry, profile in _directx_compile_jobs():
        _run(
            _directx_compile_command(
                path,
                entry,
                profile,
                output_dir=output_dir,
            )
        )


def _selected_case_dirs(case_names: list[str]) -> list[Path]:
    cases = {path.name: path for path in _case_dirs()}
    if not case_names:
        return [cases[name] for name in sorted(cases)]
    missing = sorted(name for name in case_names if name not in cases)
    if missing:
        raise SystemExit(f"Unknown demo case(s): {', '.join(missing)}")
    return [cases[name] for name in case_names]


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    print("+", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed


def _translate_case(
    case_dir: Path,
    *,
    work_dir: Path,
    targets: list[str],
    update: bool,
) -> Path:
    output_dir = work_dir / OUTPUT_DIR_NAME
    report_path = (
        Path(tempfile.mkdtemp(prefix=f"{case_dir.name}-report-")) / REPORT_NAME
        if update
        else output_dir / REPORT_NAME
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)

    command = [
        sys.executable,
        "-m",
        "crosstl._crosstl",
        "translate-project",
        str(work_dir),
        "--output-dir",
        str(output_dir),
        "--report",
        str(report_path),
        "--validate",
        "--no-format",
    ]
    for target in targets:
        command.extend(["--target", target])
    _run(command)
    return report_path


def _artifact_toolchain_failures(
    payload: dict, selected_targets: list[str]
) -> list[str]:
    """Return "target: artifact" records lacking a successful toolchain run.

    Per-artifact granularity: an available target with several translated
    artifacts must have a successful toolchain run for every artifact, not
    merely one. Artifacts that only have failed/skipped runs are reported.
    """
    validation = payload.get("validation")
    runs = validation.get("toolchainRuns") if isinstance(validation, dict) else None
    if not isinstance(runs, list):
        return []
    selected = set(selected_targets)
    ok_by_artifact: dict[tuple[str, str], bool] = {}
    for run in runs:
        if not isinstance(run, dict) or run.get("checkKind") != "artifact":
            continue
        target = run.get("target")
        path = run.get("path")
        if not isinstance(target, str) or target not in selected:
            continue
        if not isinstance(path, str) or not path:
            continue
        key = (target, path)
        ok_by_artifact.setdefault(key, False)
        if run.get("status") == "ok":
            ok_by_artifact[key] = True
    return sorted(
        f"{target}: {path}"
        for (target, path), is_ok in ok_by_artifact.items()
        if not is_ok
    )


def _validate_report(
    report_path: Path,
    *,
    run_toolchains: bool,
    require_toolchain_runs: bool,
    selected_targets: list[str],
    reports_dir: Path | None,
    case_name: str,
) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "crosstl._crosstl",
        "validate-project",
        str(report_path),
        "--format",
        "json",
    ]
    if run_toolchains:
        command.append("--run-toolchains")
    print("+", " ".join(command))
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        raise SystemExit(completed.returncode)
    payload = json.loads(completed.stdout)
    if not payload.get("success", False):
        raise SystemExit(f"{case_name}: project report validation failed")
    if run_toolchains and require_toolchain_runs:
        counts = payload.get("toolchainRunStatusCounts", {})
        ok_count = int(counts.get("ok", 0)) if isinstance(counts, dict) else 0
        failed_count = int(counts.get("failed", 0)) if isinstance(counts, dict) else 0
        run_counts_by_target = payload.get("toolchainRunStatusByTarget", {})
        target_failures = []
        for target in selected_targets:
            target_counts = (
                run_counts_by_target.get(target, {})
                if isinstance(run_counts_by_target, dict)
                else {}
            )
            target_ok_count = (
                int(target_counts.get("okCount", 0))
                if isinstance(target_counts, dict)
                else 0
            )
            target_failed_count = (
                int(target_counts.get("failedCount", 0))
                if isinstance(target_counts, dict)
                else 0
            )
            if target_ok_count <= 0 or target_failed_count:
                target_failures.append(
                    f"{target}: ok={target_ok_count}, failed={target_failed_count}"
                )
        artifact_failures = _artifact_toolchain_failures(payload, selected_targets)
        if ok_count <= 0 or failed_count or target_failures or artifact_failures:
            raise SystemExit(
                f"{case_name}: expected every selected translated artifact to "
                f"have a successful toolchain run and no failed runs, "
                f"got ok={ok_count}, failed={failed_count}, "
                f"targets={target_failures}, artifacts={artifact_failures}"
            )
    if reports_dir is not None:
        reports_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(report_path, reports_dir / f"{case_name}-{REPORT_NAME}")
        (reports_dir / f"{case_name}-validation.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    diagnostic_counts = payload.get("diagnosticCounts", {})
    run_counts = payload.get("toolchainRunStatusCounts", {})
    print(
        f"{case_name}: validation success "
        f"(diagnostics={diagnostic_counts}, toolchainRuns={run_counts})"
    )
    return payload


def _artifact_files(output_dir: Path, targets: list[str]) -> dict[Path, Path]:
    selected = set(targets)
    files: dict[Path, Path] = {}
    if not output_dir.exists():
        return files
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == REPORT_NAME:
            continue
        relative = path.relative_to(output_dir)
        if relative.parts and relative.parts[0] in selected:
            files[relative] = path
    return files


def _normalize_artifacts(output_dir: Path, targets: list[str]) -> None:
    for path in _artifact_files(output_dir, targets).values():
        _normalize_artifact(path)


def _normalize_artifact(path: Path) -> None:
    data = path.read_bytes()
    if b"\0" in data:
        return
    normalized = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    stripped = normalized.rstrip(b" \t\n")
    if not stripped:
        return
    normalized = stripped + b"\n"
    if normalized != data:
        path.write_bytes(normalized)


def _comparison_bytes(path: Path) -> bytes:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if path.suffix == ".json":
        try:
            payload = json.loads(data.rstrip(b"\n").decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return data
        # Source-map offsets (offset/endOffset/length) are compared by
        # default so generated offset drift fails demo verification. Demo
        # sources and artifacts are pinned to LF via .gitattributes so the
        # byte offsets stay identical across Linux, macOS, and Windows.
        return json.dumps(
            _normalize_json_paths(payload),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    if b"\0" in data:
        return data
    return b"\n".join(line.rstrip(b" \t") for line in data.rstrip(b"\n").split(b"\n"))


def _normalize_json_paths(value: object) -> object:
    if isinstance(value, str):
        return value.replace("\\", "/")
    if isinstance(value, list):
        return [_normalize_json_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_json_paths(item) for key, item in value.items()}
    return value


def _compare_artifacts(case_dir: Path, work_dir: Path, targets: list[str]) -> None:
    expected = _artifact_files(case_dir / OUTPUT_DIR_NAME, targets)
    actual = _artifact_files(work_dir / OUTPUT_DIR_NAME, targets)
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise SystemExit(
            f"{case_dir.name}: artifact set mismatch; "
            f"missing={missing}, extra={extra}"
        )
    changed = [
        relative
        for relative, expected_path in expected.items()
        if _comparison_bytes(expected_path) != _comparison_bytes(actual[relative])
    ]
    if changed:
        _print_artifact_diffs(case_dir, work_dir, changed)
        raise SystemExit(
            f"{case_dir.name}: generated artifacts differ from references: "
            + ", ".join(str(path) for path in changed)
        )


def _print_artifact_diffs(
    case_dir: Path, work_dir: Path, changed: list[Path], *, max_lines: int = 160
) -> None:
    for relative in changed:
        expected_path = case_dir / OUTPUT_DIR_NAME / relative
        actual_path = work_dir / OUTPUT_DIR_NAME / relative
        expected_text = _comparison_bytes(expected_path).decode("utf-8", "replace")
        actual_text = _comparison_bytes(actual_path).decode("utf-8", "replace")
        diff = list(
            difflib.unified_diff(
                expected_text.splitlines(),
                actual_text.splitlines(),
                fromfile=f"expected/{relative}",
                tofile=f"actual/{relative}",
                lineterm="",
            )
        )
        if not diff:
            continue
        print(f"{case_dir.name}: diff for {relative}", file=sys.stderr)
        truncated = diff[:max_lines]
        for line in truncated:
            print(line, file=sys.stderr)
        if len(diff) > max_lines:
            print(
                f"... diff truncated after {max_lines} of {len(diff)} lines",
                file=sys.stderr,
            )


def _copy_case(case_dir: Path, temp_root: Path) -> Path:
    work_dir = temp_root / case_dir.name
    shutil.copytree(
        case_dir,
        work_dir,
        ignore=shutil.ignore_patterns(OUTPUT_DIR_NAME),
    )
    return work_dir


def _run_case(
    case_dir: Path,
    *,
    targets: list[str],
    update: bool,
    run_toolchains: bool,
    require_toolchain_runs: bool,
    reports_dir: Path | None,
) -> None:
    _validate_corpus_manifest(case_dir)
    configured_targets = _case_targets(case_dir)
    selected_targets = targets or configured_targets
    unsupported = sorted(set(selected_targets) - set(configured_targets))
    if unsupported:
        raise SystemExit(
            f"{case_dir.name}: target(s) not configured for this demo case: "
            + ", ".join(unsupported)
        )

    if update:
        report_path = _translate_case(
            case_dir,
            work_dir=case_dir,
            targets=selected_targets,
            update=True,
        )
        _validate_report(
            report_path,
            run_toolchains=run_toolchains,
            require_toolchain_runs=require_toolchain_runs,
            selected_targets=selected_targets,
            reports_dir=reports_dir,
            case_name=case_dir.name,
        )
        _normalize_artifacts(case_dir / OUTPUT_DIR_NAME, selected_targets)
        shutil.rmtree(report_path.parent, ignore_errors=True)
        print(f"{case_dir.name}: updated {OUTPUT_DIR_NAME}")
        return

    with tempfile.TemporaryDirectory(prefix="crosstl-demo-") as temp_name:
        work_dir = _copy_case(case_dir, Path(temp_name))
        report_path = _translate_case(
            case_dir,
            work_dir=work_dir,
            targets=selected_targets,
            update=False,
        )
        _validate_report(
            report_path,
            run_toolchains=run_toolchains,
            require_toolchain_runs=require_toolchain_runs,
            selected_targets=selected_targets,
            reports_dir=reports_dir,
            case_name=case_dir.name,
        )
        _normalize_artifacts(work_dir / OUTPUT_DIR_NAME, selected_targets)
        _compare_artifacts(case_dir, work_dir, selected_targets)
        print(f"{case_dir.name}: verified {', '.join(selected_targets)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", default=[], help="Case name to run")
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Target backend subset to run; defaults to each case config",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Regenerate in a temporary copy and compare checked-in artifacts",
    )
    mode.add_argument(
        "--update",
        action="store_true",
        help="Regenerate checked-in reference artifacts",
    )
    mode.add_argument(
        "--emit-case-args",
        metavar="TARGET",
        help="Print run_demo.py --case arguments for cases configured for TARGET",
    )
    mode.add_argument(
        "--emit-artifact-paths",
        metavar="TARGET",
        help="Print checked-in artifact paths for TARGET",
    )
    mode.add_argument(
        "--emit-directx-compile-jobs",
        action="store_true",
        help="Print DirectX compile jobs as '<path> <entry> <profile>' records",
    )
    mode.add_argument(
        "--compile-directx-references",
        action="store_true",
        help="Compile every checked-in DirectX reference with DXC",
    )
    parser.add_argument(
        "--artifact-suffix",
        default="",
        help="Optional suffix filter for --emit-artifact-paths",
    )
    parser.add_argument(
        "--run-toolchains",
        action="store_true",
        help="Run configured validation toolchain smoke checks",
    )
    parser.add_argument(
        "--require-toolchain-runs",
        action="store_true",
        help="Require at least one successful toolchain smoke run per case",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        help="Optional directory for generated portability and validation reports",
    )
    parser.add_argument(
        "--compiler-output-dir",
        type=Path,
        help="Output directory for --compile-directx-references",
    )
    args = parser.parse_args(argv)

    if args.emit_case_args:
        print(" ".join(_case_args_for_target(args.emit_case_args)))
        return 0
    if args.emit_artifact_paths:
        for path in _artifact_paths_for_target(
            args.emit_artifact_paths,
            args.artifact_suffix,
        ):
            print(path)
        return 0
    if args.emit_directx_compile_jobs:
        for path, entry, profile in _directx_compile_jobs():
            print(path, entry, profile)
        return 0
    if args.compile_directx_references:
        if args.compiler_output_dir is None:
            parser.error("--compile-directx-references requires --compiler-output-dir")
        _compile_directx_references(args.compiler_output_dir)
        return 0
    if args.compiler_output_dir is not None:
        parser.error(
            "--compiler-output-dir is only valid with --compile-directx-references"
        )

    update = bool(args.update)
    for case_dir in _selected_case_dirs(args.case):
        _run_case(
            case_dir,
            targets=list(args.target),
            update=update,
            run_toolchains=bool(args.run_toolchains),
            require_toolchain_runs=bool(args.require_toolchain_runs),
            reports_dir=args.reports_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
