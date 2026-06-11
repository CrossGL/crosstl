#!/usr/bin/env python3
"""Regenerate and verify open-source project translation demos."""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parent
CASE_ROOT = DEMO_ROOT / "cases"
OUTPUT_DIR_NAME = "crosstl-out"
REPORT_NAME = "portability-report.json"
TARGET_RE = re.compile(r"(?m)^\s*targets\s*=\s*(\[[^\]]*\])")


def _case_dirs() -> list[Path]:
    return sorted(
        path for path in CASE_ROOT.iterdir() if (path / "crosstl.toml").is_file()
    )


def _case_targets(case_dir: Path) -> list[str]:
    config_text = (case_dir / "crosstl.toml").read_text(encoding="utf-8")
    match = TARGET_RE.search(config_text)
    if match is None:
        raise ValueError(f"{case_dir}/crosstl.toml does not declare project targets")
    targets = ast.literal_eval(match.group(1))
    if not isinstance(targets, list) or not all(
        isinstance(target, str) for target in targets
    ):
        raise ValueError(f"{case_dir}/crosstl.toml targets must be a string list")
    return targets


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
        if ok_count <= 0 or failed_count or target_failures:
            raise SystemExit(
                f"{case_name}: expected at least one successful toolchain run "
                f"for each selected target and no failed runs, "
                f"got ok={ok_count}, failed={failed_count}, "
                f"targets={target_failures}"
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
        if path.name.endswith(".source-remap.json"):
            payload = _drop_source_map_offsets(payload)
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


def _drop_source_map_offsets(value: object) -> object:
    if isinstance(value, list):
        return [_drop_source_map_offsets(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _drop_source_map_offsets(item)
            for key, item in value.items()
            if key not in {"endOffset", "length", "offset"}
        }
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
    args = parser.parse_args(argv)

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
