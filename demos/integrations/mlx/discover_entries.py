#!/usr/bin/env python3
"""Enumerate host-visible entries in the pinned MLX Metal kernel tree."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from crosstl.translator.entry_discovery import ENTRY_DISCOVERY_AVAILABLE
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources

MLX_COMMIT = "846d176227a0ac13d2667e58d2bb68b322109ab0"
MLX_REPOSITORY = "https://github.com/ml-explore/mlx.git"
MLX_KERNEL_ROOT = Path("mlx/backend/metal/kernels")
EXPECTED_SOURCE_UNIT_COUNT = 42
EXPECTED_ENTRY_COUNT = 17319
REPORT_KIND = "crosstl-mlx-entry-discovery"
REPORT_SCHEMA_VERSION = 1


def _git_revision(root: Path) -> str:
    if not root.is_dir():
        raise RuntimeError(f"MLX checkout does not exist: {root}")
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or str(exc)
        raise RuntimeError(f"Cannot read MLX checkout revision: {detail}") from exc
    return completed.stdout.strip()


def _verify_checkout(root: Path, expected_commit: str) -> None:
    revision = _git_revision(root)
    if revision != expected_commit:
        raise RuntimeError(
            f"MLX checkout revision mismatch: expected {expected_commit}, "
            f"found {revision}"
        )


def discover_mlx_entries(mlx_root: Path) -> dict[str, Any]:
    """Discover entries in every Metal source unit under an MLX checkout."""
    register_default_sources()
    source_spec = SOURCE_REGISTRY.get("metal")
    if source_spec is None:
        raise RuntimeError("Metal source frontend is not registered")

    kernel_root = mlx_root / MLX_KERNEL_ROOT
    source_paths = sorted(kernel_root.rglob("*.metal"))
    units = []
    total_entries = 0
    total_diagnostics = 0
    for source_path in source_paths:
        relative_path = source_path.relative_to(mlx_root).as_posix()
        discovery = source_spec.discover_entry_points(
            source_path.read_text(encoding="utf-8"),
            file_path=str(source_path),
            include_paths=[str(mlx_root)],
        )
        if discovery.status != ENTRY_DISCOVERY_AVAILABLE:
            raise RuntimeError(
                f"Entry discovery is {discovery.status} for {relative_path}"
            )
        entries = [entry.to_json() for entry in discovery.entries]
        diagnostics = [diagnostic.to_json() for diagnostic in discovery.diagnostics]
        total_entries += len(entries)
        total_diagnostics += len(diagnostics)
        units.append(
            {
                "source": relative_path,
                "sourceBackend": discovery.source_backend,
                "status": discovery.status,
                "entryCount": len(entries),
                "entries": entries,
                "diagnosticCount": len(diagnostics),
                "diagnostics": diagnostics,
            }
        )

    return {
        "schemaVersion": REPORT_SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "repository": {
            "url": MLX_REPOSITORY,
            "commit": MLX_COMMIT,
        },
        "sourceRoot": MLX_KERNEL_ROOT.as_posix(),
        "unitCount": len(units),
        "entryCount": total_entries,
        "diagnosticCount": total_diagnostics,
        "units": units,
    }


def _unit_entries(report: dict[str, Any], source: str) -> list[str]:
    for unit in report["units"]:
        if unit["source"] == source:
            return [entry["name"] for entry in unit["entries"]]
    raise RuntimeError(f"Entry discovery report is missing {source}")


def verify_pinned_discovery(
    report: dict[str, Any],
    *,
    expected_unit_count: int = EXPECTED_SOURCE_UNIT_COUNT,
    expected_entry_count: int = EXPECTED_ENTRY_COUNT,
) -> None:
    """Verify the pinned corpus totals and known false-positive boundaries."""
    if report["unitCount"] != expected_unit_count:
        raise RuntimeError(
            f"MLX Metal source unit count changed: expected {expected_unit_count}, "
            f"found {report['unitCount']}"
        )
    if report["entryCount"] != expected_entry_count:
        raise RuntimeError(
            f"MLX Metal entry count changed: expected {expected_entry_count}, "
            f"found {report['entryCount']}"
        )
    if report["diagnosticCount"] != 0:
        raise RuntimeError(
            "Pinned MLX entry discovery produced "
            f"{report['diagnosticCount']} diagnostics"
        )

    random_source = (MLX_KERNEL_ROOT / "random.metal").as_posix()
    random_entries = _unit_entries(report, random_source)
    if random_entries != ["rbitsc", "rbits"]:
        raise RuntimeError(
            "random.metal entry discovery changed: "
            f"expected ['rbitsc', 'rbits'], found {random_entries}"
        )

    binary_source = (MLX_KERNEL_ROOT / "binary.metal").as_posix()
    if "binary_int" in _unit_entries(report, binary_source):
        raise RuntimeError("binary.metal included the commented binary_int example")

    fft_source = (MLX_KERNEL_ROOT / "fft.metal").as_posix()
    if "complex_mul" in _unit_entries(report, fft_source):
        raise RuntimeError("fft.metal included the non-entry complex_mul helper")


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover entries in the pinned MLX Metal kernel tree."
    )
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-commit", default=MLX_COMMIT)
    parser.add_argument(
        "--expected-unit-count",
        type=int,
        default=EXPECTED_SOURCE_UNIT_COUNT,
    )
    parser.add_argument(
        "--expected-entry-count",
        type=int,
        default=EXPECTED_ENTRY_COUNT,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mlx_root = args.mlx_root.resolve()
    try:
        _verify_checkout(mlx_root, args.expected_commit)
        report = discover_mlx_entries(mlx_root)
        report["repository"]["commit"] = args.expected_commit
        verify_pinned_discovery(
            report,
            expected_unit_count=args.expected_unit_count,
            expected_entry_count=args.expected_entry_count,
        )
        output = args.output or (
            mlx_root / ".crosstl-mlx-porting" / "reports" / "entry-discovery.json"
        )
        _write_report(output, report)
    except (OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "report": str(output),
                "unitCount": report["unitCount"],
                "entryCount": report["entryCount"],
                "diagnosticCount": report["diagnosticCount"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
