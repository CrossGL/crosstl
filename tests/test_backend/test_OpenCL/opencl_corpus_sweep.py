"""Utility sweep for the local real-world OpenCL corpus.

This is intentionally not a pytest module; it records the ad hoc corpus slices
used while hardening the source-only OpenCL backend.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from crosstl.backend.OpenCL.OpenCLLexer import OpenCLLexer
from crosstl.backend.OpenCL.OpenCLParser import OpenCLParser

OPENCL_SUFFIXES = {".cl", ".clh", ".opencl"}
DEFAULT_SUBDIRS = (
    "arrayfire/src/backend/opencl",
    "clblas/src/library/blas",
    "clblast/src/kernels",
    "pocl/lib/CL/devices",
    "pocl/examples",
)


def collect_default_files(root):
    files = []
    for subdir in DEFAULT_SUBDIRS:
        files.extend(collect_opencl_files(root / subdir))
    return files


def collect_edge_files(root):
    files = collect_opencl_files(root / "pocl/examples")
    files.extend(collect_opencl_files(root / "pocl/tests/compiler_unit"))
    files.append(root / "pocl/lib/CL/devices/level0/memfill.cl")
    return sorted(set(files))


def collect_opencl_files(base):
    return sorted(
        path
        for path in base.rglob("*")
        if path.is_file() and path.suffix in OPENCL_SUFFIXES
    )


def parse_file(path):
    source = path.read_text(errors="replace")
    tokens = OpenCLLexer(source, file_path=str(path)).tokenize()
    return OpenCLParser(tokens).parse()


def run_sweep(root, files):
    counts = Counter()
    details = defaultdict(list)

    for path in files:
        rel = path.relative_to(root).as_posix()
        try:
            ast = parse_file(path)
        except Exception as exc:  # noqa: BLE001 - reports corpus diagnostics.
            counts["failed"] += 1
            details["failed"].append(
                f"{rel}: {type(exc).__name__}: {str(exc).splitlines()[0]}"
            )
            continue

        if ast.statements:
            counts["pass"] += 1
        else:
            counts["skipped"] += 1
            details["skipped"].append(f"{rel}: empty AST / source template")

    return counts, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--edge",
        action="store_true",
        help="Run the POCL edge slice instead of the default 207-file slice.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    files = collect_edge_files(root) if args.edge else collect_default_files(root)
    counts, details = run_sweep(root, files)

    mode = "EDGE" if args.edge else "DEFAULT"
    print(
        f"{mode} total {len(files)} pass {counts['pass']} "
        f"skipped {counts['skipped']} failed {counts['failed']}"
    )

    for category in ("skipped", "failed"):
        print(category)
        for item in details[category]:
            print(f"  {item}")


if __name__ == "__main__":
    main()
