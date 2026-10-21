"""Run the unchanged pinned ``test_qvm`` on CPU and the translated native backend.

The test multiplies vectors by untransposed affine-quantized matrices and
compares each product with ``x @ dequantize(w)``. Through the portable host the
quantized products reach the translated ``affine_qvm`` kernels, the reference
products the translated ``gemv_t`` kernel, and the error bound a translated
whole-array maximum. Every package is translated before the native worker
starts; the maximum is packaged on demand at the widths the run launches.
"""

import argparse
import json
from collections import Counter
from itertools import product
from pathlib import Path

from demos.integrations.mlx.portable_host import gemv_layout, matmul_layout
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.verify import upstream_test_sources
from demos.integrations.mlx.portable_host.verify_upstream_quantization import (
    execute,
    worker,
    write_json,
)

UPSTREAM_TESTS = ("test_quantized.TestQuantized.test_qvm",)
# test_qvm's product(group_size, bits, M, N, B), in the test's order.
GRID = ((128, 64, 32), (2, 3, 4, 5, 6, 8), (32, 128, 256), (128, 256, 67), (0, 1, 3, 8))
REDUCTIONS = ("all_reduce_maxfloat32",)
TIMEOUTS = {"random": 600, "packages": 2400, "cpu": 600, "native": 7200}


def _row_strides(shape):
    strides, step = [], 1
    for extent in reversed(shape):
        strides.append(step)
        step *= extent
    return strides[::-1]


def products():
    """``(group, bits, M, N, B, x_shape, w_shape)`` for every compared product."""
    for group, bits, m, n, b in product(*GRID):
        if m < group:
            continue
        x_shape = (1, n) if b == 0 else (b, 1, n)
        w_shape = (n, m) if b == 0 else (b, n, m)
        yield group, bits, m, n, b, x_shape, w_shape


def expected_dispatches():
    """The affine and matrix-vector entries the test dispatches, with counts.

    Each product quantizes and dequantizes ``w`` once, multiplies through
    QuantizedMatmul's untransposed route and forms the reference with Matmul.
    """
    counts = Counter()
    for group, bits, m, n, b, x_shape, w_shape in products():
        for operation in ("quantize", "dequantize"):
            counts[f"affine_{operation}_float_gs_{group}_b_{bits}"] += 1
        counts[
            matmul_layout.select(
                "float32", group, bits, 1, m, n, max(b, 1), transpose=False
            )
        ] += 1
        route = gemv_layout.route(
            "float32", x_shape, _row_strides(x_shape), w_shape, _row_strides(w_shape)
        )
        counts[route["entry"]] += 1
    return counts


def package_entries():
    """The quantized and floating matrix-vector entries to translate ahead."""
    counts = expected_dispatches()
    return (
        sorted(entry for entry in counts if entry.startswith("affine_qvm_")),
        sorted(entry for entry in counts if entry.startswith("gemv_")),
    )


def validate_trace(trace, target, count):
    if len(trace) != count or any(record.get("target") != target for record in trace):
        raise RuntimeError(
            "Upstream test_qvm trace is incomplete or uses another target"
        )
    observed = Counter(
        record["entry"]
        for record in trace
        if record["entry"].startswith(("affine_", "gemv_"))
    )
    if observed != expected_dispatches():
        raise RuntimeError(
            "Upstream test_qvm did not route every product through its translated kernel"
        )
    if not any(record["entry"] in REDUCTIONS for record in trace):
        raise RuntimeError(
            "Upstream test_qvm did not evaluate its error bound natively"
        )


def prepare_packages(root, packages, reductions, jobs):
    """Translate every matrix-vector entry and index the on-demand maximum."""
    from demos.integrations.mlx.portable_host import (
        gemv_packages,
        matmul_packages,
        reduction_packages,
    )

    target = json.loads((Path(packages) / "index.json").read_text())["target"]
    qvm, gemv = package_entries()
    records = matmul_packages.build_entries(root, packages, target, qvm, jobs)
    records += gemv_packages.build_entries(root, packages, target, gemv, jobs)
    reduction_packages.build_packages(
        root,
        reductions,
        target,
        family="all",
        entries=REDUCTIONS,
        on_demand=True,
    )
    for record in records:
        print(json.dumps(record), flush=True)


def run(root, packages, output, jobs=3):
    root, packages, output = (Path(path).resolve() for path in (root, packages, output))
    output.mkdir(parents=True, exist_ok=False)
    preparation = verify_prepared(root)
    sources = upstream_test_sources(root, UPSTREAM_TESTS)
    target = json.loads((packages / "index.json").read_text())["target"]
    write_json(
        output / "inputs.json",
        {"adaptation": preparation, "sources": sources, "target": target},
    )
    random, reductions = output / "random-packages", output / "reductions"
    module = ["-m", __package__ + ".verify_upstream_qvm"]
    common = ["--mlx-root", str(root), "--packages", str(packages)]
    commands = [
        (
            "random",
            TIMEOUTS["random"],
            [
                "-m",
                "demos.integrations.mlx.portable_host.random_packages",
                "--mlx-root",
                str(root),
                "--target",
                target,
                "--output-dir",
                str(random),
            ],
        ),
        (
            "packages",
            TIMEOUTS["packages"],
            [*module, "--prepare", *common, "--jobs", str(jobs)]
            + ["--output-dir", str(reductions)],
        ),
    ]
    for mode in ("cpu", "native"):
        commands.append(
            (
                mode,
                TIMEOUTS[mode],
                [*module, "--worker", mode, *common]
                + ["--reductions", str(reductions), "--random", str(random)]
                + ["--output-dir", str(output / mode)],
            )
        )
    failures, results = execute(output, commands, UPSTREAM_TESTS, "MLX test_qvm")
    after = verify_prepared(root)
    write_json(output / "adaptation-after.json", after)
    if preparation != after or sources != upstream_test_sources(root, UPSTREAM_TESTS):
        raise RuntimeError("MLX sources changed during upstream test_qvm verification")
    if failures:
        raise RuntimeError(
            f"Upstream test_qvm workers failed: {', '.join(failures)}; inspect {output}"
        )
    trace = [
        json.loads(line)
        for line in (output / "native/trace.jsonl").read_text().splitlines()
    ]
    validate_trace(trace, target, results["native"]["nativeDispatches"])
    evidence = {
        "commit": COMMIT,
        "target": target,
        "upstreamTestSources": sources,
        "results": results,
        "entries": dict(sorted(Counter(record["entry"] for record in trace).items())),
        "sourceUnchanged": True,
        "fullUpstreamSuite": False,
    }
    write_json(output / "evidence.json", evidence)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reductions", type=Path)
    parser.add_argument("--random", type=Path)
    parser.add_argument("--jobs", type=int, default=3)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--worker", choices=("cpu", "native"))
    mode.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare_packages(args.mlx_root, args.packages, args.output_dir, args.jobs)
    elif args.worker:
        worker(args, UPSTREAM_TESTS)
    else:
        run(args.mlx_root, args.packages, args.output_dir, args.jobs)
