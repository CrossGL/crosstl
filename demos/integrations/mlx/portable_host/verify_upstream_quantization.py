"""Run the unchanged pinned affine test on CPU and the translated native backend."""

import argparse
import json
import os
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path

from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.trace_records import read_trace_summary
from demos.integrations.mlx.portable_host.verify import upstream_test_sources

UPSTREAM_TESTS = ("test_quantized.TestQuantized.test_quantize_dequantize",)
# Worker deadlines in seconds. The native run makes about 430 dispatches; CI
# measured 5.6 s per DirectX dispatch and 4.4 s per OpenGL dispatch.
TIMEOUTS = {"random": 600, "cpu": 180, "native": 5400}
WORKER_MODES = ("cpu", "native")


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def validate_result(record, mode, tests=UPSTREAM_TESTS):
    if (
        record.get("mode") != mode
        or record.get("tests") != list(tests)
        or type(record.get("testsRun")) is not int
        or record["testsRun"] != len(tests)
        or record.get("success") is not True
        or any(record.get(key) != [] for key in ("skipped", "errors", "failures"))
        or type(record.get("nativeDispatches")) is not int
        or (
            record["nativeDispatches"] <= 0
            if mode == "native"
            else record["nativeDispatches"] != 0
        )
    ):
        raise RuntimeError(
            f"The unchanged upstream test {', '.join(tests)} did not pass"
        )


def validate_trace(trace, target, count):
    if len(trace) != count or any(record.get("target") != target for record in trace):
        raise RuntimeError(
            "Upstream quantization trace is incomplete or uses another target"
        )
    observed = Counter(
        record["entry"] for record in trace if record["entry"].startswith("affine_")
    )
    expected = {
        f"affine_{operation}_float_gs_{group}_b_{bits}": (
            3 if (group, bits) == (32, 4) else 2
        )
        for operation in ("quantize", "dequantize")
        for group in (32, 64, 128)
        for bits in (2, 3, 4, 5, 6, 8)
    }
    if observed != expected:
        raise RuntimeError(
            "Upstream quantization did not dispatch every source specialization"
        )
    large = [
        record
        for record in trace
        if record["entry"] == "all_reduce_andbool_"
        and record.get("threads") in (65536, 131072)
    ]
    if Counter(record["threads"] for record in large) != {65536: 18, 131072: 18}:
        raise RuntimeError(
            "Upstream quantization did not evaluate every large error assertion"
        )


def worker(args, tests=UPSTREAM_TESTS, *, trace_values=True):
    """Run ``tests`` unchanged on MLX's CPU or through the translated host.

    ``trace_values=False`` records long value arrays only by length and digest.
    """
    import mlx.core as mx

    args.output_dir.mkdir(parents=True, exist_ok=False)
    if mx.is_available(mx.gpu):
        raise RuntimeError("A different GPU backend is already available")
    host = None
    if args.worker == "native":
        host = HostRuntime(
            args.packages,
            args.output_dir / "trace.jsonl",
            mlx_root=args.mlx_root,
            reductions=args.reductions,
            random=args.random,
            half=getattr(args, "half", None),
            half_arithmetic=getattr(args, "half_arithmetic", None),
            bfloat=getattr(args, "bfloat", None),
            trace_values=trace_values,
        )
        host.install()
    else:
        mx.set_default_device(mx.cpu)
    os.environ["DEVICE"] = "gpu" if host else "cpu"
    sys.path.insert(0, str(args.mlx_root / "python/tests"))
    try:
        suite = unittest.TestSuite(
            unittest.defaultTestLoader.loadTestsFromName(name) for name in tests
        )
        with (args.output_dir / "upstream.log").open("w", encoding="utf-8") as stream:
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        record = {
            "mode": args.worker,
            "tests": list(tests),
            "testsRun": result.testsRun,
            "success": result.wasSuccessful(),
            "skipped": result.skipped,
            "errors": [(test.id(), message) for test, message in result.errors],
            "failures": [(test.id(), message) for test, message in result.failures],
            "nativeDispatches": host.dispatch_count if host else 0,
        }
        write_json(args.output_dir / "results.json", record)
        validate_result(record, args.worker, tests)
    finally:
        if host:
            close = getattr(host.executor.runtime_adapter.runtime, "close", None)
            if close:
                close()


def execute(output, commands, tests, label):
    """Run ``(mode, timeout, arguments)`` Python commands under a deadline.

    Each command's streams and return code are kept in ``output``. The ``cpu``
    and ``native`` workers must record a passing run of ``tests``; a failed
    preparation command stops the sequence. Returns the failed modes and the
    workers' results.
    """
    bounded = Path(__file__).resolve().parents[4] / "tools/run_bounded_command.py"
    failures, results = [], {}
    for mode, timeout, arguments in commands:
        command = [
            sys.executable,
            str(bounded),
            "--timeout-seconds",
            str(timeout),
            "--label",
            f"{label} {mode}",
            "--",
            sys.executable,
            *arguments,
        ]
        with (output / f"{mode}.stdout").open("w") as stdout, (
            output / f"{mode}.stderr"
        ).open("w") as stderr:
            process = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        write_json(
            output / f"{mode}.command.json",
            {"command": command, "returncode": process.returncode},
        )
        if process.returncode:
            failures.append(mode)
            if mode not in WORKER_MODES:
                break
        elif mode in WORKER_MODES:
            results[mode] = json.loads((output / mode / "results.json").read_text())
            validate_result(results[mode], mode, tests)
    return failures, results


def run(root, packages, reductions, output):
    root, packages, reductions, output = (
        Path(path).resolve() for path in (root, packages, reductions, output)
    )
    output.mkdir(parents=True, exist_ok=False)
    preparation = verify_prepared(root)
    sources = upstream_test_sources(root, UPSTREAM_TESTS)
    target = json.loads((packages / "index.json").read_text())["target"]
    write_json(
        output / "inputs.json",
        {"adaptation": preparation, "sources": sources, "target": target},
    )
    random = output / "random-packages"
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
        )
    ]
    for mode in ("cpu", "native"):
        commands.append(
            (
                mode,
                TIMEOUTS[mode],
                [
                    "-m",
                    __package__ + ".verify_upstream_quantization",
                    "--worker",
                    mode,
                    "--mlx-root",
                    str(root),
                    "--packages",
                    str(packages),
                    "--reductions",
                    str(reductions),
                    "--random",
                    str(random),
                    "--output-dir",
                    str(output / mode),
                ],
            )
        )
    failures, results = execute(
        output, commands, UPSTREAM_TESTS, "MLX upstream quantization"
    )
    after = verify_prepared(root)
    write_json(output / "adaptation-after.json", after)
    if preparation != after or sources != upstream_test_sources(root, UPSTREAM_TESTS):
        raise RuntimeError(
            "MLX sources changed during upstream quantization verification"
        )
    if failures:
        raise RuntimeError(
            f"Upstream quantization workers failed: {', '.join(failures)}; inspect {output}"
        )
    trace = read_trace_summary(output / "native/trace.jsonl")
    validate_trace(trace, target, results["native"]["nativeDispatches"])
    evidence = {
        "commit": COMMIT,
        "target": target,
        "upstreamTestSources": sources,
        "results": results,
        "sourceUnchanged": True,
        "fullUpstreamSuite": False,
    }
    write_json(output / "evidence.json", evidence)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--reductions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random", type=Path)
    parser.add_argument("--worker", choices=("cpu", "native"))
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        run(args.mlx_root, args.packages, args.reductions, args.output_dir)
