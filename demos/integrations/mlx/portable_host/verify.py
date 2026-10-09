"""Require real MLX evaluation through translated native compute kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from demos.integrations.mlx.portable_host import (
    binary_workloads,
    boolean_workloads,
    cast_workloads,
    copy_workloads,
    full_workloads,
    reduction_workloads,
    unary_workloads,
    view_workloads,
)
from demos.integrations.mlx.portable_host.packages import ENTRIES
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.reduction_packages import (
    ENTRIES as REDUCTION_ENTRIES,
)
from demos.integrations.mlx.portable_host.reduction_packages import (
    load_index as load_reduction_index,
)
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.verify_binary_batches import (
    run_bounded as run_binary_batches,
)
from demos.integrations.mlx.portable_host.verify_cast_batches import (
    run_bounded as run_cast_batches,
)
from demos.integrations.mlx.portable_host.verify_large_copies import (
    run_bounded as run_large_copies,
)
from demos.integrations.mlx.portable_host.verify_unary_batches import (
    run_bounded as run_unary_batches,
)

UPSTREAM_TESTS = (
    "test_ops.TestOps.test_arange_overload_dispatch",
    "test_ops.TestOps.test_arange_inferred_dtype",
    "test_ops.TestOps.test_arange_corner_cases_cast",
    "test_ops.TestOps.test_abs",
    "test_ops.TestOps.test_negative",
    "test_ops.TestOps.test_floor",
    "test_ops.TestOps.test_ceil",
    "test_ops.TestOps.test_square",
    "test_ops.TestOps.test_sqrt",
    "test_ops.TestOps.test_rsqrt",
    "test_ops.TestOps.test_exp",
    "test_ops.TestOps.test_expm1",
    "test_ops.TestOps.test_erf",
    "test_ops.TestOps.test_sin",
    "test_ops.TestOps.test_cos",
    "test_ops.TestOps.test_transpose_noargs",
    "test_ops.TestOps.test_transpose_axis",
    "test_ops.TestOps.test_broadcast",
    "test_ops.TestOps.test_split",
    "test_ops.TestOps.test_subtract",
    "test_ops.TestOps.test_multiply",
    "test_ops.TestOps.test_diff",
    "test_ops.TestOps.test_flip",
    "test_array.TestArray.test_array_type_cast",
    "test_ops.TestOps.test_bartlett_general",
    "test_ops.TestOps.test_blackman_general",
    "test_ops.TestOps.test_hamming_general",
    "test_ops.TestOps.test_hanning_general",
    "test_ops.TestOps.test_shape_overflow_error",
    "test_ops.TestOps.test_comparisons",
    "test_ops.TestOps.test_logical_not",
    "test_ops.TestOps.test_logical_xor",
    "test_ops.TestOps.test_isclose",
    "test_ops.TestOps.test_allclose",
)
DTYPES = ("float32", "int32", "uint32", "int64", "uint64")
COUNTS = (0, 1, 7, 257)
NEGATIVE_CHECKS = {
    "unsupported": "no GPU implementation",
    "over-limit": "65535",
    "missing": "artifact",
    "unary-dtype": "float32",
    "unary-allocation": "exceeds its allocation",
    "unary-strided-allocation": "exceeds its allocation",
    "unary-large-allocation": "exceeds its allocation",
    "copy-dtype": "matching float16, float32, int32, uint32, int64, uint64 or bool",
    "copy-large-allocation": "exceeds its allocation",
    "copy-allocation": "exceeds its allocation",
    "binary-dtype": "supported dtype",
    "binary-large-allocation": "exceeds its allocation",
    "cast-dtype": (
        "casts require float16, float32, int32, uint32, int64, uint64 or bool"
    ),
    "cast-large-allocation": "exceeds its allocation",
    "cast-allocation": "exceeds its allocation",
    "full-dtype": "matching float16, float32, int32, uint32, int64, uint64 or bool",
    "full-grid-limit": "65535 groups per axis",
    "full-allocation": "exceeds its allocation",
}
REDUCTION_NEGATIVE_CHECKS = {
    "reduce-dtype": "matching float32/int32/uint32",
    "reduce-limit": "65535",
    "reduce-allocation": "storage does not match",
    "reduce-empty": "No translated reduction variant",
    "reduce-row": "Small-row dispatch requires the pinned MLX source root",
    "reduce-column": "small-column and long-column reduction plans",
}


def save(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def worker(args):
    import mlx.core as mx
    import numpy as np

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if mx.metal.is_available():
        raise RuntimeError("This proof must not have a Metal backend")
    if mx.is_available(mx.gpu):
        raise RuntimeError("A different GPU backend is already available")
    runtime = None
    if args.worker != "cpu":
        runtime = HostRuntime(
            args.packages,
            output / "dispatch.jsonl",
            reductions=getattr(args, "reductions", None),
        )
        runtime.install()
        assert mx.default_device() == mx.gpu and mx.is_available(mx.gpu)
    else:
        mx.set_default_device(mx.cpu)
    os.environ["DEVICE"] = "cpu" if args.worker == "cpu" else "gpu"
    checks = {**NEGATIVE_CHECKS, **REDUCTION_NEGATIVE_CHECKS}
    if args.worker in checks:
        try:
            if args.worker == "unsupported":
                value = mx.power(
                    mx.array([-3.0, 2.0]), mx.array([1.0, 1.0]), stream=mx.gpu
                )
            elif args.worker == "over-limit":
                value = mx.arange(65536, stream=mx.gpu)
            elif args.worker == "unary-dtype":
                value = mx.abs(mx.array([-3, 2], dtype=mx.int16), stream=mx.gpu)
            elif args.worker == "unary-allocation":
                source = mx.as_strided(
                    mx.array([1.0, 2.0, 3.0]), (2,), (1,), 2, stream=mx.gpu
                )
                value = mx.abs(source, stream=mx.gpu)
            elif args.worker == "unary-strided-allocation":
                source = mx.as_strided(mx.array([1.0, 2.0, 3.0]), (2,), (-1,))
                value = mx.abs(source, stream=mx.gpu)
            elif args.worker == "unary-large-allocation":
                value = mx.abs(
                    mx.as_strided(mx.array([1.0, 2.0, 3.0]), (65536,), (1,)),
                    stream=mx.gpu,
                )
            elif args.worker == "copy-dtype":
                source = mx.array(np.arange(12, dtype=np.int16).reshape(3, 4))
                value = mx.reshape(mx.transpose(source), (12,), stream=mx.gpu)
            elif args.worker == "copy-large-allocation":
                source = mx.as_strided(
                    mx.array([1, 2, 3], dtype=mx.uint32), (256, 257), (1, 256), 2
                )
                value = mx.contiguous(source, stream=mx.gpu)
            elif args.worker == "copy-allocation":
                source = mx.as_strided(mx.array([1.0, 2.0, 3.0]), (2,), (-1,))
                value = mx.contiguous(source, stream=mx.gpu)
            elif args.worker == "binary-dtype":
                value = mx.add(
                    mx.array([1, 2], dtype=mx.int16), mx.array([2, 3], dtype=mx.int16)
                )
            elif args.worker == "binary-large-allocation":
                source = mx.as_strided(mx.array([1.0, 2.0, 3.0]), (65536,), (1,), 2)
                value = mx.add(source, source, stream=mx.gpu)
            elif args.worker == "cast-dtype":
                value = mx.array([1, 2], dtype=mx.int16).astype(mx.float32)
            elif args.worker == "cast-large-allocation":
                source = mx.as_strided(
                    mx.array([1, 2, 3], dtype=mx.int32), (65536,), (1,), 2
                )
                value = source.astype(mx.float32)
            elif args.worker == "cast-allocation":
                source = mx.as_strided(
                    mx.array([1, 2, 3], dtype=mx.int32), (2,), (1,), 2
                )
                value = source.astype(mx.float32)
            elif args.worker == "full-dtype":
                value = mx.full((3, 5), mx.array(1, dtype=mx.int16))
            elif args.worker == "full-grid-limit":
                value = mx.ones((131072,), dtype=mx.float32)
            elif args.worker == "full-allocation":
                source = mx.as_strided(mx.array([1.0, 2.0, 3.0]), (2,), (-1,))
                value = mx.full((3, 2), source)
            elif args.worker == "reduce-dtype":
                value = mx.sum(mx.array(np.ones(2, dtype=np.float16)))
            elif args.worker == "reduce-limit":
                value = mx.sum(mx.array(np.ones(65536, dtype=np.float32)))
            elif args.worker == "reduce-allocation":
                source = mx.as_strided(mx.array([1.0, 2.0, 3.0]), (5,), (1,), 2)
                value = mx.sum(source)
            elif args.worker == "reduce-empty":
                value = mx.sum(mx.array([], dtype=mx.float32))
            elif args.worker in {"reduce-row", "reduce-column"}:
                source = mx.array([[1.0, 2.0], [3.0, 4.0]])
                value = mx.sum(source, axis=1 if args.worker == "reduce-row" else 0)
            else:
                descriptor = runtime.descriptors["arangefloat32"]
                descriptor["artifact"]["packagePath"] = "artifacts/missing.glsl"
                value = mx.arange(2.0, 9.0, stream=mx.gpu)
            mx.eval(value)
        except (ValueError, RuntimeError) as error:
            message = str(error)
            expected = checks[args.worker]
            if expected not in message:
                raise
            save(output / "result.json", {"rejected": True, "message": message})
            return
        raise RuntimeError(f"{args.worker} unexpectedly executed")

    records = []
    for dtype in DTYPES:
        for count in COUNTS:
            expected = np.arange(2, 2 + 3 * count, 3, dtype=dtype)
            value = mx.arange(2, 2 + 3 * count, 3, dtype=getattr(mx, dtype))
            actual = np.array(value)
            np.testing.assert_array_equal(actual, expected)
            if str(actual.dtype) != dtype:
                raise RuntimeError("MLX readback dtype changed")
            records.append({"dtype": dtype, "count": count, "values": actual.tolist()})
    unary = unary_workloads.run(mx, np)
    views = view_workloads.run(mx, np)
    copies = copy_workloads.run(mx, np)
    binary = binary_workloads.run(mx, np)
    save(output / "binary-readbacks.json", binary)
    binary_workloads.validate(binary)
    casts = cast_workloads.run(mx, np)
    save(output / "cast-readbacks.json", casts)
    cast_workloads.validate(casts)
    full = full_workloads.run(mx, np)
    save(output / "full-readbacks.json", full)
    full_workloads.validate(full)
    booleans = boolean_workloads.run(mx, np)
    save(output / "boolean-readbacks.json", booleans)
    boolean_workloads.validate(booleans)
    sys.path.insert(0, str(args.mlx_root / "python/tests"))
    suite = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromName(name) for name in UPSTREAM_TESTS
    )
    with (output / "upstream-tests.log").open("w", encoding="utf-8") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    reductions = None
    if getattr(args, "reductions", None) is not None:
        index = json.loads((args.reductions / "index.json").read_text())

        def observe(record):
            with (output / "reduction-readbacks.jsonl").open(
                "a", encoding="utf-8"
            ) as log:
                log.write(json.dumps(record, allow_nan=False) + "\n")

        reductions = reduction_workloads.collect(
            mx,
            index["widths"],
            observe=observe,
            dispatch_count=(lambda: runtime.dispatch_count) if runtime else None,
        )
        reduction_workloads.validate(reductions, index["widths"])
    save(
        output / "result.json",
        {
            "tests": result.testsRun,
            "skipped": len(result.skipped),
            "failures": len(result.failures),
            "errors": len(result.errors),
            "arrays": records,
            "unary": unary,
            "views": views,
            "copies": copies,
            "binary": binary,
            "casts": casts,
            "full": full,
            "booleans": booleans,
            **({"reductions": reductions} if reductions is not None else {}),
        },
    )
    if (
        not result.wasSuccessful()
        or result.testsRun != len(UPSTREAM_TESTS)
        or result.skipped
    ):
        raise RuntimeError(
            f"Unchanged upstream tests failed; inspect {output / 'upstream-tests.log'}"
        )


def verify_results(result, *, cpu=False):
    if (
        type(result.get("tests")) is not int
        or result["tests"] != len(UPSTREAM_TESTS)
        or type(result.get("skipped")) is not int
        or result["skipped"] != 0
        or any(
            type(result.get(key)) is not int or result[key] != 0
            for key in ("failures", "errors")
        )
    ):
        raise RuntimeError("Incomplete upstream test results")
    expected = [
        {"dtype": dtype, "count": count, "values": [2 + 3 * i for i in range(count)]}
        for dtype in DTYPES
        for count in COUNTS
    ]
    if result.get("arrays") != expected or any(
        type(record["count"]) is not int
        or any(type(value) not in {int, float} for value in record["values"])
        for record in result["arrays"]
    ):
        raise RuntimeError("Incomplete or incorrect array readbacks")
    unary_workloads.validate(result.get("unary"), cpu=cpu)
    view_workloads.validate(result.get("views"))
    copy_workloads.validate(result.get("copies"))
    binary_workloads.validate(result.get("binary"))
    cast_workloads.validate(result.get("casts"))
    full_workloads.validate(result.get("full"))
    boolean_workloads.validate(result.get("booleans"))


def upstream_test_sources(root, tests=None):
    sources = {}
    tests = UPSTREAM_TESTS if tests is None else tests
    paths = {f"python/tests/{test.split('.')[0]}.py" for test in tests}
    for name in sorted(paths):
        path = root / name
        pristine = subprocess.check_output(
            ["git", "-C", str(root), "show", f"HEAD:{name}"], timeout=30
        )
        if path.is_symlink() or path.read_bytes() != pristine:
            raise ValueError(f"Upstream test source was modified: {name}")
        sources[name] = hashlib.sha256(pristine).hexdigest()
    return sources


def verify(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    adaptation = verify_prepared(args.mlx_root)
    save(output / "adaptation-before.json", adaptation)
    test_sources = upstream_test_sources(args.mlx_root)
    reductions = getattr(args, "reductions", None)
    reduction_index = None
    if reductions is not None:
        target = json.loads((args.packages / "index.json").read_text())["target"]
        reduction_index = load_reduction_index(reductions, target)
        if set(reduction_index["entries"]) != set(REDUCTION_ENTRIES):
            raise ValueError(
                "Reduction proof requires every supported operation and dtype"
            )
        if not {32, 64, 128}.issubset(reduction_index["widths"]):
            raise ValueError(
                "Reduction proof requires widths 32, 64 and 128 for multipass cases"
            )
    results = {}
    failed = []
    checks = {
        **NEGATIVE_CHECKS,
        **(REDUCTION_NEGATIVE_CHECKS if reductions is not None else {}),
    }
    for mode in ("cpu", "native", *checks):
        command = [
            sys.executable,
            str(Path(__file__).resolve().parents[4] / "tools/run_bounded_command.py"),
            "--timeout-seconds",
            (
                "1800"
                if reductions is not None and mode == "native"
                else "900" if mode == "native" else "180"
            ),
            "--label",
            f"MLX host {mode}",
            "--",
            sys.executable,
            "-m",
            __package__ + ".verify",
            "--worker",
            mode,
            "--mlx-root",
            str(args.mlx_root.resolve()),
            "--packages",
            str(args.packages.resolve()),
            "--output-dir",
            str(output / mode),
        ]
        if reductions is not None:
            command.extend(["--reductions", str(reductions.resolve())])
        with (output / f"{mode}.stdout").open("w") as stdout, (
            output / f"{mode}.stderr"
        ).open("w") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        save(
            output / f"{mode}.command.json",
            {"command": command, "returncode": result.returncode},
        )
        if result.returncode:
            failed.append(mode)
        else:
            results[mode] = json.loads((output / mode / "result.json").read_text())
    if failed:
        raise RuntimeError(
            f"Host proof workers failed: {', '.join(failed)}; inspect {output}"
        )
    if results["cpu"]["arrays"] != results["native"]["arrays"]:
        raise RuntimeError("Original CPU and translated GPU host results differ")
    verify_results(results["cpu"], cpu=True)
    verify_results(results["native"])
    for mode, expected in checks.items():
        if (
            results[mode].get("rejected") is not True
            or not isinstance(results[mode].get("message"), str)
            or expected not in results[mode]["message"]
        ):
            raise RuntimeError(f"Missing required rejection evidence: {mode}")
    trace = [
        json.loads(line)
        for line in (output / "native/dispatch.jsonl").read_text().splitlines()
    ]
    expected_entries = set(ENTRIES) | (
        set(REDUCTION_ENTRIES) if reductions is not None else set()
    )
    if {record["entry"] for record in trace} != expected_entries:
        raise RuntimeError("Native trace does not cover every translated entry")
    expected_dispatches = (
        [(f"arange{dtype}", count) for dtype in DTYPES for count in COUNTS if count]
        + unary_workloads.dispatches()
        + view_workloads.dispatches()
        + copy_workloads.dispatches()
        + binary_workloads.dispatches()
        + cast_workloads.dispatches()
        + full_workloads.dispatches()
        + boolean_workloads.dispatches()
    )
    if [(record["entry"], record.get("threads")) for record in trace][
        : len(expected_dispatches)
    ] != expected_dispatches or any(
        type(record.get("threads")) is not int or not 1 <= record["threads"] <= 65535
        for record in trace
    ):
        raise RuntimeError("Native trace has incomplete or invalid dispatch sizes")
    index = json.loads((args.packages / "index.json").read_text())
    if any(record["target"] != index["target"] for record in trace):
        raise RuntimeError("Native trace used an unexpected target")
    for record in trace:
        if reductions is not None and record["entry"] in REDUCTION_ENTRIES:
            continue
        count = record.get("workgroupCount")
        if (
            type(record.get("dispatchVersion")) is not int
            or record["dispatchVersion"] != 3
            or record.get("workgroupSize") != [1, 1, 1]
            or any(type(value) is not int for value in record["workgroupSize"])
            or not isinstance(count, list)
            or len(count) != 3
            or any(type(value) is not int or not 1 <= value <= 65535 for value in count)
        ):
            raise RuntimeError("Native trace has incomplete or invalid launch geometry")
    if reductions is not None:
        for mode in ("cpu", "native"):
            reduction_workloads.validate(
                results[mode].get("reductions"),
                reduction_index["widths"],
                trace=trace if mode == "native" else None,
            )
    cast_batches = (
        run_cast_batches(args.mlx_root, args.packages, output / "cast-batches")
        if getattr(args, "cast_batches", False)
        else None
    )
    large_copies = (
        run_large_copies(args.mlx_root, args.packages, output / "large-copies")
        if getattr(args, "large_copies", False)
        else None
    )
    binary_batches = (
        run_binary_batches(args.mlx_root, args.packages, output / "binary-batches")
        if getattr(args, "binary_batches", False)
        else None
    )
    unary_batches = (
        run_unary_batches(args.mlx_root, args.packages, output / "unary-batches")
        if getattr(args, "unary_batches", False)
        else None
    )
    after = verify_prepared(args.mlx_root)
    save(output / "adaptation-after.json", after)
    if after != adaptation or upstream_test_sources(args.mlx_root) != test_sources:
        raise ValueError("MLX sources changed during execution")
    evidence = {
        "schemaVersion": 2,
        "dispatchVersion": 3,
        "commit": COMMIT,
        "adaptation": adaptation,
        "target": index["target"],
        "cpuReferenceProfile": unary_workloads.cpu_reference_profile(),
        "upstreamTestSha256": test_sources["python/tests/test_ops.py"],
        "upstreamTestSources": test_sources,
        "upstreamTests": list(UPSTREAM_TESTS),
        "original": results["cpu"],
        "translated": results["native"],
        "dispatchCount": len(trace),
        "entries": sorted({record["entry"] for record in trace}),
        "negativeChecks": {mode: results[mode] for mode in checks},
        "fullUpstreamSuite": False,
        "fullTranslatedBackend": False,
        **({"reductionWidths": reduction_index["widths"]} if reduction_index else {}),
        **({"castBatches": cast_batches} if cast_batches is not None else {}),
        **({"largeCopies": large_copies} if large_copies is not None else {}),
        **({"binaryBatches": binary_batches} if binary_batches is not None else {}),
        **({"unaryBatches": unary_batches} if unary_batches is not None else {}),
    }
    save(output / "evidence.json", evidence)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--reductions", type=Path)
    parser.add_argument("--cast-batches", action="store_true")
    parser.add_argument("--large-copies", action="store_true")
    parser.add_argument("--binary-batches", action="store_true")
    parser.add_argument("--unary-batches", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--worker",
        choices=["cpu", "native", *NEGATIVE_CHECKS, *REDUCTION_NEGATIVE_CHECKS],
    )
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        verify(args)
