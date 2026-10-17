"""Verify empty reductions using unchanged MLX kernels and native host dispatch."""

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import unittest
from pathlib import Path

from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.reduction_packages import (
    INIT_ENTRIES,
    load_index,
)
from demos.integrations.mlx.portable_host.runtime import (
    BOOLEAN_GUARD,
    COPY_GUARD,
    DISPATCH_VERSION,
    HostRuntime,
)
from demos.integrations.mlx.portable_host.verify_rows import verify_artifacts

UPSTREAM_TEST = "test_reduce.TestReduce.test_zero_size"
NEGATIVE_CHECKS = {
    "missing": "No translated reduction variant",
    "over-limit": "at most 65535 outputs",
    "dtype": "initialization requires float32/int32/uint32",
}


def cases():
    for dtype in ("float32", "int32", "uint32", "bool_"):
        for operation in ("sum", "prod", "all", "any"):
            for label, shape, axes, keepdims in (
                ("scalar", [0], [0], False),
                ("tail", [0, 33], [0], False),
                ("keepdims", [3, 0, 5], [1], True),
                ("zero-output", [0, 3], [1], False),
            ):
                yield dict(
                    id=f"{label}-{operation}-{dtype}",
                    dtype=dtype,
                    operation=operation,
                    shape=shape,
                    axes=axes,
                    keepdims=keepdims,
                )
    for operation, dtype in (("sum", "float32"), ("all", "bool_")):
        yield dict(
            id=f"limit-{operation}",
            dtype=dtype,
            operation=operation,
            shape=[0, 65535],
            axes=[0],
            keepdims=False,
        )
    for operation in ("min", "max"):
        yield dict(
            id=f"zero-output-{operation}",
            dtype="float32",
            operation=operation,
            shape=[2, 0, 3],
            axes=[2],
            keepdims=False,
        )


def reference(np, case):
    data = np.empty(case["shape"], dtype=case["dtype"])
    dtype = (
        "bool_"
        if case["operation"] in {"all", "any"}
        else ("int32" if case["dtype"] == "bool_" else case["dtype"])
    )
    options = {"dtype": dtype} if case["operation"] in {"sum", "prod"} else {}
    expected = getattr(np, case["operation"])(
        data, axis=tuple(case["axes"]), keepdims=case["keepdims"], **options
    )
    return data, expected, dtype


def guard_values(dtype, target):
    if dtype == "bool_":
        return BOOLEAN_GUARD if target == "metal" else [int(v) for v in BOOLEAN_GUARD]
    if dtype == "float32":
        return [struct.unpack("<f", struct.pack("<I", v))[0] for v in COPY_GUARD]
    return COPY_GUARD


def validate(records, trace, *, native):
    import numpy as np

    required = list(cases())
    if len(records) != len(required):
        raise ValueError("Empty reduction evidence does not cover every case")
    cursor = 0
    for record, case in zip(records, required):
        data, expected, dtype = reference(np, case)
        dispatch = bool(native and expected.size)
        if (
            any(record.get(key) != value for key, value in case.items())
            or record.get("actual") != expected.tolist()
            or record.get("expected") != expected.tolist()
            or record.get("resultShape") != list(expected.shape)
            or record.get("resultDtype") != "mlx.core." + dtype.removesuffix("_")
            or record.get("inputHash") != hashlib.sha256(data.tobytes()).hexdigest()
            or record.get("inputUnchanged") is not True
            or record.get("inputConstruction")
            != ("numpy" if native else "contiguous-empty")
            or type(record.get("dispatchCount")) is not int
            or record["dispatchCount"] != int(dispatch)
        ):
            raise ValueError("Empty reduction results, layout or dispatch count differ")
        if dispatch:
            if cursor >= len(trace):
                raise ValueError("Empty reduction native trace is incomplete")
            event = trace[cursor]
            cursor += 1
            operation = {"all": "and", "any": "or"}.get(
                case["operation"], case["operation"]
            )
            if (
                event.get("entry") != f"init_reduce_{operation}{dtype}"
                or event.get("target") not in {"metal", "opengl", "directx"}
                or type(event.get("dispatchVersion")) is not int
                or event["dispatchVersion"] != DISPATCH_VERSION
                or event.get("reductionValues") != expected.reshape(-1).tolist()
                or event.get("reductionGuardValues")
                != guard_values(dtype, event["target"])
                or event.get("reductionMetadata") != {"outputSize": expected.size}
                or event.get("initializationValue")
                != int(case["operation"] in {"sum", "any"})
                or event.get("threads") != expected.size
                or event.get("workgroupCount") != [expected.size, 1, 1]
                or event.get("workgroupSize") != [1, 1, 1]
                or "threadGridSize" in event
            ):
                raise ValueError(
                    "Empty reduction native trace does not match its results"
                )
    if cursor != len(trace):
        raise ValueError("Empty reduction trace contains unexpected dispatches")


def worker(args):
    import mlx.core as mx
    import numpy as np

    args.output_dir.mkdir(parents=True)
    if mx.is_available(mx.gpu):
        raise RuntimeError("A different GPU backend is already available")
    host = None
    if args.worker != "cpu":
        host = HostRuntime(
            args.packages,
            args.output_dir / "dispatch.jsonl",
            reductions=args.reductions,
        )
        if args.worker == "missing":
            host.reduction_descriptors.clear()
        host.install()
    else:
        mx.set_default_device(mx.cpu)
    if args.worker in NEGATIVE_CHECKS:
        data = np.empty(
            (0, 65536) if args.worker == "over-limit" else (0,),
            dtype="int64" if args.worker == "dtype" else "float32",
        )
        try:
            mx.eval(mx.sum(mx.array(data), axis=0))
        except (ValueError, RuntimeError) as error:
            evidence = {
                "check": args.worker,
                "error": str(error),
                "dispatchCount": host.dispatch_count,
            }
            (args.output_dir / "rejection.json").write_text(
                json.dumps(evidence, indent=2)
            )
            if NEGATIVE_CHECKS[args.worker] not in str(error) or host.dispatch_count:
                raise RuntimeError(
                    "Empty reduction was not rejected before dispatch"
                ) from error
            return
        raise RuntimeError("Unsupported empty reduction was accepted")
    records = []
    for case in cases():
        data, expected, _ = reference(np, case)
        # NumPy's zero strides hit an invalid empty plan in the pinned CPU backend.
        source = (
            mx.array(data)
            if host is not None
            else mx.array([], dtype=getattr(mx, case["dtype"])).reshape(case["shape"])
        )
        start = host.dispatch_count if host else 0
        result = getattr(mx, case["operation"])(
            source, axis=case["axes"], keepdims=case["keepdims"]
        )
        actual = np.array(result)
        records.append(
            {
                **case,
                "actual": actual.tolist(),
                "expected": expected.tolist(),
                "resultDtype": str(result.dtype),
                "resultShape": list(actual.shape),
                "inputUnchanged": bool(np.array_equal(np.array(source), data)),
                "inputConstruction": (
                    "numpy" if host is not None else "contiguous-empty"
                ),
                "inputHash": hashlib.sha256(data.tobytes()).hexdigest(),
                "dispatchCount": host.dispatch_count - start if host else 0,
            }
        )
        (args.output_dir / "results.json").write_text(json.dumps(records, indent=2))
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise RuntimeError(f"Empty reduction numerical mismatch: {case['id']}")
        print(case["id"], flush=True)
    os.environ["DEVICE"] = "cpu" if host is None else "gpu"
    os.environ["CI"] = "1"
    sys.path.insert(0, str(args.mlx_root / "python/tests"))
    start = host.dispatch_count if host else 0
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName(UPSTREAM_TEST)
    )
    upstream = {
        "test": UPSTREAM_TEST,
        "testsRun": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skips": len(result.skipped),
        "dispatchCount": host.dispatch_count - start if host else 0,
    }
    (args.output_dir / "upstream.json").write_text(json.dumps(upstream, indent=2))
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError("The unchanged upstream empty reduction test failed")


def verify(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    before = verify_prepared(args.mlx_root)
    (output / "adaptation-before.json").write_text(json.dumps(before, indent=2))
    target = json.loads((args.packages / "index.json").read_text())["target"]
    index = load_index(args.reductions, target)
    if index["family"] != "init" or set(index["entries"]) != set(INIT_ENTRIES):
        raise ValueError("Empty reduction proof requires every initialization entry")
    results, rejections = {}, {}
    for mode in ("cpu", "native", *NEGATIVE_CHECKS):
        command = [
            sys.executable,
            str(Path(__file__).resolve().parents[4] / "tools/run_bounded_command.py"),
            "--timeout-seconds",
            "600" if mode == "native" else "90",
            "--label",
            f"MLX empty reduction {mode}",
            "--",
            sys.executable,
            "-m",
            __package__ + ".verify_empty_reductions",
            "--worker",
            mode,
            "--mlx-root",
            str(args.mlx_root.resolve()),
            "--packages",
            str(args.packages.resolve()),
            "--reductions",
            str(args.reductions.resolve()),
            "--output-dir",
            str(output / mode),
        ]
        with (output / f"{mode}.stdout").open("w") as stdout, (
            output / f"{mode}.stderr"
        ).open("w") as stderr:
            process = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        (output / f"{mode}.command.json").write_text(
            json.dumps({"command": command, "returncode": process.returncode})
        )
        if process.returncode:
            raise RuntimeError(
                f"Empty reduction {mode} worker failed; inspect {output}"
            )
        if mode in NEGATIVE_CHECKS:
            rejected = json.loads((output / mode / "rejection.json").read_text())
            if (
                rejected.get("check") != mode
                or rejected.get("dispatchCount") != 0
                or NEGATIVE_CHECKS[mode] not in rejected.get("error", "")
            ):
                raise ValueError("Empty reduction rejection evidence is incomplete")
            rejections[mode] = rejected
        else:
            results[mode] = json.loads((output / mode / "results.json").read_text())
            upstream = json.loads((output / mode / "upstream.json").read_text())
            if upstream != {
                "test": UPSTREAM_TEST,
                "testsRun": 1,
                "failures": 0,
                "errors": 0,
                "skips": 0,
                "dispatchCount": 0,
            }:
                raise ValueError("Upstream empty reduction evidence is incomplete")
    trace = [
        json.loads(line)
        for line in (output / "native/dispatch.jsonl").read_text().splitlines()
    ]
    validate(results["cpu"], [], native=False)
    validate(results["native"], trace, native=True)
    verify_artifacts(trace, args.reductions, index)
    after = verify_prepared(args.mlx_root)
    (output / "adaptation-after.json").write_text(json.dumps(after, indent=2))
    if before != after:
        raise ValueError("MLX sources changed during empty reduction verification")
    evidence = {
        "commit": COMMIT,
        "target": target,
        "adaptation": before,
        "casesPerPath": len(results["native"]),
        "dispatchCount": len(trace),
        "entries": sorted(INIT_ENTRIES),
        "upstreamTests": [UPSTREAM_TEST],
        "negativeChecks": rejections,
        "fullTranslatedBackend": False,
        "fullUpstreamSuite": False,
    }
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2))
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--reductions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", choices=("cpu", "native", *NEGATIVE_CHECKS))
    args = parser.parse_args()
    worker(args) if args.worker else verify(args)
