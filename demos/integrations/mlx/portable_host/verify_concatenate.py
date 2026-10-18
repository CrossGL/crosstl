"""Verify concatenation through unchanged copy kernels and the pinned MLX test."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from demos.integrations.mlx.portable_host.binary_workloads import mlx_operand
from demos.integrations.mlx.portable_host.packages import BOOLEAN_COPY_ENTRY, COPY_ENTRY
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.reduction_packages import load_index
from demos.integrations.mlx.portable_host.runtime import (
    BOOLEAN_GUARD,
    COPY_GUARD,
    DISPATCH_VERSION,
    HostRuntime,
)
from demos.integrations.mlx.portable_host.verify_bitwise import (
    verify_native_identity,
)
from demos.integrations.mlx.portable_host.verify_rows import verify_artifacts

UPSTREAM_TEST = "test_ops.TestOps.test_concatenate"
NEGATIVE_CHECKS = {
    "missing": "No translated package for",
    "int64": "requires matching float32/int32/uint32/bool",
    "uint8": "requires matching float32/int32/uint32/bool",
    "over-limit": "at most 65535 elements per input",
}


def cases():
    for dtype in ("float32", "int32", "uint32", "bool_"):
        for layout in (
            "dense",
            "transpose",
            "reverse",
            "broadcast",
            "empty-left",
            "empty-middle",
            "empty-all",
            "three-input",
        ):
            for axis in (0, 1):
                yield {
                    "id": f"{dtype}-{layout}-{axis}",
                    "dtype": dtype,
                    "layout": layout,
                    "axis": axis,
                }
        for layout, axis in (("flatten", None), ("large", 2)):
            yield {
                "id": f"{dtype}-{layout}",
                "dtype": dtype,
                "layout": layout,
                "axis": axis,
            }


def inputs(np, case):
    words = np.asarray(
        [0, 0x80000000, 1, 0xFFFFFFFF, 0x7FC12345, 0x7F800000, 0x3F800000, 0x55555555]
        * 8192,
        dtype="uint32",
    )
    dtype = case["dtype"]
    data = (words % 2).astype("bool") if dtype == "bool_" else words.view(dtype)
    if case["layout"] == "large":
        return [data[:32768].reshape(32, 32, 32), data[32768:].reshape(32, 32, 32)]
    base = data[:36].reshape(6, 6)
    left = np.ascontiguousarray(base[:3, :4])
    right = np.ascontiguousarray(base[2:5, 1:5])
    layout = case["layout"]
    if layout == "transpose":
        left = base[:4, :3].T
    elif layout == "reverse":
        left = base[4:1:-1, 5:1:-1]
    elif layout == "broadcast":
        left = np.broadcast_to(base[1:2, :4], (3, 4))
    elif layout in {"empty-left", "empty-middle", "empty-all"}:
        shape = list(left.shape)
        shape[case["axis"]] = 0
        empty = np.empty(shape, dtype=dtype)
        if layout == "empty-left":
            return [empty, right]
        if layout == "empty-all":
            return [empty, empty]
        return [left, empty, right]
    elif layout == "three-input":
        return [left, right, left]
    return [left, right]


def payload(np, value):
    return np.ascontiguousarray(value).tobytes().hex()


def storage(np, value):
    array = np.ascontiguousarray(value)
    return (
        array.reshape(-1).tolist()
        if array.dtype == np.bool_
        else array.view("uint32").reshape(-1).tolist()
    )


def validate(records, trace, *, native):
    import numpy as np

    required = list(cases())
    if len(records) != len(required):
        raise ValueError("Concatenation case coverage is incomplete")
    cursor = 0
    for record, case in zip(records, required):
        arrays = inputs(np, case)
        expected = np.concatenate(arrays, axis=case["axis"])
        nonempty = [array for array in arrays if array.size]
        count = len(nonempty) if native else 0
        if (
            any(record.get(key) != value for key, value in case.items())
            or record.get("inputPayloads") != [payload(np, value) for value in arrays]
            or record.get("actualPayload") != payload(np, expected)
            or record.get("resultShape") != list(expected.shape)
            or record.get("resultDtype")
            != "mlx.core." + case["dtype"].removesuffix("_")
            or record.get("inputUnchanged") is not True
            or type(record.get("dispatchCount")) is not int
            or record["dispatchCount"] != count
        ):
            raise ValueError("Concatenation result or input storage differs")
        staged = np.zeros_like(expected)
        axis = 0 if case["axis"] is None else case["axis"]
        offset = 0
        for position, source in enumerate(nonempty if native else []):
            if cursor >= len(trace):
                raise ValueError("Concatenation trace is incomplete")
            event = trace[cursor]
            cursor += 1
            source = source.reshape(-1) if case["axis"] is None else source
            selection = [slice(None)] * expected.ndim
            selection[axis] = slice(offset, offset + source.shape[axis])
            staged[tuple(selection)] = source
            shape = list(source.shape)
            while len(shape) < 2:
                shape.insert(0, 1)
            grid = [(shape[-1] + 1) // 2, shape[-2], int(np.prod(shape[:-2]))]
            boolean = case["dtype"] == "bool_"
            guard = BOOLEAN_GUARD if boolean else COPY_GUARD
            if boolean and event.get("target") != "metal":
                guard = [int(value) for value in guard]
            metadata = event.get("copyMetadata", {})
            source_strides = [
                stride // source.itemsize if size != 1 else 0
                for size, stride in zip(source.shape, source.strides)
            ]
            destination_strides = [
                stride // expected.itemsize for stride in expected.strides
            ]
            while len(source_strides) < 2:
                source_strides.insert(0, 0)
                destination_strides.insert(0, 0)
            extents = [
                (size - 1) * stride for size, stride in zip(shape, source_strides)
            ]
            low, high = sum(min(value, 0) for value in extents), sum(
                max(value, 0) for value in extents
            )
            expected_metadata = {
                "shape": shape,
                "sourceStrides": source_strides,
                "destinationStrides": destination_strides,
                "sourceOffset": -low,
                "destinationOffset": (
                    offset * (expected.strides[axis] // expected.itemsize)
                ),
                "sourceCount": high - low + 1,
                "destinationCount": expected.size,
                "preserveDestination": bool(position),
                "workgroupCount": grid,
            }
            if (
                event.get("entry") != (BOOLEAN_COPY_ENTRY if boolean else COPY_ENTRY)
                or event.get("target") not in {"metal", "opengl", "directx"}
                or event.get("dispatchVersion") != DISPATCH_VERSION
                or event.get("threads") != source.size
                or event.get("workgroupCount") != grid
                or event.get("workgroupSize") != [1, 1, 1]
                or event.get("copyValues") != storage(np, staged)
                or event.get("copyGuardWords") != guard
                or metadata != expected_metadata
            ):
                raise ValueError("Concatenation copy sequence differs")
            offset += source.shape[axis]
    if cursor != len(trace):
        raise ValueError("Concatenation trace contains unexpected dispatches")


def worker(args):
    import mlx.core as mx
    import numpy as np

    output = args.output_dir
    output.mkdir(parents=True)
    if mx.is_available(mx.gpu):
        raise RuntimeError("A different GPU backend is already available")
    host = None
    if args.worker != "cpu":
        host = HostRuntime(
            args.packages, output / "dispatch.jsonl", reductions=args.reductions
        )
        if args.worker == "missing":
            host.descriptors.pop(COPY_ENTRY)
        host.install()
    else:
        mx.set_default_device(mx.cpu)
    if args.worker in NEGATIVE_CHECKS:
        dtype = args.worker if args.worker in {"int64", "uint8"} else "int32"
        size = 65536 if args.worker == "over-limit" else 3
        a = mx.array(np.ones(size, dtype=dtype))
        try:
            mx.eval(mx.concatenate([a, a]))
        except (ValueError, RuntimeError) as error:
            record = {
                "check": args.worker,
                "error": str(error),
                "dispatchCount": host.dispatch_count,
            }
            (output / "rejection.json").write_text(json.dumps(record))
            if NEGATIVE_CHECKS[args.worker] not in str(error) or host.dispatch_count:
                raise RuntimeError(
                    "Unsupported concatenation was not rejected before dispatch"
                ) from error
            return
        raise RuntimeError("Unsupported concatenation was accepted")
    records = []
    for case in cases():
        arrays = inputs(np, case)
        operands = [mlx_operand(mx, np, value) for value in arrays]
        start = host.dispatch_count if host else 0
        result = mx.concatenate(operands, axis=case["axis"])
        actual = np.array(result)
        records.append(
            {
                **case,
                "inputPayloads": [payload(np, value) for value in arrays],
                "actualPayload": payload(np, actual),
                "resultShape": list(actual.shape),
                "resultDtype": str(result.dtype),
                "inputUnchanged": all(
                    payload(np, np.array(operand)) == payload(np, value)
                    for operand, value in zip(operands, arrays)
                ),
                "dispatchCount": host.dispatch_count - start if host else 0,
            }
        )
        (output / "results.json").write_text(json.dumps(records))
        if payload(np, actual) != payload(
            np, np.concatenate(arrays, axis=case["axis"])
        ):
            raise RuntimeError(f"Concatenation mismatch: {case['id']}")
    workload_dispatches = host.dispatch_count if host else 0
    os.environ["DEVICE"] = "gpu" if host else "cpu"
    os.environ["CI"] = "1"
    sys.path.insert(0, str(args.mlx_root / "python/tests"))
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName(UPSTREAM_TEST)
    )
    upstream = {
        "test": UPSTREAM_TEST,
        "testsRun": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skips": len(result.skipped),
        "dispatchCount": host.dispatch_count - workload_dispatches if host else 0,
        "workloadDispatchCount": workload_dispatches,
    }
    (output / "upstream.json").write_text(json.dumps(upstream))
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError("The unchanged upstream concatenation test failed")


def verify(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    before = verify_prepared(args.mlx_root)
    (output / "adaptation-before.json").write_text(json.dumps(before))
    base = json.loads((args.packages / "index.json").read_text())
    reductions = load_index(args.reductions, base["target"])
    if "w32/all_reduce_andbool_" not in reductions["descriptors"]:
        raise ValueError(
            "The upstream concatenation test requires Boolean reduction at width32"
        )
    results, upstream, rejections = {}, {}, {}
    for mode in ("cpu", "native", *NEGATIVE_CHECKS):
        command = [
            sys.executable,
            str(Path(__file__).resolve().parents[4] / "tools/run_bounded_command.py"),
            "--timeout-seconds",
            "900" if mode == "native" else "60",
            "--label",
            f"MLX concatenate {mode}",
            "--",
            sys.executable,
            "-m",
            __package__ + ".verify_concatenate",
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
            raise RuntimeError(f"Concatenation {mode} worker failed; inspect {output}")
        if mode in NEGATIVE_CHECKS:
            record = json.loads((output / mode / "rejection.json").read_text())
            if (
                record.get("check") != mode
                or record.get("dispatchCount") != 0
                or NEGATIVE_CHECKS[mode] not in record.get("error", "")
            ):
                raise ValueError("Concatenation rejection evidence is incomplete")
            rejections[mode] = record
        else:
            results[mode] = json.loads((output / mode / "results.json").read_text())
            record = json.loads((output / mode / "upstream.json").read_text())
            if any(
                record.get(key) != value
                for key, value in {
                    "test": UPSTREAM_TEST,
                    "testsRun": 1,
                    "failures": 0,
                    "errors": 0,
                    "skips": 0,
                }.items()
            ):
                raise ValueError("Upstream concatenation evidence is incomplete")
            if (
                mode == "cpu"
                and (
                    record.get("dispatchCount") != 0
                    or record.get("workloadDispatchCount") != 0
                )
            ) or (
                mode == "native"
                and (
                    type(record.get("dispatchCount")) is not int
                    or record["dispatchCount"] <= 0
                )
            ):
                raise ValueError(
                    "Upstream concatenation dispatch evidence is incomplete"
                )
            upstream[mode] = record
    trace = [
        json.loads(line)
        for line in (output / "native/dispatch.jsonl").read_text().splitlines()
    ]
    count = upstream["native"]["workloadDispatchCount"]
    if (
        type(count) is not int
        or count < 0
        or count + upstream["native"]["dispatchCount"] != len(trace)
    ):
        raise ValueError("Concatenation dispatch count differs")
    validate(results["cpu"], [], native=False)
    validate(results["native"], trace[:count], native=True)
    verify_native_identity(trace, base["target"])
    checked_events = 0
    for selected, directory, variants in (
        (base, args.packages, False),
        (reductions, args.reductions, True),
    ):
        events = [
            event
            for event in trace
            if (
                f'w{event["workgroupSize"][0]}/{event["entry"]}'
                if variants
                else event["entry"]
            )
            in selected["descriptors"]
        ]
        checked_events += len(events)
        verify_artifacts(events, directory, selected, variants=variants)
    if checked_events != len(trace):
        raise ValueError("Concatenation trace references an unknown artifact")
    after = verify_prepared(args.mlx_root)
    if before != after:
        raise ValueError("MLX sources changed during concatenation verification")
    evidence = {
        "commit": COMMIT,
        "target": base["target"],
        "adaptation": before,
        "casesPerPath": len(results["native"]),
        "dispatchCount": len(trace),
        "upstream": upstream,
        "negativeChecks": rejections,
        "upstreamTestSha256": (
            hashlib.sha256(
                (args.mlx_root / "python/tests/test_ops.py").read_bytes()
            ).hexdigest()
        ),
        "fullTranslatedBackend": False,
        "fullUpstreamSuite": False,
    }
    (output / "adaptation-after.json").write_text(json.dumps(after))
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2))
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mlx-root", "packages", "reductions", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=("cpu", "native", *NEGATIVE_CHECKS))
    args = parser.parse_args()
    worker(args) if args.worker else verify(args)
