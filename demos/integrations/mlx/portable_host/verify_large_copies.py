"""Verify large layout copies with exact storage and retained native evidence."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

from demos.integrations.mlx.portable_host.copy_workloads import source_words
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_native_execution,
    require,
)
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.runtime import (
    BOOLEAN_GUARD,
    COPY_GUARD,
    HostRuntime,
)


def cases():
    layouts = (
        ("reverse-boundary", (65535,), (-1,)),
        ("reverse-large", (65536,), (-1,)),
        ("transpose", (256, 257), (1, 256)),
        ("reverse-columns", (129, 513), (1026, -2)),
        ("permute", (2, 3, 129, 257), (33153, 66306, 1, 129)),
        ("row-broadcast", (128, 512), (0, 2)),
        ("scalar-broadcast", (256, 512), (0, 0)),
        ("wide-stride", (2, 3), (70000, 2)),
    )
    result = [
        {
            "id": dtype + "-" + name,
            "dtype": dtype,
            "shape": list(shape),
            "strides": list(strides),
        }
        for dtype in ("float32", "int32", "uint32", "bool")
        for name, shape, strides in layouts
    ]
    result.extend(
        {
            "id": "uint32-grid-" + axis,
            "dtype": "uint32",
            "shape": list(shape),
            "strides": list(strides),
        }
        for axis, shape, strides in (
            ("x", (131070,), (-1,)),
            ("y", (65535, 2), (0, 1)),
            ("z", (65535, 1, 2), (0, 0, 1)),
        )
    )
    return result


def storage(np, case):
    extents = [(size - 1) * step for size, step in zip(case["shape"], case["strides"])]
    low, high = sum(min(value, 0) for value in extents), sum(
        max(value, 0) for value in extents
    )
    raw = np.arange(high - low + 13, dtype=np.uint32) * np.uint32(2654435761)
    raw[:12] = source_words()[:12]
    values = raw % 7 == 0 if case["dtype"] == "bool" else raw.view(case["dtype"])
    return values, 5 - low, low, high


def reference(np, case):
    values, offset, _, _ = storage(np, case)
    return np.ndarray(
        tuple(case["shape"]),
        dtype=values.dtype,
        buffer=values,
        offset=offset * values.itemsize,
        strides=tuple(step * values.itemsize for step in case["strides"]),
    )


def record(np, case, before, after, actual):
    original, _, _, _ = storage(np, case)
    expected = reference(np, case)
    for value, control in ((before, original), (after, original), (actual, expected)):
        require(
            value.dtype == control.dtype
            and value.shape == control.shape
            and value.tobytes() == control.tobytes(),
            "Copy source or result storage differs",
        )
    return {
        "case": case["id"],
        "dtype": case["dtype"],
        "shape": case["shape"],
        "sourceHash": hashlib.sha256(before.tobytes()).hexdigest(),
        "resultHash": hashlib.sha256(actual.tobytes()).hexdigest(),
    }


def run_cases(mx, np, directory, host=None):
    directory.mkdir(parents=True)
    records = []
    for case in cases():
        original, offset, _, _ = storage(np, case)
        source = mx.array(original)
        view = mx.as_strided(source, case["shape"], case["strides"], offset)
        mx.eval(view)
        before = np.array(source, copy=True)
        start = host.dispatch_count if host else 0
        actual = np.array(mx.contiguous(view), copy=True)
        after = np.array(source, copy=True)
        np.savez(
            directory / (case["id"] + ".npz"), before=before, after=after, actual=actual
        )
        item = record(np, case, before, after, actual)
        count = host.dispatch_count - start if host else 0
        require(count == (1 if host else 0), "Copy dispatch count differs")
        item.update(dispatchStart=start, dispatchCount=count)
        records.append(item)
        (directory / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def require_values(np, values, expected, dtype):
    require(
        isinstance(values, list) and len(values) == len(expected),
        "Copy physical values are incomplete",
    )
    if dtype == "bool":
        require(
            all(type(value) is bool for value in values),
            "Copy Boolean values must not be coerced",
        )
    else:
        require(
            dtype in {"int32", "uint32", "int64"}, "Copy physical type is unsupported"
        )
        bounds = np.iinfo(dtype)
        require(
            all(
                type(value) is int and bounds.min <= value <= bounds.max
                for value in values
            ),
            "Copy integer values must not be coerced",
        )
    require(values == expected, "Copy physical values differ")


def audit_case(np, case, item, event, directory):
    with np.load(directory / (case["id"] + ".npz"), allow_pickle=False) as saved:
        observed = record(np, case, saved["before"], saved["after"], saved["actual"])
    require(
        {
            key: value
            for key, value in item.items()
            if key not in {"dispatchStart", "dispatchCount"}
        }
        == observed,
        "Copy retained readback identity differs",
    )
    original, offset, low, high = storage(np, case)
    padding = max(0, 2 - len(case["shape"]))
    shape = [1] * padding + case["shape"]
    strides = [0] * padding + case["strides"]
    strides = [0 if size == 1 else step for size, step in zip(shape, strides)]
    count = math.prod(shape)
    metadata = {
        "shape": shape,
        "sourceStrides": strides,
        "destinationStrides": (
            [0] * padding
            + [math.prod(case["shape"][i + 1 :]) for i in range(len(case["shape"]))]
        ),
        "sourceOffset": -low,
        "destinationOffset": 0,
        "sourceCount": high - low + 1,
        "destinationCount": count,
        "preserveDestination": False,
        "workgroupCount": [(shape[-1] + 1) // 2, shape[-2], math.prod(shape[:-2])],
    }
    logical = "bool_" if case["dtype"] == "bool" else "uint32"
    physical = "bool" if logical == "bool_" and event["target"] == "metal" else "uint32"
    entry = "ggn2_dynamic_copy" + logical + logical
    native_entry = {"opengl": "main", "directx": "CSMain"}.get(event["target"], entry)
    require(
        event["entry"] == entry
        and event["threads"] == count
        and event["dispatchVersion"] == 3
        and event["copyMetadata"] == metadata
        and event["workgroupCount"] == metadata["workgroupCount"]
        and event["workgroupSize"] == [1, 1, 1],
        "Copy launch or layout differs",
    )
    request = event["details"]["request"]
    require(
        request["entryPoint"] == request["dispatch"]["entryPoint"] == native_entry
        and request["dispatch"]["workgroupCount"] == metadata["workgroupCount"]
        and request["dispatch"]["workgroupSize"] == [1, 1, 1],
        "Copy native request differs",
    )
    inputs = {}
    for name, value in event["inputs"].items():
        binding = request["buffers"][name]
        layout = binding["binding"]["metadata"]["scalarLayout"]
        member = layout.get("memberName", name).removeprefix(entry.rstrip("_") + "_")
        require(
            member not in inputs
            and value["dtype"] == binding["dtype"] == layout["elementType"]
            and value["shape"] == binding["shape"] == [len(value["values"])]
            and layout["elementStrideBytes"] == np.dtype(value["dtype"]).itemsize,
            "Copy physical binding differs",
        )
        inputs[member] = value
    source = original[offset + low : offset + high + 1]
    expected = reference(np, case).copy().reshape(-1)
    if logical == "bool_":
        source, expected = source.astype(physical), expected.astype(physical)
        guard = np.asarray(BOOLEAN_GUARD, dtype=physical).tolist()
    else:
        source, expected = source.view(np.uint32), expected.view(np.uint32)
        guard = COPY_GUARD
    values = {
        "src": source.tolist(),
        "dst": [0] * count + guard,
        "src_shape": shape,
        "src_strides": strides,
        "dst_strides": metadata["destinationStrides"],
        "ndim": [len(shape)],
        "src_offset": [-low],
        "dst_offset": [0],
    }
    require(set(inputs) == set(values), "Copy native inputs are incomplete")
    for name, expected_values in values.items():
        require_values(
            np, inputs[name]["values"], expected_values, inputs[name]["dtype"]
        )
    require(
        inputs["src"]["dtype"] == inputs["dst"]["dtype"] == physical,
        "Copy physical source or destination type differs",
    )
    require_values(np, event["copyValues"], expected.tolist(), physical)
    require_values(np, event["copyGuardWords"], guard, physical)
    audit_native_execution(event)


def verify(root, packages, output):
    import mlx.core as mx
    import numpy as np

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    preparation = verify_prepared(root)
    require(not mx.is_available(mx.gpu), "A different GPU backend is already available")
    result = {
        "passed": False,
        "commit": COMMIT,
        "adaptation": preparation,
        "fullUpstreamSuite": False,
    }
    host = None
    try:
        mx.set_default_device(mx.cpu)
        cpu = run_cases(mx, np, output / "cpu")
        host = HostRuntime(packages, output / "trace.jsonl", retain_native_modules=True)
        host.install()
        native = run_cases(mx, np, output / "native", host)
        trace = [json.loads(line) for line in host.trace.read_text().splitlines()]
        require(
            len(trace) == host.dispatch_count == len(cases()),
            "Copy trace coverage differs",
        )
        for index, (case, original, translated, event) in enumerate(
            zip(cases(), cpu, native, trace)
        ):
            require(
                {
                    k: v
                    for k, v in original.items()
                    if k not in {"dispatchStart", "dispatchCount"}
                }
                == {
                    k: v
                    for k, v in translated.items()
                    if k not in {"dispatchStart", "dispatchCount"}
                },
                "CPU and generated copy results differ",
            )
            require(
                translated["dispatchStart"] == index
                and translated["dispatchCount"] == 1,
                "Copy trace boundary differs",
            )
            audit_case(np, case, translated, event, output / "native")
        require(preparation == verify_prepared(root), "Prepared MLX source changed")
        result.update(
            passed=True,
            target=host.target,
            casesPerPath=len(native),
            dispatchCount=len(trace),
            cpu=cpu,
            native=native,
        )
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        (output / "evidence.json").write_text(json.dumps(result, indent=2) + "\n")
        if host is not None:
            close = getattr(host.executor.runtime_adapter.runtime, "close", None)
            if close:
                close()
    return result


def run_bounded(root, packages, output):
    output = Path(output).resolve()
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[4] / "tools/run_bounded_command.py"),
        "--timeout-seconds",
        "600",
        "--label",
        "MLX large copy verification",
        "--",
        sys.executable,
        "-m",
        __name__,
        "--mlx-root",
        str(Path(root).resolve()),
        "--packages",
        str(Path(packages).resolve()),
        "--output-dir",
        str(output),
    ]
    with output.with_suffix(".stdout").open("w") as stdout, output.with_suffix(
        ".stderr"
    ).open("w") as stderr:
        process = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    output.with_suffix(".command.json").write_text(
        json.dumps({"command": command, "returncode": process.returncode}, indent=2)
        + "\n"
    )
    require(
        process.returncode == 0, "Large copy verification failed; inspect retained logs"
    )
    evidence = json.loads((output / "evidence.json").read_text())
    target = json.loads((Path(packages) / "index.json").read_text())["target"]
    require(
        evidence.get("passed") is True
        and evidence.get("commit") == COMMIT
        and evidence.get("target") == target
        and evidence.get("casesPerPath") == len(cases())
        and evidence.get("dispatchCount") == len(cases())
        and evidence.get("fullUpstreamSuite") is False,
        "Large copy evidence is incomplete",
    )
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mlx-root", "packages", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    verify(args.mlx_root, args.packages, args.output_dir)
