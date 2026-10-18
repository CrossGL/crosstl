"""Verify large MLX casts through unchanged pointwise kernels and native readbacks."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_native_execution,
    require,
)
from demos.integrations.mlx.portable_host.packages import CAST_ENTRIES
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.runtime import (
    BOOLEAN_GUARD,
    COPY_GUARD,
    HostRuntime,
)

BATCH_SIZE = 65535


def cases():
    specifications = [
        (source, destination, 65536, 3) for source, destination in CAST_ENTRIES.values()
    ]
    specifications.extend(
        (
            ("uint32", "float32", 131075, 7),
            ("float32", "int32", 65535, 5),
            ("bool_", "float32", 65536, 11),
            ("float32", "bool_", 131075, 9),
            ("uint32", "uint32", 65536, 7),
        )
    )
    return [
        {
            "id": f"{source}-{destination}-{count}-{offset}",
            "source": source,
            "destination": destination,
            "count": count,
            "offset": offset,
            "shape": [5, 26215] if count == 131075 else [count],
        }
        for source, destination, count, offset in specifications
    ]


def source_storage(np, case):
    words = np.arange(case["offset"] + case["count"] + 3, dtype=np.uint32) * np.uint32(
        104729
    )
    if case["source"] == "bool_":
        return words % 7 == 0
    if case["source"] == "float32":
        if case["destination"] == "bool_":
            return (words % 19).astype(np.float32) - np.float32(9)
        values = (words % 131071).astype(np.float32) / np.float32(16)
        return values - np.float32(4095.5) if case["destination"] == "int32" else values
    return words.view(np.dtype(case["source"]))


def reference(np, case):
    source = source_storage(np, case)[
        case["offset"] : case["offset"] + case["count"]
    ].reshape(case["shape"])
    return source, source.astype(case["destination"])


def batch_sizes(case):
    if case["source"] == case["destination"]:
        return []
    return [
        min(BATCH_SIZE, case["count"] - first)
        for first in range(0, case["count"], BATCH_SIZE)
    ]


def equal_storage(actual, expected):
    return (
        actual.dtype == expected.dtype
        and actual.shape == expected.shape
        and actual.tobytes() == expected.tobytes()
    )


def physical_values(np, values, dtype):
    require(type(values) is list, "Cast physical values must be a list")
    if dtype == "bool":
        valid = all(type(value) is bool for value in values)
    elif dtype in {"int32", "uint32"}:
        limits = np.iinfo(dtype)
        valid = all(
            type(value) is int and limits.min <= value <= limits.max for value in values
        )
    elif dtype == "float32":
        limits = np.finfo(dtype)
        valid = all(
            type(value) in {int, float}
            and np.isfinite(value)
            and abs(value) <= float(limits.max)
            and float(np.float32(value)) == value
            for value in values
        )
    else:
        valid = False
    require(valid, "Cast physical values do not match their dtype")
    return np.asarray(values, dtype=dtype)


def record(np, case, before, after, actual):
    source, expected = reference(np, case)
    require(equal_storage(before, source), "Cast source does not match its workload")
    require(equal_storage(after, before), "Cast changed its source storage")
    require(
        equal_storage(actual, expected), "Cast result differs from the exact reference"
    )
    return {
        "case": case["id"],
        "sourceDtype": case["source"],
        "destinationDtype": case["destination"],
        "shape": case["shape"],
        "sourceHash": hashlib.sha256(before.tobytes()).hexdigest(),
        "resultHash": hashlib.sha256(actual.tobytes()).hexdigest(),
        "sourcePreserved": True,
    }


def run_cases(mx, np, directory, host=None):
    directory.mkdir(parents=True)
    records = []
    for case in cases():
        storage = mx.array(source_storage(np, case))
        source = storage[case["offset"] : case["offset"] + case["count"]].reshape(
            case["shape"]
        )
        mx.eval(source)
        before = np.array(source, copy=True)
        start = host.dispatch_count if host else 0
        actual = np.array(source.astype(getattr(mx, case["destination"])), copy=True)
        after = np.array(source, copy=True)
        np.savez(
            directory / (case["id"] + ".npz"), before=before, after=after, actual=actual
        )
        item = record(np, case, before, after, actual)
        count = host.dispatch_count - start if host else 0
        require(
            count == (len(batch_sizes(case)) if host else 0),
            "Cast dispatch count differs",
        )
        item.update(dispatchStart=start, dispatchCount=count)
        records.append(item)
        (directory / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def audit_case(np, case, item, events, directory):
    with np.load(directory / (case["id"] + ".npz"), allow_pickle=False) as saved:
        observed = record(np, case, saved["before"], saved["after"], saved["actual"])
    require(
        {
            key: value
            for key, value in item.items()
            if key not in {"dispatchStart", "dispatchCount"}
        }
        == observed,
        "Cast retained readback identity differs",
    )
    sizes = batch_sizes(case)
    require(
        len(events) == len(sizes) == item["dispatchCount"],
        "Cast batch coverage differs",
    )
    source, expected = reference(np, case)
    source, expected = source.reshape(-1), expected.reshape(-1)
    first = 0
    entry = "v_copy" + case["source"] + case["destination"]
    for event, size in zip(events, sizes):
        request = event["details"]["request"]
        native_entry = {"opengl": "main", "directx": "CSMain"}.get(
            event["target"], entry
        )
        require(
            event["entry"] == entry
            and event["threads"] == size
            and event["workgroupCount"] == [size, 1, 1]
            and event["workgroupSize"] == [1, 1, 1]
            and event["dispatchVersion"] == 3,
            "Cast batch launch differs",
        )
        require(
            request["target"] == event["target"]
            and request["entryPoint"] == native_entry
            and request["dispatch"]["entryPoint"] == native_entry
            and request["dispatch"]["workgroupCount"] == [size, 1, 1]
            and request["dispatch"]["workgroupSize"] == [1, 1, 1],
            "Cast native request differs",
        )
        inputs = {}
        for name, value in event["inputs"].items():
            binding = event["details"]["request"]["buffers"][name]
            layout = binding["binding"]["metadata"]["scalarLayout"]
            member = layout.get("memberName", name).removeprefix(
                entry.rstrip("_") + "_"
            )
            require(member not in inputs, "Duplicate cast binding")
            require(
                binding["dtype"] == value["dtype"] == layout["elementType"]
                and binding["shape"] == value["shape"] == [len(value["values"])]
                and layout["elementStrideBytes"] == np.dtype(value["dtype"]).itemsize,
                "Cast physical storage differs",
            )
            inputs[member] = value
            physical_values(np, value["values"], value["dtype"])
        require(set(inputs) == {"src", "dst", "size"}, "Cast bindings are incomplete")
        require(
            inputs["size"]["dtype"] == "uint32" and inputs["size"]["values"] == [size],
            "Cast uploaded size differs",
        )
        for name, logical in (("src", case["source"]), ("dst", case["destination"])):
            physical = (
                ("bool" if event["target"] == "metal" else "uint32")
                if logical == "bool_"
                else logical
            )
            require(
                inputs[name]["dtype"] == physical,
                "Cast logical and physical dtypes disagree",
            )
        source_values = np.asarray(
            inputs["src"]["values"], dtype=inputs["src"]["dtype"]
        )
        require(
            equal_storage(
                source_values,
                source[first : first + size].astype(inputs["src"]["dtype"]),
            ),
            "Cast batch input offset differs",
        )
        physical = inputs["dst"]["dtype"]
        guard = np.asarray(
            BOOLEAN_GUARD if case["destination"] == "bool_" else COPY_GUARD,
            dtype="bool" if case["destination"] == "bool_" else "uint32",
        )
        guard = (
            guard.view("float32")
            if case["destination"] == "float32"
            else guard.astype(physical)
        )
        actual = physical_values(np, event["castValues"], physical)
        physical_values(np, event["castGuardValues"], physical)
        require(
            equal_storage(actual, expected[first : first + size].astype(physical)),
            "Native cast output differs",
        )
        require(
            event["castGuardValues"] == guard.tolist()
            and inputs["dst"]["values"] == [0] * size + guard.tolist(),
            "Native cast guard or initialization differs",
        )
        audit_native_execution(event)
        first += size


def verify(root, packages, output):
    import mlx.core as mx
    import numpy as np

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    preparation = verify_prepared(root)
    if mx.is_available(mx.gpu):
        raise RuntimeError("A different GPU backend is already available")
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
        cursor = 0
        for case, original, translated in zip(cases(), cpu, native):
            require(
                {
                    key: value
                    for key, value in original.items()
                    if key not in {"dispatchStart", "dispatchCount"}
                }
                == {
                    key: value
                    for key, value in translated.items()
                    if key not in {"dispatchStart", "dispatchCount"}
                },
                "CPU and translated cast results differ",
            )
            require(
                translated["dispatchStart"] == cursor, "Cast trace boundary differs"
            )
            end = cursor + translated["dispatchCount"]
            audit_case(np, case, translated, trace[cursor:end], output / "native")
            cursor = end
        require(
            cursor == len(trace) == host.dispatch_count,
            "Cast trace contains unexpected dispatches",
        )
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
        "MLX batched cast verification",
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
    if process.returncode:
        raise RuntimeError("Batched cast verification failed; inspect retained logs")
    evidence = json.loads((output / "evidence.json").read_text())
    target = json.loads((Path(packages) / "index.json").read_text())["target"]
    require(
        evidence.get("passed") is True
        and evidence.get("commit") == COMMIT
        and evidence.get("target") == target
        and evidence.get("casesPerPath") == len(cases())
        and evidence.get("dispatchCount")
        == sum(len(batch_sizes(case)) for case in cases())
        and evidence.get("fullUpstreamSuite") is False,
        "Batched cast evidence is incomplete",
    )
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    verify(args.mlx_root, args.packages, args.output_dir)
