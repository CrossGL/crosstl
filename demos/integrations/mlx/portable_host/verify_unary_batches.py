"""Verify unary batches without flattening MLX's stored-contiguous layouts."""

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
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.runtime import (
    BOOLEAN_GUARD,
    COPY_GUARD,
    HostRuntime,
)
from demos.integrations.mlx.portable_host.verify_cast_batches import (
    equal_storage,
    physical_values,
)

OPERATIONS = {
    "Abs": "abs",
    "Negative": "negative",
    "Square": "square",
    "Round": "round",
    "Ceil": "ceil",
    "Floor": "floor",
    "Sign": "sign",
    "Sqrt": "sqrt",
    "Rsqrt": "rsqrt",
    "LogicalNot": "logical_not",
}
BATCH_SIZE = 65535


def cases():
    specifications = [(op, 65536, "dense") for op in OPERATIONS]
    specifications += [
        ("Abs", 65535, "dense"),
        ("Negative", 131075, "dense"),
        ("LogicalNot", 131075, "dense"),
        ("Square", 256 * 257, "transpose"),
        ("LogicalNot", 256 * 257, "transpose"),
        ("Round", 3 * 65536, "broadcast"),
        ("LogicalNot", 3 * 65536, "broadcast"),
        ("Negative", 3 * 65536, "scalar-broadcast"),
    ]
    return [
        {
            "id": f"{op}-{count}-{layout}",
            "operation": op,
            "count": count,
            "layout": layout,
            "dtype": "bool_" if op == "LogicalNot" else "float32",
            "entry": (
                "v_" + op + ("bool_bool_" if op == "LogicalNot" else "float32float32")
            ),
        }
        for op, count, layout in specifications
    ]


def operand(np, case):
    shape, strides, count = [case["count"]], [1], case["count"]
    if case["layout"] == "transpose":
        shape, strides = [256, 257], [1, 256]
    elif case["layout"] == "broadcast":
        shape, strides, count = [3, 65536], [0, 1], 65536
    elif case["layout"] == "scalar-broadcast":
        strides, count = [0], 1
    offset = 7
    words = np.arange(offset + count + 5, dtype=np.uint32) * np.uint32(104729)
    if case["dtype"] == "bool_":
        storage = words % 7 == 0
    elif case["operation"] in {"Sqrt", "Rsqrt"}:
        storage = np.left_shift(np.uint32(1), (words % 12) * 2).astype(np.float32)
    else:
        storage = (words % 4093).astype(np.float32) / np.float32(16) - np.float32(128)
    view = np.ndarray(
        tuple(shape),
        storage.dtype,
        storage,
        offset * storage.itemsize,
        tuple(step * storage.itemsize for step in strides),
    )
    return storage, view, shape, strides, offset, count


def apply(np, case, values):
    operation = case["operation"]
    if operation == "Rsqrt":
        return np.float32(1) / np.sqrt(values)
    return getattr(np, OPERATIONS[operation])(values)


def batch_sizes(case):
    count = (
        1
        if case["layout"] == "scalar-broadcast"
        else (65536 if case["layout"] == "broadcast" else case["count"])
    )
    return [min(BATCH_SIZE, count - first) for first in range(0, count, BATCH_SIZE)]


def record(np, case, saved):
    storage, view, _, _, _, _ = operand(np, case)
    require(
        equal_storage(saved["before"], storage)
        and equal_storage(saved["after"], storage),
        "Unary source storage changed",
    )
    require(
        equal_storage(saved["actual"], apply(np, case, view)),
        "Unary result differs from the exact reference",
    )
    return {
        "case": case["id"],
        "sourcePreserved": True,
        "hashes": {
            name: hashlib.sha256(value.tobytes()).hexdigest()
            for name, value in saved.items()
        },
    }


def run_cases(mx, np, directory, host=None):
    directory.mkdir(parents=True)
    records = []
    for case in cases():
        storage, _, shape, strides, offset, _ = operand(np, case)
        source = mx.array(storage)
        view = mx.as_strided(source, shape, strides, offset)
        mx.eval(view)
        saved = {"before": np.array(source, copy=True)}
        start = host.dispatch_count if host else 0
        output = getattr(mx, OPERATIONS[case["operation"]])(view)
        exposed = np.asarray(output)
        saved["actual"] = exposed.copy()
        saved["after"] = np.array(source, copy=True)
        observed_strides = [step // exposed.itemsize for step in exposed.strides]
        require(observed_strides == strides, "Unary output lost its stored layout")
        np.savez(directory / (case["id"] + ".npz"), **saved)
        item = record(np, case, saved)
        count = host.dispatch_count - start if host else 0
        require(
            count == (len(batch_sizes(case)) if host else 0),
            "Unary dispatch coverage differs",
        )
        item.update(dispatchStart=start, dispatchCount=count, strides=observed_strides)
        records.append(item)
        (directory / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def physical_type(case, target):
    return (
        ("bool" if target == "metal" else "uint32")
        if case["dtype"] == "bool_"
        else "float32"
    )


def guard_values(np, case, target):
    if case["dtype"] == "bool_":
        return np.asarray(BOOLEAN_GUARD, dtype=physical_type(case, target))
    return np.asarray(COPY_GUARD, dtype="uint32").view("float32")


def audit_case(np, case, item, events, directory):
    with np.load(directory / (case["id"] + ".npz"), allow_pickle=False) as saved:
        observed = record(np, case, saved)
    require(
        {
            k: v
            for k, v in item.items()
            if k not in {"dispatchStart", "dispatchCount", "strides"}
        }
        == observed,
        "Unary retained readback identity differs",
    )
    storage, _, _, strides, offset, stored_count = operand(np, case)
    require(item["strides"] == strides, "Unary retained strides differ")
    sizes = batch_sizes(case)
    require(
        len(events) == item["dispatchCount"] == len(sizes),
        "Unary batch coverage differs",
    )
    first = 0
    for size, event in zip(sizes, events):
        target, entry = event["target"], case["entry"]
        native = {"opengl": "main", "directx": "CSMain"}.get(target, entry)
        request = event["details"]["request"]
        grid = [size, 1, 1]
        require(
            event["entry"] == entry
            and event["threads"] == size
            and event["dispatchVersion"] == 3
            and event["workgroupCount"] == grid
            and event["workgroupSize"] == [1, 1, 1]
            and "threadGridSize" not in event,
            "Unary launch differs",
        )
        require(
            request["target"] == target
            and request["entryPoint"] == request["dispatch"]["entryPoint"] == native
            and request["dispatch"]["workgroupCount"] == grid
            and request["dispatch"]["workgroupSize"] == [1, 1, 1],
            "Unary native request differs",
        )
        require(
            set(event["inputs"]) == set(request["buffers"]), "Unary bindings differ"
        )
        inputs = {}
        for name, value in event["inputs"].items():
            binding = request["buffers"][name]
            layout = binding["binding"]["metadata"]["scalarLayout"]
            member = layout.get("memberName", name).removeprefix(
                entry.rstrip("_") + "_"
            )
            member = {"in_": "in", "out_": "out"}.get(member, member)
            require(
                member not in inputs
                and value["dtype"] == binding["dtype"] == layout["elementType"]
                and value["shape"] == binding["shape"] == [len(value["values"])]
                and layout["elementStrideBytes"] == np.dtype(value["dtype"]).itemsize,
                "Unary physical binding differs",
            )
            physical_values(np, value["values"], value["dtype"])
            inputs[member] = value
        dtype = physical_type(case, target)
        require(
            set(inputs) == {"in", "out", "size"}
            and inputs["in"]["dtype"] == inputs["out"]["dtype"] == dtype
            and inputs["size"]["dtype"] == "uint32"
            and inputs["size"]["values"] == [size],
            "Unary buffer signature differs",
        )
        source = storage[offset + first : offset + first + size]
        expected = apply(np, case, source).astype(dtype)
        guard = guard_values(np, case, target)
        require(
            equal_storage(
                np.asarray(inputs["in"]["values"], dtype=dtype), source.astype(dtype)
            ),
            "Unary input batch offset differs",
        )
        require(
            equal_storage(physical_values(np, event["unaryValues"], dtype), expected),
            "Unary native output differs",
        )
        require(
            equal_storage(physical_values(np, event["unaryGuardValues"], dtype), guard)
            and equal_storage(
                np.asarray(inputs["out"]["values"], dtype=dtype),
                np.concatenate((np.zeros(size, dtype=dtype), guard)),
            ),
            "Unary output initialization or guard differs",
        )
        audit_native_execution(event)
        first += size
    require(first == stored_count, "Unary stored output coverage differs")


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
        cursor = 0
        for case, original, translated in zip(cases(), cpu, native):
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
                "CPU and translated unary results differ",
            )
            require(
                translated["dispatchStart"] == cursor, "Unary trace boundary differs"
            )
            end = cursor + translated["dispatchCount"]
            audit_case(np, case, translated, trace[cursor:end], output / "native")
            cursor = end
        require(
            cursor == len(trace) == host.dispatch_count,
            "Unary trace has unexpected dispatches",
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
        "MLX batched unary verification",
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
        raise RuntimeError("Batched unary verification failed; inspect retained logs")
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
        "Batched unary evidence is incomplete",
    )
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    verify(args.mlx_root, args.packages, args.output_dir)
