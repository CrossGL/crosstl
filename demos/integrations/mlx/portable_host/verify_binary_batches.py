"""Verify batched MLX arithmetic and comparisons against CPU and native storage."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_native_execution,
    require,
)
from demos.integrations.mlx.portable_host.packages import (
    BINARY_ENTRIES,
    COMPARISON_ENTRIES,
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
from demos.integrations.mlx.portable_host.verify_large_copies import require_values

BATCH_SIZE = 65535
OPERATIONS = {
    "Add": "add",
    "Subtract": "subtract",
    "Multiply": "multiply",
    "Minimum": "minimum",
    "Maximum": "maximum",
    "Divide": "divide",
    "Equal": "equal",
    "NotEqual": "not_equal",
    "Less": "less",
    "LessEqual": "less_equal",
    "Greater": "greater",
    "GreaterEqual": "greater_equal",
    "LogicalAnd": "logical_and",
    "LogicalOr": "logical_or",
}


def cases():
    entries = {**BINARY_ENTRIES, **COMPARISON_ENTRIES}
    specifications = [
        (entry, 65536, "dense") for entry in entries if "NaNEqual" not in entry
    ]
    specifications.extend(
        (
            ("vv_Adduint32", 65535, "dense"),
            ("vv_Subtractint32", 131075, "dense"),
            ("vv_Dividefloat32", 131075, "dense"),
            ("vv_Addfloat32", 256 * 257, "transpose"),
            ("vv_Lessuint32", 65536, "reverse"),
            ("vv_LogicalAndbool_", 256 * 512, "broadcast"),
        )
    )
    return [
        {
            "id": f"{entry}-{count}-{layout}",
            "entry": entry,
            "dtype": entries[entry],
            "count": count,
            "layout": layout,
            "operation": entry[3 : -len(entries[entry])],
        }
        for entry, count, layout in specifications
    ]


def operand(np, case, side):
    shape, strides = [case["count"]], [1]
    if case["layout"] == "transpose":
        shape, strides = [256, 257], [1, 256] if side == 0 else [257, 1]
    elif case["layout"] == "reverse" and side == 0:
        strides = [-1]
    elif case["layout"] == "broadcast":
        shape, strides = [256, 512], [0, 1] if side == 0 else [1, 0]
    extents = [(size - 1) * step for size, step in zip(shape, strides)]
    low = sum(min(value, 0) for value in extents)
    high = sum(max(value, 0) for value in extents)
    offset = (3 if side == 0 else 7) - low
    words = np.arange(offset + high + 4, dtype=np.uint32) * np.uint32(104729 + side * 8)
    if case["dtype"] == "bool_":
        storage = words % (7 if side == 0 else 11) == 0
    elif case["dtype"] == "float32":
        storage = (words % 4093).astype(np.float32) / np.float32(16) - np.float32(128)
        if side == 1 and case["operation"] == "Divide":
            storage = np.left_shift(np.uint32(1), words % 5).astype(np.float32)
    elif case["dtype"] == "int32":
        storage = (words % 2003).astype(np.int32) - np.int32(1001)
    else:
        storage = words
    view = np.ndarray(
        tuple(shape),
        dtype=storage.dtype,
        buffer=storage,
        offset=offset * storage.itemsize,
        strides=tuple(step * storage.itemsize for step in strides),
    )
    return storage, view, shape, strides, offset, low, high


def reference(np, case):
    return getattr(np, OPERATIONS[case["operation"]])(
        operand(np, case, 0)[1], operand(np, case, 1)[1]
    )


def batch_sizes(case):
    return [
        min(BATCH_SIZE, case["count"] - first)
        for first in range(0, case["count"], BATCH_SIZE)
    ]


def copy_count(case):
    return {"dense": 0, "transpose": 1, "reverse": 1, "broadcast": 2}[case["layout"]]


def record(np, case, saved):
    for side in range(2):
        storage = operand(np, case, side)[0]
        require(
            equal_storage(saved[f"before{side}"], storage)
            and equal_storage(saved[f"after{side}"], storage),
            "Binary source storage changed or differs from its workload",
        )
    require(
        equal_storage(saved["actual"], reference(np, case)),
        "Binary result differs from the exact reference",
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
        stores, views, saved = [], [], {}
        for side in range(2):
            storage, _, shape, strides, offset, _, _ = operand(np, case, side)
            source = mx.array(storage)
            view = mx.as_strided(source, shape, strides, offset)
            mx.eval(view)
            stores.append(source)
            views.append(view)
            saved[f"before{side}"] = np.array(source, copy=True)
        start = host.dispatch_count if host else 0
        saved["actual"] = np.array(
            getattr(mx, OPERATIONS[case["operation"]])(*views), copy=True
        )
        for side, source in enumerate(stores):
            saved[f"after{side}"] = np.array(source, copy=True)
        np.savez(directory / (case["id"] + ".npz"), **saved)
        item = record(np, case, saved)
        count = host.dispatch_count - start if host else 0
        require(
            count == (copy_count(case) + len(batch_sizes(case)) if host else 0),
            "Binary dispatch coverage differs",
        )
        item.update(dispatchStart=start, dispatchCount=count)
        records.append(item)
        (directory / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def bindings(np, event, entry, count, grid):
    native = {"opengl": "main", "directx": "CSMain"}.get(event["target"], entry)
    request = event["details"]["request"]
    require(
        event["entry"] == entry
        and event["threads"] == count
        and event["dispatchVersion"] == 3
        and event["workgroupCount"] == grid
        and event["workgroupSize"] == [1, 1, 1]
        and "threadGridSize" not in event,
        "Binary or materialization launch differs",
    )
    require(
        request["target"] == event["target"]
        and request["entryPoint"] == request["dispatch"]["entryPoint"] == native
        and request["dispatch"]["workgroupCount"] == grid
        and request["dispatch"]["workgroupSize"] == [1, 1, 1],
        "Binary native request differs",
    )
    require(
        set(event["inputs"]) == set(request["buffers"]),
        "Binary request bindings differ",
    )
    result = {}
    for name, value in event["inputs"].items():
        binding = request["buffers"][name]
        layout = binding["binding"]["metadata"]["scalarLayout"]
        member = layout.get("memberName", name).removeprefix(entry.rstrip("_") + "_")
        require(
            member not in result
            and value["dtype"] == binding["dtype"] == layout["elementType"]
            and value["shape"] == binding["shape"] == [len(value["values"])]
            and layout["elementStrideBytes"] == np.dtype(value["dtype"]).itemsize,
            "Binary physical binding differs",
        )
        result[member] = value
    return result


def audit_copy(np, case, side, event):
    storage, view, shape, strides, offset, low, high = operand(np, case, side)
    padding = max(0, 2 - len(shape))
    destination = [0] * padding + [math.prod(shape[i + 1 :]) for i in range(len(shape))]
    shape, strides = [1] * padding + shape, [0] * padding + strides
    grid = [(shape[-1] + 1) // 2, shape[-2], math.prod(shape[:-2])]
    metadata = {
        "shape": shape,
        "sourceStrides": strides,
        "destinationStrides": destination,
        "sourceOffset": -low,
        "destinationOffset": 0,
        "sourceCount": high - low + 1,
        "destinationCount": case["count"],
        "preserveDestination": False,
        "workgroupCount": grid,
    }
    require(event["copyMetadata"] == metadata, "Binary materialization layout differs")
    logical = "bool_" if case["dtype"] == "bool_" else "uint32"
    physical = "bool" if logical == "bool_" and event["target"] == "metal" else "uint32"
    inputs = bindings(
        np, event, "ggn2_dynamic_copy" + logical + logical, case["count"], grid
    )
    source, expected = storage[offset + low : offset + high + 1], view.copy().reshape(
        -1
    )
    if logical == "bool_":
        source, expected = source.astype(physical), expected.astype(physical)
        guard = np.asarray(BOOLEAN_GUARD, dtype=physical).tolist()
    else:
        source, expected, guard = (
            source.view(np.uint32),
            expected.view(np.uint32),
            COPY_GUARD,
        )
    values = {
        "src": source.tolist(),
        "dst": [0] * case["count"] + guard,
        "src_shape": shape,
        "src_strides": strides,
        "dst_strides": destination,
        "ndim": [len(shape)],
        "src_offset": [-low],
        "dst_offset": [0],
    }
    require(set(inputs) == set(values), "Binary materialization inputs are incomplete")
    require(
        inputs["src"]["dtype"] == inputs["dst"]["dtype"] == physical,
        "Binary materialization storage type differs",
    )
    for name, expected_values in values.items():
        require_values(
            np, inputs[name]["values"], expected_values, inputs[name]["dtype"]
        )
    require_values(np, event["copyValues"], expected.tolist(), physical)
    require_values(np, event["copyGuardWords"], guard, physical)
    audit_native_execution(event)


def audit_case(np, case, item, events, directory):
    with np.load(directory / (case["id"] + ".npz"), allow_pickle=False) as saved:
        observed = record(np, case, saved)
    require(
        {k: v for k, v in item.items() if k not in {"dispatchStart", "dispatchCount"}}
        == observed,
        "Binary retained readback identity differs",
    )
    sizes = batch_sizes(case)
    require(
        len(events) == item["dispatchCount"] == copy_count(case) + len(sizes),
        "Binary batch coverage differs",
    )
    copies = 0
    sources = []
    for side in range(2):
        view = operand(np, case, side)[1]
        sources.append(view.reshape(-1))
        if not view.flags.c_contiguous:
            audit_copy(np, case, side, events[copies])
            copies += 1
    require(copies == copy_count(case), "Binary materialization count differs")
    expected = reference(np, case).reshape(-1)
    first = 0
    for event, size in zip(events[copies:], sizes):
        inputs = bindings(np, event, case["entry"], size, [size, 1, 1])
        require(
            set(inputs) == {"a", "b", "c", "size"}, "Binary bindings are incomplete"
        )
        for value in inputs.values():
            physical_values(np, value["values"], value["dtype"])
        require(
            inputs["size"]["dtype"] == "uint32" and inputs["size"]["values"] == [size],
            "Binary uploaded size differs",
        )
        boolean = "bool" if event["target"] == "metal" else "uint32"
        source_type = boolean if case["dtype"] == "bool_" else case["dtype"]
        output_type = boolean if expected.dtype == np.bool_ else str(expected.dtype)
        require(
            inputs["a"]["dtype"] == inputs["b"]["dtype"] == source_type
            and inputs["c"]["dtype"] == output_type,
            "Binary storage width differs",
        )
        for name, source in zip(("a", "b"), sources):
            require(
                equal_storage(
                    np.asarray(inputs[name]["values"], dtype=source_type),
                    source[first : first + size].astype(source_type),
                ),
                "Binary batch input offset differs",
            )
        guard = np.asarray(
            BOOLEAN_GUARD if expected.dtype == np.bool_ else COPY_GUARD,
            dtype=boolean if expected.dtype == np.bool_ else "uint32",
        )
        guard = (
            guard.view("float32")
            if output_type == "float32"
            else guard.astype(output_type)
        )
        require(
            equal_storage(
                physical_values(np, event["binaryValues"], output_type),
                expected[first : first + size].astype(output_type),
            ),
            "Native binary output differs",
        )
        require(
            equal_storage(
                physical_values(np, event["binaryGuardValues"], output_type), guard
            )
            and inputs["c"]["values"] == [0] * size + guard.tolist(),
            "Native binary guard or initialization differs",
        )
        audit_native_execution(event)
        first += size
    require(first == case["count"], "Binary output coverage differs")


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
                "CPU and translated binary results differ",
            )
            require(
                translated["dispatchStart"] == cursor, "Binary trace boundary differs"
            )
            end = cursor + translated["dispatchCount"]
            audit_case(np, case, translated, trace[cursor:end], output / "native")
            cursor = end
        require(
            cursor == len(trace) == host.dispatch_count,
            "Binary trace contains unexpected dispatches",
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
        "MLX batched binary verification",
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
        raise RuntimeError("Batched binary verification failed; inspect retained logs")
    evidence = json.loads((output / "evidence.json").read_text())
    target = json.loads((Path(packages) / "index.json").read_text())["target"]
    require(
        evidence.get("passed") is True
        and evidence.get("commit") == COMMIT
        and evidence.get("target") == target
        and evidence.get("casesPerPath") == len(cases())
        and evidence.get("dispatchCount")
        == sum(copy_count(case) + len(batch_sizes(case)) for case in cases())
        and evidence.get("fullUpstreamSuite") is False,
        "Batched binary evidence is incomplete",
    )
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    verify(args.mlx_root, args.packages, args.output_dir)
