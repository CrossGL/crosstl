"""Verify cumulative host evaluation through unchanged translated scan entries."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from demos.integrations.mlx.portable_host import scan_dispatch, scan_layout
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_input_bindings,
    audit_native_execution,
    require,
)
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.runtime import HostRuntime

OPERATIONS = {"sum": "cumsum", "min": "cummin", "max": "cummax"}


def cases():
    return [
        {
            "id": f"{operation}-{dtype}-{axis}-{layout}",
            "operation": operation,
            "dtype": dtype,
            "axisSize": axis,
            "layout": layout,
            # backend.cpp scans a size-one axis of any stride as contiguous rows.
            "entry": (
                f"{'strided' if layout == 'strided-column' and axis > 1 else 'contig'}"
                f"_scan_inclusive_{operation}_{dtype}_{dtype}"
            ),
        } for dtype in (
            "float32",
            "int32",
        ) for operation in OPERATIONS for axis, layout in (
            (1, "row"),
            (33, "row"),
            (129, "row"),
            (257, "column"),
            *((axis, "strided-column") for axis in (1, 31, 32, 33, 65)),
        )
    ]


def operand(np, case):
    strided = case["layout"] == "strided-column"
    size = (33 if strided else 3) * case["axisSize"]
    storage = ((np.arange(size + 12, dtype=np.int32) * 7) % 19 - 9).astype(
        case["dtype"]
    )
    if case["dtype"] == "float32":
        storage *= np.float32(0.125)
    shape = (case["axisSize"], 33) if strided else (3, case["axisSize"])
    view = storage[7 : 7 + size].reshape(shape)
    if case["layout"] == "column":
        view = view.T
    return storage, view


def reference(np, case, source):
    axis = 1 if case["layout"] == "row" else 0
    op = {"sum": np.add, "min": np.minimum, "max": np.maximum}[case["operation"]]
    return op.accumulate(source, axis=axis, dtype=source.dtype)


def audit_metadata(event, names, axis, stride):
    values = {"axis_size": (2, axis)}
    if stride > 1:
        values.update(stride=(3, stride), stride_blocks=(4, (stride + 31) // 32))
    conditions = event["details"].get("metadataPreconditions", [])
    require(
        len(conditions) == len(values)
        and {item["binding"] for item in conditions}
        == {names[name] for name in values},
        "Scan checked metadata receipts are incomplete",
    )
    for name, (ordinal, value) in values.items():
        (receipt,) = (item for item in conditions if item["binding"] == names[name])
        requirement = {
            "dtype": "uint64",
            "minimum": value,
            "maximum": value,
            "sourceResource": {
                "schemaVersion": 1,
                "backend": "metal",
                "entryPoint": event["entry"],
                "parameter": name,
                "parameterIndex": ordinal,
            },
        }
        metadata = event["details"]["request"]["buffers"][names[name]]["binding"][
            "metadata"
        ]
        observed = event["details"]["inputObservations"][names[name]]["uploadSnapshot"]
        require(
            receipt["requirement"]
            == metadata["provenance"].get("metadataPrecondition")
            == requirement
            and type(receipt["value"]) is int
            and receipt["value"] == value,
            "Scan checked metadata differs from its source contract",
        )
        snapshot = receipt["uploadSnapshot"]
        require(
            snapshot["dtype"] == "uint64"
            and snapshot["shape"] == [1]
            and snapshot.get("encoding") is None
            and snapshot["elementCount"] == 1
            and snapshot["sizeBytes"] == observed["sizeBytes"] == 8
            and snapshot["sha256"] == observed["sha256"],
            "Scan checked metadata differs from its observed upload",
        )


def audit_event(np, case, event, saved):
    storage, view = operand(np, case)
    expected = reference(np, case, view)
    require(
        saved["before"].tobytes() == saved["after"].tobytes() == storage.tobytes(),
        "Scan source storage changed",
    )
    require(
        saved["actual"].dtype == expected.dtype
        and saved["actual"].shape == expected.shape
        and saved["actual"].tobytes() == expected.tobytes(),
        "Scan result differs from its independent reference",
    )
    entry = case["entry"]
    stride = 33 if entry.startswith("strided_") else 1
    execution = scan_layout.execution(entry, view.size, case["axisSize"], stride)
    request = event["details"]["request"]
    require(
        event["entry"] == entry
        and event["threads"] == view.size
        and event["dispatchVersion"] == 3,
        "Scan entry identity differs",
    )
    native_entry = {"metal": entry, "directx": "CSMain", "opengl": "main"}[
        event["target"]
    ]
    require(
        request["target"] == event["target"]
        and request["entryPoint"] == request["dispatch"]["entryPoint"] == native_entry,
        "Scan native entry differs",
    )
    require(
        all(
            event[key] == request["dispatch"][key] == value
            for key, value in execution.items()
        ),
        "Scan source launch differs",
    )
    require(
        event["scanMetadata"] == {
            "axisSize": case["axisSize"],
            "stride": stride,
            "strideBlocks": (stride + 31) // 32,
            "storageElements": view.size,
            "operation": case["operation"],
            "sourceType": case["dtype"],
            "outputType": case["dtype"],
        },
        "Scan metadata differs",
    )
    audit_input_bindings(event)
    inputs = {}
    names = {}
    for name, value in event["inputs"].items():
        binding = request["buffers"][name]
        metadata = binding["binding"]["metadata"]
        origin = metadata["provenance"]["sourceResource"]
        parameter = origin["parameter"]
        scalar = metadata["scalarLayout"]
        require(parameter not in inputs, "Duplicate scan source binding")
        require(
            binding["dtype"] == value["dtype"] == scalar["elementType"]
            and binding["shape"] == value["shape"]
            and binding.get("encoding") == value.get("encoding")
            and scalar["elementStrideBytes"] == np.dtype(value["dtype"]).itemsize
            and scalar["memberOffsetBytes"] == 0
            and origin["backend"] == "metal"
            and origin["entryPoint"] == entry
            and all(type(word) is int for word in value["values"]),
            "Scan native buffer differs",
        )
        inputs[parameter], names[parameter] = value, name
    dtype = case["dtype"]
    words = (
        storage[7 : 7 + view.size].view(np.uint32).tolist()
        if dtype == "float32"
        else storage[7 : 7 + view.size].tolist()
    )
    expected_inputs = {
        "in": scan_dispatch.packet(
            dtype, event["target"], words + scan_dispatch.guard(dtype)
        ),
        "out": scan_dispatch.packet(
            dtype,
            event["target"],
            [scan_dispatch.guard(dtype)[0]] * (view.size + 32),
        ),
        "axis_size": scan_dispatch.packet(
            "uint64", event["target"], [case["axisSize"]]
        ),
    }
    if stride > 1:
        expected_inputs.update(
            stride=scan_dispatch.packet("uint64", event["target"], [stride]),
            stride_blocks=scan_dispatch.packet(
                "uint64", event["target"], [(stride + 31) // 32]
            ),
        )
    require(
        inputs == expected_inputs,
        "Scan upload does not belong to its workload",
    )
    observations = event["details"]["inputObservations"]
    require(
        set(observations) == {names[name] for name in expected_inputs if name != "out"},
        "Scan input observations are incomplete",
    )
    for name, receipt in observations.items():
        snapshot = receipt["uploadSnapshot"]
        actual = bytes.fromhex(receipt["observedBytes"])
        value = event["inputs"][name]
        payload = np.asarray(
            value["values"], dtype="uint32" if value.get("encoding") else value["dtype"]
        ).tobytes()
        require(
            receipt["unchanged"] is True
            and hashlib.sha256(actual).hexdigest() == receipt["observedSHA256"],
            "Scan input observation is corrupt",
        )
        require(
            actual == payload
            and snapshot["sha256"] == receipt["observedSHA256"]
            and snapshot["sizeBytes"] == len(payload)
            and snapshot["shape"] == value["shape"]
            and snapshot["dtype"] == value["dtype"]
            and snapshot.get("encoding") == value.get("encoding"),
            "Scan observed input differs from its upload",
        )
    audit_metadata(event, names, case["axisSize"], stride)
    ordered = np.ascontiguousarray(
        expected.T if case["layout"] == "column" else expected
    )
    output_words = (
        ordered.view(np.uint32).ravel().tolist()
        if dtype == "float32"
        else ordered.ravel().tolist()
    )
    require(
        all(type(word) is int for word in event["outputValues"] + event["outputGuards"])
        and event["outputValues"] == output_words
        and event["outputGuards"] == scan_dispatch.guard(dtype),
        "Scan native output or guards differ",
    )
    require(
        event["outputHash"] == hashlib.sha256(ordered.tobytes()).hexdigest(),
        "Scan output storage identity differs",
    )
    audit_native_execution(event)


def run_cases(mx, np, output, host=None):
    output.mkdir(parents=True)
    results = []
    for case in cases():
        storage, view = operand(np, case)
        source = mx.array(storage)
        value = mx.as_strided(
            source, view.shape, tuple(step // view.itemsize for step in view.strides), 7
        )
        mx.eval(value)
        saved = {"before": np.array(source, copy=True)}
        start = host.dispatch_count if host else 0
        result = getattr(mx, OPERATIONS[case["operation"]])(
            value, axis=1 if case["layout"] == "row" else 0
        )
        exposed = np.asarray(result)
        saved.update(actual=exposed.copy(), after=np.array(source, copy=True))
        expected = reference(np, case, view)
        require(
            saved["actual"].dtype == expected.dtype
            and saved["actual"].shape == expected.shape
            and saved["actual"].tobytes() == expected.tobytes(),
            "Scan numerical reference differs",
        )
        require(
            saved["before"].tobytes() == saved["after"].tobytes() == storage.tobytes(),
            "Scan source changed",
        )
        require(
            not host or host.dispatch_count == start + 1,
            "Scan did not execute exactly one native entry",
        )
        np.savez(output / (case["id"] + ".npz"), **saved)
        results.append(
            {
                "case": case["id"],
                "outputHash": hashlib.sha256(saved["actual"].tobytes()).hexdigest(),
                "shape": list(exposed.shape),
            }
        )
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def verify(root, packages, output):
    import mlx.core as mx
    import numpy as np

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    prepared = verify_prepared(root)
    host = None
    result = {
        "passed": False,
        "commit": COMMIT,
        "fullUpstreamSuite": False,
        "fullScanRouting": False,
    }
    try:
        require(
            not mx.metal.is_available() and not mx.is_available(mx.gpu),
            "An original GPU backend is available",
        )
        mx.set_default_device(mx.cpu)
        cpu = run_cases(mx, np, output / "cpu")
        host = HostRuntime(
            packages, output / "trace.jsonl", mlx_root=root, retain_native_modules=True
        )
        host.install()
        native = run_cases(mx, np, output / "native", host)
        require(cpu == native, "CPU and generated scan results differ")
        events = [json.loads(line) for line in host.trace.read_text().splitlines()]
        require(
            len(events) == host.dispatch_count == len(cases()),
            "Scan trace coverage differs",
        )
        for case, event in zip(cases(), events):
            with np.load(
                output / "native" / (case["id"] + ".npz"), allow_pickle=False
            ) as saved:
                audit_event(np, case, event, saved)
        require(verify_prepared(root) == prepared, "Prepared MLX sources changed")
        result.update(
            passed=True,
            target=host.target,
            casesPerPath=len(native),
            dispatchCount=len(events),
            cpu=cpu,
            native=native,
            adaptation=prepared,
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
        # WARP needs about 17 s per case; 900 s covered only 49 of the 54
        # cases on a slow Windows runner.
        "--timeout-seconds",
        "1800",
        "--label",
        "MLX scan host verification",
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
        raise RuntimeError("Scan host verification failed; inspect retained logs")
    evidence = json.loads((output / "evidence.json").read_text())
    target = json.loads((Path(packages) / "index.json").read_text())["target"]
    require(
        evidence.get("passed") is True
        and evidence.get("commit") == COMMIT
        and evidence.get("target") == target
        and evidence.get("casesPerPath")
        == evidence.get("dispatchCount")
        == len(cases())
        and evidence.get("fullUpstreamSuite") is False
        and evidence.get("fullScanRouting") is False,
        "Scan host evidence is incomplete",
    )
    import numpy as np

    expected = []
    for case in cases():
        _, source = operand(np, case)
        result = reference(np, case, source)
        expected.append(
            {
                "case": case["id"],
                "outputHash": hashlib.sha256(result.tobytes()).hexdigest(),
                "shape": list(result.shape),
            }
        )
    require(
        evidence.get("cpu") == evidence.get("native") == expected,
        "Scan host result records differ from their references",
    )
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    verify(args.mlx_root, args.packages, args.output_dir)
