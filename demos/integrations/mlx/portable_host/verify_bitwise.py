"""Verify translated bitwise kernels through the pinned MLX host backend."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from demos.integrations.mlx.portable_host import bitwise_workloads
from demos.integrations.mlx.portable_host.packages import BITWISE_PACKAGE_ENTRIES
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.verify_rows import verify_artifacts

NEGATIVE_CHECKS = {
    "missing": "No translated package for vv_BitwiseAndint32",
    "int64": "supported dtype",
    "uint8": "supported dtype",
    "over-limit": "at most 65535",
    "negative-shift": "counts in [0, 31]",
    "large-shift": "counts in [0, 31]",
    "invert-missing": "No translated package for v_BitwiseInvertint32int32",
    "invert-int64": "requires matching int32 or uint32",
    "invert-uint8": "requires matching int32 or uint32",
    "invert-over-limit": "at most 65535",
}


def verify_native_identity(trace, target):
    required = {"dispatch-translated-artifact", "collect-runtime-outputs"} | {
        "metal": {"compile-metal-for-native-runtime", "link-metal-for-native-runtime"},
        "opengl": {"validate-glsl-for-opengl-runtime"},
        "directx": {"compile-hlsl-for-directx-runtime"},
    }[target]
    for event in trace:
        details = event.get("details", {})
        identity = details.get("artifactIdentityVerification", {})
        artifact = event["artifact"]
        expected = {key: artifact[key] for key in ("hash", "sizeBytes")}
        native = details.get("nativeRuntimeDispatch", {})
        adapter = details.get("runtimeParityAdapter", {})
        steps = details.get("adapterSteps", [])
        if (
            identity.get("verificationStatus") != "verified"
            or identity.get("target") != target
            or identity.get("expectedIdentity") != expected
            or identity.get("observedIdentity") != expected
            or native.get("artifact", {}).get("packagePath") != artifact["packagePath"]
            or native.get("artifact", {}).get("target") != target
            or any(
                native.get("dispatch", {}).get(key) != event[key]
                for key in ("workgroupCount", "workgroupSize")
            )
            or adapter.get("target") != target
            or adapter.get("runtimeAdapter") != target + "-native-runtime"
            or any(step.get("status") != "passed" for step in steps)
            or not required <= {step.get("action") for step in steps}
        ):
            raise ValueError(
                "Bitwise native compiler, dispatch or artifact identity is incomplete"
            )


def worker(args):
    import mlx.core as mx
    import numpy as np

    output = args.output_dir
    output.mkdir(parents=True)
    if mx.is_available(mx.gpu):
        raise RuntimeError("A different GPU backend is already available")
    host = None
    if args.worker == "cpu":
        mx.set_default_device(mx.cpu)
    else:
        host = HostRuntime(
            args.packages, output / "dispatch.jsonl", bitwise=args.bitwise
        )
        if args.worker in {"missing", "invert-missing"}:
            for entry in BITWISE_PACKAGE_ENTRIES:
                host.descriptors.pop(entry)
        host.install()
    if args.worker in NEGATIVE_CHECKS:
        kind = args.worker.removeprefix("invert-")
        dtype = kind if kind in {"uint8", "int64"} else "int32"
        size = 65536 if kind == "over-limit" else 3
        a = mx.array(np.ones(size, dtype=dtype))
        b = mx.array(
            np.full(
                size,
                (
                    -1
                    if args.worker == "negative-shift"
                    else 32 if args.worker == "large-shift" else 1
                ),
                dtype=dtype,
            )
        )
        operation = mx.left_shift if args.worker.endswith("shift") else mx.bitwise_and
        try:
            mx.eval(
                mx.bitwise_invert(a)
                if args.worker.startswith("invert-")
                else operation(a, b)
            )
        except (ValueError, RuntimeError) as error:
            record = {
                "check": args.worker,
                "error": str(error),
                "dispatchCount": host.dispatch_count,
            }
            (output / "rejection.json").write_text(json.dumps(record, indent=2))
            if NEGATIVE_CHECKS[args.worker] not in str(error) or host.dispatch_count:
                raise RuntimeError(
                    "Invalid bitwise input was not rejected before dispatch"
                ) from error
            return
        raise RuntimeError("Invalid bitwise input was accepted")

    def observe(record):
        with (output / "readbacks.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")

    records = bitwise_workloads.collect(
        mx,
        np,
        observe=observe,
        dispatch_count=(lambda: host.dispatch_count) if host else None,
    )
    (output / "results.json").write_text(json.dumps(records, indent=2))


def verify(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    before = verify_prepared(args.mlx_root)
    (output / "adaptation-before.json").write_text(json.dumps(before, indent=2))
    base = json.loads((args.packages / "index.json").read_text())
    index = json.loads((args.bitwise / "index.json").read_text())
    if (
        index.get("family") != "bitwise"
        or index.get("target") != base["target"]
        or set(index.get("descriptors", {})) != set(BITWISE_PACKAGE_ENTRIES)
    ):
        raise ValueError("Bitwise proof requires every entry for the base target")
    results, rejections = {}, {}
    for mode in ("cpu", "native", *NEGATIVE_CHECKS):
        command = [
            sys.executable,
            str(Path(__file__).resolve().parents[4] / "tools/run_bounded_command.py"),
            "--timeout-seconds",
            "600" if mode == "native" else "60",
            "--label",
            f"MLX bitwise {mode}",
            "--",
            sys.executable,
            "-m",
            __package__ + ".verify_bitwise",
            "--worker",
            mode,
            "--mlx-root",
            str(args.mlx_root.resolve()),
            "--packages",
            str(args.packages.resolve()),
            "--bitwise",
            str(args.bitwise.resolve()),
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
            raise RuntimeError(f"Bitwise {mode} worker failed; inspect {output}")
        if mode in NEGATIVE_CHECKS:
            record = json.loads((output / mode / "rejection.json").read_text())
            if (
                record.get("check") != mode
                or record.get("dispatchCount") != 0
                or NEGATIVE_CHECKS[mode] not in record.get("error", "")
            ):
                raise ValueError("Bitwise rejection evidence is incomplete")
            rejections[mode] = record
        else:
            results[mode] = json.loads((output / mode / "results.json").read_text())
    trace = [
        json.loads(line)
        for line in (output / "native/dispatch.jsonl").read_text().splitlines()
    ]
    bitwise_workloads.validate(results["cpu"], [], native=False)
    bitwise_workloads.validate(results["native"], trace, native=True)
    verify_native_identity(trace, base["target"])
    for selected, directory in ((index, args.bitwise), (base, args.packages)):
        verify_artifacts(
            [event for event in trace if event["entry"] in selected["descriptors"]],
            directory,
            selected,
            variants=False,
        )
    after = verify_prepared(args.mlx_root)
    (output / "adaptation-after.json").write_text(json.dumps(after, indent=2))
    if before != after:
        raise ValueError("MLX sources changed during bitwise verification")
    evidence = {
        "commit": COMMIT,
        "target": base["target"],
        "adaptation": before,
        "casesPerPath": len(results["native"]),
        "dispatchCount": len(trace),
        "entries": sorted(BITWISE_PACKAGE_ENTRIES),
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
    parser.add_argument("--bitwise", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", choices=("cpu", "native", *NEGATIVE_CHECKS))
    args = parser.parse_args()
    worker(args) if args.worker else verify(args)
