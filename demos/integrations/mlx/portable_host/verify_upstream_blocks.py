"""Run unchanged upstream block-format tests on CPU and a generated backend."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from demos.integrations.mlx.portable_host.gather_evidence import audit_native_execution
from demos.integrations.mlx.portable_host.packages import BFLOAT_ENTRIES
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.random_packages import (
    ENTRIES as RANDOM_ENTRIES,
)
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.verify import upstream_test_sources
from demos.integrations.mlx.portable_host.verify_block_quantization import compile_entry
from demos.integrations.mlx.portable_host.verify_half import write_json

UPSTREAM_TESTS = tuple(
    "test_quantized.TestQuantized." + name
    for name in (
        "test_mxfp4_quantize_dequantize",
        "test_mxfp8_quantize_dequantize",
        "test_mxfp8_block_scale_does_not_saturate",
        "test_nvfp4_quantize_dequantize",
    )
)


def make_host(args, trace):
    return HostRuntime(
        args.packages,
        trace,
        mlx_root=args.mlx_root,
        random=args.random,
        bfloat=args.bfloat,
        reductions=args.reductions,
        retain_native_modules=True,
    )


def validate_results(records, *, native):
    if not isinstance(records, list) or len(records) != len(UPSTREAM_TESTS):
        raise ValueError("Upstream block test inventory differs")
    cursor = 0
    for name, record in zip(UPSTREAM_TESTS, records):
        if (
            record.get("test") != name
            or type(record.get("testsRun")) is not int
            or record["testsRun"] != 1
            or record.get("success") is not True
            or any(
                record.get(key) != []
                for key in (
                    "errors",
                    "failures",
                    "skipped",
                    "expectedFailures",
                    "unexpectedSuccesses",
                )
            )
            or type(record.get("dispatchStart")) is not int
            or record["dispatchStart"] != cursor
            or type(record.get("dispatchCount")) is not int
            or (
                record["dispatchCount"] <= 0 if native else record["dispatchCount"] != 0
            )
        ):
            raise ValueError(
                "An unchanged upstream block test failed or did not execute"
            )
        cursor += record["dispatchCount"]
    return cursor


def worker(args):
    import mlx.core as mx

    args.output_dir.mkdir(parents=True, exist_ok=False)
    if mx.is_available(mx.gpu):
        raise RuntimeError("A different GPU backend is already available")
    host = None
    if args.worker == "native":
        host = make_host(args, args.output_dir / "trace.jsonl")
        host.install()
    else:
        mx.set_default_device(mx.cpu)
    os.environ["DEVICE"] = "gpu" if host else "cpu"
    sys.path.insert(0, str(args.mlx_root / "python/tests"))
    records = []
    try:
        with (args.output_dir / "upstream.log").open("w", encoding="utf-8") as stream:
            for name in UPSTREAM_TESTS:
                start = host.dispatch_count if host else 0
                result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
                    unittest.defaultTestLoader.loadTestsFromName(name)
                )
                records.append(
                    {
                        "test": name,
                        "testsRun": result.testsRun,
                        "success": result.wasSuccessful(),
                        "errors": [
                            (test.id(), message) for test, message in result.errors
                        ],
                        "failures": [
                            (test.id(), message) for test, message in result.failures
                        ],
                        "skipped": [
                            (test.id(), message) for test, message in result.skipped
                        ],
                        "expectedFailures": [
                            (test.id(), message)
                            for test, message in result.expectedFailures
                        ],
                        "unexpectedSuccesses": [
                            test.id() for test in result.unexpectedSuccesses
                        ],
                        "dispatchStart": start,
                        "dispatchCount": host.dispatch_count - start if host else 0,
                    }
                )
                write_json(args.output_dir / "results.json", records)
                stream.flush()
        validate_results(records, native=host is not None)
    finally:
        if host:
            close = getattr(host.executor.runtime_adapter.runtime, "close", None)
            if close:
                close()


def event_packages(event, host):
    entry = event["entry"]
    if entry.startswith("all_reduce_"):
        key = f'w{event["workgroupSize"][0]}/{entry}'
        return [
            (
                host.reduction_descriptors[key],
                host.reduction_directories[key] / "package",
            )
        ]
    if entry.startswith("row_reduce_small_"):
        return host.small_rows.get(
            entry,
            thread_grid_size=event["threadGridSize"],
            workgroup_size=event["workgroupSize"],
        )
    if entry.startswith(("mxfp4_", "mxfp8_", "nvfp4_")):
        return [host.quantization.get(entry)]
    if entry.startswith("gather"):
        return [host.gathers.get(entry, 0)]
    directory = (
        host.bfloat_directory
        if entry in BFLOAT_ENTRIES
        else host.random_directory if entry in RANDOM_ENTRIES else host.directory
    )
    return [(host.descriptors[entry], directory / "package")]


def audit_event(event, packages, output, *, target):
    if (
        event["target"] != target
        or event["dispatchVersion"] != 3
        or event["artifact"] != packages[0][0]["artifact"]
    ):
        raise ValueError("Upstream block dispatch identity differs")
    for descriptor, root in packages:
        artifact = descriptor["artifact"]
        source = (root / artifact["packagePath"]).resolve()
        if not source.is_relative_to(root.resolve()):
            raise ValueError("Upstream block artifact escapes its package")
        data = source.read_bytes()
        if (
            len(data) != artifact["sizeBytes"]
            or hashlib.sha256(data).hexdigest() != artifact["hash"]["value"]
        ):
            raise ValueError("Upstream block source identity differs")
    modules = output.resolve() / "native/native-modules"
    details = event["details"]
    if "regions" in details:
        if len(details["regions"]) != len(packages):
            raise ValueError("Upstream block region coverage differs")
        for number, (region, (descriptor, root)) in enumerate(
            zip(details["regions"], packages)
        ):
            geometry = descriptor["provenance"]["dispatchRegion"]
            module = Path(region["moduleFile"]).resolve()
            if (
                region["artifact"] != descriptor["artifact"]
                or region["provenance"] != descriptor["provenance"]
                or Path(region["packageRoot"]).resolve() != root.resolve()
                or any(
                    region[key] != geometry[key]
                    for key in ("workgroupCount", "workgroupSize")
                )
                or not module.is_relative_to(modules)
                or hashlib.sha256(module.read_bytes()).hexdigest()
                != region["moduleHash"]
            ):
                raise ValueError("Upstream block region identity differs")
            compile_entry(
                target,
                f'{event["entry"]}-{number}',
                descriptor,
                root,
                output / "compilation",
            )
        return
    if len(packages) != 1 or (
        "packageRoot" in event
        and Path(event["packageRoot"]).resolve() != packages[0][1].resolve()
    ):
        raise ValueError("Upstream block package identity differs")
    request = details["request"]
    if (
        request["target"] != target
        or request["entryPoint"]
        != {"metal": event["entry"], "opengl": "main", "directx": "CSMain"}[target]
        or any(
            request["dispatch"][key] != event[key]
            for key in ("workgroupCount", "workgroupSize")
        )
    ):
        raise ValueError("Upstream block native launch differs")
    for module in [details["module"], *details["validationModules"]]:
        if not Path(module["file"]).resolve().is_relative_to(modules):
            raise ValueError(
                "Upstream block native module escapes the evidence directory"
            )
    audit_native_execution({**event, "packageRoot": str(packages[0][1])})


def verify(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    before = verify_prepared(args.mlx_root)
    sources = upstream_test_sources(args.mlx_root, UPSTREAM_TESTS)
    host = make_host(args, output / "audit.jsonl")
    evidence = {
        "passed": False,
        "commit": COMMIT,
        "target": host.target,
        "adaptation": before,
        "upstreamTestSources": sources,
        "upstreamTests": list(UPSTREAM_TESTS),
        "fullUpstreamSuite": False,
    }
    write_json(output / "summary.json", evidence)
    try:
        records = {}
        for mode in ("cpu", "native"):
            command = [
                sys.executable,
                str(
                    Path(__file__).resolve().parents[4] / "tools/run_bounded_command.py"
                ),
                "--timeout-seconds",
                "1800" if mode == "native" else "180",
                "--label",
                "MLX upstream blocks " + mode,
                "--",
                sys.executable,
                "-m",
                __package__ + ".verify_upstream_blocks",
                "--worker",
                mode,
            ]
            for name in ("mlx_root", "packages", "random", "bfloat"):
                command.extend(
                    ("--" + name.replace("_", "-"), str(getattr(args, name).resolve()))
                )
            for directory in args.reductions:
                command.extend(("--reductions", str(directory.resolve())))
            command.extend(("--output-dir", str(output / mode)))
            with (output / f"{mode}.stdout").open("w") as stdout, (
                output / f"{mode}.stderr"
            ).open("w") as stderr:
                process = subprocess.run(
                    command, stdout=stdout, stderr=stderr, check=False
                )
            write_json(
                output / f"{mode}.command.json",
                {"command": command, "returncode": process.returncode},
            )
            if process.returncode:
                raise RuntimeError(
                    f"Upstream block {mode} worker failed; inspect {output}"
                )
            records[mode] = json.loads((output / mode / "results.json").read_text())
            validate_results(records[mode], native=mode == "native")
        count = validate_results(records["native"], native=True)
        observed = set()
        number = 0
        with (output / "native/trace.jsonl").open() as stream:
            for number, line in enumerate(stream, 1):
                event = json.loads(line)
                audit_event(
                    event, event_packages(event, host), output, target=host.target
                )
                observed.add(event["entry"])
        if (
            number != count
            or not {
                "all_reduce_maxbfloat16",
                "all_reduce_maxfloat32",
                "all_reduce_andbool_",
                "row_reduce_small_1_reduce_maxbfloat16",
                "row_reduce_small_1_reduce_maxfloat32",
            }
            <= observed
        ):
            raise ValueError("Upstream block dispatch coverage differs")
        evidence.update(passed=True, results=records, nativeDispatches=count)
    except Exception as error:
        evidence["error"] = str(error)
        raise
    finally:
        after = verify_prepared(args.mlx_root)
        unchanged = before == after and sources == upstream_test_sources(
            args.mlx_root, UPSTREAM_TESTS
        )
        evidence.update(adaptationAfter=after, sourceUnchanged=unchanged)
        if not unchanged:
            evidence.update(
                passed=False, error="Upstream block sources changed during verification"
            )
        write_json(output / "summary.json", evidence)
        if not unchanged:
            raise ValueError(evidence["error"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mlx-root", "packages", "random", "bfloat", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reductions", type=Path, action="append", required=True)
    parser.add_argument("--worker", choices=("cpu", "native"))
    args = parser.parse_args()
    worker(args) if args.worker else verify(args)
