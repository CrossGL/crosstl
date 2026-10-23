"""Run unchanged upstream op tests on the small-row job's package families.

Each test runs unmodified from the pinned ``python/tests`` tree, once on the
MLX CPU and once through the translated host. The native run installs the
base host packages with the families the small-row job already translates:
random bits, Boolean and integer bitwise operations, selection, absolute
values, reduction initialization and the checked Boolean and float32
conjunctions. General gather and scatter kernels are specialized from the
pinned MLX sources at dispatch time, and each of their dispatches is
recomputed from its retained inputs and native modules. DivMod's
``binary_two`` kernels are specialized the same way, and every DivMod dispatch
is recomputed from its retained operands. Every other dispatch
must come from one of those indexes, so a test that reaches another family
(scans, the 64-bit or 16-bit package families) fails the trace instead of
passing on an unchecked package.
"""

import argparse
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from demos.integrations.mlx.portable_host import (
    divmod_evidence,
    gather_evidence,
    random_evidence,
    scatter_axis_evidence,
    scatter_evidence,
)
from demos.integrations.mlx.portable_host.divmod_dispatch import (
    ENTRY_PREFIXES as DIVMOD_PREFIXES,
)
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.random_packages import (
    ENTRIES,
)
from demos.integrations.mlx.portable_host.random_packages import (
    load_index as load_random,
)
from demos.integrations.mlx.portable_host.reduction_packages import (
    load_index as load_reductions,
)
from demos.integrations.mlx.portable_host.reduction_packages import variant_keys
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.verify import upstream_test_sources
from demos.integrations.mlx.portable_host.verify_bitwise import verify_native_identity
from demos.integrations.mlx.portable_host.verify_rows import verify_artifacts

UPSTREAM_TESTS = tuple(
    "test_ops.TestOps." + name
    for name in (
        "test_trunc",
        "test_erfinv",
        "test_logical_and",
        "test_logical_or",
        "test_full_ones_zeros",
        "test_atleast_1d",
        "test_atleast_2d",
        "test_atleast_3d",
        "test_tile",
        "test_repeat",
        "test_roll",
        "test_roll_errors",
        "test_tri",
        "test_tril",
        "test_triu",
        "test_as_strided",
        "test_view",
        "test_depends",
        "test_eye",
        "test_diag",
        "test_diagonal",
        "test_divmod",
    )
)
# Source-specialized indexing kernels by entry prefix, longest prefix first.
INDEXING_PREFIXES = ("gather", "scatter_axis", "scatter")
# Entry-keyed package families; reductions are keyed by width and entry.
FAMILIES = ("packages", "random", "bitwise", "selection", "absolute")
REQUIRED_REDUCTIONS = frozenset(("w32/all_reduce_andbool_", "w256/all_reduce_andbool_"))
NEGATIVE_CHECKS = {"missing": "No translated package for vv_Equalint32"}
# The CI step bounds the three workers and the audit at 1,680 seconds.
TIMEOUTS = {"cpu": 120, "native": 1440, "missing": 60}


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def install(args, output):
    host = HostRuntime(
        args.packages,
        output / "dispatch.jsonl",
        random=args.random,
        bitwise=args.bitwise,
        selection=args.selection,
        absolute=args.absolute,
        reductions=list(args.reductions),
        mlx_root=args.mlx_root,
    )
    if args.worker == "missing":
        host.descriptors.pop("vv_Equalint32")
    host.install()
    return host


def reject(mx, host, output, check):
    try:
        mx.eval(mx.array_equal(mx.array([1, 2, 3]), mx.array([1, 2, 4])))
    except (ValueError, RuntimeError) as error:
        record = {
            "check": check,
            "error": str(error),
            "dispatchCount": host.dispatch_count,
        }
        write_json(output / "rejection.json", record)
        if NEGATIVE_CHECKS[check] not in str(error) or host.dispatch_count:
            raise RuntimeError(
                "The missing package was not refused before dispatch"
            ) from error
        return
    raise RuntimeError("A comparison without its package was accepted")


def worker(args):
    import mlx.core as mx

    output = args.output_dir
    output.mkdir(parents=True)
    if mx.is_available(mx.gpu):
        raise RuntimeError("A different GPU backend is already available")
    host = None
    if args.worker == "cpu":
        mx.set_default_device(mx.cpu)
    else:
        host = install(args, output)
    if args.worker in NEGATIVE_CHECKS:
        return reject(mx, host, output, args.worker)
    os.environ["DEVICE"] = "gpu" if host else "cpu"
    os.environ["CI"] = "1"
    sys.path.insert(0, str(args.mlx_root / "python/tests"))
    records = []
    for name in UPSTREAM_TESTS:
        start = host.dispatch_count if host else 0
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromName(name)
        )
        records.append(
            {
                "test": name,
                "testsRun": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skips": len(result.skipped),
                "dispatchCount": host.dispatch_count - start if host else 0,
            }
        )
        write_json(output / "upstream.json", {"tests": records})
    validate_upstream({"tests": records}, native=host is not None)


def validate_upstream(record, *, native):
    tests = record.get("tests", [])
    if [item.get("test") for item in tests] != list(UPSTREAM_TESTS):
        raise ValueError("Upstream op test inventory is incomplete")
    for item in tests:
        expected = {"testsRun": 1, "failures": 0, "errors": 0, "skips": 0}
        count = item.get("dispatchCount")
        if (
            any(item.get(key) != value for key, value in expected.items())
            or type(count) is not int
            or (count <= 0 if native else count != 0)
        ):
            raise ValueError(f"Unchanged upstream test failed: {item.get('test')}")


def load_indexes(args):
    base = json.loads((args.packages / "index.json").read_text())
    target = base["target"]
    indexes = {"packages": base}
    indexes["random"] = {
        "target": target,
        "descriptors": load_random(args.random, target),
    }
    for name in ("bitwise", "selection", "absolute"):
        index = json.loads((getattr(args, name) / "index.json").read_text())
        if index.get("target") != target:
            raise ValueError(f"The {name} packages target another backend")
        indexes[name] = index
    reductions = [load_reductions(path, target) for path in args.reductions]
    keys = set().union(*(variant_keys(index) for index in reductions))
    if REQUIRED_REDUCTIONS - keys:
        raise ValueError(
            "Upstream op assertions require reduction packages: "
            + ", ".join(sorted(REQUIRED_REDUCTIONS - keys))
        )
    return target, indexes, reductions


def indexing_auditor(entry):
    """Return the evidence check for a source-specialized indexing entry."""
    auditors = {
        "gather": gather_evidence.audit_event,
        "scatter_axis": scatter_axis_evidence.audit_event,
        "scatter": scatter_evidence.audit_event,
    }
    prefix = next((item for item in INDEXING_PREFIXES if entry.startswith(item)), None)
    return auditors.get(prefix)


def audit_trace(np, args, trace, target, indexes, reductions):
    """Tie every dispatch to exactly one translated package."""
    random_events = [event for event in trace if event["entry"] in ENTRIES]
    for event in random_events:
        random_evidence.audit_event(np, event)
    if random_events:
        random_evidence.verify_packages(
            random_events, args.mlx_root, args.random, target
        )
    owners = [0] * len(trace)
    indexing = 0
    for i, event in enumerate(trace):
        auditor = indexing_auditor(event["entry"])
        if auditor is None:
            continue
        if event["target"] != target:
            raise ValueError("Upstream op indexing dispatch targets another backend")
        # Recomputes the result from the retained inputs and native modules.
        auditor(np, event)
        owners[i] += 1
        indexing += 1
    for i, event in enumerate(trace):
        if not event["entry"].startswith(DIVMOD_PREFIXES):
            continue
        if event["target"] != target:
            raise ValueError("Upstream op DivMod dispatch targets another backend")
        # Recomputes both outputs and checks the retained native modules.
        divmod_evidence.audit_event(event)
        owners[i] += 1
    verify_native_identity(
        [
            event
            for event in trace
            if event["entry"] not in ENTRIES
            and indexing_auditor(event["entry"]) is None
            and not event["entry"].startswith(DIVMOD_PREFIXES)
        ],
        target,
    )
    for name in FAMILIES:
        if name == "random":
            selected = [i for i, event in enumerate(trace) if event["entry"] in ENTRIES]
        else:
            descriptors = indexes[name]["descriptors"]
            selected = [
                i for i, event in enumerate(trace) if event["entry"] in descriptors
            ]
            verify_artifacts(
                [trace[i] for i in selected],
                getattr(args, name),
                indexes[name],
                variants=False,
            )
        for i in selected:
            owners[i] += 1
    for path, index in zip(args.reductions, reductions):
        keys = variant_keys(index)
        selected = [
            i
            for i, event in enumerate(trace)
            if f'w{event["workgroupSize"][0]}/{event["entry"]}' in keys
        ]
        verify_artifacts(
            [trace[i] for i in selected],
            path,
            index,
            variants=True,
            mlx_root=args.mlx_root,
        )
        for i in selected:
            owners[i] += 1
    if any(count != 1 for count in owners):
        raise ValueError("Upstream op trace contains an unknown or ambiguous package")
    if not indexing:
        raise ValueError("Upstream indexing tests did not reach gather or scatter")
    return indexing


def run_worker(args, output, mode):
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[4] / "tools/run_bounded_command.py"),
        "--timeout-seconds",
        str(TIMEOUTS[mode]),
        "--label",
        f"MLX upstream ops {mode}",
        "--",
        sys.executable,
        "-m",
        __package__ + ".verify_upstream_ops",
        "--worker",
        mode,
    ]
    for name in ("mlx_root", "packages", "random", "bitwise", "selection", "absolute"):
        command.extend(
            ("--" + name.replace("_", "-"), str(getattr(args, name).resolve()))
        )
    for path in args.reductions:
        command.extend(("--reductions", str(path.resolve())))
    command.extend(("--output-dir", str(output / mode)))
    with (output / f"{mode}.stdout").open("w") as stdout, (
        output / f"{mode}.stderr"
    ).open("w") as stderr:
        process = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    record = {"command": command, "returncode": process.returncode}
    write_json(output / f"{mode}.command.json", record)
    return record


def verify(args):
    import numpy as np

    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    before = verify_prepared(args.mlx_root)
    sources = upstream_test_sources(args.mlx_root, UPSTREAM_TESTS)
    write_json(output / "adaptation-before.json", before)
    target, indexes, reductions = load_indexes(args)
    evidence = {
        "commit": COMMIT,
        "target": target,
        "passed": False,
        "adaptation": before,
        "upstreamTestSources": sources,
        "upstreamTests": list(UPSTREAM_TESTS),
        "workers": {},
        "fullTranslatedBackend": False,
        "fullUpstreamSuite": False,
    }
    try:
        for mode in ("cpu", "native", *NEGATIVE_CHECKS):
            evidence["workers"][mode] = run_worker(args, output, mode)
            write_json(output / "evidence.json", evidence)
        if any(value["returncode"] for value in evidence["workers"].values()):
            raise RuntimeError("An upstream op worker failed; inspect retained logs")
        upstream = {}
        for mode in ("cpu", "native"):
            upstream[mode] = json.loads((output / mode / "upstream.json").read_text())
            validate_upstream(upstream[mode], native=mode == "native")
        trace = [
            json.loads(line)
            for line in (output / "native/dispatch.jsonl").read_text().splitlines()
        ]
        if sum(item["dispatchCount"] for item in upstream["native"]["tests"]) != len(
            trace
        ):
            raise ValueError("Upstream op trace count differs")
        indexing = audit_trace(np, args, trace, target, indexes, reductions)
        rejections = {}
        for mode, error in NEGATIVE_CHECKS.items():
            item = json.loads((output / mode / "rejection.json").read_text())
            if (
                item.get("check") != mode
                or item.get("dispatchCount") != 0
                or error not in item.get("error", "")
            ):
                raise ValueError("Upstream op rejection evidence is incomplete")
            rejections[mode] = item
        if before != verify_prepared(args.mlx_root) or sources != upstream_test_sources(
            args.mlx_root, UPSTREAM_TESTS
        ):
            raise ValueError("MLX source changed during upstream op verification")
        evidence.update(
            passed=True,
            dispatchCount=len(trace),
            indexingDispatchCount=indexing,
            upstream=upstream,
            negativeChecks=rejections,
        )
    except Exception as error:
        evidence["error"] = str(error)
        raise
    finally:
        write_json(output / "evidence.json", evidence)
    return evidence


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "mlx-root",
        "packages",
        "random",
        "bitwise",
        "selection",
        "absolute",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reductions", type=Path, action="append", required=True)
    parser.add_argument("--worker", choices=("cpu", "native", *NEGATIVE_CHECKS))
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse()
    worker(arguments) if arguments.worker else verify(arguments)
