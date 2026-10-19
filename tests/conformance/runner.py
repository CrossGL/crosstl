"""Translate, validate and execute conformance cases on every selected runtime without conflating outcomes.

Each cell is (case, target, route, compile configuration, runtime). Translation and validation happen once per
(case, target, route, configuration); execution then happens on every runtime selected for the target, batched
per runtime so container runtimes start once. Outcomes:

- ``refused``: the translator raised a typed diagnostic; ``translator-error``: it failed without one;
- ``invalid``: the target's native checker rejected the artifact; ``driver-rejected``: the runtime's own compiler
  or pipeline creation rejected an artifact the checker accepted; ``interface-changed``: the artifact no longer
  exposes its source's two-buffer interface;
- ``preserved`` / ``diverged``: the execution matched or contradicted the source-rule oracle; ``observed``: the
  source rule leaves the result undefined, so the result is recorded without an obligation;
- ``memory-corruption``: the kernel wrote the input buffer or past the end of the output buffer;
- ``compiled-only``, ``unavailable``, ``timeout``, ``runtime-error``, ``harness-error``: no execution evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .cases import BY_ID, CANARY, CASES, compare
from .interface import InterfaceChanged, interface
from .process import run
from .runtimes import REPOSITORY, RUNTIMES, execute, provision
from .toolchains import profile
from .validators import BINARIES, CONFIGS, ENVIRONMENTS, HLSL_PROFILE, TARGETS, validate

GUARD_WORDS = 32


def digest(data):
    return hashlib.sha256(data).hexdigest()


def compile_only_runtime():
    return f"compile-only@{platform.system().lower()}"


def isolated(job, path, timeout):
    path.write_text(json.dumps(job, indent=2) + "\n")
    process = run(
        [sys.executable, "-m", "tests.conformance.worker", path], timeout=timeout
    )
    result_path = path.with_suffix(".result.json")
    if process.status != "accepted" or not result_path.is_file():
        return {
            "status": (
                process.status
                if process.status in {"timeout", "unavailable"}
                else "worker-error"
            ),
            "process": process.to_json(),
        }
    return json.loads(result_path.read_text())


@dataclass
class Cell:
    """One (case, target, route, configuration) after translation and native validation."""

    case: object
    target: str
    config: str
    directory: Path
    base: dict
    outcome: str | None = None
    artifact: Path | None = None
    binary: Path | None = None
    abi: dict = field(default_factory=dict)


def translate(case, target, route, root):
    directory = root / case.id / target / route
    directory.mkdir(parents=True, exist_ok=False)
    artifact = directory / ("shader" + TARGETS[target])
    if route == "reference":
        artifact.write_bytes(case.src.read_bytes())
        return artifact, {"status": "reference"}
    job = dict(
        operation="translate",
        source=str(case.src),
        language=case.lang,
        target=target,
        artifact=str(artifact),
    )
    return artifact, isolated(job, directory / "translate.json", 120)


def _resources(abi, reflection):
    """Bind HLSL buffers with the element stride DXC reflected, falling back to the declared scalar type."""
    resources = []
    for resource in abi["resources"]:
        stride = reflection.get("strides", {}).get(resource["name"], resource["stride"])
        if resource["kind"] == "structured" and not stride:
            raise InterfaceChanged(
                f"Unknown structured-buffer stride for {resource['name']}"
            )
        resources.append(dict(resource, stride=stride))
    return resources


def prepare(
    case, target, route, config, artifact, translation, *, environment, profile
):
    directory = artifact.parent / config
    directory.mkdir()
    base = dict(
        case=case.id,
        language=case.lang,
        target=target,
        config=config,
        route=route,
        readings=case.readings,
        rule=case.rule,
        sourceSHA256=digest(case.src.read_bytes()),
        inputSHA256=digest(case.inputs.tobytes()),
        environment=environment,
        host=platform.system(),
        translation=translation,
    )
    cell = Cell(case, target, config, directory, base)
    if translation["status"] not in {"reference", "translated"}:
        cell.outcome = translation["status"]
        return cell
    base["artifactSHA256"] = digest(artifact.read_bytes())
    try:
        abi = interface(artifact, target, case.group)
    except InterfaceChanged as error:
        cell.outcome = "harness-error" if route == "reference" else "interface-changed"
        base["message"] = str(error)
        return cell
    binary = directory / ("shader" + BINARIES[target])
    validations, reflection = validate(
        artifact,
        target,
        binary,
        abi["entry"],
        config=config,
        environment=environment,
        profile=profile,
    )
    base["validation"] = [item.to_json() for item in validations]
    failed = next((item for item in validations if item.status != "accepted"), None)
    if failed:
        cell.outcome = {"rejected": "invalid"}.get(failed.status, failed.status)
        return cell
    if target == "hlsl":
        try:
            abi["resources"] = _resources(abi, reflection)
        except InterfaceChanged as error:
            cell.outcome, base["message"] = "harness-error", str(error)
            return cell
    cell.artifact, cell.binary, cell.abi = artifact, binary, abi
    return cell


def stage(cell, runtime):
    """Write the buffers and the execution job of one cell for one runtime."""
    case, directory = cell.case, cell.directory / runtime.id
    directory.mkdir()
    initial = (
        case.out_init.tobytes()
        if case.out_init is not None
        else np.full(
            case.out_count * np.dtype(case.out_dtype).itemsize, 0xA5, dtype=np.uint8
        ).tobytes()
    )
    guard = np.full(GUARD_WORDS, CANARY, dtype=np.uint32).tobytes()
    (directory / "input.bin").write_bytes(case.inputs.tobytes())
    (directory / "output.bin").write_bytes(initial + guard)
    job = dict(
        operation="execute",
        adapter=runtime.adapter,
        runtime=runtime.id,
        expect=runtime.expect,
        artifact=str(cell.artifact),
        binary=str(cell.binary),
        directory=str(directory),
        target=cell.target,
        config=cell.config,
        threads=case.threads,
        **cell.abi,
    )
    path = directory / "execute.json"
    path.write_text(json.dumps(job, indent=2) + "\n")
    return path, len(initial)


def evaluate(cell, job, size, result):
    case, directory = cell.case, job.parent
    if result["status"] != "executed":
        return {"outcome": result["status"], "execution": result}
    output = (directory / "output.bin").read_bytes()
    unchanged = (directory / "input.bin").read_bytes() == case.inputs.tobytes()
    guard = np.full(GUARD_WORDS, CANARY, dtype=np.uint32).tobytes()
    guarded = len(output) == size + len(guard) and output[size:] == guard
    matched, detail = compare(case, output[:size])
    outcome = (
        ("preserved" if matched else "diverged")
        if any(case.readings.values())
        else "observed"
    )
    if not (unchanged and guarded):
        outcome = "memory-corruption"
    return dict(
        outcome=outcome,
        execution=result,
        numericalMatch=bool(matched),
        comparison=detail,
        inputUnchanged=unchanged,
        outputGuardIntact=guarded,
        outputSHA256=digest(output),
    )


def select_runtimes(spec, targets):
    """Map each target to the (runtime, provision) pairs to execute on; explicit runtimes are mandatory."""
    if spec == "none":
        return {target: [] for target in targets}
    if spec == "available":
        chosen = [runtime for runtime in RUNTIMES.values() if runtime.target in targets]
        chosen = [runtime for runtime in chosen if provision(runtime).available]
    else:
        unknown = set(spec.split(",")) - set(RUNTIMES)
        if unknown:
            raise ValueError(f"Unknown runtimes: {sorted(unknown)}")
        chosen = [RUNTIMES[name] for name in spec.split(",")]
    return {
        target: [
            (runtime, provision(runtime))
            for runtime in chosen
            if runtime.target == target
        ]
        for target in targets
    }


def run_suite(
    cases,
    targets,
    configs,
    selection,
    root,
    *,
    reference=False,
    environment="vulkan1.1",
    profile=HLSL_PROFILE,
):
    """Execute the whole matrix; returns one record per (case, target, route, configuration, runtime)."""
    cells = []
    for case in cases:
        for target in targets:
            route = "reference" if reference else "translated"
            if reference and target != case.lang:
                continue
            artifact, translation = translate(case, target, route, root)
            for config in configs:
                cells.append(
                    prepare(
                        case,
                        target,
                        route,
                        config,
                        artifact,
                        translation,
                        environment=environment,
                        profile=profile,
                    )
                )
    records, staged = [], {}
    for cell in cells:
        runtimes = selection.get(cell.target, [])
        if not runtimes:
            outcome = cell.outcome or "compiled-only"
            records.append(
                dict(cell.base, runtime=compile_only_runtime(), outcome=outcome)
            )
            continue
        for runtime, provided in runtimes:
            if cell.outcome is not None:
                records.append(
                    dict(cell.base, runtime=runtime.id, outcome=cell.outcome)
                )
            elif not provided.available:
                records.append(
                    dict(
                        cell.base,
                        runtime=runtime.id,
                        outcome="unavailable",
                        message=provided.reason,
                    )
                )
            else:
                job, size = stage(cell, runtime)
                index = len(records)
                records.append(None)
                staged.setdefault(runtime.id, []).append((index, cell, job, size))
    for name, jobs in staged.items():
        results = execute(provision(name), [job for _, _, job, _ in jobs], root)
        for (index, cell, job, size), result in zip(jobs, results):
            try:
                evaluated = evaluate(cell, job, size, result)
            except (
                Exception
            ) as error:  # noqa: BLE001 - a broken comparison is a harness defect of the cell
                evaluated = dict(
                    outcome="harness-error",
                    exception=type(error).__name__,
                    message=str(error),
                )
            records[index] = dict(cell.base, runtime=name, **evaluated)
    return records


def summary(records):
    counts = Counter((record["runtime"], record["outcome"]) for record in records)
    lines = []
    for runtime in sorted({runtime for runtime, _ in counts}):
        parts = ", ".join(
            f"{outcome} {count}"
            for (name, outcome), count in sorted(counts.items())
            if name == runtime
        )
        lines.append(f"{runtime}: {parts}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory for artifacts and results",
    )
    parser.add_argument(
        "--cases", help="Comma-separated case identifiers; omitted means every fixture"
    )
    parser.add_argument(
        "--heldout-per-family",
        type=int,
        default=0,
        help="Also generate this many held-out cases per family and language",
    )
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--configs", default="strict")
    parser.add_argument(
        "--runtimes",
        default="available",
        help="'available', 'none', or a comma-separated list that must all be provisioned",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Execute each source on its own language",
    )
    parser.add_argument("--environment", choices=ENVIRONMENTS, default="vulkan1.1")
    parser.add_argument("--hlsl-profile", default=HLSL_PROFILE)
    parser.add_argument(
        "--baseline", type=Path, help="Scoreboard to check the results against"
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Replace the baseline's cells in this run's scope with the new results",
    )
    args = parser.parse_args(argv)
    cases = (
        [BY_ID[name] for name in args.cases.split(",")] if args.cases else list(CASES)
    )
    targets, configs = args.targets.split(","), args.configs.split(",")
    if set(targets) - set(TARGETS) or set(configs) - set(CONFIGS):
        parser.error("Unknown target or compile configuration")
    selection = select_runtimes(args.runtimes, targets)
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.output.resolve()
    if args.heldout_per_family:
        from .heldout import heldout_cases

        cases += heldout_cases(
            root / "heldout-sources", n_per_family=args.heldout_per_family
        )
    provisions = [
        provided.to_json() for pairs in selection.values() for _, provided in pairs
    ]
    (root / "run.json").write_text(
        json.dumps(
            {
                "argv": sys.argv[1:],
                "repository": str(REPOSITORY),
                "profile": profile(),
                "provisions": provisions,
            },
            indent=2,
        )
        + "\n"
    )
    records = run_suite(
        cases,
        targets,
        configs,
        selection,
        root,
        reference=args.reference,
        environment=args.environment,
        profile=args.hlsl_profile,
    )
    with (root / "results.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    print(summary(records))
    if not args.baseline:
        return 0
    from .scoreboard import check, improvements, merge

    runtimes = {runtime.id for pairs in selection.values() for runtime, _ in pairs}
    if any(not pairs for pairs in selection.values()):
        runtimes.add(compile_only_runtime())
    scope = {
        "case": {case.id for case in cases},
        "target": set(targets),
        "config": set(configs),
        "route": {"reference" if args.reference else "translated"},
        "environment": {args.environment},
        "runtime": runtimes,
    }
    baseline = json.loads(args.baseline.read_text()) if args.baseline.is_file() else {}
    if args.update_baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(
            json.dumps(merge(baseline, records, scope), indent=2, sort_keys=True) + "\n"
        )
        print(f"Updated {args.baseline}")
        return 0
    if not baseline:
        print(
            f"No scoreboard at {args.baseline} for profile {profile()}", file=sys.stderr
        )
        return 1
    for message in improvements(baseline, records):
        print(message)
    failures = check(baseline, records, scope=scope)
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
