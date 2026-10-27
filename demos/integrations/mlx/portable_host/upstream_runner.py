"""Run one unchanged pinned upstream matrix-product test on CPU and natively.

A runner module (``verify_upstream_qvm``, ``verify_upstream_qmv``) describes its
test and passes itself here. It provides:

- ``MODULE``: its importable name, for the bounded subprocesses;
- ``UPSTREAM_TESTS`` and ``LABEL``: the unchanged test and its log label;
- ``TIMEOUTS``: the deadline of each bounded command, by mode;
- ``prepare_packages(root, packages, reductions, jobs)``: the ``--prepare`` step;
- ``validate_trace(trace, target, count)``: the native trace's requirements;
- ``REDUCTIONS``: the whole-array maxima of its error bounds. Each reduction
  family's maxima sit in ``maxima_directories(reductions, REDUCTIONS)``, which
  the native worker loads;
- optionally ``COLUMN_SUMS``: column reduction entries ``prepare_packages``
  translates into ``column_directory(reductions)``, which the native worker
  also loads;
- optionally ``OPERATORS``: element-wise package families (``packages.py``'s
  ``half``, ``half-arithmetic`` and ``bfloat``) for a test that computes in,
  or converts from, a 16-bit float type. ``prepare_packages`` translates each into
  ``operator_directory(reductions, family)`` and the native worker loads them.

``run`` translates the random packages, then the module's packages, then runs
the test on MLX's CPU and through the translated host, and checks that the
pinned sources and adaptation are the same afterwards.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.trace_records import read_trace_summary
from demos.integrations.mlx.portable_host.verify import upstream_test_sources
from demos.integrations.mlx.portable_host.verify_upstream_quantization import (
    execute,
    worker,
    write_json,
)

RANDOM_PACKAGES = "demos.integrations.mlx.portable_host.random_packages"
# Whole-array reduction families, in the order the worker loads them.
MAXIMA_FAMILIES = ("all", "half", "bfloat")
# The worker option that loads each element-wise package family.
OPERATOR_OPTIONS = {
    "half": "--half",
    "half-arithmetic": "--half-arithmetic",
    "bfloat": "--bfloat",
}


def row_strides(shape):
    strides, step = [], 1
    for extent in reversed(shape):
        strides.append(step)
        step *= extent
    return strides[::-1]


def check_trace(trace, target, count, *, label, expected, prefixes, reductions):
    """Require ``trace`` to hold exactly ``expected``'s routed dispatches.

    ``count`` is the native worker's dispatch count; every record must name
    ``target``. The dispatches whose entries start with ``prefixes`` must match
    ``expected`` exactly, and one of ``reductions`` must have run natively.
    """
    if len(trace) != count or any(record.get("target") != target for record in trace):
        raise RuntimeError(
            f"Upstream {label} trace is incomplete or uses another target"
        )
    observed = Counter(
        record["entry"] for record in trace if record["entry"].startswith(prefixes)
    )
    if observed != expected:
        raise RuntimeError(
            f"Upstream {label} did not route every product through its translated kernel"
        )
    if not any(record["entry"] in reductions for record in trace):
        raise RuntimeError(
            f"Upstream {label} did not evaluate its error bound natively"
        )


def column_directory(reductions):
    """Where a runner's static column reduction packages sit beside its maxima."""
    reductions = Path(reductions)
    return reductions.with_name(reductions.name + "-column")


def maxima_directories(reductions, maxima):
    """Each reduction family of ``maxima`` and the directory that packages it.

    The float32 family keeps ``reductions`` itself; a 16-bit family sits
    beside it, as ``reductions-half`` or ``reductions-bfloat``.
    """
    from demos.integrations.mlx.portable_host.reduction_packages import ENTRY_GROUPS

    reductions = Path(reductions)
    families = {}
    for entry in maxima:
        family = next(
            (name for name in MAXIMA_FAMILIES if entry in ENTRY_GROUPS[name]), None
        )
        if family is None:
            raise ValueError(f"No whole-array reduction family packages {entry}")
        families.setdefault(family, []).append(entry)
    return {
        family: (
            (
                reductions
                if family == "all"
                else reductions.with_name(f"{reductions.name}-{family}")
            ),
            tuple(families[family]),
        )
        for family in MAXIMA_FAMILIES
        if family in families
    }


def operator_directory(reductions, family):
    """Where element-wise package ``family`` sits beside a runner's maxima."""
    if family not in OPERATOR_OPTIONS:
        raise ValueError(f"Unsupported upstream operator package family: {family}")
    return Path(reductions).with_name(f"{family}-packages")


def translate_ahead(
    root, packages, reductions, jobs, groups, maxima, columns=(), operators=()
):
    """Translate each ``(builder, entries)`` group and index ``maxima`` on demand.

    ``builder`` is a package module with ``build_entries``; the maxima are
    packaged when the run first launches them, at the widths it launches, in
    their family's ``maxima_directories`` entry. ``columns`` are column
    reduction entries translated into ``column_directory(reductions)``, and
    ``operators`` element-wise families translated into ``operator_directory``.
    """
    from demos.integrations.mlx.portable_host import packages as operator_packages
    from demos.integrations.mlx.portable_host import reduction_packages

    target = json.loads((Path(packages) / "index.json").read_text())["target"]
    records = []
    for builder, entries in groups:
        records += builder.build_entries(root, packages, target, entries, jobs)
    for family, (directory, entries) in maxima_directories(reductions, maxima).items():
        reduction_packages.build_packages(
            root,
            directory,
            target,
            family=family,
            entries=entries,
            on_demand=True,
        )
    for family in operators:
        operator_packages.build_packages(
            root, operator_directory(reductions, family), target, family=family
        )
    if columns:
        reduction_packages.build_packages(
            root,
            column_directory(reductions),
            target,
            family="column",
            entries=tuple(columns),
            jobs=jobs,
        )
    for record in records:
        print(json.dumps(record), flush=True)


def _commands(spec, root, packages, output, target, jobs):
    random, reductions = output / "random-packages", output / "reductions"
    loaded = []
    for directory, _ in maxima_directories(reductions, spec.REDUCTIONS).values():
        loaded += ["--reductions", str(directory)]
    if getattr(spec, "COLUMN_SUMS", ()):
        loaded += ["--reductions", str(column_directory(reductions))]
    for family in getattr(spec, "OPERATORS", ()):
        loaded += [
            OPERATOR_OPTIONS[family],
            str(operator_directory(reductions, family)),
        ]
    module = ["-m", spec.MODULE]
    common = ["--mlx-root", str(root), "--packages", str(packages)]
    commands = [
        (
            "random",
            spec.TIMEOUTS["random"],
            ["-m", RANDOM_PACKAGES, "--mlx-root", str(root), "--target", target]
            + ["--output-dir", str(random)],
        ),
        (
            "packages",
            spec.TIMEOUTS["packages"],
            [*module, "--prepare", *common, "--jobs", str(jobs)]
            + ["--output-dir", str(reductions)],
        ),
    ]
    for mode in ("cpu", "native"):
        commands.append(
            (
                mode,
                spec.TIMEOUTS[mode],
                [*module, "--worker", mode, *common]
                + [*loaded, "--random", str(random)]
                + ["--output-dir", str(output / mode)],
            )
        )
    return commands


def run(spec, root, packages, output, jobs=3):
    """Run ``spec``'s test end to end and return the evidence it records."""
    root, packages, output = (Path(path).resolve() for path in (root, packages, output))
    output.mkdir(parents=True, exist_ok=False)
    tests, label = spec.UPSTREAM_TESTS, spec.LABEL
    preparation = verify_prepared(root)
    sources = upstream_test_sources(root, tests)
    target = json.loads((packages / "index.json").read_text())["target"]
    write_json(
        output / "inputs.json",
        {"adaptation": preparation, "sources": sources, "target": target},
    )
    commands = _commands(spec, root, packages, output, target, jobs)
    failures, results = execute(output, commands, tests, f"MLX {label}")
    after = verify_prepared(root)
    write_json(output / "adaptation-after.json", after)
    if preparation != after or sources != upstream_test_sources(root, tests):
        raise RuntimeError(f"MLX sources changed during upstream {label} verification")
    if failures:
        raise RuntimeError(
            f"Upstream {label} workers failed: {', '.join(failures)}; inspect {output}"
        )
    trace = read_trace_summary(output / "native/trace.jsonl")
    spec.validate_trace(trace, target, results["native"]["nativeDispatches"])
    evidence = {
        "commit": COMMIT,
        "target": target,
        "upstreamTestSources": sources,
        "results": results,
        "entries": dict(sorted(Counter(record["entry"] for record in trace).items())),
        "sourceUnchanged": True,
        "fullUpstreamSuite": False,
    }
    write_json(output / "evidence.json", evidence)
    return evidence


def main(spec, argv=None):
    """The command line every runner module shares."""
    parser = argparse.ArgumentParser(description=spec.__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reductions", type=Path, action="append")
    parser.add_argument("--random", type=Path)
    parser.add_argument("--half", type=Path)
    parser.add_argument("--half-arithmetic", type=Path)
    parser.add_argument("--bfloat", type=Path)
    parser.add_argument("--jobs", type=int, default=3)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--worker", choices=("cpu", "native"))
    mode.add_argument("--prepare", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare:
        spec.prepare_packages(args.mlx_root, args.packages, args.output_dir, args.jobs)
    elif args.worker:
        # Thousands of dispatches: full value arrays would exceed 4 GB of trace.
        worker(args, spec.UPSTREAM_TESTS, trace_values=False)
    else:
        run(spec, args.mlx_root, args.packages, args.output_dir, args.jobs)
