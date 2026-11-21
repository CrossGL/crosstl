"""Check that translation output does not depend on ``PYTHONHASHSEED``.

The engineering criterion is that output is identical across hash seeds: emission must not depend on the
iteration order of sets or dicts keyed by strings, which changes with the seed, or on object addresses. This
check translates every shader source under ``ROOTS`` to every backend in ``backends``, once per seed, with each
seed in its own interpreter, and compares the results.

A result is the SHA-256 of the translated text or, when translation raises, the exception's type and the SHA-256
of its message. A refusal therefore has to be as stable as an output: the same input must be refused with the
same diagnostic under every seed. Inputs are discovered by the source extensions the translator registers, so a
new fixture or demo is covered without editing this file. Paths are passed relative to the repository root, and
the workers run from it, so that diagnostics that quote a path are compared as they are.

The conformance workflow's "Hash-seed determinism" job runs this check. Pull requests
translate to CrossGL and the graphics targets under two seeds (the defaults); pushes to main, the nightly schedule
and manual runs translate to every backend under three::

    python -m tests.conformance.determinism --seeds 0,1,4242 --backends all

The command prints each difference and exits with status 1 when any result differs between seeds.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]

# Repository sources translated by the check. The integration demos under demos/integrations belong to the corpus
# suites that run them (MLX and ComputeSharp) and are left to those suites.
ROOTS = (
    "examples",
    "demos/readme",
    "demos/open-source-porting",
    "tests/conformance/cases",
    "tests/fixtures",
)

# The targets of the pull-request test: CrossGL itself (the importers' output) and the graphics targets.
GRAPHICS_BACKENDS = ("cgl", "metal", "directx", "opengl", "vulkan")
DEFAULT_SEEDS = ("0", "1")

# A worker is given this long for its whole shard; a hang fails the check rather than the job's timeout.
WORKER_TIMEOUT_SECONDS = 1800


def source_extensions() -> frozenset[str]:
    """The file extensions the translator accepts as sources."""
    from crosstl.translator.source_registry import (
        SOURCE_REGISTRY,
        register_default_sources,
    )

    register_default_sources()
    return frozenset(SOURCE_REGISTRY.extensions())


def all_backends() -> tuple[str, ...]:
    """CrossGL and every registered code generator, in a stable order."""
    from crosstl.translator.codegen.registry import BACKEND_REGISTRY

    return ("cgl", *sorted(BACKEND_REGISTRY.names()))


def discover_inputs(
    roots: Sequence[str] = ROOTS, repository: Path = REPOSITORY
) -> list[str]:
    """Repository-relative POSIX paths of every source file under ``roots``, sorted."""
    extensions = source_extensions()
    found = set()
    for root in roots:
        base = repository / root
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in extensions:
                found.add(path.relative_to(repository).as_posix())
    return sorted(found)


def translate_result(path: str, backend: str) -> str:
    """One input's result for one backend, as a hash that is comparable across processes."""
    import crosstl

    try:
        text = crosstl.translate(path, backend=backend)
    except Exception as error:  # every refusal and crash is part of the result
        digest = hashlib.sha256(str(error).encode("utf-8", "replace")).hexdigest()
        return f"error {type(error).__name__} {digest}"
    if not isinstance(text, str):
        text = repr(text)
    return "ok " + hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def run_worker(inputs: Sequence[str], backends: Sequence[str]) -> dict[str, str]:
    """Translate in this interpreter; keys are ``path<TAB>backend``."""
    return {
        f"{path}\t{backend}": translate_result(path, backend)
        for path in inputs
        for backend in backends
    }


def _worker_environment(seed: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONHASHSEED": seed,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    environment.pop("PYTEST_XDIST_WORKER", None)
    return environment


def default_jobs(seed_count: int) -> int:
    """Worker processes per seed: share the CPUs, and use one each under pytest-xdist, which already does."""
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return 1
    return max(1, (os.cpu_count() or 1) // max(1, seed_count))


def results_by_seed(
    inputs: Sequence[str],
    backends: Sequence[str],
    seeds: Sequence[str],
    jobs: int | None = None,
    repository: Path = REPOSITORY,
) -> dict[str, dict[str, str]]:
    """Run ``jobs`` workers per seed, each over every ``jobs``-th input, and merge their results per seed."""
    jobs = jobs or default_jobs(len(seeds))
    request = {"backends": list(backends)}
    processes = []
    for seed in seeds:
        for shard in range(jobs):
            shard_inputs = list(inputs[shard::jobs])
            if not shard_inputs:
                continue
            payload = json.dumps({**request, "inputs": shard_inputs})
            process = subprocess.Popen(
                [sys.executable, "-m", "tests.conformance.determinism", "--worker"],
                cwd=repository,
                env=_worker_environment(seed),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            processes.append((seed, shard, process, payload))
    merged: dict[str, dict[str, str]] = {seed: {} for seed in seeds}
    failures = []
    for seed, shard, process, payload in processes:
        try:
            stdout, stderr = process.communicate(
                payload, timeout=WORKER_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            failures.append(f"seed {seed} shard {shard}: timed out")
            continue
        if process.returncode != 0:
            tail = "\n".join(stderr.strip().splitlines()[-20:])
            failures.append(
                f"seed {seed} shard {shard}: exit {process.returncode}\n{tail}"
            )
            continue
        merged[seed].update(json.loads(stdout))
    if failures:
        raise RuntimeError("determinism workers failed:\n" + "\n".join(failures))
    return merged


def differences(by_seed: dict[str, dict[str, str]]) -> list[str]:
    """One line for every result that is not the same under every seed, and for every result a seed lacks."""
    seeds = list(by_seed)
    keys = sorted(set().union(*(set(results) for results in by_seed.values())))
    lines = []
    for key in keys:
        values = [by_seed[seed].get(key, "missing") for seed in seeds]
        if len(set(values)) > 1:
            path, backend = key.split("\t")
            detail = ", ".join(
                f"seed {seed}: {value[:40]}" for seed, value in zip(seeds, values)
            )
            lines.append(f"{path} -> {backend}: {detail}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--seeds",
        default=",".join(DEFAULT_SEEDS),
        help="comma-separated PYTHONHASHSEED values",
    )
    parser.add_argument(
        "--backends",
        default=",".join(GRAPHICS_BACKENDS),
        help="comma-separated backends, or 'all'",
    )
    parser.add_argument(
        "--jobs", type=int, default=None, help="worker processes per seed"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="repository-relative sources (default: every source under ROOTS)",
    )
    args = parser.parse_args(argv)
    if args.worker:
        request = json.loads(sys.stdin.read())
        # Anything the translator prints goes to stderr, so that stdout carries only the results.
        with contextlib.redirect_stdout(sys.stderr):
            results = run_worker(request["inputs"], request["backends"])
        json.dump(results, sys.stdout)
        return 0
    seeds = [seed for seed in args.seeds.split(",") if seed]
    if len(seeds) < 2:
        parser.error("give at least two seeds")
    backends = (
        all_backends()
        if args.backends == "all"
        else tuple(name for name in args.backends.split(",") if name)
    )
    inputs = args.inputs or discover_inputs()
    by_seed = results_by_seed(inputs, backends, seeds, args.jobs)
    found = differences(by_seed)
    for line in found:
        print(line)
    print(
        f"{len(inputs)} inputs x {len(backends)} backends under seeds {', '.join(seeds)}: "
        f"{len(found)} results differ",
        file=sys.stderr,
    )
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
