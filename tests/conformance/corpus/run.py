"""Translate the third-party corpus to every target, check each artifact natively and score the result.

Every (unit, target) cell gets one outcome:

- ``usable``: the target's checker accepts the artifact, it declares an entry point and carries no fallback marker;
- ``degraded``: accepted, but a fallback marker shows a lost construct, or no entry point survived;
- ``refused``: CrossTL raised a typed diagnostic;
- ``invalid``: the target's checker rejects the artifact;
- ``translator-error``, ``timeout``, ``worker-crash``: an untyped exception, the time limit, or a dead process;
- ``validator-timeout``, ``unvalidated``, ``harness-error``: the cell could not be checked.

A cell is feasible when the target language can express every stage of the unit, partial when it can express
some, and infeasible otherwise; usable shares are computed over feasible cells. Translation latency is measured
inside a persistent worker around ``crosstl.translate`` alone, and its p95 per source language is checked
against ``budgets.json``. The scoreboard records one outcome code per cell; a run fails when a cell in its scope
is worse than recorded, a cell cannot be checked, or a latency budget is exceeded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import queue
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..toolchains import CACHE, profile
from . import checks
from .manifest import HERE, MANIFEST, STAGES, load, manifest_digest, rebuild

REPOSITORY = HERE.parents[2]
BUDGETS = HERE / "budgets.json"
SCOREBOARDS = HERE / "scoreboards"
SCHEMA = 1
TARGETS = ("metal", "hlsl", "glsl", "essl", "spirv")
BACKENDS = {
    "metal": "metal",
    "hlsl": "directx",
    "glsl": "opengl",
    "essl": "webgl",
    "spirv": "vulkan",
}
SOURCE_BACKENDS = {
    "glsl": "opengl",
    "hlsl": "directx",
    "metal": "metal",
    "spirv": "vulkan",
}
EXTENSIONS = {
    "metal": ".metal",
    "hlsl": ".hlsl",
    "glsl": ".glsl",
    "essl": ".essl.glsl",
    "spirv": ".spvasm",
}
GRAPHICS = {"vertex", "fragment"}
# Stages each target language can express (MSL 4: no geometry stage; its tessellation and ray tracing use other
# programming models; desktop GLSL 4.60 core has no mesh or ray-tracing stages; GLSL ES 3.00 is WebGL 2).
TARGET_STAGES = {
    "metal": {"compute", "mesh", "task"} | GRAPHICS,
    "hlsl": set(STAGES),
    "glsl": {"compute", "geometry", "tess_control", "tess_eval"} | GRAPHICS,
    "essl": set(GRAPHICS),
    "spirv": set(STAGES),
}
CODES = {
    "usable": "U",
    "refused": "R",
    "degraded": "D",
    "invalid": "I",
    "translator-error": "E",
    "timeout": "T",
    "worker-crash": "C",
    "validator-timeout": "V",
    "unvalidated": "N",
    "harness-error": "H",
}
OUTCOMES = {code: outcome for outcome, code in CODES.items()}
# Lower is better. A typed refusal beats any output that is wrong or unusable; unchecked cells rank last.
RANK = {"U": 0, "R": 1, "D": 2, "I": 3, "E": 4, "T": 4, "C": 4, "V": 5, "N": 5, "H": 5}
UNCHECKED = {"V", "N", "H"}


def feasibility(stages, target):
    expressible = set(stages) & TARGET_STAGES[target]
    if expressible == set(stages):
        return "feasible"
    return "partial" if expressible else "infeasible"


def cell_id(unit, target):
    return f"{unit.id}|{target}"


class Worker:
    """One persistent translation process; a timed-out or dead process is replaced before the next job."""

    def __init__(self, timeout, module="tests.conformance.corpus.worker"):
        self.timeout = timeout
        self.module = module
        self.child = None
        self.lines = None

    def _start(self):
        environment = {**os.environ, "PYTHONPATH": str(REPOSITORY)}
        self.child = subprocess.Popen(
            [sys.executable, "-m", self.module],
            cwd=REPOSITORY,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        )
        self.lines = queue.Queue()
        threading.Thread(
            target=self._read, args=(self.child, self.lines), daemon=True
        ).start()
        ready = self.lines.get(timeout=300)
        if ready is None:
            raise RuntimeError(f"The worker {self.module} exited during start-up")

    @staticmethod
    def _read(child, lines):
        for line in child.stdout:
            lines.put(line)
        lines.put(None)

    def stop(self):
        if self.child and self.child.poll() is None:
            self.child.kill()
            self.child.wait()
        self.child = None

    def translate(self, job):
        if self.child is None or self.child.poll() is not None:
            self._start()
        self.child.stdin.write(json.dumps(job) + "\n")
        self.child.stdin.flush()
        try:
            line = self.lines.get(timeout=self.timeout)
        except queue.Empty:
            self.stop()
            return {"status": "timeout", "seconds": float(self.timeout)}
        if line is None:
            code = self.child.wait()
            self.child = None
            return {
                "status": "worker-crash",
                "message": f"worker exited with status {code}",
            }
        return json.loads(line)


def load_translations(journal):
    """Translation results already appended to ``journal`` by an earlier, possibly interrupted, run."""
    done = {}
    if journal is not None and journal.is_file():
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut short when the run was killed
            done[entry.pop("id")] = entry
    return done


def translate_all(jobs, workers, timeout, progress=None, journal=None):
    """Run every job on ``workers`` persistent processes; the longest-expected jobs start first.

    Each result is appended to ``journal`` as it arrives, and jobs already recorded there are not run again, so
    an interrupted measurement resumes where it stopped.
    """
    results, lock = load_translations(journal), threading.Lock()
    artifacts = {job["id"]: Path(job["artifact"]) for job in jobs}
    # A recorded translation counts only while its artifact is still on disk.
    results = {
        identity: entry
        for identity, entry in results.items()
        if identity in artifacts
        and (entry.get("status") != "translated" or artifacts[identity].is_file())
    }
    pending = queue.Queue()
    for job in sorted(
        jobs, key=lambda job: (job["sourceLanguage"] != "metal", job["id"])
    ):
        if job["id"] not in results:
            pending.put(job)
    handle = journal.open("a", encoding="utf-8") if journal is not None else None
    if (
        handle is not None
        and handle.tell()
        and not journal.read_bytes().endswith(b"\n")
    ):
        handle.write("\n")  # end the line a killed run left unfinished

    def drain():
        worker = Worker(timeout)
        try:
            while True:
                try:
                    job = pending.get_nowait()
                except queue.Empty:
                    return
                result = worker.translate(job)
                with lock:
                    results[job["id"]] = result
                    if handle is not None:
                        handle.write(json.dumps({"id": job["id"], **result}) + "\n")
                        handle.flush()
                    if progress:
                        progress(len(results), len(jobs))
        finally:
            worker.stop()

    threads = [threading.Thread(target=drain) for _ in range(max(1, workers))]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        if handle is not None:
            handle.close()
    return results


def classify(translation, verdict, artifact, target):
    """Outcome of one cell from its translation result and native verdict."""
    status = translation["status"]
    if status != "translated":
        return status
    if verdict["verdict"] == "accepted":
        text = Path(artifact).read_text(errors="replace")
        entries = checks.count_entries(text, target)
        return (
            "usable"
            if translation.get("markers", 0) == 0 and entries > 0
            else "degraded"
        )
    return {
        "rejected": "invalid",
        "timeout": "validator-timeout",
        "unavailable": "unvalidated",
    }.get(verdict["verdict"], "harness-error")


def measure(units, targets, output, workers, timeout, progress=None, resume=False):
    """Translate, check and classify every (unit, target) cell; returns the cell records.

    Translation results are journalled in ``translations.jsonl``; ``resume`` keeps the ones already there.
    """
    journal = output / "translations.jsonl"
    if not resume and journal.exists():
        journal.unlink()
    jobs, cells = [], {}
    for unit in units:
        for target in targets:
            identity = cell_id(unit, target)
            artifact = (
                output / "artifacts" / unit.slice / f"{unit.path}{EXTENSIONS[target]}"
            )
            cells[identity] = (unit, target, artifact)
            jobs.append(
                {
                    "id": identity,
                    "source": str(unit.file),
                    "sourceLanguage": unit.language,
                    "sourceBackend": SOURCE_BACKENDS[unit.language],
                    "backend": BACKENDS[target],
                    "artifact": str(artifact),
                    "includeDirs": list(unit.include_dirs),
                    "defines": unit.defines,
                }
            )
    translations = translate_all(jobs, workers, timeout, progress, journal)

    def one(identity):
        unit, target, artifact = cells[identity]
        translation = translations[identity]
        verdict = (
            checks.check(artifact, target, unit.stages)
            if translation["status"] == "translated"
            else {"verdict": None, "commands": [], "log": ""}
        )
        record = {
            "cell": identity,
            "slice": unit.slice,
            "unit": unit.path,
            "language": unit.language,
            "tag": unit.tag,
            "target": target,
            "stages": list(unit.stages),
            "feasibility": feasibility(unit.stages, target),
            "outcome": classify(translation, verdict, artifact, target),
            "seconds": translation.get("seconds"),
            "cpuSeconds": translation.get("cpuSeconds"),
            "markers": translation.get("markers"),
            "diagnosticCode": translation.get("diagnosticCode"),
            "exception": translation.get("exception"),
            "message": (translation.get("message") or verdict["log"] or "")[-600:],
            "commands": verdict["commands"],
        }
        return record

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(one, sorted(cells)))


def percentile(values, fraction):
    """Nearest-rank percentile; ``None`` for no values."""
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def summarize(records, timeout):
    pairs = defaultdict(Counter)
    seconds = defaultdict(list)
    cpu_seconds = defaultdict(list)
    for record in records:
        pair = f"{record['language']}->{record['target']}"
        pairs[pair][record["feasibility"]] += 1
        if record["feasibility"] == "feasible":
            pairs[pair][record["outcome"]] += 1
        # A timed-out or crashed translation took at least the time limit; it is never dropped from latency.
        value = (
            record["seconds"]
            if record["outcome"] not in ("timeout", "worker-crash")
            else timeout
        )
        seconds[record["language"]].append(value if value is not None else timeout)
        seconds[pair].append(value if value is not None else timeout)
        # CPU time exists only for a translation that returned; a timeout is charged the limit here too.
        cpu = record.get("cpuSeconds")
        cpu_seconds[record["language"]].append(
            cpu
            if record["outcome"] not in ("timeout", "worker-crash") and cpu is not None
            else timeout
        )
    summary = {}
    for pair, counts in sorted(pairs.items()):
        feasible = counts["feasible"]
        summary[pair] = {
            **{name: counts[name] for name in ("feasible", "partial", "infeasible")},
            "outcomes": {name: counts[name] for name in CODES if counts[name]},
            "usableShare": round(counts["usable"] / feasible, 4) if feasible else None,
            "p95Seconds": round(percentile(seconds[pair], 0.95), 4),
        }
    latency = {
        language: {
            "cells": len(values),
            "p50Seconds": round(percentile(values, 0.5), 4),
            "p95Seconds": round(percentile(values, 0.95), 4),
            "maxSeconds": round(max(values), 4),
            "p95CpuSeconds": round(percentile(cpu_seconds[language], 0.95), 4),
        }
        for language, values in sorted(seconds.items())
        if "->" not in language
    }
    return {"pairs": summary, "latency": latency}


def encode(records, manifest):
    """Scoreboard cells: per slice and target, one outcome code per unit in manifest order ('-' unmeasured)."""
    by_cell = {record["cell"]: CODES[record["outcome"]] for record in records}
    cells = {}
    for name, entry in manifest["slices"].items():
        for target in TARGETS:
            row = "".join(
                by_cell.get(f"{name}:{unit['path']}|{target}", "-")
                for unit in entry["units"]
            )
            if row.strip("-"):
                cells.setdefault(name, {})[target] = row
    return cells


def decode(cells, manifest):
    decoded = {}
    for name, targets in cells.items():
        units = manifest["slices"][name]["units"]
        for target, row in targets.items():
            if len(row) != len(units):
                raise ValueError(
                    f"Scoreboard row {name}/{target} does not match the manifest"
                )
            for unit, code in zip(units, row):
                if code != "-":
                    decoded[f"{name}:{unit['path']}|{target}"] = code
    return decoded


def compare(baseline, records, manifest):
    """Failures and improvements of this run's cells against the recorded scoreboard."""
    if baseline.get("manifestSHA256") != manifest_digest():
        return ["The scoreboard was recorded for a different corpus manifest"], []
    expected = decode(baseline["cells"], manifest)
    failures, better = [], []
    for record in records:
        actual, recorded = CODES[record["outcome"]], expected.get(record["cell"])
        if actual in UNCHECKED:
            failures.append(
                f"Unchecked {record['cell']}: {record['outcome']}: {record['message'][-200:]}"
            )
        elif recorded is None:
            failures.append(f"Unrecorded cell {record['cell']}: {record['outcome']}")
        elif RANK[actual] > RANK[recorded]:
            failures.append(
                f"Regression {record['cell']}: {OUTCOMES[recorded]} -> {record['outcome']}"
            )
        elif actual != recorded:
            better.append(
                f"Improved {record['cell']}: {OUTCOMES[recorded]} -> {record['outcome']}"
            )
    return failures, better


def check_budgets(summary, budgets):
    """Wall-time budgets bound what a user waits; CPU-time budgets catch a slowdown on a loaded runner."""
    failures = []
    for language, measured in summary["latency"].items():
        for key, kind in (("p95Seconds", "wall"), ("p95CpuSeconds", "CPU")):
            if key not in budgets:
                continue
            limit = budgets[key].get(language)
            if limit is None:
                failures.append(
                    f"No {kind} latency budget for source language {language}"
                )
            elif measured.get(key) is not None and measured[key] > limit:
                failures.append(
                    f"Latency budget exceeded for {language}: {kind} p95 {measured[key]} s > {limit} s"
                )
    return failures


def pair_shares(cells, manifest):
    """Usable share per (source, target) pair over the feasible cells of a decoded scoreboard."""
    stages = {
        f"{name}:{unit['path']}": (entry["language"], unit["stages"])
        for name, entry in manifest["slices"].items()
        for unit in entry["units"]
    }
    counts = defaultdict(Counter)
    for identity, code in cells.items():
        unit, target = identity.rsplit("|", 1)
        language, unit_stages = stages[unit]
        if feasibility(unit_stages, target) == "feasible":
            counts[f"{language}->{target}"][OUTCOMES[code]] += 1
    return {
        pair: {
            "feasible": sum(found.values()),
            "outcomes": dict(sorted(found.items())),
            "usableShare": round(found["usable"] / sum(found.values()), 4),
        }
        for pair, found in sorted(counts.items())
    }


def scoreboard(records, manifest, previous=None):
    """The committed scoreboard: earlier cells outside this run are kept, this run's cells replace theirs."""
    cells = decode(previous["cells"], manifest) if previous else {}
    cells.update({record["cell"]: CODES[record["outcome"]] for record in records})
    merged = [
        {"cell": identity, "outcome": OUTCOMES[code]}
        for identity, code in cells.items()
    ]
    return {
        "schemaVersion": SCHEMA,
        "manifestSHA256": manifest_digest(),
        "codes": OUTCOMES,
        "cells": encode(merged, manifest),
        "pairs": pair_shares(cells, manifest),
    }


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=CACHE / "corpus")
    parser.add_argument("--slices", help="Comma-separated slice names (default: all)")
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument(
        "--limit", type=int, help="First N units per slice (a smoke subset)"
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2)
    )
    parser.add_argument("--timeout", type=float)
    parser.add_argument(
        "--scoreboard", type=Path, default=SCOREBOARDS / f"{profile()}.json"
    )
    parser.add_argument("--update-scoreboard", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep the translations an interrupted run journalled in OUTPUT/translations.jsonl",
    )
    parser.add_argument("--budgets", type=Path, default=BUDGETS)
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Measure only; do not compare or check budgets",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    targets = [t for t in args.targets.split(",") if t]
    unknown = set(targets) - set(TARGETS)
    if unknown:
        raise SystemExit(f"Unknown targets: {sorted(unknown)}")
    manifest = load(MANIFEST)
    budgets = json.loads(args.budgets.read_text())
    timeout = args.timeout or budgets["timeoutSeconds"]
    slices = [name for name in (args.slices or "").split(",") if name] or None
    units, problems = rebuild(args.cache, manifest, slices)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    if args.limit:
        kept = Counter()
        units = [
            u for u in units if kept.update([u.slice]) or kept[u.slice] <= args.limit
        ]
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def progress(done, total):
        if done % 250 == 0 or done == total:
            print(f"translated {done}/{total} cells", file=sys.stderr, flush=True)

    records = measure(
        units, targets, args.output, args.workers, timeout, progress, args.resume
    )
    summary = summarize(records, timeout)
    with (args.output / "results.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    run = {
        "profile": profile(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "workers": args.workers,
        "timeoutSeconds": timeout,
        "units": len(units),
        "targets": targets,
        "wallSeconds": round(time.perf_counter() - started, 1),
        "resumed": args.resume,
        "manifestSHA256": manifest_digest(),
        **summary,
    }
    (args.output / "summary.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n"
    )
    for pair, data in summary["pairs"].items():
        print(
            f"{pair:16s} feasible {data['feasible']:4d} usable {data['usableShare']} p95 {data['p95Seconds']} s"
        )
    previous = (
        json.loads(args.scoreboard.read_text()) if args.scoreboard.is_file() else None
    )
    if previous and previous.get("manifestSHA256") != manifest_digest():
        previous_cells = None
    else:
        previous_cells = previous
    # The candidate is the committed scoreboard with this run's cells replaced; commit it to record a change.
    board = (
        json.dumps(
            scoreboard(records, manifest, previous_cells), indent=1, sort_keys=True
        )
        + "\n"
    )
    (args.output / f"{profile()}.json").write_text(board)
    if args.update_scoreboard:
        args.scoreboard.parent.mkdir(parents=True, exist_ok=True)
        args.scoreboard.write_text(board)
        print(f"Updated {args.scoreboard}")
        return 0
    if args.no_gate:
        return 0
    if previous is None:
        print(
            f"No corpus scoreboard at {args.scoreboard} for profile {profile()}",
            file=sys.stderr,
        )
        return 1
    failures, better = compare(previous, records, manifest)
    failures += check_budgets(summary, budgets)
    for message in better:
        print(message)
    for message in failures:
        print(message, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
