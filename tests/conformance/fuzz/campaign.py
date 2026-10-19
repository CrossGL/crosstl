"""Crash-fuzzing campaigns for every CrossTL importer, with a ratcheting baseline of known crashes.

``python -m tests.conformance.fuzz --importers all --seconds 600 --output DIR`` fuzzes each importer for the
given time (or ``--examples`` inputs) in a persistent worker process with a per-input time limit. Hypothesis
generates the inputs in seeded batches. When an input crashes with a signature the baseline does not list,
hypothesis shrinks it, the minimal input is written to ``DIR/findings/``, and fuzzing continues with the next
batch. ``DIR/summary.json`` counts every (input kind, route, status) per importer.

The baseline (``tests/conformance/fuzz/baseline.json``) lists the known crash signatures, each with a committed
minimal reproducer under ``reproducers/``. Every run first replays the reproducers. The run fails when a new
signature is found, or when a reproducer no longer crashes (the crash was fixed, so the baseline must shrink).
``--update-baseline`` applies both changes instead of failing. The goal state is an empty baseline.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

from hypothesis import HealthCheck, Phase, Verbosity, given, seed, settings
from hypothesis.errors import HypothesisException

from ..corpus.run import Worker
from .outcomes import CRASH_STATUSES, signature
from .targets import IMPORTERS, case_text

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "baseline.json"
BATCH = 200
WORKER_MODULE = "tests.conformance.fuzz.worker"


class _Stop(BaseException):
    """The budget is spent. A BaseException, so hypothesis propagates it without shrinking."""


class _Found(Exception):
    """A crash with a new signature; hypothesis shrinks the input that raised it."""


def run_job(worker, job):
    outcome = worker.translate(job)
    if outcome["status"] in ("timeout", "worker-crash"):
        outcome["signature"] = signature(
            job["importer"], job["backend"], outcome["status"]
        )
    return outcome


class Campaign:
    def __init__(
        self,
        target,
        worker,
        known,
        *,
        seconds,
        examples,
        seed_value,
        shrink_seconds,
        backends,
    ):
        self.target = target
        self.worker = worker
        self.known = set(known)
        self.deadline = time.monotonic() + seconds if seconds else None
        self.examples = examples
        self.seed = seed_value
        self.shrink_seconds = shrink_seconds
        self.backends = backends or target.backends
        self.generated = 0
        self.stats = collections.Counter()
        self.crashes = collections.Counter()
        self.rejections = collections.Counter()
        self.findings = {}

    def exhausted(self):
        if self.examples is not None and self.generated >= self.examples:
            return True
        return self.deadline is not None and time.monotonic() >= self.deadline

    def run(self):
        batch = 0
        while not self.exhausted():
            self._batch(batch)
            batch += 1
        return self

    def _batch(self, batch):
        state = {"target": None, "until": None}

        @settings(
            max_examples=BATCH,
            database=None,
            deadline=None,
            phases=(Phase.generate, Phase.shrink),
            report_multiple_bugs=False,
            suppress_health_check=list(HealthCheck),
            verbosity=Verbosity.quiet,
        )
        @seed(self.seed * 1_000_003 + batch)
        @given(self.target.strategy())
        def probe(case):
            shrinking = state["target"] is not None
            if shrinking and time.monotonic() >= state["until"]:
                raise _Stop()
            if not shrinking:
                if self.exhausted():
                    raise _Stop()
                self.generated += 1
            for backend in self.backends:
                outcome = run_job(self.worker, self.target.job(case, backend))
                if not shrinking:
                    self._count(case, backend, outcome)
                key = outcome.get("signature")
                if not key or key in self.known:
                    continue
                self._record(key, case, backend, outcome)
                if state["target"] in (None, key):
                    if state["target"] is None:
                        state["target"] = key
                        state["until"] = time.monotonic() + self.shrink_seconds
                    raise _Found(key)

        try:
            probe()
        except (_Found, _Stop, HypothesisException):
            pass
        if state["target"]:
            self.known.add(state["target"])

    def _count(self, case, backend, outcome):
        status = outcome["status"]
        self.stats[f"{case[0]}|{backend}|{status}"] += 1
        if status in CRASH_STATUSES:
            self.crashes[outcome["signature"]] += 1
        elif status == "rejected":
            self.rejections[f"{backend}|{outcome.get('exception')}"] += 1

    def _record(self, key, case, backend, outcome):
        text = case_text(case)
        previous = self.findings.get(key)
        if previous is None or len(text) < len(previous["text"]):
            self.findings[key] = {
                "text": text,
                "importer": self.target.name,
                "backend": backend,
                "extension": case[1],
                "input": case[0],
                "exception": outcome.get("exception"),
                "message": outcome.get("message", ""),
            }

    def summary(self):
        return {
            "inputs": self.generated,
            "outcomes": dict(sorted(self.stats.items())),
            "crashSignatures": dict(sorted(self.crashes.items())),
            "untypedRejections": dict(sorted(self.rejections.items())),
            "newSignatures": sorted(self.findings),
        }


def reproducer_name(finding):
    digest = hashlib.sha256(finding["text"].encode("utf-8")).hexdigest()[:12]
    return f"{finding['importer']}-{digest}{finding['extension']}"


def reproducers(baseline_path):
    """Reproducers live next to their baseline."""
    return Path(baseline_path).parent / "reproducers"


def replay_job(entry, directory):
    text = (directory / entry["reproducer"]).read_text(encoding="utf-8")
    if entry["importer"] == "preprocessor":
        return {
            "kind": "preprocess",
            "importer": "preprocessor",
            "backend": entry["backend"],
            "unit": json.loads(text),
        }
    return {
        "kind": "translate",
        "importer": entry["importer"],
        "extension": entry["extension"],
        "backend": entry["backend"],
        "source": text,
    }


def replay(baseline, worker, importers, directory):
    """Signatures whose reproducer no longer reproduces them, with what happened instead."""
    fixed = {}
    for key, entry in sorted(baseline["crashes"].items()):
        if entry["importer"] not in importers:
            continue
        outcome = run_job(worker, replay_job(entry, directory))
        if outcome.get("signature") != key:
            fixed[key] = outcome.get("signature") or outcome["status"]
    return fixed


def load_baseline(path):
    if path and Path(path).is_file():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return {"schemaVersion": 1, "crashes": {}}


def write_baseline(path, baseline, findings, fixed):
    directory = reproducers(path)
    directory.mkdir(exist_ok=True)
    for key in fixed:
        entry = baseline["crashes"].pop(key)
        (directory / entry["reproducer"]).unlink(missing_ok=True)
    for key, finding in findings.items():
        name = reproducer_name(finding)
        (directory / name).write_text(finding["text"], encoding="utf-8")
        baseline["crashes"][key] = {
            "importer": finding["importer"],
            "backend": finding["backend"],
            "extension": finding["extension"],
            "input": finding["input"],
            "reproducer": name,
            "message": finding["message"][:200],
        }
    baseline["crashes"] = dict(sorted(baseline["crashes"].items()))
    Path(path).write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--importers", default="all", help="comma-separated, or all")
    parser.add_argument(
        "--seconds", type=float, default=None, help="budget per importer"
    )
    parser.add_argument(
        "--examples", type=int, default=None, help="inputs per importer"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=30.0, help="seconds per input")
    parser.add_argument("--shrink-seconds", type=float, default=300.0)
    parser.add_argument(
        "--backends", default=None, help="override each importer's routes"
    )
    parser.add_argument("--baseline", default=str(BASELINE))
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    if arguments.seconds is None and arguments.examples is None:
        parser.error("give --seconds or --examples")
    names = (
        list(IMPORTERS)
        if arguments.importers == "all"
        else arguments.importers.split(",")
    )
    unknown = sorted(set(names) - set(IMPORTERS))
    if unknown:
        parser.error(f"unknown importers {unknown}; known: {sorted(IMPORTERS)}")
    arguments.importers = names
    return arguments


def main(argv=None):
    arguments = parse_arguments(argv)
    output = Path(arguments.output)
    (output / "findings").mkdir(parents=True, exist_ok=True)
    baseline = load_baseline(arguments.baseline)
    backends = tuple(arguments.backends.split(",")) if arguments.backends else None
    worker = Worker(arguments.timeout, module=WORKER_MODULE)
    summary, findings = {"seed": arguments.seed, "importers": {}}, {}
    try:
        fixed = replay(
            baseline, worker, arguments.importers, reproducers(arguments.baseline)
        )
        for name in arguments.importers:
            started = time.monotonic()
            campaign = Campaign(
                IMPORTERS[name],
                worker,
                baseline["crashes"],
                seconds=arguments.seconds,
                examples=arguments.examples,
                seed_value=arguments.seed,
                shrink_seconds=arguments.shrink_seconds,
                backends=backends,
            ).run()
            summary["importers"][name] = campaign.summary()
            summary["importers"][name]["seconds"] = round(time.monotonic() - started, 1)
            findings.update(campaign.findings)
            print(
                f"{name}: {campaign.generated} inputs, {sum(campaign.crashes.values())} crashing, "
                f"{len(campaign.findings)} new signatures",
                flush=True,
            )
    finally:
        worker.stop()
    for key, finding in findings.items():
        (output / "findings" / reproducer_name(finding)).write_text(
            finding["text"], encoding="utf-8"
        )
    summary["fixed"] = fixed
    summary["new"] = {
        key: reproducer_name(finding) for key, finding in sorted(findings.items())
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    for key in sorted(findings):
        print(f"NEW CRASH {key}: {findings[key]['message'][:160]!r}")
    for key, now in sorted(fixed.items()):
        print(f"FIXED {key} (now {now}): remove it from the baseline")
    if arguments.update_baseline:
        write_baseline(arguments.baseline, baseline, findings, fixed)
        return 0
    return 1 if findings or fixed else 0


if __name__ == "__main__":
    sys.exit(main())
