"""The Metal sin probe and the fake conformance harness the execution tests share."""

import numpy as np

from tests.conformance import runner as harness
from tests.conformance import runtimes
from tools.coverage import (
    probes,
)
from tools.coverage.model import VOID, Feature, Param, Result, Signature, Type

SIN = Feature(
    "metal",
    "function",
    "sin",
    Signature(Type("float"), (Param(Type("float")),)),
    crossgl="sin",
)
NATIVE = probes.build(SIN, "metal").source


def supported(translated=NATIVE, direction="output"):
    return Result(SIN, direction, "supported", sources=(NATIVE, translated))


def metal_runs():
    try:
        return runtimes.provision(runtimes.RUNTIMES["metal-apple"]).available
    except Exception:  # noqa: BLE001 - an unprovisionable runtime is an unavailable one
        return False


def fake_harness(
    monkeypatch,
    translated_outcome,
    written,
    runtime="metal-apple",
    environments=None,
    message="",
):
    """Make run_suite report the native probe as run and the translation as ``translated_outcome``.

    ``environments``, when given, collects the Vulkan environment of each run. ``message`` is what the runtime says
    about the translation.
    """

    class Provided:
        available = True

    runtime = runtimes.RUNTIMES[runtime]
    monkeypatch.setattr(
        harness,
        "select_runtimes",
        lambda spec, targets: {t: [(runtime, Provided())] for t in targets},
    )

    def run_suite(
        cases, targets, configs, selection, root, *, reference=False, environment
    ):
        assert reference and targets == [runtime.target] and configs == ["strict"]
        if environments is not None:
            environments.append(environment)
        records = []
        for case in cases:
            assert "#version 460" not in case.src.read_text()
            role = case.id.split("-")[1]
            outcome = "preserved" if role == "native" else translated_outcome
            words = np.zeros(case.out_count, dtype=np.uint32)
            if role == "translated":
                words[:] = written
            if outcome in ("preserved", "memory-corruption"):
                case.check(case.inputs, words)
            record = {"case": case.id, "runtime": runtime.id, "outcome": outcome}
            if role == "translated" and message:
                record["execution"] = {"status": outcome, "message": message}
            records.append(record)
        return records

    monkeypatch.setattr(harness, "run_suite", run_suite)


# The native Metal probe with its buffers moved from slots 0 and 1, as a host binding the native probe's buffers
# would not find them.
REBOUND = NATIVE.replace("[[buffer(0)]]", "[[buffer(2)]]").replace(
    "[[buffer(1)]]", "[[buffer(3)]]"
)


def constants_given(monkeypatch):
    """Collect the constants block of every case the fake harness runs, by case id."""
    given = {}
    original = harness.run_suite

    def run_suite(cases, *args, **kwargs):
        given.update((case.id, case.constants) for case in cases)
        return original(cases, *args, **kwargs)

    monkeypatch.setattr(harness, "run_suite", run_suite)
    return given


def hlsl_result(translated=None):
    feature = Feature("hlsl", "function", "sin", SIN.signature, crossgl="sin")
    native = probes.build(feature, "hlsl").source
    return Result(
        feature, "output", "supported", sources=(native, translated or native)
    )


# A call that takes no argument, whose probe declares an input it never reads.
CALLS = {
    "hlsl": ("WaveGetLaneIndex", Type("uint")),
    "slang": ("WaveGetLaneIndex", Type("uint")),
    "wgsl": ("workgroupBarrier", VOID),
    "rust": ("workgroup_memory_barrier_with_group_sync", VOID),
}
