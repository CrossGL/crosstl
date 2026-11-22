"""Compute probes that run although their CrossGL probe calls a counterpart overload or their construct reads nothing."""

import functools

import pytest

from tools.coverage import execution, features, probes, runner, validators

from .execution_cases import metal_runs


@functools.cache
def _metal(name: str):
    [feature] = [f for f in features.load("metal", offline=True) if f.name == name]
    return feature


def _supported_everywhere(monkeypatch):
    monkeypatch.setattr(validators, "available", lambda language, target="": True)
    monkeypatch.setattr(
        runner, "_native_rejection", lambda feature, language: (feature, "")
    )
    monkeypatch.setattr(runner, "translate", lambda *args: "translated metal")
    monkeypatch.setattr(
        runner, "_validated", lambda *args, **kwargs: validators.Verdict(True)
    )


def test_a_counterpart_probe_records_the_elements_its_call_reads():
    feature = _metal("atomic_fetch_add_explicit")
    assert feature.counterpart is not None
    native = probes.build(feature, "metal")
    crossgl = probes.build(feature, "crossgl")
    # The memory order is a constant, so both calls read one int and update the output buffer atomically.
    assert native.reads == crossgl.reads
    assert native.reads[0] == "int" and native.reads[-1].startswith("-> atomic ")


def test_a_counterpart_probe_that_stores_another_result_reads_differently():
    # simd_ballot stores a simd_vote as one word, where CrossGL's subgroupBallot stores a uvec4.
    feature = _metal("simd_ballot")
    assert (
        probes.build(feature, "metal").reads != probes.build(feature, "crossgl").reads
    )


def test_the_output_row_runs_a_counterpart_probe_only_when_it_reads_alike(
    monkeypatch,
):
    _supported_everywhere(monkeypatch)
    added = runner.run_output(_metal("atomic_fetch_add_explicit"), "metal")
    assert added.outcome == "supported"
    assert added.sources == (
        probes.build(_metal("atomic_fetch_add_explicit"), "metal").source,
        "translated metal",
    )
    assert runner.run_output(_metal("simd_ballot"), "metal").sources == ()


@pytest.mark.parametrize(
    "name", ["local-invocation-id", "subgroup-size", "workgroup-size"]
)
def test_a_compute_construct_that_reads_nothing_declares_an_unread_input(name):
    feature = _metal(name)
    for language in ("metal", "crossgl"):
        source = probes.build(feature, language).source
        assert execution.input_buffers(source) == ("fin",), language


def test_a_construct_in_a_language_the_harness_cannot_run_declares_no_input():
    [feature] = [
        f
        for f in features.load("cuda", offline=True)
        if f.name == "local-invocation-id"
    ]
    assert not execution.runs("cuda")
    assert execution.input_buffers(probes.build(feature, "cuda").source) == ()


@pytest.mark.skipif(
    not metal_runs() or not validators.available("metal"), reason="no Metal runtime"
)
def test_metal_runs_counterpart_and_construct_probes_with_a_verdict():
    results = [
        runner.run_output(_metal(name), "metal")
        for name in ("atomic_fetch_add_explicit", "local-invocation-id")
    ]
    assert all(execution.executable(result) for result in results)
    ran, used = execution.run(results, "metal-apple")
    assert used == ["metal-apple"]
    assert [(r.outcome, r.runtime, r.unrun) for r in ran] == [
        ("supported", "metal-apple", "")
    ] * 2
