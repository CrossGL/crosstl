"""Compute probes that run although their CrossGL probe calls a counterpart overload or their construct reads nothing."""

import pytest

from tools.coverage import execution, features, probes, runner, validators
from tools.coverage.model import Feature, Param, Signature, Type
from tools.coverage.upstream import UpstreamError

from .execution_cases import metal_runs

INT = Type("int")
BOOL = Type("bool")
ADD = Signature(INT, (Param(INT, "inout"), Param(INT, "in")))
# The pinned features these tests use, written out so that the tests need no feature sources; the last test checks
# them against the pinned lists where those are cached.
FEATURES = {
    ("metal", "atomic_fetch_add_explicit"): Feature(
        "metal",
        "function",
        "atomic_fetch_add_explicit",
        Signature(INT, (*ADD.params, Param(Type("", opaque="memory_order"), "const"))),
        group="atomic",
        crossgl="atomicAdd",
        stage="compute",
        counterpart=ADD,
    ),
    ("metal", "simd_ballot"): Feature(
        "metal",
        "function",
        "simd_ballot",
        Signature(Type("", opaque="simd_vote"), (Param(BOOL, "in"),)),
        group="simdgroup",
        crossgl="subgroupBallot",
        stage="compute",
        counterpart=Signature(Type("uint", (4,)), (Param(BOOL, "in"),)),
    ),
    **{
        (language, name): Feature(
            language, "compute", name, None, group=f"BuiltIn.{group}", stage="compute"
        )
        for language, name, group in (
            ("metal", "local-invocation-id", "LocalInvocationId"),
            ("metal", "subgroup-size", "SubgroupSize"),
            ("metal", "workgroup-size", "WorkgroupSize"),
            ("cuda", "local-invocation-id", "LocalInvocationId"),
        )
    },
}


def _metal(name: str) -> Feature:
    return FEATURES["metal", name]


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
    feature = FEATURES["cuda", "local-invocation-id"]
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


@pytest.mark.parametrize("language", sorted({language for language, _ in FEATURES}))
def test_the_features_match_the_pinned_lists(language):
    try:
        pinned = {f.name: f for f in features.load(language, offline=True)}
    except UpstreamError:
        pytest.skip(f"the {language} feature sources are not cached")
    for (owner, name), feature in FEATURES.items():
        if owner == language:
            assert pinned[name] == feature, name
