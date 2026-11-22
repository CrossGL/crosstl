"""The CrossGL probes of Metal's ballots that return a simd_vote or quad_vote mask."""

import pytest

from tools.coverage import probes, validators
from tools.coverage.features import metal

VOTES = """metal_simdgroup simd_active_threads_mask simd_vote()
metal_simdgroup simd_ballot simd_vote(bool)
metal_quadgroup quad_active_threads_mask quad_vote()
metal_quadgroup quad_ballot quad_vote(bool)
"""


def votes():
    return {feature.name: feature for feature in metal.parse(VOTES)}


@pytest.mark.parametrize(
    "name, stored",
    [
        (
            "simd_active_threads_mask",
            "uint64_t r = (uint64(WaveActiveBallot(true).x)"
            " | (uint64(WaveActiveBallot(true).y) << 32));",
        ),
        (
            "quad_active_threads_mask",
            "uint r = ((WaveActiveBallot(true)[WaveGetLaneIndex() >> 5u]"
            " >> (WaveGetLaneIndex() & 28u)) & 0xFu);",
        ),
        (
            "quad_ballot",
            "uint r = ((WaveActiveBallot(a0)[WaveGetLaneIndex() >> 5u]"
            " >> (WaveGetLaneIndex() & 28u)) & 0xFu);",
        ),
    ],
)
def test_crossgl_reads_a_vote_mask_as_its_vote_t_integer(name, stored):
    feature = votes()[name]
    # The CrossGL probe reads the inputs the Metal probe reads, so the two can run on the same inputs.
    assert feature.counterpart is None
    source = probes.build(feature, "crossgl").source
    assert stored in source
    native = probes.build(feature, "metal").source
    assert "static_cast<" in native and "::vote_t>(" in native


def test_simd_ballot_keeps_its_uvec4_counterpart():
    feature = votes()["simd_ballot"]
    assert feature.counterpart is not None
    assert "uvec4 r = subgroupBallot(a0);" in probes.build(feature, "crossgl").source


@pytest.mark.skipif(
    not validators.available("metal"), reason="the Metal toolchain is not installed"
)
@pytest.mark.parametrize(
    "name", ["simd_active_threads_mask", "quad_active_threads_mask", "quad_ballot"]
)
def test_vote_mask_probe_translates_to_valid_metal(tmp_path, name):
    import crosstl

    path = tmp_path / "probe.cgl"
    path.write_text(probes.build(votes()[name], "crossgl").source)
    verdict = validators.validate(
        "metal", crosstl.translate(str(path), backend="metal")
    )
    assert verdict.accepted, verdict.log
