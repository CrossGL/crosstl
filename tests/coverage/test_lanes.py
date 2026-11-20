"""Which probes run as one workgroup of invocations, and which leave some invocations inactive."""

import pytest

from tests.conformance import runner as harness
from tools.coverage import execution, lanes, probes
from tools.coverage.model import VOID, Feature, Param, Result, Signature, Type

from .execution_cases import metal_runs

FLOAT = Type("float")
UINT = Type("uint")
UNARY = Signature(FLOAT, (Param(FLOAT),))
LANE = Signature(FLOAT, (Param(FLOAT), Param(UINT)))


def feature(language, name, signature=UNARY, crossgl="", group="", stage="compute"):
    return Feature(
        language, "function", name, signature, group=group, crossgl=crossgl, stage=stage
    )


@pytest.mark.parametrize(
    "language, name, crossgl",
    [
        ("hlsl", "WaveActiveSum", "WaveActiveSum"),
        ("glsl", "subgroupInclusiveAdd", "subgroupInclusiveAdd"),
        ("glsl", "subgroupBroadcastFirst", "subgroupBroadcastFirst"),
        ("metal", "simd_sum", "WaveActiveSum"),
        ("metal", "quad_broadcast_first", ""),
        ("rust", "subgroup_f_add", "WaveActiveSum"),
        ("opencl", "sub_group_non_uniform_reduce_add", ""),
        ("hip", "__ballot", ""),
        ("spirv", "OpGroupNonUniformFAdd", "subgroupAdd"),
    ],
)
def test_reductions_votes_and_first_lane_reads_skip_every_third_invocation(
    language, name, crossgl
):
    operation = feature(language, name, crossgl=crossgl)
    assert lanes.cooperative(operation) and lanes.divergent(operation)


@pytest.mark.parametrize(
    "language, name, crossgl",
    [
        ("hlsl", "WaveReadLaneAt", "WaveReadLaneAt"),
        ("hlsl", "QuadReadAcrossX", "QuadReadAcrossX"),
        ("glsl", "subgroupShuffleUp", "subgroupShuffleUp"),
        ("glsl", "subgroupBroadcast", "subgroupBroadcast"),
        ("metal", "simd_shuffle_rotate_down", ""),
        ("wgsl", "quadSwapX", "QuadReadAcrossX"),
        ("cuda", "__shfl_down_sync", ""),
        ("hip", "__hip_ds_bpermute", ""),
        ("slang", "WaveRotate", "WaveReadLaneAt({0}, ({1}) % WaveGetLaneCount())"),
    ],
)
def test_reads_of_a_named_lane_keep_every_invocation_active(language, name, crossgl):
    operation = feature(language, name, LANE, crossgl=crossgl)
    assert lanes.cooperative(operation) and not lanes.divergent(operation)


def test_operations_every_invocation_must_call_keep_every_invocation_active():
    reduce = feature("opencl", "sub_group_reduce_add", group="cl_khr_subgroups")
    assert lanes.cooperative(reduce) and not lanes.divergent(reduce)


@pytest.mark.parametrize(
    "operation",
    [
        feature("glsl", "sin", crossgl="sin"),
        # GLSL's all() and any() reduce a vector, not a subgroup; HIP's __all and __any are warp votes.
        feature("glsl", "all", Signature(Type("bool"), (Param(Type("bool", (3,))),))),
        feature("metal", "simdgroup_multiply", crossgl=""),
        # A barrier returns nothing to compare.
        feature("glsl", "subgroupBarrier", Signature(VOID, ()), "subgroupBarrier"),
        feature("hlsl", "WaveActiveSum", crossgl="WaveActiveSum", stage="fragment"),
    ],
)
def test_other_features_run_one_invocation_per_workgroup(operation):
    assert not lanes.cooperative(operation)


def test_hip_warp_votes_are_subgroup_operations():
    vote = Signature(Type("int"), (Param(Type("int")),))
    assert lanes.cooperative(feature("hip", "__all", vote))
    assert lanes.cooperative(feature("cuda", "__any_sync", vote))


def test_subgroup_probes_declare_a_workgroup_and_skip_every_third_invocation():
    add = feature("glsl", "subgroupAdd", crossgl="subgroupAdd")
    glsl = probes.build(add, "glsl").source
    assert "layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;" in glsl
    assert "    if (i % 3u != 0u) {\n        float a0 = fin[i + 0u];" in glsl
    crossgl = probes.build(add, "crossgl").source
    assert "layout(local_size_x = 64," in crossgl and "if (i % 3u != 0u) {" in crossgl
    # The baseline makes no call but skips the same invocations, so the branch is not part of the call's id.
    assert "if (i % 3u != 0u) {" in probes.build(add, "crossgl", baseline=True).source
    wave = feature("hlsl", "WaveActiveSum", crossgl="WaveActiveSum")
    assert "[numthreads(64, 1, 1)]" in probes.build(wave, "hlsl").source
    slang = probes.build(
        feature("slang", "WaveActiveSum", crossgl="WaveActiveSum"), "slang"
    )
    assert "[numthreads(64, 1, 1)]" in slang.source
    wgsl = probes.build(feature("wgsl", "subgroupAdd", crossgl="WaveActiveSum"), "wgsl")
    assert "@workgroup_size(64, 1, 1)" in wgsl.source
    assert "if (i % 3u != 0u) {" in wgsl.source
    rust = probes.build(
        feature("rust", "subgroup_f_add", crossgl="WaveActiveSum"), "rust"
    )
    assert "#[spirv(compute(threads(64)))]" in rust.source
    assert "    if i % 3 != 0 {\n" in rust.source


def test_a_shuffle_probe_runs_every_invocation_and_other_probes_run_alone():
    shuffle = probes.build(
        feature("glsl", "subgroupShuffle", LANE, "subgroupShuffle"), "glsl"
    ).source
    assert "local_size_x = 64," in shuffle and "% 3u" not in shuffle
    sine = probes.build(feature("glsl", "sin", crossgl="sin"), "glsl").source
    assert "local_size_x = 1," in sine and "% 3u" not in sine
    assert (
        "[numthreads(1, 1, 1)]" in probes.build(feature("hlsl", "sin"), "hlsl").source
    )


def test_the_harness_runs_a_subgroup_probe_as_one_workgroup(monkeypatch, tmp_path):
    groups = {}

    def run_suite(cases, *args, **kwargs):
        groups.update((case.id, case.group) for case in cases)
        return []

    monkeypatch.setattr(harness, "run_suite", run_suite)
    for name in ("simd_sum", "sin"):
        operation = feature("metal", name)
        native = probes.build(operation, "metal").source
        result = Result(operation, "output", "supported", sources=(native, native))
        execution._execute(
            "metal", [(0, result)], {"metal": []}, tmp_path / name, False
        )
        assert set(groups.values()) == {execution.THREADS if name != "sin" else 1}
        groups.clear()


@pytest.mark.skipif(not metal_runs(), reason="no Metal runtime")
def test_a_single_lane_stand_in_for_a_reduction_is_wrong():
    operation = feature("metal", "simd_sum", crossgl="WaveActiveSum")
    native = probes.build(operation, "metal").source
    assert "simd_sum(a0)" in native
    stand_in = native.replace("simd_sum(a0)", "a0")
    # A reduction over every lane, active or not, is as wrong as one over a single lane.
    every_lane = native.replace(
        "if (i % 3u != 0u) {", "float s = simd_sum(fin[i]);\n    if (i % 3u != 0u) {"
    ).replace("simd_sum(a0)", "s")
    same, single, every = (
        Result(operation, "output", "supported", sources=(native, translated))
        for translated in (native, stand_in, every_lane)
    )
    (same, single, every), _ = execution.run([same, single, every])
    assert same.outcome == "supported" and same.runtime == "metal-apple"
    assert single.outcome == "wrong" and execution.DIVERGED in single.detail
    assert every.outcome == "wrong" and execution.DIVERGED in every.detail
