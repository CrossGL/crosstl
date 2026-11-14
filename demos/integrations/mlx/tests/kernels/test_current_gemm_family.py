"""Execute every real-typed pinned ``steel_gemm_fused`` entry natively.

``steel_gemm_fused.metal`` instantiates four element types, six tile shapes
and four transpositions. This gate translates each float32, float16 and
bfloat16 entry with three frozen function-constant configurations (unaligned,
aligned, and broadcast batches with ``alpha * AB + beta * C``) and compares
every output word, guards included, with an exact reference; on Metal it also
compares with the original source. ``test_current_gemm.py`` runs all six
configurations for one representative entry. complex64 entries are not part
of this gate; #2261 tracks their partial-specialization translation.
"""

import os
from collections import Counter

import pytest

from demos.integrations.mlx.tests.kernels import gemm_fused_model as model
from demos.integrations.mlx.tests.kernels.test_current_gemm import (
    GUARDS,
    _aligned_cases,
    _exact,
    _frozen_values,
    _loader_access_assertions,
    _opengl_index_assertions,
    _variant_data,
    execute_entry,
    prepare_gemm_source,
)

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_GEMM_FAMILY"
SHARD_INDEX_ENV = "CROSTL_MLX_CURRENT_GEMM_FAMILY_SHARD_INDEX"
SHARD_COUNT_ENV = "CROSTL_MLX_CURRENT_GEMM_FAMILY_SHARD_COUNT"
# Five is coprime with the three variants and the four transpositions, so
# every shard mixes all of them.
CI_SHARD_COUNT = 5
REAL_TYPES = ("float16", "bfloat16", "float32")
ENTRIES = tuple(entry for entry in model.entries() if entry.dtype in REAL_TYPES)
CASES = {
    # Partial tiles in M, N and K, batches with padding and spare threadgroups,
    # and several tiles along every axis for every shape.
    "plain": ((4, 5, 7, 1, 0, 0), (33, 35, 17, 2, 3, 1), (70, 69, 41, 1, 1, 0)),
    "broadcast-axpby": ((33, 35, 17, 4, 3, 1),),
}
VARIANTS = ("plain", "aligned", "broadcast-axpby")
SCOPE = (
    "Every float32, float16 and bfloat16 steel_gemm_fused entry with three "
    "frozen constant configurations; complex64 entries and the upstream "
    "suite are not covered."
)


def _cases(entry, variant):
    return _aligned_cases(entry) if variant == "aligned" else CASES[variant]


PAIRS = tuple((entry, variant) for entry in ENTRIES for variant in VARIANTS)


def _selected_pairs():
    index, count = os.environ.get(SHARD_INDEX_ENV), os.environ.get(SHARD_COUNT_ENV)
    if index is None and count is None:
        return PAIRS
    if index is None or count is None:
        raise RuntimeError(f"{SHARD_INDEX_ENV} and {SHARD_COUNT_ENV} go together")
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise RuntimeError(f"GEMM family shard {index} is outside [0, {count})")
    return PAIRS[index::count]


SELECTED_PAIRS = _selected_pairs()


def _pair_id(pair):
    entry, variant = pair
    return f"{entry.name}-{variant}"


def test_gemm_family_names_every_real_entry():
    assert len(ENTRIES) == 3 * 6 * 4
    assert len({entry.name for entry in ENTRIES}) == len(ENTRIES)
    assert Counter(entry.dtype for entry in ENTRIES) == {
        dtype: 24 for dtype in REAL_TYPES
    }
    assert Counter(entry.transpose for entry in ENTRIES) == {
        name: 18 for name, _, _ in model.TRANSPOSES
    }
    assert all(set(_frozen_values(variant)) for variant in VARIANTS)


def test_gemm_family_shards_partition_the_pairs(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_pairs())
    assert sorted(map(_pair_id, (p for shard in shards for p in shard))) == sorted(
        map(_pair_id, PAIRS)
    )
    assert {len(shard) for shard in shards} == {43, 44}
    for shard in shards:
        assert {variant for _, variant in shard} == set(VARIANTS)
        assert {entry.dtype for entry, _ in shard} == set(REAL_TYPES)
        assert {entry.transpose for entry, _ in shard} == {
            name for name, _, _ in model.TRANSPOSES
        }
    monkeypatch.setenv(SHARD_INDEX_ENV, str(CI_SHARD_COUNT))
    with pytest.raises(RuntimeError):
        _selected_pairs()
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_pairs()


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.name)
def test_gemm_family_assertions_follow_the_entry(entry):
    accesses = _loader_access_assertions(entry)
    assert [item["function"] for item in accesses] == [
        loader.function for loader in entry.loaders
    ]
    assert all(item["entryPoint"] == entry.name for item in accesses)
    ranges = _opengl_index_assertions(entry)
    reads = max(loader.reads for loader in entry.loaders)
    device = [item for item in ranges if item["maximum"] == 2**31 - 1]
    assert len(device) == reads + 3
    workgroup = [item for item in ranges if item["maximum"] != 2**31 - 1]
    assert [(item["indexed"], item["maximum"]) for item in workgroup] == [
        ("crosstl_workgroup_As", accesses[0]["maximum"]),
        ("crosstl_workgroup_Bs", accesses[1]["maximum"]),
    ]


@pytest.mark.parametrize("transpose", [name for name, _, _ in model.TRANSPOSES])
@pytest.mark.parametrize("dtype", REAL_TYPES)
def test_gemm_family_references_are_exact_and_guarded(dtype, transpose):
    """Every reference value is exact in the output type, and every slot no
    output element names keeps its guard, for each layout and variant."""
    entry = model.BY_NAME[
        f"steel_gemm_fused_{transpose}_{dtype}_{dtype}_bm64_bn32_bk8_wm4_wn1"
    ]
    guard = GUARDS[dtype]
    assert _exact(dtype, guard)
    for variant in VARIANTS:
        for case in _cases(entry, variant):
            m, n, k, batches, padding, _ = case
            resources, expected, geometry = _variant_data(case, variant, entry)
            params = resources["params"]
            assert geometry["workgroupSize"] == [32, 1, 4]
            assert geometry["workgroupCount"][:2] == [
                -(-n // 32) + case[5],
                -(-m // 64) + case[5],
            ]
            live = {
                batch * params["batch_stride_d"] + row * params["ldd"] + col
                for batch in range(batches)
                for row in range(m)
                for col in range(n)
            }
            assert all(
                (expected[index] == guard) == (index not in live)
                for index in range(len(expected))
            ), (variant, case)
            assert all(_exact(dtype, value) for value in expected)
            for name in ("A", "B", "C"):
                assert all(_exact(dtype, value) for value in resources[name])
            # A transposed operand stores K rows of M (or N rows of K) elements.
            assert params["lda"] == (m if entry.transpose_a else k) + padding
            assert params["ldb"] == (k if entry.transpose_b else n) + padding


def test_gemm_family_transposed_layouts_hold_the_same_matrix():
    case = (33, 35, 17, 2, 3, 1)
    products = {}
    for name, _, _ in model.TRANSPOSES:
        entry = model.BY_NAME[
            f"steel_gemm_fused_{name}_float32_float32_bm32_bn32_bk16_wm2_wn2"
        ]
        resources, expected, _ = _variant_data(case, "plain", entry)
        params = resources["params"]
        products[name] = [
            expected[batch * params["batch_stride_d"] + row * params["ldd"] + col]
            for batch in range(2)
            for row in range(33)
            for col in range(35)
        ]
    assert len({tuple(values) for values in products.values()}) == 1
    assert len(set(products["nn"])) > 16


@pytest.fixture(scope="module")
def gemm_family_source(tmp_path_factory):
    return prepare_gemm_source(tmp_path_factory, REQUIRE_ENV)


@pytest.mark.parametrize("pair", SELECTED_PAIRS, ids=_pair_id)
def test_current_gemm_family_executes(tmp_path, gemm_family_source, pair):
    entry, variant = pair
    execute_entry(
        tmp_path,
        gemm_family_source,
        entry,
        variant,
        _cases(entry, variant),
        SCOPE,
    )
