"""Multi-wave index-value reductions break ties by the lower index on each target.

A 416-invocation workgroup with an exact subgroup width of 32 has 13
subgroups. Each invocation holds one ``{index, value}`` pair. A subgroup
reduces its pairs with ``simd_shuffle_down`` under one total order (the larger
value, then the lower index), lane zero publishes the subgroup's pair to
workgroup memory, and the first subgroup reduces the published pairs again.
The largest value occurs in several subgroups and lanes, so only the index
order decides the result. Lanes past the published count keep their own
partial, as an arg reduction's do, and the order still makes them harmless.

Every invocation records its subgroup identity, lane, suffix reduction,
published pair and, for invocation zero, the final index, so a wrong value
move, a lost or torn workgroup store or a wrong comparison is attributable
from the readback alone. The pairs carry ``uint``, ``float`` and ``half``
values; a ``half`` member is narrower than the ``uint`` beside it in workgroup
memory.

The DirectX translation runs on hardware waves of exactly 32 lanes
(``[WaveSize(32)]``) and lowers ``simd_shuffle_down`` to ``WaveReadLaneAt``
with an in-range per-lane index, which DXC itself lowers to a SPIR-V
``OpGroupNonUniformShuffle``. No WARP release executes it: the system
library rejects the pipeline, 1.0.13 writes nothing, 1.0.14 to 1.0.18 remove
the device during the dispatch, and 1.0.19 and 1.0.21 return, for lanes 0 to 15 only, pairs of lanes below the
reader, which no shuffle down can reach, while lanes 16 to 31 stay exact. The
executions on a Direct3D device therefore remain opt-in, and the Metal host
requires them.
"""

import json
import struct
import subprocess

import pytest

from tests.test_translator.test_directx_float_atomics import _compile
from tests.test_translator.test_directx_multiwave_reduction import (
    metal_reduction_runner as _metal_reduction_runner,
)
from tests.test_translator.test_directx_software_reductions import (
    _execute_words,
)
from tests.test_translator.test_directx_software_reductions import (
    directx_runtime as _directx_runtime,
)
from tests.test_translator.test_metal_builtin_ownership import (
    _compile as _compile_metal,
)

directx_runtime = _directx_runtime
metal_reduction_runner = _metal_reduction_runner
METAL_REQUIRE_ENV = "CROSTL_REQUIRE_METAL_MULTIWAVE_REDUCTION"
SOURCE_NAME = "kernels/index_ties.metal"
GROUP_SIZE = 416
WIDTH = 32
WAVES = GROUP_SIZE // WIDTH
GROUPS = 3
RECORD = 7
SENTINEL = 0xFFFFFFFF
REPEATS = 8
VALUE_TYPES = ("uint", "float", "half")

SOURCE = f"""#include <metal_stdlib>
using namespace metal;

template <typename T>
struct IndexedValue {{
  uint index;
  T value;
}};

inline uint decode_value(uint word, uint) {{
  return word;
}}
inline float decode_value(uint word, float) {{
  return as_type<float>(word);
}}
inline half decode_value(uint word, half) {{
  return as_type<half>(ushort(word));
}}
inline uint encode_value(uint value) {{
  return value;
}}
inline uint encode_value(float value) {{
  return as_type<uint>(value);
}}
inline uint encode_value(half value) {{
  return uint(as_type<ushort>(value));
}}

template <typename T>
IndexedValue<T> shuffle_pair_down(IndexedValue<T> item, ushort delta) {{
  return IndexedValue<T>{{
      simd_shuffle_down(item.index, delta), simd_shuffle_down(item.value, delta)}};
}}

template <typename T>
IndexedValue<T> prefer(IndexedValue<T> best, IndexedValue<T> other) {{
  if (best.value < other.value ||
      (best.value == other.value && best.index > other.index)) {{
    return other;
  }}
  return best;
}}

template <typename T, uint GROUP_SIZE>
[[kernel]] void index_ties(
    const device uint* inputWords [[buffer(0)]],
    device uint* outputWords [[buffer(1)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint group [[threadgroup_position_in_grid]]) {{
  threadgroup IndexedValue<T> partials[32];
  uint index = group * GROUP_SIZE + tid;
  IndexedValue<T> best{{tid, decode_value(inputWords[index], T(0))}};
  for (uint offset = simd_size / 2; offset > 0; offset /= 2) {{
    best = prefer(best, shuffle_pair_down(best, ushort(offset)));
  }}
  if (lane == 0u) {{
    partials[simd_id] = best;
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint record = index * {RECORD}u;
  outputWords[record] = simd_id;
  outputWords[record + 1u] = lane;
  outputWords[record + 2u] = best.index;
  outputWords[record + 3u] = encode_value(best.value);
  outputWords[record + 4u] = 0xFFFFFFFFu;
  outputWords[record + 5u] = 0xFFFFFFFFu;
  outputWords[record + 6u] = 0xFFFFFFFFu;
  if (simd_id != 0u) {{
    return;
  }}
  uint waves = (GROUP_SIZE + simd_size - 1u) / simd_size;
  if (lane < waves) {{
    best = partials[lane];
    outputWords[record + 4u] = best.index;
    outputWords[record + 5u] = encode_value(best.value);
  }}
  for (uint offset = simd_size / 2; offset > 0; offset /= 2) {{
    best = prefer(best, shuffle_pair_down(best, ushort(offset)));
  }}
  if (tid == 0u) {{
    outputWords[record + 6u] = best.index;
  }}
}}

#define instantiate_index_ties(name, type)                       \\
  template [[host_name("index_ties_" #name)]] [[kernel]]         \\
  void index_ties<type, {GROUP_SIZE}>(                            \\
      const device uint* inputWords [[buffer(0)]],               \\
      device uint* outputWords [[buffer(1)]],                    \\
      uint tid [[thread_position_in_threadgroup]],               \\
      uint lane [[thread_index_in_simdgroup]],                   \\
      uint simd_id [[simdgroup_index_in_threadgroup]],           \\
      uint simd_size [[threads_per_simdgroup]],                  \\
      uint group [[threadgroup_position_in_grid]]);

instantiate_index_ties(uint, uint)
instantiate_index_ties(float, float)
instantiate_index_ties(half, half)
"""


def _value(group, tid):
    """Small integers, exact in every value type; the maximum, 22, first
    occurs past the first two subgroups and then in most of the others."""

    return (tid * 37 + group * 11) % (23 if tid >= 2 * WIDTH else 19)


def _word(value, value_type):
    if value_type == "uint":
        return value
    if value_type == "float":
        return struct.unpack("<I", struct.pack("<f", float(value)))[0]
    return struct.unpack("<H", struct.pack("<e", float(value)))[0]


def _inputs(value_type):
    return [
        _word(_value(group, tid), value_type)
        for group in range(GROUPS)
        for tid in range(GROUP_SIZE)
    ]


def _best(pairs):
    """The pair with the largest value, then the lowest index."""

    return min(pairs, key=lambda pair: (-pair[1], pair[0]))


def _suffixes(group):
    """Each invocation's pair after the shuffle-down reduction of its subgroup.

    A lane whose source lane is past the subgroup keeps its own pair, so lane
    ``k`` ends with the reduction of lanes ``k`` to 31.
    """

    pairs = [(tid, _value(group, tid)) for tid in range(GROUP_SIZE)]
    return [_best(pairs[tid : (tid // WIDTH + 1) * WIDTH]) for tid in range(GROUP_SIZE)]


def _reference(value_type):
    """Metal reference: subgroup ``k`` of every workgroup is ``tid / 32 == k``."""

    words = []
    for group in range(GROUPS):
        suffixes = _suffixes(group)
        final = _best(suffixes[wave * WIDTH] for wave in range(WAVES))
        for tid, (index, value) in enumerate(suffixes):
            wave, lane = divmod(tid, WIDTH)
            published = [SENTINEL, SENTINEL]
            if tid < WAVES:
                pair = suffixes[tid * WIDTH]
                published = [pair[0], _word(pair[1], value_type)]
            words.extend(
                [
                    wave,
                    lane,
                    index,
                    _word(value, value_type),
                    *published,
                    final[0] if tid == 0 else SENTINEL,
                ]
            )
    return words


def _differences(actual, value_type, *, ordered_subgroups):
    """Return every invariant the readback violates.

    Metal numbers subgroups in invocation order. The DirectX translation
    numbers physical waves in arrival order, so there the identities of a
    workgroup must be a permutation of ``0..12`` that is uniform within each
    wave, and lane ``k`` of the first subgroup publishes the pair of the wave
    whose identity is ``k``. Lanes and wave membership follow invocation
    order on both, and the final index never depends on the numbering.
    """

    differences = []
    assert len(actual) == GROUPS * GROUP_SIZE * RECORD
    reference = _reference(value_type)
    for group in range(GROUPS):
        base = group * GROUP_SIZE * RECORD
        records = [actual[base + tid * RECORD :][:RECORD] for tid in range(GROUP_SIZE)]
        expected_records = [
            reference[base + tid * RECORD :][:RECORD] for tid in range(GROUP_SIZE)
        ]
        identities = [records[wave * WIDTH][0] for wave in range(WAVES)]
        if sorted(identities) != list(range(WAVES)):
            differences.append({"group": group, "subgroupIdentities": identities})
            continue
        if ordered_subgroups and identities != list(range(WAVES)):
            differences.append({"group": group, "subgroupIdentities": identities})
            continue
        wave_by_identity = {identity: wave for wave, identity in enumerate(identities)}
        first = wave_by_identity[0]
        for tid, (record, wanted) in enumerate(zip(records, expected_records)):
            wave, lane = divmod(tid, WIDTH)
            expected = [identities[wave], *wanted[1:4], SENTINEL, SENTINEL, SENTINEL]
            if wave == first and lane < WAVES:
                published = expected_records[wave_by_identity[lane] * WIDTH]
                expected[4:6] = published[2:4]
            if wave == first and lane == 0:
                expected[6] = expected_records[0][6]
            if record != expected:
                differences.append(
                    {"group": group, "tid": tid, "expected": expected, "actual": record}
                )
    return differences


def _translate(tmp_path, target, value_type):
    from crosstl.project import ProjectConfig, translate_project

    repo = tmp_path / "repo"
    source = repo / SOURCE_NAME
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(SOURCE, encoding="utf-8")
    entry = f"index_ties_{value_type}"
    rules = {}
    if target != "metal":
        rules["subgroup_width_rules"] = {SOURCE_NAME: WIDTH}
    if target == "directx":
        # The reference keeps an upper lane's own pair, as simd_shuffle_down
        # defines; HLSL leaves an out-of-range WaveReadLaneAt undefined.
        rules["source_options"] = {
            "metal": {
                "target_options": {
                    "directx": {"relative_wave_shuffle_out_of_range": "self"}
                }
            }
        }
    report = translate_project(
        ProjectConfig(
            root=repo,
            include_patterns=(SOURCE_NAME,),
            targets=(target,),
            output_dir="out",
            entry_points={SOURCE_NAME: (entry,)},
            entry_workgroup_size_rules={SOURCE_NAME: {entry: [GROUP_SIZE, 1, 1]}},
            **rules,
        ),
        format_output=False,
    )
    payload = report.to_json()
    assert payload["diagnostics"] == []
    assert payload["summary"]["translatedCount"] == 1
    (artifact,) = payload["artifacts"]
    return (repo / artifact["path"]).read_text(encoding="utf-8"), artifact


def test_reference_breaks_ties_by_the_lower_index():
    reference = _reference("half")
    assert _differences(reference, "half", ordered_subgroups=True) == []
    for group in range(GROUPS):
        values = [_value(group, tid) for tid in range(GROUP_SIZE)]
        maxima = [tid for tid, value in enumerate(values) if value == max(values)]
        # The maximum occurs in several subgroups, none of them the first two.
        assert len({tid // WIDTH for tid in maxima}) > 2 and maxima[0] >= 2 * WIDTH
        assert reference[group * GROUP_SIZE * RECORD + 6] == maxima[0]
    torn = list(reference)
    # A published value that lost its bits loses the tie to a later subgroup.
    torn[2 * RECORD + 5] = 0
    assert _differences(torn, "half", ordered_subgroups=True) == [
        {
            "group": 0,
            "tid": 2,
            "expected": reference[2 * RECORD :][:RECORD],
            "actual": torn[2 * RECORD :][:RECORD],
        }
    ]


@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_index_ties_translate_to_exact_width_hlsl(tmp_path, value_type):
    generated, artifact = _translate(tmp_path, "directx", value_type)
    assert artifact["entryPoint"]["target"] == "CSMain"
    assert "[numthreads(416, 1, 1)]" in generated
    assert "[WaveSize(32)]" in generated
    assert "simd_id = crossglPhysicalSubgroupID;" in generated
    assert "groupshared IndexedValue_" in generated
    assert "__crossgl_wave_shuffle_down_self_" in generated
    assert "(WaveGetLaneIndex() + uint(" not in generated
    _compile(generated, tmp_path, profile="cs_6_6")


@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_index_ties_execute_repeatedly(tmp_path, directx_runtime, value_type):
    generated, _ = _translate(tmp_path, "directx", value_type)
    artifact, module = _compile(generated, tmp_path, profile="cs_6_6")
    words = _inputs(value_type)
    expected = _reference(value_type)
    failures = {}
    for repeat in range(REPEATS):
        work = tmp_path / f"dispatch-{repeat}"
        work.mkdir()
        actual = _execute_words(
            work,
            generated,
            words,
            expected,
            value_type="uint",
            workgroup_size=(GROUP_SIZE, 1, 1),
            workgroup_count=(GROUPS, 1, 1),
            module_compiler=lambda *_args, **_kwargs: (artifact, module),
        )
        differences = _differences(actual, value_type, ordered_subgroups=False)
        if differences:
            failures[repeat] = {"count": len(differences), "first": differences[:8]}
    (tmp_path / "failures.json").write_text(json.dumps(failures, indent=2))
    assert not failures, failures


@pytest.mark.parametrize("value_type", VALUE_TYPES)
@pytest.mark.parametrize("variant", ["original", "translated"])
def test_metal_index_ties_match_reference(
    tmp_path, metal_reduction_runner, variant, value_type
):
    entry = f"index_ties_{value_type}"
    if variant == "original":
        source = SOURCE
    else:
        source, artifact = _translate(tmp_path, "metal", value_type)
        assert artifact["entryPoint"]["target"] == entry
    _, module = _compile_metal(source, "metal", tmp_path)
    guard = [0x6A15BEEF] * WIDTH
    expected = _reference(value_type)
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "inputs": [_inputs(value_type) + guard],
                "outputCount": len(expected) + len(guard),
                "threadCount": GROUPS,
                "threadsPerThreadgroup": GROUP_SIZE,
            }
        ),
        encoding="utf-8",
    )
    failures = {}
    for repeat in range(REPEATS):
        result = subprocess.run(
            [str(metal_reduction_runner), str(module), entry, str(request)],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        data = json.loads(result.stdout)
        assert data["threadExecutionWidth"] == WIDTH, data["device"]
        actual = data["values"]
        assert actual[len(expected) :] == [0xDEADBEEF] * len(guard)
        if actual[: len(expected)] != expected:
            differences = _differences(
                actual[: len(expected)], value_type, ordered_subgroups=True
            )
            failures[repeat] = {"count": len(differences), "first": differences[:8]}
    assert not failures, failures


def test_index_ties_are_required_with_the_multiwave_controls():
    from pathlib import Path

    from tests import workflow_text

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text(encoding="utf-8")
    text = workflow_text.workflow_job_step_section(
        workflow, "metal-host", "Validate multi-wave workgroup reductions"
    )
    assert "tests/test_translator/test_multiwave_index_ties.py" in text
    assert "if:" not in text and "continue-on-error" not in text
