"""Multi-wave workgroup reductions keep every subgroup partial on each target.

A 512-invocation workgroup with an exact subgroup width of 32 has 16
subgroups. Each subgroup reduces its own values, lane zero publishes the
partial to workgroup memory, and the first 16 invocations reduce the
partials again. The kernel records the subgroup identity, lane, subgroup
partial, published partial and final reduction of every invocation, so a
wrong subgroup mapping, a missing producer, a barrier fault, an uninitialized
partial or a corrupt upload is attributable from the readback alone.
"""

import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from crosstl.project import ProjectConfig, translate_project
from tests.test_translator.test_directx_float_atomics import _compile
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
ROOT = Path(__file__).resolve().parents[2]
METAL_REQUIRE_ENV = "CROSTL_REQUIRE_METAL_MULTIWAVE_REDUCTION"
SOURCE_NAME = "kernels/multiwave.metal"
GROUP_SIZE = 512
WIDTH = 32
WAVES = GROUP_SIZE // WIDTH
GROUPS = 3
# The last workgroup ends in a tail: subgroup 13 reads 19 values and
# subgroups 14 and 15 read none.
COUNT = GROUPS * GROUP_SIZE - 77
RECORD = 5
SENTINEL = 0xFFFFFFFF
REPEATS = 8
VALUE_TYPES = ("uint", "float")

SOURCE = f"""#include <metal_stdlib>
using namespace metal;

template <typename T, uint GROUP_SIZE, uint WAVES>
[[kernel]] void multiwave_reduce(
    const device uint* inputWords [[buffer(0)]],
    device uint* outputWords [[buffer(1)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {{
  threadgroup T partials[WAVES];
  uint count = inputWords[0];
  uint index = group * GROUP_SIZE + tid;
  T value = index < count ? as_type<T>(inputWords[1u + index]) : T(0);
  T waveTotal = simd_sum(value);
  if (lane == 0u) {{
    partials[simd_id] = waveTotal;
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint observedWord = 0xFFFFFFFFu;
  uint totalWord = 0xFFFFFFFFu;
  if (tid < WAVES) {{
    T observed = partials[tid];
    T total = simd_sum(observed);
    observedWord = as_type<uint>(observed);
    totalWord = as_type<uint>(total);
  }}
  uint record = index * {RECORD}u;
  outputWords[record] = simd_id;
  outputWords[record + 1u] = lane;
  outputWords[record + 2u] = as_type<uint>(waveTotal);
  outputWords[record + 3u] = observedWord;
  outputWords[record + 4u] = totalWord;
}}

#define instantiate_multiwave(name, type)                      \\
  template [[host_name("multiwave_reduce_" #name)]] [[kernel]] \\
  void multiwave_reduce<type, {GROUP_SIZE}, {WAVES}>(          \\
      const device uint* inputWords [[buffer(0)]],             \\
      device uint* outputWords [[buffer(1)]],                  \\
      uint tid [[thread_position_in_threadgroup]],             \\
      uint lane [[thread_index_in_simdgroup]],                 \\
      uint simd_id [[simdgroup_index_in_threadgroup]],         \\
      uint group [[threadgroup_position_in_grid]]);

instantiate_multiwave(uint, uint)
instantiate_multiwave(float, float)
"""


def _float_word(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _values(value_type):
    """Distinct values per subgroup, lane and workgroup.

    Every float value and every partial or total is an integer below 2**24,
    so float reductions are exact in any association order.
    """

    values = []
    for index in range(COUNT):
        group, local = divmod(index, GROUP_SIZE)
        wave, lane = divmod(local, WIDTH)
        if value_type == "uint":
            values.append(((wave + 1) << 16) + lane * 3 + group + 1)
        else:
            values.append((wave + 1) * 64 + lane + group)
    return values


def _word(value, value_type):
    return value & SENTINEL if value_type == "uint" else _float_word(value)


def _inputs(value_type):
    return [COUNT] + [_word(value, value_type) for value in _values(value_type)]


def _wave_totals(value_type):
    values = _values(value_type) + [0] * (GROUPS * GROUP_SIZE - COUNT)
    return [
        [
            sum(values[base : base + WIDTH])
            for base in range(group * GROUP_SIZE, (group + 1) * GROUP_SIZE, WIDTH)
        ]
        for group in range(GROUPS)
    ]


def _reference(value_type):
    """Metal reference: subgroup ``k`` of every workgroup is ``tid / 32 == k``."""

    words = []
    for totals in _wave_totals(value_type):
        group_total = sum(totals)
        for tid in range(GROUP_SIZE):
            wave, lane = divmod(tid, WIDTH)
            published = tid < WAVES
            words.extend(
                [
                    wave,
                    lane,
                    _word(totals[wave], value_type),
                    _word(totals[tid], value_type) if published else SENTINEL,
                    _word(group_total, value_type) if published else SENTINEL,
                ]
            )
    return words


def _differences(actual, value_type, *, ordered_subgroups):
    """Return every invariant the readback violates.

    Metal numbers subgroups in invocation order. The DirectX translation
    numbers physical waves in arrival order, so there the subgroup identities
    of a workgroup must be a permutation of ``0..15`` that is uniform within
    each wave. Lanes and wave membership must still follow invocation order,
    because the source reduces invocations ``0..15`` as lanes of one subgroup.
    """

    differences = []
    assert len(actual) == GROUPS * GROUP_SIZE * RECORD
    for group, totals in enumerate(_wave_totals(value_type)):
        group_total = sum(totals)
        records = [
            actual[(group * GROUP_SIZE + tid) * RECORD :][:RECORD]
            for tid in range(GROUP_SIZE)
        ]
        identities = []
        for wave in range(WAVES):
            members = records[wave * WIDTH : (wave + 1) * WIDTH]
            identity = members[0][0]
            identities.append(identity)
            for lane, record in enumerate(members):
                tid = wave * WIDTH + lane
                expected = {
                    "subgroup": wave if ordered_subgroups else identity,
                    "lane": lane,
                    "subgroupTotal": _word(totals[wave], value_type),
                }
                observed = dict(zip(expected, record[:3]))
                if observed != expected:
                    differences.append(
                        {
                            "group": group,
                            "tid": tid,
                            "expected": expected,
                            "actual": observed,
                        }
                    )
        if sorted(identities) != list(range(WAVES)):
            differences.append({"group": group, "subgroupIdentities": identities})
            continue
        wave_by_identity = {identity: wave for wave, identity in enumerate(identities)}
        for tid, record in enumerate(records):
            if tid < WAVES:
                expected = [
                    _word(totals[wave_by_identity[tid]], value_type),
                    _word(group_total, value_type),
                ]
            else:
                expected = [SENTINEL, SENTINEL]
            if record[3:] != expected:
                differences.append(
                    {
                        "group": group,
                        "tid": tid,
                        "expectedPublishedAndTotal": expected,
                        "actualPublishedAndTotal": record[3:],
                    }
                )
    return differences


def _translate(tmp_path, target, value_type):
    repo = tmp_path / "repo"
    source = repo / SOURCE_NAME
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(SOURCE, encoding="utf-8")
    entry = f"multiwave_reduce_{value_type}"
    rules = {}
    if target != "metal":
        # Metal cannot enforce an exact width; the Metal controls check the
        # pipeline's execution width instead.
        rules["subgroup_width_rules"] = {SOURCE_NAME: WIDTH}
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
    report.write_json(tmp_path / "report.json")
    payload = report.to_json()
    assert payload["diagnostics"] == []
    assert payload["summary"]["translatedCount"] == 1
    (artifact,) = payload["artifacts"]
    return (repo / artifact["path"]).read_text(encoding="utf-8"), artifact


@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_multiwave_reduction_translates_to_exact_width_hlsl(tmp_path, value_type):
    generated, artifact = _translate(tmp_path, "directx", value_type)
    assert artifact["entryPoint"]["target"] == "CSMain"
    assert "[numthreads(512, 1, 1)]" in generated
    assert "[WaveSize(32)]" in generated
    assert "uint lane = WaveGetLaneIndex();" in generated
    assert "simd_id = crossglPhysicalSubgroupID;" in generated
    assert generated.count("WaveActiveSum(") == 2
    # Two barriers number the physical waves; one publishes the partials.
    assert generated.count("GroupMemoryBarrierWithGroupSync();") == 3
    assert f"groupshared {value_type} multiwave_reduce_{value_type}_partials[16];" in (
        generated
    )
    _compile(generated, tmp_path, profile="cs_6_6")


def test_reference_checker_rejects_lost_partials():
    reference = _reference("uint")
    assert _differences(reference, "uint", ordered_subgroups=True) == []
    corrupt = list(reference)
    corrupt[3 * RECORD + 3] = 0xD8E52060
    assert _differences(corrupt, "uint", ordered_subgroups=False) == [
        {
            "group": 0,
            "tid": 3,
            "expectedPublishedAndTotal": reference[3 * RECORD + 3 :][:2],
            "actualPublishedAndTotal": [0xD8E52060, reference[3 * RECORD + 4]],
        }
    ]
    swapped = list(reference)
    for lane in range(WIDTH):
        swapped[lane * RECORD] = 1
        swapped[(WIDTH + lane) * RECORD] = 0
    for tid in (0, 1):
        swapped[tid * RECORD + 3] = reference[(1 - tid) * RECORD + 3]
    assert _differences(swapped, "uint", ordered_subgroups=False) == []
    assert _differences(swapped, "uint", ordered_subgroups=True) != []


@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_multiwave_reduction_executes_repeatedly(tmp_path, directx_runtime, value_type):
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
            value_type=value_type,
            workgroup_size=(GROUP_SIZE, 1, 1),
            workgroup_count=(GROUPS, 1, 1),
            module_compiler=lambda *_args, **_kwargs: (artifact, module),
        )
        differences = _differences(actual, value_type, ordered_subgroups=False)
        if differences:
            failures[repeat] = {"count": len(differences), "first": differences[:8]}
    (tmp_path / "failures.json").write_text(json.dumps(failures, indent=2))
    assert not failures, failures


@pytest.fixture(scope="module")
def metal_reduction_runner(tmp_path_factory):
    if os.environ.get(METAL_REQUIRE_ENV) != "1":
        pytest.skip(f"set {METAL_REQUIRE_ENV}=1 to require native Metal execution")
    assert sys.platform == "darwin", "Native Metal execution requires macOS"
    runner = tmp_path_factory.mktemp("metal-reduction-runner") / "readback"
    subprocess.run(
        [
            "swiftc",
            str(
                ROOT / "tests/fixtures/runtime_verification/metal_uint32_buffers.swift"
            ),
            "-o",
            str(runner),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return runner


@pytest.mark.parametrize("value_type", VALUE_TYPES)
@pytest.mark.parametrize("variant", ["original", "translated"])
def test_metal_multiwave_reduction_matches_reference(
    tmp_path, metal_reduction_runner, variant, value_type
):
    entry = f"multiwave_reduce_{value_type}"
    if variant == "original":
        source = SOURCE
    else:
        source, artifact = _translate(tmp_path, "metal", value_type)
        assert artifact["entryPoint"]["target"] == entry
    artifact_path, module = _compile_metal(source, "metal", tmp_path)
    assert module.is_file()
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
        (tmp_path / f"readback-{repeat}.json").write_text(result.stdout)
        data = json.loads(result.stdout)
        assert data["threadExecutionWidth"] == WIDTH, data["device"]
        assert data["buffers"][0] == _inputs(value_type) + guard
        actual = data["values"]
        assert actual[len(expected) :] == [0xDEADBEEF] * len(guard)
        if actual[: len(expected)] != expected:
            differences = _differences(
                actual[: len(expected)], value_type, ordered_subgroups=True
            )
            failures[repeat] = {"count": len(differences), "first": differences[:8]}
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "artifactSha256": (
                    hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                ),
                "moduleSha256": hashlib.sha256(module.read_bytes()).hexdigest(),
                "device": data["device"],
                "failures": failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert not failures, failures
