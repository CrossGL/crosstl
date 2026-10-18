"""Direct3D runtime controls for partially active waves and wide workgroups.

Every kernel here is fixed HLSL text, so CrossTL code generation is not
involved: a failure isolates the DXC and Direct3D runtime pair that executed
it. Each control asserts only what the HLSL specification requires.

* Wave intrinsics (Shader Model 6.0): ``WaveActiveSum`` returns the sum of its
  operand over the *active* lanes of the current wave. Inactive lanes do not
  contribute.
* ``[WaveSize(N)]`` (Shader Model 6.6): the shader executes with exactly
  ``N`` lanes per wave, or pipeline creation fails.

The oracles derive wave membership from the readback, because the
specification does not fix how invocations are packed into waves.
"""

import struct

import pytest

from tests.test_translator.test_directx_float_atomics import _compile
from tests.test_translator.test_directx_software_reductions import (
    _execute_words,
)
from tests.test_translator.test_directx_software_reductions import (
    directx_runtime as _directx_runtime,
)

directx_runtime = _directx_runtime
WIDTH = 32
ACTIVE = 16
SENTINEL = 0xFFFFFFFF
VALUE_TYPES = ("uint", "int", "float")
OPERAND_SOURCES = ("buffer", "groupshared")
WIDE_GROUP = 512
WIDE_GROUPS = 3

_HALF_WAVE = """StructuredBuffer<uint> inputWords : register(t0);
RWStructuredBuffer<uint> outputWords : register(u1);
groupshared uint staged[{group}];

[WaveSize(32)]
[numthreads({group}, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint group : SV_GroupID) {{
    uint index = group * {group}u + tid;
    uint lane = WaveGetLaneIndex();
    uint wave = WaveReadLaneFirst(index);
    staged[tid] = inputWords.Load(index);
    GroupMemoryBarrierWithGroupSync();
    uint total = 0xFFFFFFFFu;
    if (lane < 16u) {{
        {value_type} operand = {cast}({operand});
        total = asuint(WaveActiveSum(operand));
    }}
    outputWords[index * 3u] = wave;
    outputWords[index * 3u + 1u] = lane;
    outputWords[index * 3u + 2u] = total;
}}
"""

_WIDE_EXECUTION = """StructuredBuffer<uint> inputWords : register(t0);
RWStructuredBuffer<uint> outputWords : register(u1);

{wave_size}[numthreads(512, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint group : SV_GroupID) {{
    uint index = group * 512u + tid;
    outputWords[index * 2u] = inputWords.Load(0) | index;
    outputWords[index * 2u + 1u] = WaveGetLaneCount();
}}
"""

# The arrival-order wave numbering that a 512-invocation workgroup needs when
# a source language exposes the subgroup index within the workgroup.
_ARRIVAL_COUNTER = """StructuredBuffer<uint> inputWords : register(t0);
RWStructuredBuffer<uint> outputWords : register(u1);
groupshared uint arrivals;

[WaveSize(32)]
[numthreads(512, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint group : SV_GroupID) {
    if (tid == 0u) {
        arrivals = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    uint wave = 0u;
    if (WaveIsFirstLane()) {
        InterlockedAdd(arrivals, 1u, wave);
    }
    wave = WaveReadLaneFirst(wave);
    GroupMemoryBarrierWithGroupSync();
    uint index = group * 512u + tid;
    outputWords[index * 2u] = inputWords.Load(0) | wave;
    outputWords[index * 2u + 1u] = WaveGetLaneIndex();
}
"""


def _float_word(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _operand_words(value_type, count):
    """Distinct operands whose partial sums are exact in binary32."""

    if value_type == "uint":
        return [
            ((index + 1) * 0x01010101 + index * 7) & SENTINEL for index in range(count)
        ]
    if value_type == "int":
        return [
            ((index * 977 - 40000) * (1 if index % 3 else -1)) & SENTINEL
            for index in range(count)
        ]
    return [_float_word((index % 61) * 4 + 1 + (index // 61)) for index in range(count)]


def _operand_value(word, value_type):
    if value_type == "uint":
        return word
    if value_type == "int":
        return word - (1 << 32) if word & 0x80000000 else word
    return struct.unpack("<f", struct.pack("<I", word))[0]


def _sum_word(words, value_type):
    total = sum(_operand_value(word, value_type) for word in words)
    return _float_word(total) if value_type == "float" else total & SENTINEL


def _half_wave_differences(actual, words, value_type, invocations):
    """Check every wave's half-active sum against its own active lanes."""

    records = [actual[index * 3 : index * 3 + 3] for index in range(invocations)]
    waves = {}
    for index, (wave, lane, _total) in enumerate(records):
        waves.setdefault(wave, []).append((lane, index))
    differences = []
    for wave, members in sorted(waves.items()):
        lanes = sorted(lane for lane, _ in members)
        if lanes != list(range(WIDTH)):
            differences.append({"wave": wave, "lanes": lanes})
            continue
        active = [index for lane, index in members if lane < ACTIVE]
        expected = _sum_word([words[index] for index in active], value_type)
        for lane, index in members:
            want = expected if lane < ACTIVE else SENTINEL
            if records[index][2] != want:
                differences.append(
                    {
                        "wave": wave,
                        "lane": lane,
                        "expected": want,
                        "actual": records[index][2],
                    }
                )
    return differences


def _run(tmp_path, source, words, expected_length, *, group, groups):
    artifact, module = _compile(source, tmp_path, profile="cs_6_6")
    return _execute_words(
        tmp_path,
        source,
        words,
        [SENTINEL] * expected_length,
        workgroup_size=(group, 1, 1),
        workgroup_count=(groups, 1, 1),
        module_compiler=lambda *_args, **_kwargs: (artifact, module),
    )


def _half_wave_source(value_type, operand_source, group):
    operand = (
        "staged[tid]" if operand_source == "groupshared" else "inputWords.Load(index)"
    )
    cast = {"uint": "", "int": "asint", "float": "asfloat"}[value_type]
    return _HALF_WAVE.format(
        group=group, value_type=value_type, cast=cast, operand=operand
    )


@pytest.mark.parametrize("operand_source", OPERAND_SOURCES)
@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_half_wave_sum_compiles(tmp_path, value_type, operand_source):
    source = _half_wave_source(value_type, operand_source, WIDTH)
    assert source.count("WaveActiveSum(") == 1
    _compile(source, tmp_path, profile="cs_6_6")


def test_half_wave_oracle_rejects_inactive_lane_contributions():
    words = _operand_words("uint", WIDTH)
    readback = []
    for lane in range(WIDTH):
        total = _sum_word(words[:ACTIVE], "uint") if lane < ACTIVE else SENTINEL
        readback.extend([0, lane, total])
    assert _half_wave_differences(readback, words, "uint", WIDTH) == []
    polluted = list(readback)
    polluted[2] = _sum_word(words, "uint")
    assert _half_wave_differences(polluted, words, "uint", WIDTH) == [
        {"wave": 0, "lane": 0, "expected": readback[2], "actual": polluted[2]}
    ]


@pytest.mark.parametrize(
    "group", (WIDTH, WIDE_GROUP), ids=("one-wave", "sixteen-waves")
)
@pytest.mark.parametrize("operand_source", OPERAND_SOURCES)
@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_half_wave_sum_executes(
    tmp_path, directx_runtime, value_type, operand_source, group
):
    groups = 1 if group == WIDTH else WIDE_GROUPS
    invocations = group * groups
    words = _operand_words(value_type, invocations)
    source = _half_wave_source(value_type, operand_source, group)
    actual = _run(tmp_path, source, words, invocations * 3, group=group, groups=groups)
    differences = _half_wave_differences(actual, words, value_type, invocations)
    assert not differences, differences[:8]


@pytest.mark.parametrize(
    "wave_size", (None, WIDTH), ids=("default-width", "wave-size-32")
)
def test_wide_workgroup_executes(tmp_path, directx_runtime, wave_size):
    attribute = "" if wave_size is None else f"[WaveSize({wave_size})]\n"
    source = _WIDE_EXECUTION.format(wave_size=attribute)
    marker = 0xC0DE0000
    actual = _run(
        tmp_path,
        source,
        [marker],
        WIDE_GROUPS * WIDE_GROUP * 2,
        group=WIDE_GROUP,
        groups=WIDE_GROUPS,
    )
    markers = actual[0::2]
    widths = sorted(set(actual[1::2]))
    assert markers == [marker | index for index in range(WIDE_GROUPS * WIDE_GROUP)]
    assert len(widths) == 1, widths
    if wave_size is not None:
        assert widths == [wave_size]


def test_wide_workgroup_arrival_counter_executes(tmp_path, directx_runtime):
    marker = 0xA0000000
    actual = _run(
        tmp_path,
        _ARRIVAL_COUNTER,
        [marker],
        WIDE_GROUPS * WIDE_GROUP * 2,
        group=WIDE_GROUP,
        groups=WIDE_GROUPS,
    )
    for group in range(WIDE_GROUPS):
        records = actual[group * WIDE_GROUP * 2 : (group + 1) * WIDE_GROUP * 2]
        identities = records[0::2]
        assert all(word & 0xFFFF0000 == marker for word in identities), identities[:4]
        waves = {}
        for identity, lane in zip(identities, records[1::2]):
            waves.setdefault(identity & 0xFFFF, []).append(lane)
        assert sorted(waves) == list(range(WIDE_GROUP // WIDTH)), sorted(waves)
        assert all(sorted(lanes) == list(range(WIDTH)) for lanes in waves.values())
