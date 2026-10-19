"""Direct3D runtime controls for guarded groupshared writes read back by slot.

Every kernel here is fixed HLSL text, so CrossTL code generation is not
involved: a failure isolates the DXC and Direct3D runtime pair that executed
it. The kernels follow the shape of a 128-invocation blocked scan that
emulates 32-lane subgroups through groupshared scratch:

* each invocation scans four elements and exchanges partials with an
  inclusive shuffle-up through a scratch array, four barriers per step;
* lane 31 of each 32-invocation group publishes its total to a slot;
* group 0 alone reads every slot, scans them and writes them back under the
  same guard (slots 4..31 are never published and hold unspecified values);
* after a barrier every invocation reads the slot of its own group.

HLSL requires ``GroupMemoryBarrierWithGroupSync`` to make every groupshared
write before it visible to every invocation of the group after it, so the
slot read must return the value group 0 wrote there. The guarded-slot control
keeps only that guarded write, barrier and uniform-slot read.
"""

from string import Template

import pytest

from tests.test_translator.test_directx_float_atomics import _compile
from tests.test_translator.test_directx_software_reductions import (
    _execute_words,
)
from tests.test_translator.test_directx_software_reductions import (
    directx_runtime as _directx_runtime,
)

directx_runtime = _directx_runtime
GROUP = 128
WIDTH = 32
READS = 4
SENTINEL = 0xFFFFFFFF
# Three rows of a 37-element axis: one long run of set values, one row whose
# first set value lies in lane 4 and one almost entirely set row.
ROWS = (
    "1011110010110000100000110100101000111",
    "0000000000000000001000000000010000000",
    "1101110111111111111111111111111111111",
)
AXIS = len(ROWS[0])
SCAN_VARIANTS = {
    # (element type, slot storage type)
    "bool-in-uint-slots": ("bool", "uint"),
    "bool-in-bool-slots": ("bool", "bool"),
    "uint-in-uint-slots": ("uint", "uint"),
}

_SCAN = Template("""StructuredBuffer<uint> inputWords : register(t0);
RWStructuredBuffer<uint> outputWords : register(u1);

static uint softwareInvocation;
groupshared $slot slots[32];
groupshared uint scratch[128];

uint shuffleAndFillUpWords(uint value, uint fill, uint delta, uint modulo, uint invocation) {
    uint lane = invocation % 32u;
    uint subgroupBase = invocation - lane;
    scratch[invocation] = value;
    GroupMemoryBarrierWithGroupSync();
    bool validModulo = modulo == 2u || modulo == 4u || modulo == 8u || modulo == 16u || modulo == 32u;
    uint segmentWidth = validModulo ? modulo : 32u;
    uint segmentLane = lane % segmentWidth;
    bool useValue = segmentLane >= delta;
    uint sourceLane = lane;
    if (validModulo && delta <= segmentWidth) {
        sourceLane = useValue ? lane - delta : lane + segmentWidth - delta;
    }
    bool sourceValid = validModulo && delta <= segmentWidth && sourceLane < 32u;
    sourceLane = sourceValid ? sourceLane : lane;
    uint result = scratch[subgroupBase + sourceLane];
    GroupMemoryBarrierWithGroupSync();
    scratch[invocation] = fill;
    GroupMemoryBarrierWithGroupSync();
    if (sourceValid && !useValue) {
        result = scratch[subgroupBase + sourceLane];
    }
    GroupMemoryBarrierWithGroupSync();
    return result;
}

$value shuffleAndFillUp($value data, $value filling, uint16_t delta) {
    return $from_word(shuffleAndFillUpWords(uint(data), uint(filling), uint((uint(delta) & 65535u)), 32u, softwareInvocation));
}

$value combine($value a, $value b) {
    return ((a >= b) ? a : b);
}

$value inclusiveScan($value x) {
    for (int i = 1; (i <= 16); i *= 2) {
        $value other = shuffleAndFillUp(x, $init, uint16_t(i));
        x = combine(x, other);
    }
    return x;
}

$value exclusiveScan($value x) {
    x = inclusiveScan(x);
    return shuffleAndFillUp(x, $init, uint16_t(1));
}

[numthreads(128, 1, 1)]
void CSMain(uint3 lid : SV_GroupThreadID, uint3 group : SV_GroupID, uint groupIndex : SV_GroupIndex) {
    uint lane = groupIndex % 32u;
    uint wave = groupIndex / 32u;
    softwareInvocation = groupIndex;
    uint axis = inputWords[0];
    uint rowOffset = group.x * axis;
    uint waves = 128u / 32u;
    $value prefix = $init;
    $value values[4];
    for (uint r = 0; r < (axis + 511u) / 512u; r++) {
        uint offset = r * 512u + lid.x * 4u;
        for (int i = 0; i < 4; i++) {
            values[i] = (offset + i < axis) ? $from_word(inputWords[1u + rowOffset + offset + i]) : $init;
        }
        for (int i = 1; i < 4; i++) {
            values[i] = combine(values[i], values[i - 1]);
        }
        $value prevThread = exclusiveScan(values[3]);
        GroupMemoryBarrierWithGroupSync();
        if (lane == 31u) {
            slots[wave] = combine(prevThread, values[3]);
        }
        GroupMemoryBarrierWithGroupSync();
        {
            bool enabled = wave == 0u;
            $value slotInput = $init;
            if (enabled) {
                slotInput = slots[lane];
            }
            $value prevWave = exclusiveScan(slotInput);
            if (enabled) {
                slots[lane] = prevWave;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        for (int i = 0; i < 4; i++) {
            values[i] = combine(values[i], prefix);
            values[i] = combine(values[i], slots[wave]);
            values[i] = combine(values[i], prevThread);
        }
        for (int i = 0; i < 4; i++) {
            if (offset + i < axis) {
                outputWords[rowOffset + offset + i] = uint(values[i]);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if ((wave == (waves - 1u)) && (lane == 31u)) {
            slots[0] = values[3];
        }
        GroupMemoryBarrierWithGroupSync();
        prefix = slots[0];
    }
}
""")

_GUARDED_SLOT = """StructuredBuffer<uint> inputWords : register(t0);
RWStructuredBuffer<uint> outputWords : register(u1);
groupshared uint slots[32];

[numthreads(128, 1, 1)]
void CSMain(uint3 group : SV_GroupID, uint groupIndex : SV_GroupIndex) {
    uint lane = groupIndex % 32u;
    uint wave = groupIndex / 32u;
    uint index = group.x * 128u + groupIndex;
    bool value = inputWords[index] != 0u;
    GroupMemoryBarrierWithGroupSync();
    if (wave == 0u) {
        slots[lane] = uint(value);
    }
    GroupMemoryBarrierWithGroupSync();
    outputWords[index] = slots[wave];
}
"""


def _scan_source(variant):
    value, slot = SCAN_VARIANTS[variant]
    return _SCAN.substitute(
        value=value,
        slot=slot,
        init="false" if value == "bool" else "0u",
        from_word="bool" if value == "bool" else "uint",
    )


def _scan_words():
    return [AXIS] + [int(bit) for row in ROWS for bit in row]


def _scan_expected():
    expected = []
    for row in ROWS:
        running = 0
        for bit in row:
            running |= int(bit)
            expected.append(running)
    return expected


def _guarded_slot_words():
    pattern = "0110100110010110"
    return [int(pattern[index % len(pattern)]) for index in range(GROUP * 3)]


def _guarded_slot_expected(words):
    return [
        words[group * GROUP + (tid // WIDTH)]
        for group in range(3)
        for tid in range(GROUP)
    ]


def _differences(actual, expected):
    return [
        (index, want, got)
        for index, (got, want) in enumerate(zip(actual, expected))
        if got != want
    ]


def _execute(tmp_path, source, words, expected):
    artifact, module = _compile(
        source, tmp_path, profile="cs_6_2", flags=("-enable-16bit-types",)
    )
    return _execute_words(
        tmp_path,
        source,
        words,
        [SENTINEL] * len(expected),
        workgroup_size=(GROUP, 1, 1),
        workgroup_count=(3, 1, 1),
        module_compiler=lambda *_args, **_kwargs: (artifact, module),
    )


def test_scan_oracle_reads_the_first_set_value_of_each_row():
    expected = _scan_expected()
    # Row 1 first sets index 18, which the fourth element of lane 4 holds.
    assert expected[AXIS : AXIS + 19] == [0] * 18 + [1]
    assert len(expected) == 3 * AXIS and len(_scan_words()) == 3 * AXIS + 1


def test_guarded_slot_oracle_rejects_own_slot_values():
    words = _guarded_slot_words()
    expected = _guarded_slot_expected(words)
    own = [
        words[group * GROUP + tid % WIDTH] for group in range(3) for tid in range(GROUP)
    ]
    assert _differences(own, expected)


@pytest.mark.parametrize("variant", sorted(SCAN_VARIANTS))
def test_guarded_slot_scan_compiles(tmp_path, variant):
    source = _scan_source(variant)
    # Four barriers per scratch shuffle and five in the block loop.
    assert source.count("GroupMemoryBarrierWithGroupSync();") == 9
    assert source.count("slots[lane] = prevWave;") == 1
    _compile(source, tmp_path, profile="cs_6_2", flags=("-enable-16bit-types",))


@pytest.mark.parametrize("variant", sorted(SCAN_VARIANTS))
def test_guarded_slot_scan_executes(tmp_path, directx_runtime, variant):
    expected = _scan_expected()
    actual = _execute(tmp_path, _scan_source(variant), _scan_words(), expected)
    differences = _differences(actual, expected)
    assert not differences, (len(differences), differences[:8])


def test_guarded_slot_readback_executes(tmp_path, directx_runtime):
    words = _guarded_slot_words()
    expected = _guarded_slot_expected(words)
    actual = _execute(tmp_path, _GUARDED_SLOT, words, expected)
    differences = _differences(actual, expected)
    assert not differences, (len(differences), differences[:8])
