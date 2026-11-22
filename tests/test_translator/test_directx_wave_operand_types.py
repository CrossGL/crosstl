"""HLSL wave intrinsic operands take their own type context.

Only the value operand of a value-preserving wave intrinsic shares the call's
result type. A lane index is ``uint`` and a partition mask is ``uint4``
whatever the result type is, so a ``WaveGetLaneIndex()`` or ``WaveMatch()``
inside them must not be checked against a ``float4`` or ``float`` result.
"""

import pytest

from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from tests.test_translator.test_directx_float_atomics import _compile

SOURCE = """
shader WaveOperands {{
    compute {{
        RWStructuredBuffer<float4> data @buffer(0);
        RWStructuredBuffer<uint4> masks @buffer(1);
        RWStructuredBuffer<uint> counts @buffer(2);
        RWStructuredBuffer<float> sums @buffer(3);
        @stage_entry
        @workgroup_size(32, 1, 1)
        void k(uvec3 id @gl_GlobalInvocationID) {{
            {statement}
        }}
    }}
}}
"""

CASES = {
    "read-lane-at": (
        "data[id.x] = WaveReadLaneAt(data[id.x], "
        "(WaveGetLaneIndex() + 1u) & (WaveGetLaneCount() - 1u));",
        "WaveReadLaneAt(data[id.x], ((WaveGetLaneIndex() + 1u) & "
        "(WaveGetLaneCount() - 1u)))",
    ),
    "quad-read-lane-at": (
        "data[id.x] = QuadReadLaneAt(data[id.x], WaveGetLaneIndex() & 3u);",
        "QuadReadLaneAt(data[id.x], (WaveGetLaneIndex() & 3u))",
    ),
    "multi-prefix-mask": (
        "sums[id.x] = WaveMultiPrefixSum(sums[id.x], "
        "WaveMatch(WaveGetLaneIndex() & 1u));",
        "WaveMultiPrefixSum(sums[id.x], WaveMatch((WaveGetLaneIndex() & 1u)))",
    ),
    "ballot-condition": (
        "masks[id.x] = WaveActiveBallot(WaveGetLaneIndex() < 4u);",
        "WaveActiveBallot((WaveGetLaneIndex() < 4u))",
    ),
    "count-bits-condition": (
        "counts[id.x] = WaveActiveCountBits(WaveGetLaneIndex() > 1u);",
        "WaveActiveCountBits((WaveGetLaneIndex() > 1u))",
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_wave_operand_keeps_its_own_type(case, tmp_path):
    statement, expected = CASES[case]
    generated = HLSLCodeGen().generate(parse(SOURCE.format(statement=statement)))

    assert expected in generated
    _compile(generated, tmp_path, profile="cs_6_5")
