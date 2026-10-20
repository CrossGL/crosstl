"""64-bit integer elements of HLSL structured buffers translated to Metal.

The Metal type map spells 64-bit scalars ``int64_t``/``uint64_t`` while the
HLSL layout planner works with the Metal ``long``/``ulong`` keywords. Both
spellings must reach the planner as the same 8-byte scalar. The expected
offsets are the DXC 1.9 ``cs_6_6`` reflection of ``Wide``: ``Tag`` 0,
``Value`` 8, ``Pair`` 16, element size 32.
"""

from tests.test_backend.test_directx.test_hlsl_buffer_layouts import (
    compile_metal_with_layout,
    generate_metal,
)

WIDE_INTEGER_BUFFERS = """
    struct Wide
    {
        uint Tag;
        int64_t Value;
        uint64_t2 Pair;
    };

    RWStructuredBuffer<int64_t> signedValues : register(u0);
    RWStructuredBuffer<uint64_t> unsignedValues : register(u1);
    RWStructuredBuffer<int64_t2> pairs : register(u2);
    RWStructuredBuffer<Wide> items : register(u3);

    [numthreads(1, 1, 1)]
    void Execute(uint3 id : SV_DispatchThreadID)
    {
        signedValues[id.x] = signedValues[id.x] * 3;
        unsignedValues[id.x] = unsignedValues[id.x] + 1u;
        pairs[id.x] = pairs[id.x].yx;
        items[id.x].Value = items[id.x].Value + (int64_t)items[id.x].Tag;
        items[id.x].Pair = items[id.x].Pair + unsignedValues[id.x];
    }
"""


def test_metal_structured_buffers_accept_64_bit_integer_scalars(tmp_path):
    metal = generate_metal(WIDE_INTEGER_BUFFERS)

    assert "device long* signedValues [[buffer(0)]]" in metal
    assert "device ulong* unsignedValues [[buffer(1)]]" in metal
    assert "device long2* pairs [[buffer(2)]]" in metal
    assert "device Wide* items [[buffer(3)]]" in metal
    compile_metal_with_layout(
        metal,
        tmp_path,
        offsets=[("Wide", "Tag", 0), ("Wide", "Value", 8), ("Wide", "Pair", 16)],
        sizes=[("Wide", 32)],
    )


def test_metal_struct_members_place_64_bit_integers_at_hlsl_offsets():
    metal = generate_metal(WIDE_INTEGER_BUFFERS)

    assert (
        "struct Wide {\n    uint Tag;\n    char _pad1[4];\n    long Value;\n"
        "    packed_ulong2 Pair;\n" in metal
    )
