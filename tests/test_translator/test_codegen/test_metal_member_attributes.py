"""Argument buffer ``[[id(n)]]`` attributes on array members.

An attribute after an array declarator's extent appertains to the array type, so the Metal compiler rejects
``device float* params[4] [[id(0)]];`` ("'id' attribute cannot be applied to types"). The member's attribute goes
right after its name: ``device float* params [[id(0)]][4];``.
"""

import crosstl
from crosstl.translator.codegen.metal_member_attributes import attach_member_attributes

SOURCE = """#include <metal_stdlib>
using namespace metal;
struct Args {
  metal::array<device float*, 4> params [[id(0)]];
  device float* grads[4] [[id(4)]];
  device float* single [[id(8)]];
};
kernel void k(device Args& args [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  args.params[i][0] = args.grads[i][0] + args.single[0];
}
"""


def test_attributes_follow_the_declarator_name():
    assert (
        attach_member_attributes("device float *params[4]", "params", " [[id(0)]]")
        == "device float *params [[id(0)]][4]"
    )
    assert (
        attach_member_attributes("device float *single", "single", " [[id(8)]]")
        == "device float *single [[id(8)]]"
    )
    assert attach_member_attributes("float v[2]", "v", "") == "float v[2]"


def test_metal_array_members_keep_their_argument_buffer_ids(tmp_path):
    path = tmp_path / "args.metal"
    path.write_text(SOURCE)
    metal = crosstl.translate(
        str(path), backend="metal", format_output=False, source_backend="metal"
    )
    assert "device float* params [[id(0)]][4];" in metal
    assert "device float* grads [[id(4)]][4];" in metal
    assert "device float* single [[id(8)]];" in metal
    assert "[4] [[id(" not in metal
