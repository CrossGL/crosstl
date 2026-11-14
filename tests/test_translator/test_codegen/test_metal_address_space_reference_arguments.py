"""Reference arguments of parameters in an explicit address space.

CrossGL ``inout device float v`` is Metal's ``device float& v``: it binds the caller's device lvalue in place. The
Metal generator copied such arguments into thread temporaries and stored them back, as it does for CrossGL
``inout`` parameters it lowers to ``thread T&``. A thread temporary cannot bind to a ``device`` or
``threadgroup`` reference, so the Metal compiler rejected every such call.
"""

import crosstl

SOURCE = """#include <metal_stdlib>
using namespace metal;
inline void bump(device float& v) { v += 1.0f; }
inline void tg(threadgroup float& v) { v *= 2.0f; }
inline void th(thread float& v) { v -= 1.0f; }
kernel void k(device float* data [[buffer(0)]], uint i [[thread_position_in_grid]],
              uint l [[thread_index_in_threadgroup]]) {
  threadgroup float tile[64];
  device float* p = data + i;
  bump(*p);
  bump(data[i + 1]);
  tile[l] = data[i];
  tg(tile[l]);
  float x = tile[l];
  th(x);
  data[i] = x;
}
"""


def test_device_and_threadgroup_references_bind_their_lvalues(tmp_path):
    path = tmp_path / "kernel.metal"
    path.write_text(SOURCE)
    metal = crosstl.translate(
        str(path), backend="metal", format_output=False, source_backend="metal"
    )
    assert "bump(*p);" in metal
    assert "bump(data[i + 1]);" in metal
    assert "tg(tile[l]);" in metal
    assert "th(x);" in metal
    assert "__crossgl_ref" not in metal


def test_crossgl_device_reference_parameter(tmp_path):
    path = tmp_path / "shader.cgl"
    path.write_text("""
shader S {
    void bump(inout device float v) { v += 1.0; }
    compute {
        void main(RWStructuredBuffer<float> data @buffer(0), uint i @gl_GlobalInvocationID) {
            device float* p = data + i;
            bump((*p));
        }
    }
}
""")
    metal = crosstl.translate(str(path), backend="metal", format_output=False)
    assert "void bump(device float& v)" in metal
    assert "bump(*p);" in metal
    assert "__crossgl_ref" not in metal
