"""``const auto`` locals initialized from pointers.

``const auto p = data + i;`` deduces ``device float* const``: the ``const`` qualifies the local pointer object, not
the storage it points to ([dcl.type.auto.deduct]/2, [dcl.spec.auto]/7). The importer used to spell it
``const device float* p``, so stores through ``p`` became stores through a pointer to const, which the Metal target
refuses and other compilers reject. ``const auto* t = data + i;`` still qualifies the pointee.
"""

import crosstl

SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void k(device float* data [[buffer(0)]], const device float* cdata [[buffer(1)]],
              constant float* kdata [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  const auto p = data + i;
  const auto q = cdata + i;
  const auto r = kdata + i;
  auto s = cdata + i;
  const auto* t = data + i;
  const auto v = data[i];
  *p = q[0] + r[0] + s[0] + t[0] + v;
}
"""


def test_const_auto_pointer_keeps_its_pointee_writable(tmp_path):
    path = tmp_path / "kernel.metal"
    path.write_text(SOURCE)
    cgl = crosstl.translate(
        str(path), backend="cgl", format_output=False, source_backend="metal"
    )
    assert "device float* p = data + i;" in cgl
    assert "const device float* p" not in cgl
    assert "const device float* q = cdata + i;" in cgl
    assert "constant float* r = kdata + i;" in cgl
    assert "const device float* s = cdata + i;" in cgl
    assert "const device float* t = data + i;" in cgl
    assert "const float v = buffer_load(data, i);" in cgl
