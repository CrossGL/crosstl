"""Member types named through a type alias of their struct.

``using kern = KH;`` makes ``kern`` a synonym of ``KH`` ([dcl.typedef]/2), so ``kern::acc_type`` and
``typename kern::acc_type`` name ``KH``'s member alias. The importer looked such members up by struct name only and
emitted the bare, undeclared ``acc_type``.
"""

import crosstl

SOURCE = """#include <metal_stdlib>
using namespace metal;
struct KH { using acc_type = float; static constant constexpr int mem = 32; };
using GK = KH;
kernel void k(const device half* in [[buffer(0)]], device half* out [[buffer(1)]],
              uint i [[thread_position_in_threadgroup]]) {
  using kern = KH;
  threadgroup typename kern::acc_type tgp[kern::mem];
  typename kern::acc_type x = in[i];
  kern::acc_type y = x;
  GK::acc_type z = y;
  typename GK::acc_type w = z;
  tgp[i] = w;
  out[i] = half(tgp[i]);
}
"""


def test_member_aliases_resolve_through_global_and_local_aliases(tmp_path):
    path = tmp_path / "kernel.metal"
    path.write_text(SOURCE)
    cgl = crosstl.translate(
        str(path), backend="cgl", format_output=False, source_backend="metal"
    )
    assert "threadgroup float[32] tgp;" in cgl
    for name in ("x", "y", "z", "w"):
        assert f"float {name} = " in cgl
    assert "acc_type" not in cgl.split("compute {", 1)[1]


TEMPLATE_SOURCE = """#include <metal_stdlib>
using namespace metal;
template <typename T> struct DefaultAccT { using type = float; };
template <typename T, const int N, typename AccT = typename DefaultAccT<T>::type>
struct Kern {
  using acc_type = AccT;
  static constant constexpr int mem = N;
  static METAL_FUNC void run(threadgroup AccT* tgp, const device T* in, device T* out, uint i) {
    tgp[i] = static_cast<AccT>(in[i]);
    out[i] = static_cast<T>(tgp[i]);
  }
};
template <typename T, const int N>
[[kernel]] void k(const device T* in [[buffer(0)]], device T* out [[buffer(1)]],
                  uint i [[thread_position_in_threadgroup]]) {
  using kern = Kern<T, N>;
  threadgroup typename kern::acc_type tgp[kern::mem];
  kern::run(tgp, in, out, i);
}
template [[host_name("k_half")]] [[kernel]] void k<half, 32>(
    const device half* in [[buffer(0)]], device half* out [[buffer(1)]],
    uint i [[thread_position_in_threadgroup]]);
"""


def test_member_alias_of_an_instantiated_struct_template(tmp_path):
    # The instantiation Kern<half, 32> is materialized after the kernel that names it.
    path = tmp_path / "kernel.metal"
    path.write_text(TEMPLATE_SOURCE)
    cgl = crosstl.translate(
        str(path), backend="cgl", format_output=False, source_backend="metal"
    )
    assert "threadgroup float[32] tgp;" in cgl
