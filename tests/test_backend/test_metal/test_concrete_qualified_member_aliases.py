"""A dependent member type substituted into a materialized function resolves to its alias target.

``twiddle<scalar_T>`` inside the kernel template below is materialized with
``scalar_T`` replaced by ``typename IOTraits<float2, float2>::scalar_T`` before
``IOTraits<float2, float2>`` itself is materialized.  The helper body then
spelled ``typename IOTraits_float2_float2::scalar_T(0)``, a functional cast the
Metal parser rejects, so the Metal-to-Metal translation failed although the
owner was concrete and its alias names ``float``.
"""

import re

import pytest

import crosstl

_SOURCE = """
#include <metal_stdlib>
using namespace metal;

template <typename storage_T>
struct StorageTraits {
  using scalar_T = storage_T;
};

template <typename T>
struct StorageTraits<vec<T, 2>> {
  using scalar_T = T;
};

template <typename in_T, typename out_T>
struct IOTraits {
  using scalar_T = typename StorageTraits<in_T>::scalar_T;
};

template <typename T>
METAL_FUNC vec<T, 2> twiddle(int k, int p) {
  using phase_T = decltype(M_PI_F * T(0));
  phase_T theta = -phase_T(2) * phase_T(k) * phase_T(M_PI_F) / phase_T(p);
  return {T(metal::fast::cos(theta)), T(metal::fast::sin(theta))};
}

template <typename in_T, typename out_T>
[[kernel]] void run(device out_T* out [[buffer(0)]],
                    uint i [[thread_position_in_grid]]) {
  using scalar_T = typename IOTraits<in_T, out_T>::scalar_T;
  vec<scalar_T, 2> value = twiddle<scalar_T>(int(i), 8);
  out[i] = out_T(value);
}

template [[host_name("run_float2")]] [[kernel]] void run<float2, float2>(
    device float2*, uint);
template [[host_name("run_half2")]] [[kernel]] void run<half2, half2>(
    device half2*, uint);
"""


def _translate(tmp_path, backend):
    source = tmp_path / "twiddle.metal"
    source.write_text(_SOURCE)
    return crosstl.translate(
        str(source), backend=backend, source_backend="metal", format_output=False
    )


@pytest.mark.parametrize("backend", ["metal", "cgl", "directx"])
def test_concrete_owner_member_alias_resolves_in_materialized_helper(tmp_path, backend):
    output = _translate(tmp_path, backend)

    assert "IOTraits_float2_float2::" not in output
    assert "IOTraits_half2_half2::" not in output
    assert "scalar_T" not in output.replace("_scalar_T", "")


def test_metal_output_casts_to_each_concrete_scalar(tmp_path):
    output = _translate(tmp_path, "metal")

    assert "float(metal::fast::cos(theta))" in output
    assert "half(metal::fast::cos(theta))" in output
    assert re.search(r"\btypename\s", output) is None


_LOCAL_OWNER_SOURCE = """
#include <metal_stdlib>
using namespace metal;

template <typename T, typename AccT = float>
struct Kern {
  using acc_type = AccT;
  static constant constexpr const int tgp_mem_size = 4;
  static METAL_FUNC void run(const device T* in, device T* out,
                             threadgroup AccT* scratch, uint lid) {
    scratch[lid] = AccT(in[lid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[lid] = T(scratch[3 - lid]);
  }
};

template <typename T>
[[kernel]] void reverse(const device T* in [[buffer(0)]],
                        device T* out [[buffer(1)]],
                        uint lid [[thread_position_in_threadgroup]]) {
  using kernel_t = Kern<T>;
  threadgroup typename kernel_t::acc_type
      scratch[kernel_t::tgp_mem_size == 0 ? 1 : kernel_t::tgp_mem_size];
  kernel_t::run(in, out, scratch, lid);
}

template [[host_name("reverse_half")]] [[kernel]] void reverse<half>(
    const device half*, device half*, uint);
"""


@pytest.mark.parametrize("backend", ["metal", "cgl", "directx"])
def test_member_type_of_local_owner_alias_resolves(tmp_path, backend):
    source = tmp_path / "reverse.metal"
    source.write_text(_LOCAL_OWNER_SOURCE)

    output = crosstl.translate(
        str(source), backend=backend, source_backend="metal", format_output=False
    )

    assert "acc_type" not in output
    assert "kernel_t" not in output
    if backend == "metal":
        assert "threadgroup float scratch[" in output
