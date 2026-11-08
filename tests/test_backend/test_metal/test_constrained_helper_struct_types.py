"""Struct templates in the signature of a materialized constrained helper name their concrete structure.

``template <typename T, enable_if_t<P<T>, bool> = true> void add(device Wrapper<T>* object, ...)`` is materialized
after the pass that replaces concrete struct template-ids with their specializations. The helper therefore kept
``device Wrapper<float>*`` while its caller passed ``device Wrapper_float_void*``: Metal rejected the call, and the
other targets named an undeclared template. Each case below fails on that code.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl

SOURCE = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    template <typename T>
    constexpr constant bool is_atomic_scalar = is_same_v<T, int> || is_same_v<T, uint> || is_same_v<T, float>;

    template <typename T, typename = void>
    struct Wrapper;

    template <typename T>
    struct Wrapper<T, enable_if_t<is_atomic_scalar<T>>> {
      atomic<T> val;
    };

    template <typename T, enable_if_t<is_atomic_scalar<T>, bool> = true>
    void add(device Wrapper<T>* object, T val, size_t offset) {
      atomic_fetch_add_explicit(&(object[offset].val), val, memory_order_relaxed);
    }

    template <typename T>
    [[kernel]] void accumulate(
        device Wrapper<T>* out [[buffer(0)]],
        device const T* in [[buffer(1)]],
        uint i [[thread_position_in_grid]]) {
      auto base = out + i;
      add(base, in[i], 0);
    }

    template [[host_name("accumulate_float")]] [[kernel]] decltype(accumulate<float>) accumulate<float>;
""")


def _translate(tmp_path, backend):
    path = tmp_path / "accumulate.metal"
    path.write_text(SOURCE)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_the_helper_and_its_caller_name_one_structure(tmp_path):
    output = _translate(tmp_path, "metal")

    assert "Wrapper<" not in output
    assert "void add_float(device Wrapper_float_void* object" in output
    assert "device Wrapper_float_void* out_ [[buffer(0)]]" in output
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")
    artifact = tmp_path / "out.metal"
    artifact.write_text(output)
    result = subprocess.run(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(artifact), "-o", "/dev/null"],
        capture_output=True,
        text=True,
    )
    if "unable to find utility" in result.stderr:
        pytest.skip("the Metal toolchain is not installed")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("backend", ["cgl", "directx"])
def test_no_target_names_the_template(tmp_path, backend):
    assert "Wrapper<" not in _translate(tmp_path, backend)
